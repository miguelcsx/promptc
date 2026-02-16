from __future__ import annotations

import json
from pathlib import Path

from promptc.engines.dspy_optimize import DSPyOptimizeEngine
from promptc.engines.evaluate import EvalEngine
from promptc.engines.intent import IntentEngine
from promptc.engines.optimize import OptimizeEngine
from promptc.engines.profile import ProfileEngine
from promptc.engines.render import OutputRenderer
from promptc.engines.signature import SignatureEngine
from promptc.engines.workflow import WorkflowEngine
from promptc.models import (
    CompileRequest,
    ConvergenceInfo,
    EvalExample,
    EvalReport,
    PolicyConfig,
    PromptArtifact,
    RuntimeConfig,
)
from promptc.storage.interfaces import (
    ArtifactRepo,
    CacheRepo,
    ProfileRepo,
    SignatureRepo,
    WorkflowRepo,
)
from promptc.utils import now_utc, stable_hash


class CompilerService:
    def __init__(
        self,
        profile_repo: ProfileRepo,
        workflow_repo: WorkflowRepo,
        signature_repo: SignatureRepo,
        artifact_repo: ArtifactRepo,
        cache_repo: CacheRepo,
        intent_engine: IntentEngine,
        profile_engine: ProfileEngine,
        signature_engine: SignatureEngine,
        workflow_engine: WorkflowEngine,
        optimize_engine: OptimizeEngine,
        eval_engine: EvalEngine,
        dspy_optimize_engine: DSPyOptimizeEngine,
        output_renderer: OutputRenderer,
        runtime: RuntimeConfig,
        policy: PolicyConfig,
    ) -> None:
        self.profile_repo = profile_repo
        self.workflow_repo = workflow_repo
        self.signature_repo = signature_repo
        self.artifact_repo = artifact_repo
        self.cache_repo = cache_repo
        self.intent_engine = intent_engine
        self.profile_engine = profile_engine
        self.signature_engine = signature_engine
        self.workflow_engine = workflow_engine
        self.optimize_engine = optimize_engine
        self.eval_engine = eval_engine
        self.dspy_optimize_engine = dspy_optimize_engine
        self.output_renderer = output_renderer
        self.runtime = runtime
        self.policy = policy

    def compile(self, req: CompileRequest) -> tuple[PromptArtifact, str]:
        profile = self.profile_repo.get(req.profile_id)
        workflow = self.workflow_repo.get(req.workflow_id)
        workflow_issues = self.workflow_engine.validate(workflow)

        cache_key = stable_hash(
            {
                "request": req.model_dump(mode="json"),
                "profile_snapshot": profile.model_dump(mode="json"),
                "workflow_snapshot": workflow.model_dump(mode="json"),
                "policy_snapshot": self.policy.model_dump(mode="json"),
                "runtime_snapshot": {
                    "provider": self.runtime.provider,
                    "model": self.runtime.model,
                    "role_providers": self.runtime.role_providers,
                    "role_models": self.runtime.role_models,
                    "role_temperatures": self.runtime.role_temperatures,
                    "temperature": self.runtime.temperature,
                    "judge_count": self.runtime.judge_count,
                    "optimizer": self.runtime.optimizer,
                    "rubric_version": self.runtime.rubric_version,
                },
                "prompt_pipeline_version": 7,
            }
        )
        cached_path = self.cache_repo.get(cache_key) if self.runtime.use_cache else None
        if cached_path:
            cached_artifact = self.artifact_repo.get_by_path(cached_path)
            cached_artifact.scores["cache_hit"] = 1.0
            return cached_artifact, cached_path

        outcome = self.intent_engine.infer(req.intent, req.raw_context, req.workflow_id)

        suitable, reasons = self.profile_engine.assess_suitability(profile, outcome)
        if not suitable:
            profile = self.profile_engine.refine_profile(
                profile, outcome, reasons, workflow=workflow
            )
            self.profile_repo.put(profile)

        signature = self.signature_engine.build(outcome, workflow)
        prev = self.signature_repo.get_latest(signature.signature_id)
        if prev:
            signature = self.signature_engine.bump(prev, outcome)
        self.signature_repo.put(signature)

        self.dspy_optimize_engine.compile_if_configured(
            optimizer_name=self.runtime.optimizer,
            dataset_path=self.runtime.eval_dataset_path,
        )

        candidates = self.optimize_engine.generate_candidates(
            outcome=outcome,
            profile=profile,
            workflow=workflow,
            output_format=req.output_format,
            n=req.variants,
        )

        best_candidate = candidates[0]
        best_score = self.policy.evaluation.score_floor
        no_improve_rounds = 0
        patience = self.policy.search.plateau_patience
        convergence = ConvergenceInfo(
            iterations=0, stopped_reason="max_iters", frontier_size=0
        )

        for i in range(1, req.max_iters + 1):
            scored = self.eval_engine.score_many(candidates, outcome, profile, workflow)
            frontier = _pareto_frontier(scored)
            convergence.frontier_size = len(frontier)
            current_best = sorted(frontier, key=lambda c: c.total_score, reverse=True)[
                0
            ]

            if (
                current_best.total_score
                > best_score + self.policy.search.improve_epsilon
            ):
                best_score = current_best.total_score
                best_candidate = current_best
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1

            convergence.iterations = i
            if no_improve_rounds >= patience:
                convergence.stopped_reason = "score_plateau"
                break

            refined_pool = [best_candidate]
            for c in scored[: req.variants - 1]:
                refined_pool.append(
                    self.optimize_engine.refine(
                        candidate=c,
                        outcome=outcome,
                        profile=profile,
                        workflow=workflow,
                        output_format=req.output_format,
                        iteration=i,
                    )
                )
            candidates = refined_pool

        rendered_prompt = self.output_renderer.render(
            best_candidate.prompt, req.output_format
        )
        artifact_id = stable_hash(
            {
                "intent": req.intent,
                "workflow": workflow.id,
                "profile": profile.id,
                "prompt": rendered_prompt,
                "time": now_utc().isoformat(),
                "seed": req.seed,
            }
        )

        lineage_depth = 0
        if req.parent_artifact_id:
            parent = self.artifact_repo.get(req.parent_artifact_id)
            lineage_depth = parent.lineage_depth + 1

        tests = (
            self._emit_tests(rendered_prompt, workflow.required_sections)
            if req.emit_tests
            else []
        )

        artifact = PromptArtifact(
            artifact_id=artifact_id,
            created_at=now_utc(),
            intent=req.intent,
            workflow=workflow.id,
            profile=f"{profile.id}@{profile.version}",
            output_format=req.output_format,
            assumptions=outcome.assumptions,
            constraints=outcome.constraints,
            prompt_final=rendered_prompt,
            prompt_variants=[
                self.output_renderer.render(c.prompt, req.output_format)
                for c in candidates
            ],
            scores={
                "final": round(
                    best_candidate.total_score, self.policy.search.score_precision
                ),
                "quality": round(
                    best_candidate.judge_scores.get(
                        "quality", self.policy.evaluation.score_floor
                    ),
                    self.policy.search.score_precision,
                ),
                "clarity": round(
                    best_candidate.judge_scores.get(
                        "clarity", self.policy.evaluation.score_floor
                    ),
                    self.policy.search.score_precision,
                ),
                "constraint_fit": round(
                    best_candidate.judge_scores.get(
                        "constraint_fit", self.policy.evaluation.score_floor
                    ),
                    self.policy.search.score_precision,
                ),
                "bloat_penalty": best_candidate.deterministic_penalties.get(
                    "bloat", self.policy.evaluation.score_floor
                ),
                "ambiguity_penalty": best_candidate.deterministic_penalties.get(
                    "ambiguity",
                    self.policy.evaluation.score_floor,
                ),
                "redundancy_penalty": best_candidate.deterministic_penalties.get(
                    "redundancy",
                    self.policy.evaluation.score_floor,
                ),
                "technique_penalty": best_candidate.deterministic_penalties.get(
                    "missing_techniques",
                    self.policy.evaluation.score_floor,
                ),
                "friction_penalty": best_candidate.deterministic_penalties.get(
                    "friction",
                    self.policy.evaluation.score_floor,
                ),
                "workflow_issue_count": float(len(workflow_issues)),
            },
            convergence=convergence,
            tests=tests,
            runtime=self.runtime,
            signature=signature,
            parent_artifact_id=req.parent_artifact_id,
            lineage_depth=lineage_depth,
        )

        artifact_path = self.artifact_repo.save(artifact)
        if self.runtime.use_cache:
            self.cache_repo.put(cache_key, artifact_path)

        return artifact, artifact_path

    def evaluate_dataset(
        self, dataset_path: Path, max_iters: int | None = None
    ) -> EvalReport:
        rows = _load_eval_dataset(dataset_path)
        details: list[dict[str, object]] = []
        pass_count = 0
        total_score = 0.0
        effective_max_iters = (
            max_iters if max_iters is not None else self.runtime.default_max_iters
        )

        for row in rows:
            workflow_id = row.workflow or self.runtime.default_workflow_id
            profile_id = row.profile or self.runtime.default_profile_id
            req = CompileRequest(
                intent=row.intention,
                raw_context=row.raw_context,
                workflow_id=workflow_id,
                profile_id=profile_id,
                output_format=self.runtime.default_output_format,
                variants=self.runtime.default_variants,
                max_iters=effective_max_iters,
                emit_tests=self.runtime.default_emit_tests,
                seed=self.runtime.seed,
            )
            artifact, _ = self.compile(req)
            score = float(artifact.scores.get("final", 0.0))
            total_score += score
            text = artifact.prompt_final.lower()
            sections_ok = all(sec.lower() in text for sec in row.expected_sections)
            terms_ok = all(term.lower() in text for term in row.required_terms)
            score_ok = score >= max(0.0, row.min_score)
            passed = sections_ok and terms_ok and score_ok
            if passed:
                pass_count += 1
            details.append(
                {
                    "intent": row.intention,
                    "workflow": workflow_id,
                    "profile": profile_id,
                    "score": score,
                    "sections_ok": sections_ok,
                    "terms_ok": terms_ok,
                    "score_ok": score_ok,
                    "pass": passed,
                }
            )

        n = len(rows)
        return EvalReport(
            created_at=now_utc(),
            dataset_path=str(dataset_path),
            total_examples=n,
            pass_rate=(pass_count / n) if n else self.policy.evaluation.score_floor,
            avg_score=(total_score / n) if n else self.policy.evaluation.score_floor,
            details=details,
        )

    def _emit_tests(
        self, prompt: str, required_sections: list[str]
    ) -> list[dict[str, str | bool]]:
        text = prompt.lower()
        tests: list[dict[str, str | bool]] = []
        for section in required_sections:
            tests.append(
                {
                    "name": f"contains_section_{section}",
                    "pass": section.lower() in text,
                    "type": "string_contains",
                }
            )
        tests.append(
            {
                "name": "has_output_contract",
                "pass": "output contract" in text,
                "type": "string_contains",
            }
        )
        return tests


def load_context_value(context: str | None, context_file: Path | None) -> str:
    if context_file:
        return context_file.read_text()
    return context or ""


def _load_eval_dataset(path: Path) -> list[EvalExample]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        return []
    return [EvalExample.model_validate(r) for r in rows if isinstance(r, dict)]


def _pareto_frontier(candidates: list[object]) -> list[object]:
    frontier: list[object] = []
    for c in candidates:
        dominated = False
        for d in candidates:
            if d is c:
                continue
            if _dominates(d, c):
                dominated = True
                break
        if not dominated:
            frontier.append(c)
    return frontier or candidates


def _dominates(a: object, b: object) -> bool:
    a_total = float(getattr(a, "total_score", 0.0))
    b_total = float(getattr(b, "total_score", 0.0))

    a_clarity = float(getattr(a, "judge_scores", {}).get("clarity", 0.0))
    b_clarity = float(getattr(b, "judge_scores", {}).get("clarity", 0.0))

    a_bloat = float(getattr(a, "deterministic_penalties", {}).get("bloat", 0.0))
    b_bloat = float(getattr(b, "deterministic_penalties", {}).get("bloat", 0.0))

    no_worse = a_total >= b_total and a_clarity >= b_clarity and a_bloat <= b_bloat
    strictly_better = a_total > b_total or a_clarity > b_clarity or a_bloat < b_bloat
    return no_worse and strictly_better
