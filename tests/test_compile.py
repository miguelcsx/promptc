from pathlib import Path

import json

from promptc.compiler import CompilerService
from promptc.engines.dspy_optimize import DSPyOptimizeEngine
from promptc.engines.evaluate import EvalEngine
from promptc.engines.intent import IntentEngine
from promptc.engines.optimize import OptimizeEngine
from promptc.engines.profile import ProfileEngine
from promptc.engines.render import OutputRenderer
from promptc.engines.signature import SignatureEngine
from promptc.engines.workflow import WorkflowEngine
from promptc.models import CompileRequest, OutputFormat, PolicyConfig, RuntimeConfig
from promptc.storage.fs import FsArtifactRepo, FsCacheRepo, FsCalibrationRepo, FsProfileRepo, FsSignatureRepo, FsWorkflowRepo


class FakeProgram:
    def infer_outcome(self, intention: str, raw_context: str, workflow_hint: str):
        _ = raw_context, workflow_hint
        return {
            "outcome_type": "analyze",
            "objective": intention,
            "success_criteria": ["clear", "constrained"],
            "assumptions": ["test assumption"],
            "constraints": ["test constraint"],
            "ambiguity_flags": [],
        }

    def candidate(self, outcome_spec_json: str, profile_json: str, workflow_json: str, workflow_id: str, output_format: str):
        _ = outcome_spec_json, profile_json, workflow_json, workflow_id, output_format
        return {
            "compiled_prompt": "# INSTRUCTION HIERARCHY\n# INPUT BOUNDARY\n# FAILURE POLICY\n# OUTPUT CONTRACT\n# FINAL SELF-CHECK\nOutput contract",
            "rationale": "fake",
        }

    def judge(
        self,
        candidate_prompt: str,
        outcome_spec_json: str,
        workflow_json: str,
        workflow_rubric_json: str = "",
    ):
        _ = candidate_prompt, outcome_spec_json, workflow_json, workflow_rubric_json
        return {"score_quality": 0.8, "score_clarity": 0.8, "score_constraint_fit": 0.8, "risk_flags": []}

    def refine_candidate(
        self,
        candidate_prompt: str,
        judge_feedback_json: str,
        deterministic_penalties_json: str,
        outcome_spec_json: str,
        profile_json: str,
        workflow_json: str,
        output_format: str,
    ):
        _ = judge_feedback_json, deterministic_penalties_json, outcome_spec_json, profile_json, workflow_json, output_format
        return {"refined_prompt": candidate_prompt, "refinement_notes": "fake-refine"}

    def generate_profile(self, profile_id: str, description: str, intent: str, workflow_json: str):
        _ = workflow_json
        data = {
            "id": profile_id,
            "version": 1,
            "description": description or f"suggested for {intent}",
            "style": {"verbosity": "concise", "tone": "direct", "stance": "architect"},
            "assumptions": {},
            "prompt_policies": {
                "require_tradeoffs": True,
                "forbid_handwavy_claims": True,
                "require_explicit_output_contract": True,
            },
            "constraints": {"max_prompt_tokens": 900, "structure": ["goal", "constraints", "output_contract"]},
            "suitability_rules": {"best_for": ["analyze"], "avoid_for": []},
            "context_patterns": [
                "trusted_instruction_source_only",
                "explicit_delimiter_blocks",
                "abstain_on_missing_context",
            ],
        }
        return {"profile_yaml": json.dumps(data), "suitability": 0.9, "reasons": ["fit"]}

    def refine_profile(self, intent: str, workflow_json: str, existing_profile_json: str, reasons_json: str):
        _ = workflow_json, reasons_json
        data = json.loads(existing_profile_json)
        data["description"] = f"{data.get('description', '')} refined for {intent}".strip()
        data["version"] = int(data.get("version", 1)) + 1
        return {"profile_yaml": json.dumps(data), "suitability": 0.95, "reasons": ["better_fit"]}

    def maybe_optimize(self, optimizer_name, trainset, metric):
        _ = optimizer_name, trainset, metric
        return True


def policy() -> PolicyConfig:
    return PolicyConfig.model_validate(
        {
            "search": {
                "improve_epsilon": 0.001,
                "plateau_patience": 2,
                "score_precision": 4,
                "weight_sum_tolerance": 0.001,
                "eval_weight_target": 1.0,
                "default_artifact_list_limit": 20,
                "max_artifact_list_limit": 200,
                "max_variants": 8,
                "max_iters": 10,
            },
            "profile": {"min_prompt_tokens": 250},
            "evaluation": {
                "variability_penalty_cap": 0.15,
                "variability_penalty_scale": 0.5,
                "score_floor": 0.0,
                "score_ceiling": 1.0,
                "round_digits": 4,
            },
            "optimizer_metric": {
                "non_empty_weight": 0.4,
                "required_terms_weight": 0.6,
                "output_contract_bonus": 0.3,
            },
        }
    )


def build_service(root: Path) -> CompilerService:
    runtime = RuntimeConfig(provider="local", model="dummy", optimizer="none", judge_count=1)
    p = FakeProgram()
    pol = policy()
    return CompilerService(
        profile_repo=FsProfileRepo(root / "profiles"),
        workflow_repo=FsWorkflowRepo(root / "workflows"),
        signature_repo=FsSignatureRepo(root / "signatures"),
        artifact_repo=FsArtifactRepo(root / "artifacts"),
        cache_repo=FsCacheRepo(root / "cache"),
        intent_engine=IntentEngine(program=p),
        profile_engine=ProfileEngine(program=p, policy=pol),
        signature_engine=SignatureEngine(),
        workflow_engine=WorkflowEngine(policy=pol),
        optimize_engine=OptimizeEngine(program=p),
        eval_engine=EvalEngine(program=p, runtime=runtime, calibration_repo=FsCalibrationRepo(root / "calibrations"), policy=pol),
        dspy_optimize_engine=DSPyOptimizeEngine(program=p, policy=pol),
        output_renderer=OutputRenderer(),
        runtime=runtime,
        policy=pol,
    )


def seed_defaults(root: Path) -> None:
    profiles = root / "profiles"
    workflows = root / "workflows"
    profiles.mkdir(parents=True)
    workflows.mkdir(parents=True)

    (profiles / "expert.yaml").write_text(
        """
id: expert
version: 1
style: {verbosity: concise, tone: direct, stance: architect}
assumptions: {}
prompt_policies: {require_tradeoffs: true, forbid_handwavy_claims: true, require_explicit_output_contract: true}
constraints: {max_prompt_tokens: 900, structure: [goal, constraints, output_contract]}
suitability_rules: {best_for: [analyze], avoid_for: []}
context_patterns: [trusted_instruction_source_only, explicit_delimiter_blocks, abstain_on_missing_context]
""".strip()
    )
    (workflows / "analyze.yaml").write_text(
        """
id: analyze
signature_template: input_context, objective, constraints -> compiled_prompt, checks
required_sections: [goal, method, output_contract]
required_prompt_blocks: [instruction_hierarchy, input_boundary, failure_policy, output_contract, final_self_check]
eval_weights: {quality: 0.45, clarity: 0.25, constraint_fit: 0.2, brevity: 0.1}
""".strip()
    )


def test_compile_writes_artifact_and_signature(tmp_path: Path) -> None:
    seed_defaults(tmp_path)
    service = build_service(tmp_path)

    artifact, artifact_path = service.compile(
        CompileRequest(
            intent="I want to analyze migration risks",
            raw_context="legacy monolith",
            workflow_id="analyze",
            profile_id="expert",
            output_format=OutputFormat.CHATML,
            variants=3,
            max_iters=4,
            emit_tests=False,
            seed=0,
        )
    )

    assert artifact.prompt_final
    assert Path(artifact_path).exists()
    assert artifact.workflow == "analyze"
    assert artifact.signature is not None
    assert (tmp_path / "signatures").exists()


def test_eval_dataset(tmp_path: Path) -> None:
    seed_defaults(tmp_path)
    service = build_service(tmp_path)

    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        """
[
  {
    "intention": "I want to analyze a deployment plan",
    "raw_context": "k8s to ecs",
    "workflow": "analyze",
    "profile": "expert",
    "required_terms": ["output contract"]
  }
]
""".strip()
    )

    report = service.evaluate_dataset(dataset)
    assert report.total_examples == 1
    assert 0.0 <= report.avg_score <= 1.0


def test_workflow_validation(tmp_path: Path) -> None:
    from promptc.models import WorkflowConfig

    w = WorkflowConfig(
        id="broken",
        signature_template="x->y",
        required_sections=["goal"],
        required_prompt_blocks=[
            "instruction_hierarchy",
            "input_boundary",
            "failure_policy",
            "output_contract",
            "final_self_check",
        ],
    )
    issues = WorkflowEngine(policy=policy()).validate(w)
    assert issues


def test_compile_cache_short_circuit(tmp_path: Path) -> None:
    seed_defaults(tmp_path)
    service = build_service(tmp_path)

    req = CompileRequest(
        intent="I want to analyze migration risks",
        raw_context="legacy monolith",
        workflow_id="analyze",
        profile_id="expert",
        output_format=OutputFormat.CHATML,
        variants=3,
        max_iters=4,
        emit_tests=False,
        seed=0,
    )

    artifact1, path1 = service.compile(req)
    artifact2, path2 = service.compile(req)

    assert path1 == path2
    assert artifact2.scores.get("cache_hit") == 1.0
    assert artifact1.artifact_id == artifact2.artifact_id
