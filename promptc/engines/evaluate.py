from __future__ import annotations

import json
import re
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor

from promptc.dspy.programs import PromptCompilerProgram
from promptc.models import CognitiveProfile, JudgeCalibration, OutcomeSpec, PolicyConfig, PromptCandidate, RuntimeConfig, WorkflowConfig
from promptc.storage.interfaces import CalibrationRepo
from promptc.utils import estimate_tokens, now_utc


class EvalEngine:
    def __init__(
        self,
        program: PromptCompilerProgram | None = None,
        runtime: RuntimeConfig | None = None,
        calibration_repo: CalibrationRepo | None = None,
        policy: PolicyConfig | None = None,
    ) -> None:
        self.program = program
        self.runtime = runtime or RuntimeConfig()
        self.calibration_repo = calibration_repo
        self._calibration_lock = threading.Lock()
        if policy is None:
            raise ValueError("EvalEngine requires policy")
        self.policy = policy

    def score_many(
        self,
        candidates: list[PromptCandidate],
        outcome: OutcomeSpec,
        profile: CognitiveProfile,
        workflow: WorkflowConfig,
    ) -> list[PromptCandidate]:
        if len(candidates) <= 1 or self.runtime.parallel_workers <= 1:
            return [self.score(c, outcome, profile, workflow) for c in candidates]

        workers = min(len(candidates), max(1, int(self.runtime.parallel_workers)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(self.score, c, outcome, profile, workflow) for c in candidates]
            return [f.result() for f in futures]

    def score(
        self,
        candidate: PromptCandidate,
        outcome: OutcomeSpec,
        profile: CognitiveProfile,
        workflow: WorkflowConfig,
    ) -> PromptCandidate:
        penalties = self._deterministic_penalties(candidate.prompt, profile, workflow)
        candidate.deterministic_penalties = penalties

        judge, variance_penalty, risk_flags = self._judge(candidate.prompt, outcome, workflow)
        candidate.judge_scores = judge
        candidate.judge_feedback = risk_flags

        w = workflow.eval_weights
        quality = judge.get("quality", 0.0)
        clarity = judge.get("clarity", 0.0)
        fit = judge.get("constraint_fit", 0.0)
        floor = self.policy.evaluation.score_floor
        ceiling = self.policy.evaluation.score_ceiling
        brevity = ceiling - min(ceiling, penalties.get("bloat", 0.0) + penalties.get("redundancy", 0.0))

        base = quality * w.quality + clarity * w.clarity + fit * w.constraint_fit + brevity * w.brevity
        total_penalty = (
            penalties.get("ambiguity", 0.0)
            + penalties.get("missing_sections", 0.0)
            + penalties.get("missing_techniques", 0.0)
            + penalties.get("friction", 0.0)
            + variance_penalty
        )
        candidate.total_score = max(floor, min(ceiling, base - total_penalty))
        return candidate

    def _deterministic_penalties(self, text: str, profile: CognitiveProfile, workflow: WorkflowConfig) -> dict[str, float]:
        text_lc = text.lower()
        tokens = estimate_tokens(text)
        max_tokens = max(1, profile.constraints.max_prompt_tokens)
        bloat = max(0.0, (tokens - max_tokens) / max_tokens)

        lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
        redundancy = 0.0
        if lines:
            unique_ratio = len(set(lines)) / len(lines)
            redundancy = max(0.0, 1 - unique_ratio)

        ambiguity = 0.0
        vague_markers = workflow.vague_markers
        for marker in vague_markers:
            if re.search(rf"\b{re.escape(marker)}\b", text_lc):
                ambiguity += workflow.ambiguity_penalty_per_marker

        missing_sections = 0.0
        required_hits = sum(1 for section in workflow.required_sections if section.lower() in text_lc)
        if workflow.required_sections:
            missing_sections = (
                max(0.0, 1 - required_hits / len(workflow.required_sections))
                * workflow.missing_sections_weight
            )

        missing_techniques = 0.0
        required_markers = workflow.required_technique_markers
        if required_markers:
            marker_hits = sum(1 for marker in required_markers if marker in text_lc)
            missing_techniques = (
                max(0.0, 1 - marker_hits / len(required_markers))
                * workflow.missing_techniques_weight
            )

        friction = 0.0
        for marker in workflow.forbidden_patterns:
            if marker and marker in text_lc:
                friction += 0.08
        for marker in workflow.friction_markers:
            if marker and marker in text_lc:
                friction += 0.1

        return {
            "bloat": round(bloat, self.policy.evaluation.round_digits),
            "redundancy": round(redundancy, self.policy.evaluation.round_digits),
            "ambiguity": round(min(workflow.ambiguity_penalty_cap, ambiguity), self.policy.evaluation.round_digits),
            "missing_sections": round(missing_sections, self.policy.evaluation.round_digits),
            "missing_techniques": round(missing_techniques, self.policy.evaluation.round_digits),
            "friction": round(min(0.5, friction), self.policy.evaluation.round_digits),
        }

    def _judge(self, prompt: str, outcome: OutcomeSpec, workflow: WorkflowConfig) -> tuple[dict[str, float], float, list[str]]:
        judge_count = max(1, int(self.runtime.judge_count))

        quality_scores: list[float] = []
        clarity_scores: list[float] = []
        fit_scores: list[float] = []
        risk_flags: list[str] = []

        for _ in range(judge_count):
            item = self._judge_once(prompt, outcome, workflow)
            quality_scores.append(item["quality"])
            clarity_scores.append(item["clarity"])
            fit_scores.append(item["constraint_fit"])
            for risk in item["risk_flags"]:
                risk_text = str(risk).strip()
                if risk_text:
                    risk_flags.append(risk_text)

        floor = self.policy.evaluation.score_floor
        ceiling = self.policy.evaluation.score_ceiling
        quality = _mean(quality_scores, floor, ceiling)
        clarity = _mean(clarity_scores, floor, ceiling)
        constraint_fit = _mean(fit_scores, floor, ceiling)

        variance_penalty = self.policy.evaluation.score_floor
        if judge_count > 1:
            variance_penalty = min(
                self.policy.evaluation.variability_penalty_cap,
                self.policy.evaluation.variability_penalty_scale
                * (_stdev(quality_scores) + _stdev(clarity_scores) + _stdev(fit_scores)),
            )

        self._update_calibration(quality, clarity, constraint_fit)

        return {
            "quality": quality,
            "clarity": clarity,
            "constraint_fit": constraint_fit,
        }, round(variance_penalty, self.policy.evaluation.round_digits), sorted(set(risk_flags))

    def _judge_once(self, prompt: str, outcome: OutcomeSpec, workflow: WorkflowConfig) -> dict[str, float | list[str]]:
        if self.program is None:
            raise ValueError("EvalEngine requires a DSPy program")
        floor = self.policy.evaluation.score_floor
        ceiling = self.policy.evaluation.score_ceiling
        out = self.program.judge(
            candidate_prompt=prompt,
            outcome_spec_json=json.dumps(outcome.model_dump(mode="json")),
            workflow_json=json.dumps(workflow.model_dump(mode="json")),
        )
        return {
            "quality": max(floor, min(ceiling, float(out.get("score_quality", floor)))),
            "clarity": max(floor, min(ceiling, float(out.get("score_clarity", floor)))),
            "constraint_fit": max(floor, min(ceiling, float(out.get("score_constraint_fit", floor)))),
            "risk_flags": [str(x) for x in out.get("risk_flags", []) if str(x).strip()],
        }

    def _update_calibration(self, quality: float, clarity: float, constraint_fit: float) -> None:
        if not self.calibration_repo:
            return

        model = self.runtime.model
        rubric = self.runtime.rubric_version
        with self._calibration_lock:
            cal = self.calibration_repo.get(model, rubric)
            if not cal:
                cal = JudgeCalibration(
                    model=model,
                    rubric_version=rubric,
                    runs=0,
                    mean_quality=self.policy.evaluation.score_floor,
                    mean_clarity=self.policy.evaluation.score_floor,
                    mean_constraint_fit=self.policy.evaluation.score_floor,
                    updated_at=now_utc(),
                )

            n = cal.runs
            cal.runs = n + 1
            cal.mean_quality = ((cal.mean_quality * n) + quality) / cal.runs
            cal.mean_clarity = ((cal.mean_clarity * n) + clarity) / cal.runs
            cal.mean_constraint_fit = ((cal.mean_constraint_fit * n) + constraint_fit) / cal.runs
            cal.updated_at = now_utc()
            self.calibration_repo.put(cal)


def _mean(values: list[float], floor: float, ceiling: float) -> float:
    if not values:
        return floor
    return max(floor, min(ceiling, sum(values) / len(values)))


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.pstdev(values))
