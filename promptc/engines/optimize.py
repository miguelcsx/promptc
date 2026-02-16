from __future__ import annotations

import json

from promptc.dspy.programs import PromptCompilerProgram
from promptc.engines.intent import outcome_to_json
from promptc.engines.techniques import normalize_prompt_text
from promptc.models import CognitiveProfile, OutcomeSpec, OutputFormat, PromptCandidate, WorkflowConfig
from promptc.utils import estimate_tokens


class OptimizeEngine:
    def __init__(self, program: PromptCompilerProgram) -> None:
        self.program = program

    def generate_candidates(
        self,
        outcome: OutcomeSpec,
        profile: CognitiveProfile,
        workflow: WorkflowConfig,
        output_format: OutputFormat,
        n: int,
    ) -> list[PromptCandidate]:
        candidates: list[PromptCandidate] = []
        for i in range(max(1, n)):
            prompt_text, rationale = self._synthesize(
                outcome,
                profile,
                workflow,
                output_format,
                variant=i,
            )
            candidates.append(
                PromptCandidate(
                    prompt=prompt_text,
                    rationale=rationale,
                    estimated_tokens=estimate_tokens(prompt_text),
                )
            )
        return candidates

    def refine(
        self,
        candidate: PromptCandidate,
        outcome: OutcomeSpec,
        profile: CognitiveProfile,
        workflow: WorkflowConfig,
        output_format: OutputFormat,
        iteration: int,
    ) -> PromptCandidate:
        feedback = {
            "iteration": iteration,
            "judge_scores": candidate.judge_scores,
            "risk_flags": candidate.judge_feedback,
        }
        out = self.program.refine_candidate(
            candidate_prompt=candidate.prompt,
            judge_feedback_json=json.dumps(feedback),
            deterministic_penalties_json=json.dumps(candidate.deterministic_penalties),
            outcome_spec_json=outcome_to_json(outcome),
            profile_json=json.dumps(profile.model_dump(mode="json")),
            workflow_json=json.dumps(workflow.model_dump(mode="json")),
            output_format=output_format.value,
        )
        refined_text = normalize_prompt_text(out.get("refined_prompt", "").strip())
        if not refined_text:
            raise ValueError("Candidate refinement returned empty prompt")
        notes = out.get("refinement_notes", f"DSPy refinement iteration {iteration}")

        return PromptCandidate(
            prompt=refined_text,
            rationale=notes,
            estimated_tokens=estimate_tokens(refined_text),
        )

    def _synthesize(
        self,
        outcome: OutcomeSpec,
        profile: CognitiveProfile,
        workflow: WorkflowConfig,
        output_format: OutputFormat,
        variant: int,
    ) -> tuple[str, str]:
        out = self.program.candidate(
            outcome_spec_json=outcome_to_json(outcome),
            profile_json=json.dumps(profile.model_dump(mode="json")),
            workflow_json=json.dumps(workflow.model_dump(mode="json")),
            workflow_id=workflow.id,
            output_format=output_format.value,
        )
        text = out.get("compiled_prompt", "").strip()
        if not text:
            raise ValueError("Candidate generation returned empty prompt")
        return normalize_prompt_text(text), out.get("rationale", f"DSPy candidate {variant}")
