from __future__ import annotations

import json

import yaml

from promptc.dspy.programs import PromptCompilerProgram
from promptc.models import CognitiveProfile, OutcomeSpec, PolicyConfig, WorkflowConfig


class ProfileEngine:
    def __init__(self, program: PromptCompilerProgram, policy: PolicyConfig) -> None:
        self.program = program
        self.policy = policy

    def assess_suitability(self, profile: CognitiveProfile, outcome: OutcomeSpec) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        outcome_id = outcome.outcome_type.value

        if profile.suitability_rules.best_for and outcome_id not in profile.suitability_rules.best_for:
            reasons.append(f"Profile {profile.id} is not marked best_for={outcome_id}")

        if outcome_id in profile.suitability_rules.avoid_for:
            reasons.append(f"Profile {profile.id} is marked avoid_for={outcome_id}")

        if profile.constraints.max_prompt_tokens < self.policy.profile.min_prompt_tokens:
            reasons.append("Profile max_prompt_tokens is below policy minimum")

        if profile.prompt_policies.require_explicit_output_contract is False:
            reasons.append("Profile disables explicit output contract requirement")

        return (len(reasons) == 0), reasons

    def refine_profile(
        self,
        profile: CognitiveProfile,
        outcome: OutcomeSpec,
        reasons: list[str],
        workflow: WorkflowConfig,
    ) -> CognitiveProfile:
        out = self.program.refine_profile(
            intent=outcome.objective,
            workflow_json=json.dumps(workflow.model_dump(mode="json")),
            existing_profile_json=json.dumps(profile.model_dump(mode="json")),
            reasons_json=json.dumps(reasons),
        )
        yml = out.get("profile_yaml", "")
        if not yml.strip():
            raise ValueError("Profile suggestion returned empty YAML")

        data = yaml.safe_load(yml)
        if not isinstance(data, dict):
            raise ValueError("Profile suggestion did not return YAML mapping")

        parsed = CognitiveProfile.model_validate(data)
        parsed.version = max(parsed.version, profile.version + 1)
        if reasons:
            suffix = "; ".join(reasons)
            parsed.description = f"{parsed.description} [{suffix}]".strip()
        return self._enforce_profile_contract(parsed, outcome=outcome, workflow=workflow)

    def suggest_profile(
        self,
        profile_id: str,
        description: str,
        intent: str,
        workflow: WorkflowConfig,
    ) -> CognitiveProfile:
        out = self.program.generate_profile(
            profile_id=profile_id,
            description=description,
            intent=intent,
            workflow_json=json.dumps(workflow.model_dump(mode="json")),
        )
        yml = out.get("profile_yaml", "")
        if not yml.strip():
            raise ValueError("Profile suggestion returned empty YAML")

        data = yaml.safe_load(yml)
        if not isinstance(data, dict):
            raise ValueError("Profile suggestion did not return YAML mapping")

        parsed = CognitiveProfile.model_validate(data)
        parsed.id = profile_id
        pseudo_outcome = OutcomeSpec(
            outcome_type=outcome_from_workflow(workflow.id),
            objective=intent,
            success_criteria=[],
            assumptions=[],
            constraints=[],
            ambiguity_flags=[],
        )
        return self._enforce_profile_contract(parsed, outcome=pseudo_outcome, workflow=workflow)

    def generate_profile(self, profile_id: str, description: str) -> CognitiveProfile:
        return CognitiveProfile(
            id=profile_id,
            version=1,
            description=description,
            context_patterns=[
                "trusted_instruction_source_only",
                "explicit_delimiter_blocks",
                "abstain_on_missing_context",
            ],
        )

    def _enforce_profile_contract(
        self,
        profile: CognitiveProfile,
        outcome: OutcomeSpec,
        workflow: WorkflowConfig,
    ) -> CognitiveProfile:
        p = profile.model_copy(deep=True)
        if p.constraints.max_prompt_tokens < self.policy.profile.min_prompt_tokens:
            p.constraints.max_prompt_tokens = self.policy.profile.min_prompt_tokens
        if not p.prompt_policies.require_explicit_output_contract:
            p.prompt_policies.require_explicit_output_contract = True
        if not p.context_patterns:
            p.context_patterns = [
                "trusted_instruction_source_only",
                "explicit_delimiter_blocks",
                "abstain_on_missing_context",
            ]
        if workflow.id and workflow.id not in p.suitability_rules.best_for:
            p.suitability_rules.best_for.append(workflow.id)
        outcome_key = outcome.outcome_type.value
        if outcome_key and outcome_key not in p.suitability_rules.best_for:
            p.suitability_rules.best_for.append(outcome_key)
        return p


def outcome_from_workflow(workflow_id: str):
    from promptc.models import OutcomeType

    key = (workflow_id or "").strip().lower()
    if key == "explain":
        return OutcomeType.EXPLAIN
    if key == "summarize":
        return OutcomeType.SUMMARIZE
    if key == "decide":
        return OutcomeType.DECIDE
    if key == "compile":
        return OutcomeType.COMPILE
    return OutcomeType.ANALYZE
