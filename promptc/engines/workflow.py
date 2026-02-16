from __future__ import annotations

from promptc.models import CANONICAL_PROMPT_BLOCKS, PolicyConfig, WorkflowConfig


class WorkflowEngine:
    def __init__(self, policy: PolicyConfig) -> None:
        self.policy = policy

    def section_checklist(self, workflow: WorkflowConfig) -> list[str]:
        return workflow.required_sections

    def validate(self, workflow: WorkflowConfig) -> list[str]:
        issues: list[str] = []
        if not workflow.required_sections:
            issues.append("required_sections is empty")
        if "output_contract" not in [s.lower() for s in workflow.required_sections]:
            issues.append("required_sections should include output_contract")
        if not workflow.prompt_style.strip():
            issues.append("prompt_style is empty")
        if not workflow.prompt_directives:
            issues.append("prompt_directives is empty")
        if not workflow.quality_checklist:
            issues.append("quality_checklist is empty")
        if not workflow.clarification_policy:
            issues.append("clarification_policy is empty")
        if not workflow.ambiguity_resolution_rules:
            issues.append("ambiguity_resolution_rules is empty")
        if not workflow.forbidden_patterns:
            issues.append("forbidden_patterns is empty")
        weight_sum = (
            workflow.eval_weights.quality
            + workflow.eval_weights.clarity
            + workflow.eval_weights.constraint_fit
            + workflow.eval_weights.brevity
        )
        if abs(weight_sum - self.policy.search.eval_weight_target) > self.policy.search.weight_sum_tolerance:
            issues.append(f"eval_weights must sum to 1.0, got {weight_sum:.3f}")
        missing_blocks = [b for b in CANONICAL_PROMPT_BLOCKS if b not in workflow.required_prompt_blocks]
        if missing_blocks:
            issues.append(f"required_prompt_blocks missing canonical blocks: {missing_blocks}")
        rubric = workflow.judge_rubric
        if not rubric.priorities:
            issues.append("judge_rubric.priorities is empty")
        if not rubric.quality_checks:
            issues.append("judge_rubric.quality_checks is empty")
        if not rubric.clarity_checks:
            issues.append("judge_rubric.clarity_checks is empty")
        if not rubric.constraint_checks:
            issues.append("judge_rubric.constraint_checks is empty")
        if not workflow.required_technique_markers:
            issues.append("required_technique_markers is empty")
        if not workflow.vague_markers:
            issues.append("vague_markers is empty")
        if workflow.ambiguity_penalty_per_marker <= 0.0:
            issues.append("ambiguity_penalty_per_marker must be > 0")
        if workflow.ambiguity_penalty_cap <= 0.0:
            issues.append("ambiguity_penalty_cap must be > 0")
        if workflow.missing_sections_weight <= 0.0:
            issues.append("missing_sections_weight must be > 0")
        if workflow.missing_techniques_weight <= 0.0:
            issues.append("missing_techniques_weight must be > 0")
        return issues
