from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

CANONICAL_PROMPT_BLOCKS = (
    "instruction_hierarchy",
    "input_boundary",
    "failure_policy",
    "output_contract",
    "final_self_check",
)


class OutcomeType(str, Enum):
    EXPLAIN = "explain"
    SUMMARIZE = "summarize"
    ANALYZE = "analyze"
    DECIDE = "decide"
    COMPILE = "compile"


class OutputFormat(str, Enum):
    PLAIN = "plain"
    CHATML = "chatml"
    JSON_SCHEMA = "json_schema"


class CompileRequest(BaseModel):
    intent: str
    raw_context: str = ""
    workflow_id: str
    profile_id: str
    output_format: OutputFormat
    variants: int
    max_iters: int
    emit_tests: bool
    seed: int
    parent_artifact_id: str | None = None


class RuntimeConfig(BaseModel):
    provider: str = "openai"
    model: str = "openai/gpt-4o-mini"
    role_providers: dict[str, str] = Field(default_factory=dict)
    role_models: dict[str, str] = Field(default_factory=dict)
    role_temperatures: dict[str, float] = Field(default_factory=dict)
    base_url: str | None = None
    provider_base_url: dict[str, str] = Field(default_factory=dict)
    api_key: str | None = None
    api_key_env: str | None = None
    provider_api_key: dict[str, str] = Field(default_factory=dict)
    provider_api_key_env: dict[str, str] = Field(default_factory=dict)
    provider_force_json_object: dict[str, bool] = Field(default_factory=dict)
    strip_v1_model_prefixes: list[str] = Field(default_factory=list)
    optimizer: str = "none"
    judge_count: int = 1
    temperature: float = 0.2
    max_retries: int = 2
    seed: int = 0
    parallel_workers: int = 4
    rubric_version: str = "v1"
    eval_dataset_path: str | None = None
    use_cache: bool = True
    default_workflow_id: str = "analyze"
    default_profile_id: str = "expert"
    default_output_format: OutputFormat = OutputFormat.CHATML
    default_variants: int = 3
    default_max_iters: int = 4
    default_emit_tests: bool = False
    default_show_metadata: bool = False
    default_profile_suggest_workflow_id: str = "analyze"
    default_profile_refine_workflow_id: str = "analyze"
    default_profile_refine_outcome_type: OutcomeType = OutcomeType.ANALYZE


class SearchPolicy(BaseModel):
    improve_epsilon: float
    plateau_patience: int
    score_precision: int
    weight_sum_tolerance: float
    eval_weight_target: float
    default_artifact_list_limit: int
    max_artifact_list_limit: int
    max_variants: int
    max_iters: int


class ProfilePolicy(BaseModel):
    min_prompt_tokens: int


class EvaluationPolicy(BaseModel):
    variability_penalty_cap: float
    variability_penalty_scale: float
    score_floor: float
    score_ceiling: float
    round_digits: int


class OptimizerMetricPolicy(BaseModel):
    non_empty_weight: float
    required_terms_weight: float
    output_contract_bonus: float


class PolicyConfig(BaseModel):
    search: SearchPolicy
    profile: ProfilePolicy
    evaluation: EvaluationPolicy
    optimizer_metric: OptimizerMetricPolicy


class OutcomeSpec(BaseModel):
    outcome_type: OutcomeType
    objective: str
    success_criteria: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    ambiguity_flags: list[str] = Field(default_factory=list)


class SignatureSpec(BaseModel):
    signature_id: str
    version: int
    intent: str
    workflow: str
    fields_in: list[str]
    fields_out: list[str]
    assumptions: list[str] = Field(default_factory=list)
    created_at: datetime


class ProfileStyle(BaseModel):
    verbosity: str = "concise"
    tone: str = "direct"
    stance: str = "architect"


class ProfilePromptPolicies(BaseModel):
    require_tradeoffs: bool = True
    forbid_handwavy_claims: bool = True
    require_explicit_output_contract: bool = True


class ProfileConstraints(BaseModel):
    max_prompt_tokens: int = 4500
    structure: list[str] = Field(default_factory=list)


class SuitabilityRules(BaseModel):
    best_for: list[str] = Field(default_factory=list)
    avoid_for: list[str] = Field(default_factory=list)


class CognitiveProfile(BaseModel):
    id: str
    version: int = 1
    description: str = ""
    style: ProfileStyle = Field(default_factory=ProfileStyle)
    assumptions: dict[str, Any] = Field(default_factory=dict)
    prompt_policies: ProfilePromptPolicies = Field(
        default_factory=ProfilePromptPolicies
    )
    constraints: ProfileConstraints = Field(default_factory=ProfileConstraints)
    suitability_rules: SuitabilityRules = Field(default_factory=SuitabilityRules)
    context_patterns: list[str]

    @field_validator("context_patterns")
    @classmethod
    def _validate_context_patterns(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("context_patterns must not be empty")
        normalized = [v.strip() for v in value if v and v.strip()]
        if not normalized:
            raise ValueError("context_patterns must contain non-empty items")
        return normalized


class WorkflowEvalWeights(BaseModel):
    quality: float = 0.45
    clarity: float = 0.25
    constraint_fit: float = 0.2
    brevity: float = 0.1


class WorkflowJudgeRubric(BaseModel):
    priorities: list[str] = Field(default_factory=list)
    quality_checks: list[str] = Field(default_factory=list)
    clarity_checks: list[str] = Field(default_factory=list)
    constraint_checks: list[str] = Field(default_factory=list)
    risk_checks: list[str] = Field(default_factory=list)


class WorkflowConfig(BaseModel):
    id: str
    description: str = ""
    signature_template: str
    prompt_style: str = "contract"
    prompt_directives: list[str] = Field(default_factory=list)
    quality_checklist: list[str] = Field(default_factory=list)
    clarification_policy: list[str] = Field(default_factory=list)
    ambiguity_resolution_rules: list[str] = Field(default_factory=list)
    forbidden_patterns: list[str] = Field(default_factory=list)
    friction_markers: list[str] = Field(default_factory=list)
    required_sections: list[str] = Field(default_factory=list)
    required_prompt_blocks: list[str] = Field(default_factory=list)
    required_technique_markers: list[str] = Field(default_factory=list)
    vague_markers: list[str] = Field(default_factory=list)
    ambiguity_penalty_per_marker: float = 0.04
    ambiguity_penalty_cap: float = 0.35
    missing_sections_weight: float = 0.25
    missing_techniques_weight: float = 0.2
    eval_weights: WorkflowEvalWeights = Field(default_factory=WorkflowEvalWeights)
    judge_rubric: WorkflowJudgeRubric = Field(default_factory=WorkflowJudgeRubric)
    # Per-workflow guidance injected into the compiler/refiner LLM calls.
    # When non-empty, replaces the default universal principles.
    compiler_principles: str = ""
    refine_principles: str = ""

    @field_validator("required_prompt_blocks")
    @classmethod
    def _validate_required_prompt_blocks(cls, value: list[str]) -> list[str]:
        return [v.strip().lower() for v in value if v and v.strip()]

    @field_validator("required_technique_markers")
    @classmethod
    def _validate_required_technique_markers(cls, value: list[str]) -> list[str]:
        return [v.strip().lower() for v in value if v and v.strip()]

    @field_validator("vague_markers")
    @classmethod
    def _validate_vague_markers(cls, value: list[str]) -> list[str]:
        return [v.strip().lower() for v in value if v and v.strip()]

    @field_validator("prompt_directives")
    @classmethod
    def _validate_prompt_directives(cls, value: list[str]) -> list[str]:
        return [v.strip() for v in value if v and v.strip()]

    @field_validator("quality_checklist", "clarification_policy", "ambiguity_resolution_rules")
    @classmethod
    def _validate_nonempty_list_items(cls, value: list[str]) -> list[str]:
        return [v.strip() for v in value if v and v.strip()]

    @field_validator("forbidden_patterns")
    @classmethod
    def _validate_forbidden_patterns(cls, value: list[str]) -> list[str]:
        return [v.strip().lower() for v in value if v and v.strip()]

    @field_validator("friction_markers")
    @classmethod
    def _validate_friction_markers(cls, value: list[str]) -> list[str]:
        return [v.strip().lower() for v in value if v and v.strip()]

    @field_validator("missing_sections_weight", "missing_techniques_weight")
    @classmethod
    def _validate_penalty_weight(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("penalty weights must be >= 0")
        return value

    @field_validator("ambiguity_penalty_per_marker", "ambiguity_penalty_cap")
    @classmethod
    def _validate_ambiguity_penalty(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("ambiguity penalty values must be >= 0")
        return value


class PromptCandidate(BaseModel):
    prompt: str
    rationale: str = ""
    estimated_tokens: int = 0
    deterministic_penalties: dict[str, float] = Field(default_factory=dict)
    judge_scores: dict[str, float] = Field(default_factory=dict)
    judge_feedback: list[str] = Field(default_factory=list)
    total_score: float = 0.0


class ConvergenceInfo(BaseModel):
    iterations: int = 0
    stopped_reason: str = "max_iters"
    frontier_size: int = 0


class PromptArtifact(BaseModel):
    artifact_id: str
    created_at: datetime
    intent: str
    workflow: str
    profile: str
    output_format: OutputFormat
    assumptions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    prompt_final: str
    prompt_variants: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    convergence: ConvergenceInfo = Field(default_factory=ConvergenceInfo)
    tests: list[dict[str, Any]] = Field(default_factory=list)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    signature: SignatureSpec | None = None
    parent_artifact_id: str | None = None
    lineage_depth: int = 0


class ArtifactIndexEntry(BaseModel):
    artifact_id: str
    path: str
    created_at: datetime
    parent_artifact_id: str | None = None
    lineage_depth: int = 0


class EvalExample(BaseModel):
    intention: str
    raw_context: str = ""
    workflow: str | None = None
    profile: str | None = None
    required_terms: list[str] = Field(default_factory=list)
    expected_sections: list[str] = Field(default_factory=list)
    min_score: float = 0.0


class EvalReport(BaseModel):
    created_at: datetime
    dataset_path: str
    total_examples: int
    pass_rate: float
    avg_score: float
    details: list[dict[str, Any]] = Field(default_factory=list)


class JudgeCalibration(BaseModel):
    model: str
    rubric_version: str
    runs: int = 0
    mean_quality: float = 0.0
    mean_clarity: float = 0.0
    mean_constraint_fit: float = 0.0
    updated_at: datetime
