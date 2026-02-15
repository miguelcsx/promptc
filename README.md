# promptc

> Compile optimized, production-quality system prompts from vague intent

`promptc` is a local DSPy-based prompt compiler.
It generates optimized prompts and metadata artifacts for use in external LLM systems. It does not execute the end-user task.

## Setup (uv)

1. Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Sync project environment:

```bash
uv sync
```

This project is configured for `uv` as the default workflow (runtime + dev dependencies).

## Quick Start

```bash
uv run promptc analyze \
  --intent "I want to analyze migration risks" \
  --context "legacy monolith, zero downtime" \
  --profile expert \
  --output chatml
```

Generic mode:

```bash
uv run promptc compile \
  --intent "I want to explain this architecture" \
  --workflow explain \
  --profile beginner_teacher
```

## How It Works

`promptc` turns a vague user intent into a production-quality system prompt through a multi-stage pipeline:

1. **Intent Analysis** — Extracts a precise outcome contract (objective, success criteria, constraints, assumptions) from the raw intent
2. **Candidate Generation** — Generates multiple prompt candidates using prompt engineering principles embedded in the compiler guidance
3. **Evaluation** — Scores candidates on quality, clarity, and constraint fit using both LLM judges and deterministic penalties
4. **Refinement** — Iteratively improves the best candidates using targeted feedback
5. **Rendering** — Outputs the final prompt in the requested format (`plain`, `chatml`, or `json_schema`)

### Prompt Engineering Principles

The compiler guidance embeds established prompt engineering techniques into every generated prompt:

- **Role & Identity** — Opens with a clear role definition that anchors downstream behavior
- **Specificity Over Vagueness** — Concrete, measurable criteria instead of vague qualifiers
- **Chain-of-Thought Scaffolding** — Step-by-step reasoning structure for complex tasks
- **Instruction Hierarchy** — Critical rules first, marked with IMPORTANT/CRITICAL
- **Positive Then Negative Framing** — Desired behavior first, then guardrails
- **Output Contract** — Explicit format, success criteria, and failure modes
- **Context Boundaries** — Clear separation between system instructions and user input
- **No Unnecessary Friction** — No invented procedural gates unless explicitly requested

### Evaluation & Quality Control

Each candidate is scored on multiple dimensions:

- **Quality** (0–1) — Usefulness, specificity, technique depth, task fitness
- **Clarity** (0–1) — Unambiguous, well-structured, readable, non-redundant
- **Constraint Fit** (0–1) — Required sections present, forbidden patterns absent, output contract testable

Deterministic penalties catch issues LLM judges might miss:
- **Bloat** — Token count exceeding profile limits
- **Redundancy** — Repeated lines or instructions
- **Ambiguity** — Vague markers like "etc", "as needed", "maybe"
- **Missing Sections** — Required workflow sections not present
- **Missing Techniques** — Required technique markers absent
- **Friction** — Invented procedural gates (configurable per workflow via `friction_markers`)

## Strict Mode (Important)

`promptc` is fail-fast:
- no silent fallback generation
- no silent fallback judging
- invalid/missing policy/workflow/profile contracts cause errors

You must have a working DSPy + provider configuration.

## Config

On first run, defaults are copied into `.promptc/`.

### Runtime

`.promptc/config.yaml`

```yaml
provider: local
model: ollama/gpt-oss:120b-cloud
base_url: http://localhost:11434/v1
provider_base_url:
  openrouter: https://openrouter.ai/api/v1
api_key: null
api_key_env: null
provider_api_key: {}
provider_api_key_env:
  openai: OPENAI_API_KEY
  anthropic: ANTHROPIC_API_KEY
  openrouter: OPENROUTER_API_KEY
provider_force_json_object:
  local: true
strip_v1_model_prefixes:
  - ollama/
optimizer: none
judge_count: 1
temperature: 0.2
max_retries: 2
seed: 0
parallel_workers: 4
rubric_version: v1
eval_dataset_path: .promptc/evals/basic.json
use_cache: true
default_workflow_id: analyze
default_profile_id: expert
default_output_format: chatml
default_variants: 3
default_max_iters: 4
default_emit_tests: false
default_show_metadata: false
default_profile_suggest_workflow_id: analyze
default_profile_refine_workflow_id: analyze
default_profile_refine_outcome_type: analyze
```

CLI flags override runtime config.

Provider settings are declarative:
- `provider` and `model` can be any DSPy/LiteLLM-supported values.
- Configure credentials globally (`api_key`, `api_key_env`) or per provider (`provider_api_key`, `provider_api_key_env`).
- Configure per-provider base URLs with `provider_base_url`.

### Policy (declarative tuning)

`.promptc/policy.yaml`
- search convergence
- evaluation penalties/markers
- optimizer metric weights
- profile/workflow validation thresholds

### Profiles

Cognitive profiles control prompt style and constraints:

| Profile | Token Limit | Style | Best For |
|---|---|---|---|
| `expert` | 2500 | concise, direct | analyze, decide, compile |
| `beginner_teacher` | 3000 | verbose, supportive | explain, summarize |
| `critic` | 2500 | concise, critical | analyze, decide |

### Workflows

Workflows define the prompt contract structure:

| Workflow | Style | Key Sections |
|---|---|---|
| `analyze` | structured behavior contract | goal, knowns, unknowns, method, evidence_bar, tradeoffs, output_contract |
| `explain` | pedagogical behavior contract | goal, knowns, unknowns, explanation_method, examples, output_contract |
| `summarize` | faithful compression contract | goal, source_scope, compression_ratio, exclusions, output_contract |
| `decide` | decision contract | goal, options, decision_criteria, scoring_method, recommendation_rule, output_contract |

Each workflow includes:
- `prompt_directives` — Generation rules (role definition, CoT scaffolding, specificity)
- `quality_checklist` — Verification criteria for the generated prompt
- `friction_markers` — Patterns that indicate unwanted procedural friction (scored as penalties)
- `forbidden_patterns` — Patterns that must never appear in output
- `required_sections` — Sections that must be present
- `judge_rubric` — Scoring priorities and checks for the LLM judge

## Core Commands

- `uv run promptc compile|explain|summarize|analyze|decide`
- `uv run promptc profile list|create|suggest|refine`
- `uv run promptc workflow list|validate|show|create`
- `uv run promptc artifact list|show|diff|lineage`
- `uv run promptc eval run --dataset .promptc/evals/basic.json`
- `uv run promptc eval calibration --model <model> --rubric-version <version>`
- `uv run promptc cache clear`

## Dev Commands (uv)

- Run tests: `uv run pytest`
- Run one test file: `uv run pytest tests/test_compile.py`

## Output and Storage

Artifacts are stored under `.promptc/`:
- `profiles/`, `workflows/`
- `signatures/*.vN.json`
- `artifacts/YYYY-MM-DD/*.json`
- `artifacts/index.json`, `artifacts/index_map.json`
- `calibrations/*.json`
- `eval_reports/*.json`
- `cache/cache.json`

## Architecture

```
intent ─► IntentEngine ─► OutcomeSpec
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         ProfileEngine   SignatureEngine  WorkflowEngine
              │               │               │
              ▼               ▼               ▼
         OptimizeEngine (candidate generation + refinement)
              │
              ▼
         EvalEngine (LLM judge + deterministic penalties)
              │
              ▼
         OutputRenderer (plain / chatml / json_schema)
              │
              ▼
         PromptArtifact (stored + cached)
```

## Notes

- Prompt quality structure is schema-enforced via workflow/profile contracts (`required_prompt_blocks`, `context_patterns`).
- Supported output renderers: `plain`, `chatml`, `json_schema`.
- Pipeline version is tracked in cache keys — upgrading the compiler automatically invalidates stale cached artifacts.
- Friction detection is fully workflow-driven via `friction_markers` in each workflow YAML, not hardcoded.
