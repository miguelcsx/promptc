# promptc

`promptc` is a local prompt compiler.
It takes an intent plus context and outputs a system-prompt artifact (and metadata) for use in other LLM systems.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

## Quick Start

Analyze-style compile:

```bash
uv run promptc analyze \
  --intent "Analyze migration risks" \
  --context "legacy monolith, zero downtime" \
  --profile expert \
  --output chatml
```

Generic compile:

```bash
uv run promptc compile \
  --intent "Explain this architecture" \
  --workflow explain \
  --profile beginner_teacher
```

## Pipeline

1. Parse intent into an outcome contract (objective, constraints, assumptions).
2. Generate prompt candidates.
3. Score candidates with judge models + deterministic checks.
4. Refine top candidates.
5. Render output (`plain`, `chatml`, or `json_schema`).

## Strict Mode

`promptc` is fail-fast:

- no silent generation fallback
- no silent judge fallback
- invalid policy/workflow/profile contracts error out

You need a working DSPy/LiteLLM provider setup.

## Configuration

On first run, defaults are copied to `.promptc/`.

Runtime config: `.promptc/config.yaml`

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

CLI flags override config values.

Policy config: `.promptc/policy.yaml`

- convergence settings
- penalties and marker sets
- optimizer weights
- validation thresholds

## Profiles

Profiles control tone/constraints:

| Profile | Token Limit | Style | Best For |
|---|---|---|---|
| `expert` | 2500 | concise, direct | analyze, decide, compile |
| `beginner_teacher` | 3000 | verbose, supportive | explain, summarize |
| `critic` | 2500 | concise, critical | analyze, decide |

## Workflows

Workflows define required prompt sections:

| Workflow | Style | Key Sections |
|---|---|---|
| `analyze` | structured contract | goal, knowns, unknowns, method, evidence_bar, tradeoffs, output_contract |
| `explain` | teaching contract | goal, knowns, unknowns, explanation_method, examples, output_contract |
| `summarize` | compression contract | goal, source_scope, compression_ratio, exclusions, output_contract |
| `decide` | decision contract | goal, options, decision_criteria, scoring_method, recommendation_rule, output_contract |

Each workflow includes:

- `prompt_directives`
- `quality_checklist`
- `friction_markers`
- `forbidden_patterns`
- `required_sections`
- `judge_rubric`

## Commands

```bash
uv run promptc compile|explain|summarize|analyze|decide
uv run promptc profile list|create|suggest|refine
uv run promptc workflow list|validate|show|create
uv run promptc artifact list|show|diff|lineage
uv run promptc eval run --dataset .promptc/evals/basic.json
uv run promptc eval calibration --model <model> --rubric-version <version>
uv run promptc cache clear
```

## Development

```bash
uv run pytest
uv run pytest tests/test_compile.py
```

## Artifacts

Stored in `.promptc/`:

- `profiles/`, `workflows/`
- `signatures/*.vN.json`
- `artifacts/YYYY-MM-DD/*.json`
- `artifacts/index.json`, `artifacts/index_map.json`
- `calibrations/*.json`
- `eval_reports/*.json`
- `cache/cache.json`

## Architecture

```text
intent -> IntentEngine -> OutcomeSpec
                        |
        +---------------+---------------+
        |               |               |
        v               v               v
   ProfileEngine   SignatureEngine  WorkflowEngine
        |               |               |
        +---------------+---------------+
                        |
                        v
                 OptimizeEngine
                        |
                        v
                   EvalEngine
                        |
                        v
                 OutputRenderer
                        |
                        v
                  PromptArtifact
```
