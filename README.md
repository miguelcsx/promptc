# promptc

Local prompt compiler. Takes an intent and outputs a system prompt.

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

## TUI

```bash
uv run promptc
```

Type an intent to compile. Slash commands:

```
/provider anthropic ANTHROPIC_API_KEY   set provider (persisted)
/model anthropic/claude-haiku-4-5-20251001
/variants 2
/iters 3
/context This is a coding assistant    set extra context for next compile
/workflow [list|use|show|create|validate]
/profile  [list|use|show|refine|history]
/artifacts [list|show|diff|lineage]
/config show
/config set temperature 0.3
/format [plain|chatml|json_schema]
/cache clear
/help
```

Config changes from the TUI persist to `.promptc/config.yaml`.

## CLI

```bash
uv run promptc compile --intent "..." --workflow analyze --profile expert --output chatml
uv run promptc analyze  --intent "..."
uv run promptc explain  --intent "..."
uv run promptc summarize --intent "..."
uv run promptc decide   --intent "..."

uv run promptc profile list|show|create|suggest|refine
uv run promptc workflow list|show|validate|create
uv run promptc artifact list|show|diff|lineage
uv run promptc eval run --dataset .promptc/evals/basic.json
uv run promptc cache clear
```

## Config

On first run, defaults are written to `.promptc/config.yaml`. Key fields:

```yaml
provider: ollama
model: ollama/llama3
base_url: http://localhost:11434
api_key_env: null
temperature: 0.2
default_workflow_id: analyze
default_profile_id: expert
default_output_format: chatml
default_variants: 3
default_max_iters: 4
use_cache: true
```

## Development

```bash
uv run pytest
```
