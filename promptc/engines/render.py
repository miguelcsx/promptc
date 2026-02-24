from __future__ import annotations

import json

from promptc.models import OutputFormat


class OutputRenderer:
    def render(self, prompt: str, output_format: OutputFormat) -> str:
        if output_format == OutputFormat.PLAIN:
            return prompt.strip()
        if output_format == OutputFormat.CHATML:
            return self._chatml(prompt)
        if output_format == OutputFormat.JSON_SCHEMA:
            return self._json_schema(prompt)
        return prompt.strip()

    def _chatml(self, prompt: str) -> str:
        return (
            "<|im_start|>system\n"
            f"{prompt.strip()}\n"
            "<|im_end|>"
        )

    def _json_schema(self, prompt: str) -> str:
        schema = {
            "instruction": prompt.strip(),
            "contract": {
                "type": "object",
                "required": ["result"],
                "properties": {
                    "result": {"type": "string"},
                    "notes": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": False,
            },
        }
        return json.dumps(schema, indent=2)
