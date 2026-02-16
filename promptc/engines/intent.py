from __future__ import annotations

import json

from promptc.dspy.programs import PromptCompilerProgram
from promptc.models import OutcomeSpec


class IntentEngine:
    def __init__(self, program: PromptCompilerProgram) -> None:
        self.program = program

    def infer(self, intention: str, raw_context: str, workflow_hint: str) -> OutcomeSpec:
        result = self.program.infer_outcome(intention, raw_context, workflow_hint)
        return OutcomeSpec.model_validate(result)


def outcome_to_json(outcome: OutcomeSpec) -> str:
    return json.dumps(outcome.model_dump(mode="json"), ensure_ascii=True)
