from __future__ import annotations

from promptc.models import OutcomeSpec, SignatureSpec, WorkflowConfig
from promptc.utils import now_utc, stable_hash


class SignatureEngine:
    def build(self, outcome: OutcomeSpec, workflow: WorkflowConfig) -> SignatureSpec:
        fields_in, fields_out = _parse_signature_template(workflow.signature_template)
        signature_id = stable_hash(
            {
                "workflow": workflow.id,
                "outcome": outcome.outcome_type.value,
                "objective": outcome.objective,
                "template": workflow.signature_template,
                "in": fields_in,
                "out": fields_out,
            }
        )
        return SignatureSpec(
            signature_id=signature_id,
            version=1,
            intent=outcome.objective,
            workflow=workflow.id,
            fields_in=fields_in,
            fields_out=fields_out,
            assumptions=outcome.assumptions,
            created_at=now_utc(),
        )

    def bump(self, previous: SignatureSpec, outcome: OutcomeSpec) -> SignatureSpec:
        spec = previous.model_copy(deep=True)
        spec.version += 1
        spec.intent = outcome.objective
        spec.assumptions = outcome.assumptions
        spec.created_at = now_utc()
        return spec


def _parse_signature_template(template: str) -> tuple[list[str], list[str]]:
    if "->" not in template:
        raise ValueError(f"Invalid signature_template, missing '->': {template}")
    left, right = template.split("->", 1)
    fields_in = [x.strip() for x in left.split(",") if x.strip()]
    fields_out = [x.strip() for x in right.split(",") if x.strip()]
    if not fields_in:
        raise ValueError(f"Invalid signature_template, empty input fields: {template}")
    if not fields_out:
        raise ValueError(f"Invalid signature_template, empty output fields: {template}")
    return fields_in, fields_out
