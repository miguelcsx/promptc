from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from promptc.dspy.programs import PromptCompilerProgram
from promptc.models import PolicyConfig

try:
    import dspy
except Exception:  # pragma: no cover
    dspy = None


class DSPyOptimizeEngine:
    def __init__(self, program: PromptCompilerProgram, policy: PolicyConfig) -> None:
        self.program = program
        self.policy = policy

    def compile_if_configured(self, optimizer_name: str, dataset_path: str | None) -> bool:
        if not optimizer_name or optimizer_name == "none":
            return False

        trainset = self._load_dataset(dataset_path)
        if not trainset:
            raise ValueError("Optimizer requested but dataset is empty")

        def metric(example: Any, pred: Any, trace: Any | None = None) -> float:
            del trace
            text = getattr(pred, "compiled_prompt", "") if pred is not None else ""
            text_lc = str(text).lower()

            score = self.policy.evaluation.score_floor
            if text_lc.strip():
                score += self.policy.optimizer_metric.non_empty_weight

            expected_terms = _extract_terms(example)
            if expected_terms:
                hits = sum(1 for t in expected_terms if t in text_lc)
                score += self.policy.optimizer_metric.required_terms_weight * (hits / max(1, len(expected_terms)))
            elif "output contract" in text_lc:
                score += self.policy.optimizer_metric.output_contract_bonus

            return max(self.policy.evaluation.score_floor, min(self.policy.evaluation.score_ceiling, score))

        return self.program.maybe_optimize(optimizer_name=optimizer_name, trainset=trainset, metric=metric)

    def _load_dataset(self, dataset_path: str | None) -> list[Any]:
        rows = _read_dataset_rows(dataset_path)
        if not rows:
            return []

        out: list[Any] = []
        for row in rows:
            out.append(_to_dspy_example(row))
        return out


def _read_dataset_rows(dataset_path: str | None) -> list[dict[str, Any]]:
    if not dataset_path:
        return []
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError("JSON dataset must be an array")
        return [x for x in data if isinstance(x, dict)]

    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _to_dspy_example(row: dict[str, Any]) -> Any:
    if dspy is None:
        raise RuntimeError("dspy-ai is required for optimizer datasets")

    intention = str(row.get("intention", ""))
    raw_context = str(row.get("raw_context", ""))
    workflow = str(row.get("workflow", ""))

    example = dspy.Example(
        intention=intention,
        raw_context=raw_context,
        workflow_hint=workflow,
        expected_terms=row.get("required_terms", []),
    )
    return example.with_inputs("intention", "raw_context", "workflow_hint")


def _extract_terms(example: Any) -> list[str]:
    if isinstance(example, dict):
        terms = example.get("required_terms") or example.get("expected_terms") or []
        return [str(t).lower() for t in terms]

    terms = getattr(example, "required_terms", None) or getattr(example, "expected_terms", None) or []
    return [str(t).lower() for t in terms]
