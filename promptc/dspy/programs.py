from __future__ import annotations

import ast
import json
import re
import time
from typing import Any

from promptc.dspy.lm import get_lm_json_adapter
from promptc.dspy.signatures import CandidatePrompt, CandidateRefine, IntentToOutcome, JudgePrompt, ProfileGenerate, ProfileRefine

try:
    import dspy
except Exception:  # pragma: no cover
    dspy = None

try:
    from dspy.utils.exceptions import AdapterParseError
except Exception:  # pragma: no cover
    AdapterParseError = None


class DSPyUnavailableError(RuntimeError):
    pass


class PromptCompilerProgram:
    def __init__(self, max_retries: int = 2, role_lms: dict[str, Any] | None = None) -> None:
        if not dspy:
            raise DSPyUnavailableError("dspy-ai is not available")
        self.max_retries = max(0, max_retries)
        self.role_lms = role_lms or {}
        self.intent_predict = dspy.ChainOfThought(IntentToOutcome)
        self.candidate_predict = dspy.ChainOfThought(CandidatePrompt)
        self.candidate_predict_by_workflow: dict[str, Any] = {}
        self.judge_predict = dspy.Predict(JudgePrompt)
        self.refine_predict = dspy.ChainOfThought(CandidateRefine)
        self.refine_predict_by_workflow: dict[str, Any] = {}
        self.profile_generate_predict = dspy.ChainOfThought(ProfileGenerate)
        self.profile_refine_predict = dspy.ChainOfThought(ProfileRefine)

    def infer_outcome(self, intention: str, raw_context: str, workflow_hint: str) -> dict[str, Any]:
        out = self._call_with_role(
            "intent",
            self.intent_predict,
            intention=intention,
            raw_context=raw_context,
            workflow_hint=workflow_hint,
        )
        return {
            "outcome_type": str(_required_attr(out, "outcome_type")),
            "objective": str(_required_attr(out, "objective")),
            "success_criteria": _required_json_list(_required_attr(out, "success_criteria")),
            "assumptions": _required_json_list(_required_attr(out, "assumptions")),
            "constraints": _required_json_list(_required_attr(out, "constraints")),
            "ambiguity_flags": _required_json_list(_required_attr(out, "ambiguity_flags")),
        }

    def candidate(
        self,
        outcome_spec_json: str,
        profile_json: str,
        workflow_json: str,
        workflow_id: str,
        output_format: str,
    ) -> dict[str, Any]:
        compiler_guidance = _compiler_guidance_text(output_format=output_format, workflow_json=workflow_json)
        predictor = self._candidate_predictor(workflow_id)
        try:
            out = self._call_with_role(
                "candidate",
                predictor,
                compiler_guidance=compiler_guidance,
                workflow_id=workflow_id,
                outcome_spec_json=outcome_spec_json,
                profile_json=profile_json,
                workflow_json=workflow_json,
                output_format=output_format,
            )
            return {
                "compiled_prompt": str(_required_attr(out, "compiled_prompt")),
                "rationale": str(_required_attr(out, "rationale")),
            }
        except Exception as exc:
            partial = _extract_partial_from_adapter_error(exc)
            if partial and "compiled_prompt" in partial:
                return {
                    "compiled_prompt": str(partial["compiled_prompt"]),
                    "rationale": str(partial.get("rationale", partial.get("reasoning", ""))),
                }
            raise

    def judge(self, candidate_prompt: str, outcome_spec_json: str, workflow_json: str) -> dict[str, Any]:
        rubric_json = _workflow_rubric_json(workflow_json=workflow_json, outcome_spec_json=outcome_spec_json)
        out = self._call_with_role(
            "judge",
            self.judge_predict,
            candidate_prompt=candidate_prompt,
            outcome_spec_json=outcome_spec_json,
            workflow_json=workflow_json,
            workflow_rubric_json=rubric_json,
        )
        return {
            "score_quality": _required_float(_required_attr(out, "score_quality")),
            "score_clarity": _required_float(_required_attr(out, "score_clarity")),
            "score_constraint_fit": _required_float(_required_attr(out, "score_constraint_fit")),
            "risk_flags": _required_json_list(_required_attr(out, "risk_flags")),
        }

    def refine_candidate(
        self,
        candidate_prompt: str,
        judge_feedback_json: str,
        deterministic_penalties_json: str,
        outcome_spec_json: str,
        profile_json: str,
        workflow_json: str,
        output_format: str,
    ) -> dict[str, Any]:
        workflow_id = _workflow_id_from_json(workflow_json)
        predictor = self._refine_predictor(workflow_id)
        refine_guidance = _refine_guidance_text(workflow_json=workflow_json, output_format=output_format)
        try:
            out = self._call_with_role(
                "refine",
                predictor,
                candidate_prompt=candidate_prompt,
                workflow_id=workflow_id,
                refine_guidance=refine_guidance,
                judge_feedback_json=judge_feedback_json,
                deterministic_penalties_json=deterministic_penalties_json,
                outcome_spec_json=outcome_spec_json,
                profile_json=profile_json,
                workflow_json=workflow_json,
                output_format=output_format,
            )
            return {
                "refined_prompt": str(_required_attr(out, "refined_prompt")),
                "refinement_notes": str(_required_attr(out, "refinement_notes")),
            }
        except Exception as exc:
            partial = _extract_partial_from_adapter_error(exc)
            if partial and "refined_prompt" in partial:
                return {
                    "refined_prompt": str(partial["refined_prompt"]),
                    "refinement_notes": str(partial.get("refinement_notes", partial.get("reasoning", ""))),
                }
            raise

    def generate_profile(self, profile_id: str, description: str, intent: str, workflow_json: str) -> dict[str, Any]:
        out = self._call_with_role(
            "profile",
            self.profile_generate_predict,
            profile_id=profile_id,
            description=description,
            intent=intent,
            workflow_json=workflow_json,
        )
        return {
            "profile_yaml": str(_required_attr(out, "profile_yaml")),
            "suitability": _required_float(_required_attr(out, "suitability")),
            "reasons": _required_json_list(_required_attr(out, "reasons")),
        }

    def refine_profile(
        self,
        intent: str,
        workflow_json: str,
        existing_profile_json: str,
        reasons_json: str,
    ) -> dict[str, Any]:
        out = self._call_with_role(
            "profile",
            self.profile_refine_predict,
            intent=intent,
            workflow_json=workflow_json,
            existing_profile_json=existing_profile_json,
            reasons_json=reasons_json,
        )
        return {
            "profile_yaml": str(_required_attr(out, "profile_yaml")),
            "suitability": _required_float(_required_attr(out, "suitability")),
            "reasons": _required_json_list(_required_attr(out, "reasons")),
        }

    def maybe_optimize(self, optimizer_name: str, trainset: list[Any], metric: Any) -> bool:
        if not optimizer_name or optimizer_name == "none":
            return False
        if not trainset:
            raise ValueError("optimizer requested without trainset")

        name = optimizer_name.lower()
        tele = getattr(dspy, "teleprompt", None)
        if tele is None:
            raise RuntimeError("dspy.teleprompt is unavailable")

        if name == "mipro" and hasattr(tele, "MIPROv2"):
            optimizer = tele.MIPROv2(metric=metric)
            self.candidate_predict = optimizer.compile(self.candidate_predict, trainset=trainset)
            for workflow_id, predictor in self.candidate_predict_by_workflow.items():
                compiled = optimizer.compile(predictor, trainset=trainset)
                self.candidate_predict_by_workflow[workflow_id] = compiled
            return True
        if name == "bootstrap" and hasattr(tele, "BootstrapFewShotWithRandomSearch"):
            optimizer = tele.BootstrapFewShotWithRandomSearch(metric=metric)
            self.candidate_predict = optimizer.compile(self.candidate_predict, trainset=trainset)
            for workflow_id, predictor in self.candidate_predict_by_workflow.items():
                compiled = optimizer.compile(predictor, trainset=trainset)
                self.candidate_predict_by_workflow[workflow_id] = compiled
            return True
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _retry_call(self, fn: Any, **kwargs: Any) -> Any:
        last_err: Exception | None = None
        for i in range(self.max_retries + 1):
            try:
                return fn(**kwargs)
            except Exception as exc:  # pragma: no cover
                last_err = exc
                if i < self.max_retries:
                    time.sleep(0.4 * (2**i))
        if last_err:
            raise last_err
        raise RuntimeError("unknown DSPy call failure")

    def _call_with_role(self, role: str, fn: Any, **kwargs: Any) -> Any:
        lm = self.role_lms.get(role)
        if lm is None:
            return self._retry_call(fn, **kwargs)
        context_fn = getattr(dspy, "context", None)
        if context_fn is None:
            return self._retry_call(fn, **kwargs)
        adapter = get_lm_json_adapter(lm)
        if adapter is None:
            with context_fn(lm=lm):
                return self._retry_call(fn, **kwargs)
        with context_fn(lm=lm, adapter=adapter):
            return self._retry_call(fn, **kwargs)

    def _candidate_predictor(self, workflow_id: str) -> Any:
        key = (workflow_id or "").strip().lower()
        if not key:
            return self.candidate_predict
        predictor = self.candidate_predict_by_workflow.get(key)
        if predictor is not None:
            return predictor
        predictor = dspy.ChainOfThought(CandidatePrompt)
        self.candidate_predict_by_workflow[key] = predictor
        return predictor

    def _refine_predictor(self, workflow_id: str) -> Any:
        key = (workflow_id or "").strip().lower()
        if not key:
            return self.refine_predict
        predictor = self.refine_predict_by_workflow.get(key)
        if predictor is not None:
            return predictor
        predictor = dspy.ChainOfThought(CandidateRefine)
        self.refine_predict_by_workflow[key] = predictor
        return predictor


def _required_attr(obj: Any, field: str) -> Any:
    if not hasattr(obj, field):
        raise ValueError(f"Missing DSPy output field: {field}")
    return getattr(obj, field)


def _required_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return _clean_list(value)
    if not isinstance(value, str):
        raise ValueError("Expected JSON list string")
    text = value.strip()
    if not text:
        return []

    # Preferred path: strict JSON list.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return _clean_list(parsed)
    except json.JSONDecodeError:
        pass

    # Common model output fallback: Python-style list with single quotes.
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return _clean_list(parsed)
    except (ValueError, SyntaxError):
        pass

    # Pragmatic fallback for local models: bullets/newlines/comma-separated text.
    if "\n" in text:
        pieces = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line = line.lstrip("-* \t")
            if line:
                pieces.append(line)
        if pieces:
            return _clean_list(pieces)

    if "," in text:
        parts = [p.strip() for p in text.split(",")]
        parts = [p for p in parts if p]
        if parts:
            return _clean_list(parts)

    return _clean_list([text])


def _clean_list(items: list[Any]) -> list[str]:
    out = [str(x).strip() for x in items if str(x).strip()]
    return out


def _required_float(value: Any) -> float:
    n = _coerce_float(value)
    if n < 0.0 or n > 1.0:
        raise ValueError("Expected float in [0, 1]")
    return n


def _coerce_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise ValueError(f"Expected float-like value, got: {type(value)}")

    text = value.strip().lower()
    if not text:
        raise ValueError("Empty float-like string")

    try:
        return float(text)
    except ValueError:
        pass

    # Common local-model artifact: "0. nine" / "1. zero"
    m = re.match(r"^(0|1)\.\s*([a-z]+)$", text)
    if m:
        whole = m.group(1)
        frac_word = m.group(2)
        digit = _word_to_digit(frac_word)
        if digit is not None:
            return float(f"{whole}.{digit}")

    # Fallback: first numeric token in the string.
    token = re.search(r"(?:^|[^0-9.])([01](?:\.\d+)?)", f" {text}")
    if token:
        return float(token.group(1))

    raise ValueError(f"Could not parse float-like value: {value!r}")


def _word_to_digit(word: str) -> str | None:
    mapping = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
    }
    return mapping.get(word)


def _compiler_guidance_text(output_format: str, workflow_json: str) -> str:
    workflow = _workflow_from_json(workflow_json)
    workflow_id = str(workflow.get("id", "")).strip().lower()
    prompt_style = str(workflow.get("prompt_style", "")).strip() or "contract"
    workflow_description = str(workflow.get("description", "")).strip()
    required_sections = _as_clean_list(workflow.get("required_sections", []))
    prompt_directives = _as_clean_list(workflow.get("prompt_directives", []))
    quality_checklist = _as_clean_list(workflow.get("quality_checklist", []))
    clarification_policy = _as_clean_list(workflow.get("clarification_policy", []))
    ambiguity_rules = _as_clean_list(workflow.get("ambiguity_resolution_rules", []))
    forbidden_patterns = _as_clean_list(workflow.get("forbidden_patterns", []))
    workflow_guidance = _workflow_guidance_text(
        workflow_id=workflow_id,
        prompt_style=prompt_style,
        workflow_description=workflow_description,
        required_sections=required_sections,
        prompt_directives=prompt_directives,
        quality_checklist=quality_checklist,
        clarification_policy=clarification_policy,
        ambiguity_rules=ambiguity_rules,
        forbidden_patterns=forbidden_patterns,
    )
    return (
        "You are an expert prompt engineer. Generate a single, production-quality "
        "system prompt that functions as a complete behavior contract.\n\n"
        "## Core Principles\n\n"
        "1. **Role & Identity**: Open with a clear role definition that establishes "
        "who the AI is and what domain expertise it brings. A strong identity anchors "
        "all downstream behavior.\n"
        "2. **Constraint Fidelity**: ONLY include constraints, rules, and policies that "
        "are explicitly stated or directly implied by the user's intent and outcome spec. "
        "NEVER invent platform policies (e.g., 'OpenAI policy', 'academic dishonesty'), "
        "safety restrictions, word limits, citation rules, formatting requirements, or "
        "behavioral restrictions that are not present in the outcome spec. If the user "
        "didn't ask for it, don't add it. The prompt must faithfully represent the user's "
        "actual intent, not what you think they should want.\n"
        "3. **Format Agnosticism**: Do NOT specify output format (LaTeX, Markdown, JSON, "
        "plain text, etc.) unless the user's intent or constraints EXPLICITLY request a "
        "specific format. Let the AI choose the best presentation for the task. Do not "
        "prescribe word counts, section counts, or structural templates unless the user "
        "asked for them.\n"
        "4. **Honor Knowledge Level Deeply**: When the intent says 'explain from scratch', "
        "'assume I know nothing', or similar, this is a HARD constraint. It means: every "
        "single concept must be explained from absolute basics using everyday language. "
        "Do not use ANY technical term without first explaining it as if to someone who "
        "has never encountered it. Do not assume familiarity with ANY domain vocabulary. "
        "Build understanding layer by layer from zero.\n"
        "5. **Specificity Over Vagueness**: Every instruction must be concrete and "
        "actionable. Replace qualifiers like 'thorough' or 'detailed' with measurable "
        "criteria (e.g., 'cover at least 3 distinct failure modes', 'compare minimum "
        "2 alternatives'). Use exact thresholds where they come from the intent.\n"
        "6. **Structured Decomposition**: Break the prompt into clearly labeled sections. "
        "Use Markdown headings (##, ###) for top-level organization AND semantic XML tags "
        "to delimit functional blocks (e.g., `<constraints>`, `<examples>`, "
        "`<reasoning_steps>`, `<failure_policy>`, `<context>`). This mirrors how "
        "production system prompts organize dense instruction sets. Never use bracket-style "
        "pseudo-markup like [Section] or [[block]].\n"
        "7. **Chain-of-Thought Scaffolding**: For tasks requiring reasoning, embed "
        "explicit step-by-step methodology. Tell the model what to think about and in "
        "what order, not just what to output.\n"
        "8. **Instruction Hierarchy**: Place the most critical behavioral rules first. "
        "Mark non-negotiable constraints with IMPORTANT or CRITICAL. Less critical "
        "preferences go later.\n"
        "9. **Positive Then Negative Framing**: State what to do (primary behavior), "
        "then what to avoid (guardrails). Lead with desired behavior, not prohibitions.\n"
        "10. **Output Contract**: End with success/failure criteria that are testable "
        "against the user's actual intent. Do not over-specify format unless requested.\n"
        "11. **Context Boundaries**: Use clear delimiters (XML tags or Markdown sections) "
        "to separate system instructions from user-provided input.\n"
        "12. **Few-Shot When Valuable**: When the task benefits from demonstration, "
        "include 1-2 concrete input/output examples inside an `<examples>` tag.\n"
        "13. **No Friction Unless Requested**: Do not invent procedural gates, "
        "format requirements, word limits, or citation styles not in the intent.\n\n"
        "## Quality Bar\n\n"
        "The generated prompt should rival the density of production system prompts. "
        "Be comprehensive — fill the token budget with actionable, specific instructions "
        "that faithfully serve the user's intent. Every sentence must do work. But density "
        "means depth on WHAT THE USER ASKED FOR, not inventing extra rules they didn't.\n\n"
        "## XML Tag Usage\n\n"
        "Use semantic XML tags (snake_case) to organize prompt sections. Good tags include:\n"
        "- `<role>` — identity and expertise\n"
        "- `<constraints>` — hard behavioral rules FROM the intent\n"
        "- `<method>` or `<reasoning_steps>` — chain-of-thought scaffold\n"
        "- `<examples>` — few-shot demonstrations\n"
        "- `<failure_policy>` — what to do when constraints can't be met\n"
        "- `<context>` — background information or assumptions\n"
        "- `<guardrails>` — safety and boundary rules FROM the intent\n"
        "Tags should wrap content, not replace headings. Combine with Markdown headings.\n\n"
        "## CRITICAL Output Rules\n\n"
        "- Output the prompt text ONLY — no commentary, no wrapper.\n"
        "- Do NOT wrap the output in `<system>`, `<prompt>`, or similar envelope tags. "
        "The rendering layer handles framing. Your output IS the prompt content directly.\n"
        f"- Target format: `{output_format}`.\n\n"
        f"## Workflow Context\n\n{workflow_guidance}"
    )


def _workflow_guidance_text(
    workflow_id: str,
    prompt_style: str,
    workflow_description: str,
    required_sections: list[str],
    prompt_directives: list[str],
    quality_checklist: list[str],
    clarification_policy: list[str],
    ambiguity_rules: list[str],
    forbidden_patterns: list[str],
) -> str:
    parts = []
    if workflow_id:
        parts.append(f"Workflow id: `{workflow_id}`.")
    if prompt_style:
        parts.append(f"Prompt style: `{prompt_style}`.")
    if workflow_description:
        parts.append(f"Workflow intent: {workflow_description}.")
    if required_sections:
        parts.append(f"Required sections to preserve: {_join_items(required_sections)}.")
    if prompt_directives:
        parts.append(f"Workflow directives: {_join_items(prompt_directives)}.")
    if quality_checklist:
        parts.append(f"Quality checklist: {_join_items(quality_checklist)}.")
    if clarification_policy:
        parts.append(f"Ambiguity handling policy: {_join_items(clarification_policy)}.")
    if ambiguity_rules:
        parts.append(f"Assumption rules: {_join_items(ambiguity_rules)}.")
    if forbidden_patterns:
        parts.append(f"Avoid forbidden patterns: {_join_items(forbidden_patterns)}.")
    if not parts:
        return "Emphasize explicit objective, constraints, and output contract."
    return " ".join(parts)


def _refine_guidance_text(workflow_json: str, output_format: str) -> str:
    workflow = _workflow_from_json(workflow_json)
    workflow_id = str(workflow.get("id", "")).strip().lower()
    prompt_directives = _as_clean_list(workflow.get("prompt_directives", []))
    quality_checklist = _as_clean_list(workflow.get("quality_checklist", []))
    forbidden_patterns = _as_clean_list(workflow.get("forbidden_patterns", []))
    required_sections = _as_clean_list(workflow.get("required_sections", []))
    return (
        "You are refining an existing prompt candidate. Apply surgical, targeted "
        "improvements — do not rewrite from scratch.\n\n"
        "## Refinement Protocol\n\n"
        "1. **Diagnose First**: Read the judge feedback and deterministic penalties "
        "carefully. Identify the 2-3 highest-impact weaknesses.\n"
        "2. **Preserve What Works**: Keep sections that scored well. Only modify "
        "what the feedback specifically criticizes.\n"
        "3. **Priority Order**: Fix constraint violations first, then clarity issues, "
        "then quality/depth improvements. Never sacrifice correctness for style.\n"
        "4. **Strip Hallucinated Constraints**: Remove any constraint, policy, or rule "
        "that is NOT present in the outcome spec or user intent. Common hallucinations: "
        "platform policies ('OpenAI policy', 'academic dishonesty'), invented word limits, "
        "forced citation formats, format requirements the user didn't request. If the user "
        "didn't ask for it, remove it.\n"
        "5. **Fix Format Over-Specification**: If the prompt prescribes output format "
        "(LaTeX, Markdown, JSON, word counts, section templates) but the user's intent "
        "didn't request a specific format, remove those prescriptions. Let the AI choose.\n"
        "6. **Honor Knowledge Level**: If the intent says 'assume no knowledge' or similar, "
        "verify EVERY technical term in the prompt is explained from absolute basics. "
        "If the prompt uses domain jargon without everyday-language definitions, fix it.\n"
        "7. **Technique Checklist**: Verify the refined prompt covers:\n"
        "   - Clear role/identity definition\n"
        "   - Specific, measurable instructions from the actual intent\n"
        "   - Step-by-step reasoning structure where appropriate\n"
        "   - Output contract tied to the user's actual success criteria\n"
        "   - Semantic XML tags for structural organization\n"
        "8. **Remove Friction**: Strip invented procedural gates not in the intent.\n"
        "9. **Remove Bracket Markup**: Replace bracket-style tags ([Section], [[block]]) "
        "with Markdown headings or semantic XML tags. Keep proper XML tags.\n"
        "10. **Remove Envelope Tags**: Strip `<system>`, `<prompt>` envelope wrappers.\n\n"
        f"## Constraints\n\n"
        f"- Required sections: {_join_items(required_sections)}\n"
        f"- Workflow directives: {_join_items(prompt_directives)}\n"
        f"- Quality checklist: {_join_items(quality_checklist)}\n"
        f"- Forbidden patterns: {_join_items(forbidden_patterns)}\n"
        f"- Output format: `{output_format}` | Workflow: `{workflow_id}`"
    )


def _workflow_id_from_json(workflow_json: str) -> str:
    try:
        data = json.loads(workflow_json)
    except json.JSONDecodeError:
        return ""
    if not isinstance(data, dict):
        return ""
    value = data.get("id", "")
    return str(value).strip()


def _workflow_rubric_json(workflow_json: str, outcome_spec_json: str) -> str:
    workflow: dict[str, Any] = _workflow_from_json(workflow_json)
    outcome: dict[str, Any] = {}
    try:
        o = json.loads(outcome_spec_json)
        if isinstance(o, dict):
            outcome = o
    except json.JSONDecodeError:
        outcome = {}

    judge_rubric = workflow.get("judge_rubric", {})
    if not isinstance(judge_rubric, dict):
        judge_rubric = {}

    rubric = {
        "workflow_id": str(workflow.get("id", "")),
        "workflow_description": str(workflow.get("description", "")),
        "prompt_style": str(workflow.get("prompt_style", "")),
        "prompt_directives": workflow.get("prompt_directives", []),
        "quality_checklist": workflow.get("quality_checklist", []),
        "clarification_policy": workflow.get("clarification_policy", []),
        "ambiguity_resolution_rules": workflow.get("ambiguity_resolution_rules", []),
        "forbidden_patterns": workflow.get("forbidden_patterns", []),
        "required_sections": workflow.get("required_sections", []),
        "required_prompt_blocks": workflow.get("required_prompt_blocks", []),
        "eval_weights": workflow.get("eval_weights", {}),
        "judge_rubric": judge_rubric,
        "objective": outcome.get("objective", ""),
        "constraints": outcome.get("constraints", []),
        "ambiguity_flags": outcome.get("ambiguity_flags", []),
    }
    return json.dumps(rubric, ensure_ascii=True)


def _workflow_from_json(workflow_json: str) -> dict[str, Any]:
    try:
        parsed = json.loads(workflow_json)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _as_clean_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _join_items(items: list[str]) -> str:
    if not items:
        return "(none)"
    return "; ".join(items)


def _extract_partial_from_adapter_error(exc: Exception) -> dict[str, Any] | None:
    """Extract partial output fields from a DSPy AdapterParseError.

    When the LLM produces most required fields but drops a non-critical one
    (e.g., refinement_notes), DSPy rejects the entire response. This helper
    recovers the usable fields so callers can provide defaults for the missing ones.
    """
    if AdapterParseError is None:
        return None
    if not isinstance(exc, AdapterParseError):
        return None

    # Try parsed_result first (dict of fields that DSPy successfully parsed)
    parsed = getattr(exc, "parsed_result", None)
    if isinstance(parsed, dict) and parsed:
        return parsed

    # Fall back to parsing the raw LM response JSON
    raw = getattr(exc, "lm_response", None)
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    return None
