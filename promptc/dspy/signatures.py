from __future__ import annotations

try:
    import dspy
except Exception:  # pragma: no cover
    dspy = None


if dspy:

    class IntentToOutcome(dspy.Signature):
        """Convert a raw user intent into a precise outcome contract for prompt compilation.

        Analyze the intent to extract:
        - A single, concrete objective sentence that passes the "could I verify this was achieved?" test
        - Measurable success criteria — each one should be independently testable (not vague like "good quality")
        - Explicit assumptions introduced to fill gaps in the user's request
        - Hard behavioral/content constraints that the final prompt must enforce
        - Unresolved ambiguities that could materially change the prompt if clarified
        """

        intention = dspy.InputField(desc="Raw user request, possibly vague or underspecified")
        raw_context = dspy.InputField(desc="Optional contextual details from the user")
        workflow_hint = dspy.InputField(desc="Preferred workflow if user provides one (e.g., analyze, explain, decide)")
        outcome_type = dspy.OutputField(desc="One of explain/summarize/analyze/decide/compile — choose based on the dominant task verb")
        objective = dspy.OutputField(desc="Single concrete objective sentence; must be specific enough that a third party could verify completion")
        success_criteria = dspy.OutputField(desc="JSON list of measurable outcomes; each must be independently testable, not subjective")
        assumptions = dspy.OutputField(desc="JSON list of explicit assumptions introduced to compensate for missing information")
        constraints = dspy.OutputField(desc="JSON list of hard behavioral or content constraints the prompt must enforce — include knowledge level requirements (e.g., 'explain from absolute basics, assume zero domain knowledge') when the user indicates them")
        ambiguity_flags = dspy.OutputField(desc="JSON list of unresolved ambiguities that could materially change the prompt")


    class CandidatePrompt(dspy.Signature):
        """Generate a production-quality behavior-contract prompt.

        The prompt must function as a complete specification for AI behavior. Key criteria:

        1. Opens with a clear role/identity that anchors all downstream behavior
        2. ONLY includes constraints from the outcome spec — never invent platform policies,
           safety rules, format requirements, or behavioral restrictions not in the intent
        3. Does NOT specify output format unless the user explicitly requested one — let the
           AI choose the best presentation (Markdown, LaTeX, etc.)
        4. When the intent says "assume no knowledge" or similar, EVERY concept must be
           explained from absolute basics in everyday language, no domain jargon assumed
        5. Complex tasks include step-by-step reasoning scaffolding (chain-of-thought)
        6. Uses semantic XML tags for structural organization (<constraints>, <method>, etc.)
        7. Critical rules from the intent appear first; preferences come later
        8. Ends with success/failure criteria tied to the user's actual intent
        9. Zero filler — every sentence does work; no hallucinated constraints or rules
        """

        compiler_guidance = dspy.InputField(desc="Detailed prompt engineering principles and workflow-specific generation rules")
        workflow_id = dspy.InputField(desc="Workflow id from filesystem config (e.g., analyze, explain, decide, summarize)")
        outcome_spec_json = dspy.InputField(desc="Outcome spec JSON: objective, success_criteria, assumptions, constraints")
        profile_json = dspy.InputField(desc="Cognitive profile JSON: style preferences, constraint limits, prompt policies")
        workflow_json = dspy.InputField(desc="Workflow JSON: required sections, directives, forbidden patterns, quality checklist")
        output_format = dspy.InputField(desc="Target output format: plain/chatml/json_schema")
        compiled_prompt = dspy.OutputField(
            desc=(
                "The complete prompt text. Must open with a role definition, use Markdown headings "
                "AND semantic XML tags (e.g., <constraints>, <examples>, <method>) for structure, "
                "include all required workflow sections, and close with testable success criteria. "
                "ONLY include constraints from the outcome spec — never hallucinate policies or "
                "format requirements not in the intent. Do NOT specify output format unless the "
                "intent explicitly requests one. Do NOT wrap in <system> or <prompt> envelope tags. "
                "No commentary — prompt content only."
            )
        )
        rationale = dspy.OutputField(desc="Brief explanation of design choices: which techniques were applied and why they fit this task")


    class JudgePrompt(dspy.Signature):
        """Score a prompt candidate on three dimensions using the workflow rubric.

        Scoring rubric:
        - quality (0..1): Does the prompt produce useful, specific, task-fit outputs?
          High: concrete methods, measurable criteria, domain-appropriate depth.
          Low: vague instructions, generic advice, missing key task requirements.
        - clarity (0..1): Is the prompt unambiguous, readable, and non-redundant?
          High: each instruction is distinct, well-ordered, no contradictions.
          Low: overlapping instructions, unclear scope, verbose filler.
        - constraint_fit (0..1): Does the prompt enforce all hard constraints?
          High: all required sections present, forbidden patterns absent, output contract testable.
          Low: missing sections, includes forbidden patterns, vague output spec.
        """

        candidate_prompt = dspy.InputField(desc="The prompt candidate to evaluate")
        outcome_spec_json = dspy.InputField(desc="Target objective, success criteria, and constraints")
        workflow_json = dspy.InputField(desc="Workflow contract: required sections, directives, eval weights")
        workflow_rubric_json = dspy.InputField(desc="Structured scoring rubric derived from workflow: priorities, quality/clarity/constraint/risk checks")
        score_quality = dspy.OutputField(desc="0..1 float: usefulness, specificity, technique depth, and task fitness")
        score_clarity = dspy.OutputField(desc="0..1 float: unambiguous, well-structured, readable, non-redundant")
        score_constraint_fit = dspy.OutputField(desc="0..1 float: required sections present, forbidden patterns absent, output contract testable")
        risk_flags = dspy.OutputField(desc="JSON list of specific risks: e.g., 'missing output contract', 'vague method section', 'contradictory constraints'")


    class CandidateRefine(dspy.Signature):
        """Surgically refine a prompt candidate using evaluator feedback.

        Refinement principles:
        - Diagnose the 2-3 highest-impact weaknesses from judge scores and deterministic penalties
        - Apply targeted fixes to weak areas; preserve sections that already score well
        - Fix constraint violations first, then clarity, then depth
        - Verify technique coverage: role definition, specificity, reasoning structure, output contract
        - Remove friction artifacts (invented gates, custom markup) unless workflow-required
        """

        candidate_prompt = dspy.InputField(desc="Current prompt candidate to improve")
        workflow_id = dspy.InputField(desc="Workflow id from filesystem config")
        refine_guidance = dspy.InputField(desc="Detailed refinement protocol: diagnosis steps, priority order, technique checklist")
        judge_feedback_json = dspy.InputField(desc="Structured feedback: quality/clarity/constraint_fit scores and risk_flags list")
        deterministic_penalties_json = dspy.InputField(desc="Penalty breakdown: bloat, redundancy, ambiguity, missing_sections, missing_techniques, friction")
        outcome_spec_json = dspy.InputField(desc="Target outcome spec with objective and constraints")
        profile_json = dspy.InputField(desc="Active cognitive profile with style and constraint limits")
        workflow_json = dspy.InputField(desc="Workflow contract: required sections, directives, forbidden patterns")
        output_format = dspy.InputField(desc="Target output format: plain/chatml/json_schema")
        refined_prompt = dspy.OutputField(
            desc="Improved prompt text. Must fix identified weaknesses while preserving strong sections. Use semantic XML tags for structure. Do NOT wrap in <system> envelope tags. Prompt content only."
        )
        refinement_notes = dspy.OutputField(desc="Brief summary: what was changed and why, referencing specific feedback items")


    class ProfileGenerate(dspy.Signature):
        """Generate a new cognitive profile from intent/workflow requirements."""

        profile_id = dspy.InputField(desc="Target profile id")
        description = dspy.InputField(desc="Desired profile description")
        intent = dspy.InputField(desc="User intent or objective for the profile")
        workflow_json = dspy.InputField(desc="Workflow contract")
        profile_yaml = dspy.OutputField(desc="YAML content for a complete profile")
        suitability = dspy.OutputField(desc="0..1")
        reasons = dspy.OutputField(desc="JSON list")


    class ProfileRefine(dspy.Signature):
        """Refine an existing cognitive profile using suitability reasons."""

        intent = dspy.InputField(desc="Current intent/objective")
        workflow_json = dspy.InputField(desc="Workflow contract")
        existing_profile_json = dspy.InputField(desc="Existing profile JSON")
        reasons_json = dspy.InputField(desc="JSON list of suitability/refinement reasons")
        profile_yaml = dspy.OutputField(desc="YAML content for the refined profile")
        suitability = dspy.OutputField(desc="0..1")
        reasons = dspy.OutputField(desc="JSON list")
