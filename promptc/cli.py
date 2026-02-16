from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
import yaml

from promptc.compiler import CompilerService, load_context_value
from promptc.config import load_policy_config, load_runtime_config
from promptc.dspy.lm import build_dspy_lm, configure_dspy_lm
from promptc.dspy.programs import PromptCompilerProgram
from promptc.engines.dspy_optimize import DSPyOptimizeEngine
from promptc.engines.evaluate import EvalEngine
from promptc.engines.intent import IntentEngine
from promptc.engines.optimize import OptimizeEngine
from promptc.engines.profile import ProfileEngine
from promptc.engines.render import OutputRenderer
from promptc.engines.signature import SignatureEngine
from promptc.engines.workflow import WorkflowEngine
from promptc.models import CompileRequest, OutputFormat, RuntimeConfig, WorkflowConfig
from promptc.storage.fs import (
    FsArtifactRepo,
    FsCacheRepo,
    FsCalibrationRepo,
    FsProfileRepo,
    FsSignatureRepo,
    FsWorkflowRepo,
)

app = typer.Typer(help="promptc: local DSPy-based prompt compiler")
profile_app = typer.Typer(help="manage cognitive profiles")
workflow_app = typer.Typer(help="manage workflows")
artifact_app = typer.Typer(help="inspect compiled artifacts")
eval_app = typer.Typer(help="run evaluation datasets")

app.add_typer(profile_app, name="profile")
app.add_typer(workflow_app, name="workflow")
app.add_typer(artifact_app, name="artifact")
app.add_typer(eval_app, name="eval")


def service_profile_seed(profile_id: str, description: str):
    from promptc.models import CognitiveProfile

    return CognitiveProfile(
        id=profile_id,
        version=1,
        description=description,
        context_patterns=[
            "trusted_instruction_source_only",
            "explicit_delimiter_blocks",
            "abstain_on_missing_context",
        ],
    )


def service_workflow_seed(workflow_id: str, description: str, signature_template: str) -> WorkflowConfig:
    return WorkflowConfig(
        id=workflow_id,
        description=description,
        signature_template=signature_template,
        prompt_style="contract",
        prompt_directives=[
            "State objective and constraints explicitly before any method details",
            "Resolve ambiguity by asking minimal clarifying questions when required inputs are missing",
            "Prefer direct natural language over synthetic template wrappers",
        ],
        quality_checklist=[
            "objective is concrete and testable",
            "constraints are explicit and enforceable",
            "output contract is explicit and verifiable",
            "style matches target audience and profile assumptions",
        ],
        clarification_policy=[
            "Ask only for information needed to safely execute the task",
            "If ambiguity remains, state assumptions clearly before proceeding",
        ],
        ambiguity_resolution_rules=[
            "Prefer explicit assumptions over silent guessing",
            "When constraints conflict, prioritize safety and user-stated hard constraints",
        ],
        forbidden_patterns=[
            "etc",
            "as needed",
            "single-message only",
            "exactly 3 rounds",
            "arbitrary word limit",
            "[section]",
            "[[block]]",
            "<section>",
        ],
        required_sections=[
            "goal",
            "knowns",
            "unknowns",
            "method",
            "evidence_bar",
            "tradeoffs",
            "output_contract",
        ],
        required_prompt_blocks=[
            "instruction_hierarchy",
            "input_boundary",
            "failure_policy",
            "output_contract",
            "final_self_check",
        ],
        required_technique_markers=["objective", "constraints", "method", "evidence", "output contract"],
        vague_markers=["etc", "as needed", "maybe", "could be", "various"],
    )


def _ensure_defaults(project_root: Path) -> None:
    defaults_root = Path(__file__).parent / "defaults"
    for folder in ["profiles", "workflows", "evals"]:
        src = defaults_root / folder
        dst = project_root / folder
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.glob("*"):
            if not f.is_file():
                continue
            out = dst / f.name
            if not out.exists():
                out.write_text(f.read_text())
            elif folder in {"workflows"} and f.suffix.lower() in {".yaml", ".yml"}:
                _merge_yaml_defaults(defaults_path=f, target_path=out)

    for folder in ["artifacts", "signatures", "cache", "calibrations"]:
        (project_root / folder).mkdir(parents=True, exist_ok=True)

    policy_src = defaults_root / "policy.yaml"
    policy_dst = project_root / "policy.yaml"
    if not policy_dst.exists():
        policy_dst.write_text(policy_src.read_text())
    else:
        _merge_yaml_defaults(defaults_path=policy_src, target_path=policy_dst)


def _merge_yaml_defaults(defaults_path: Path, target_path: Path) -> None:
    defaults = yaml.safe_load(defaults_path.read_text()) or {}
    current = yaml.safe_load(target_path.read_text()) or {}
    if not isinstance(defaults, dict) or not isinstance(current, dict):
        return
    merged = _deep_merge(defaults, current)
    if merged != current:
        target_path.write_text(yaml.safe_dump(merged, sort_keys=False))


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _build_service(
    project_root: Path,
    runtime_overrides: dict[str, object],
) -> CompilerService:
    _ensure_defaults(project_root)
    runtime = load_runtime_config(project_root, runtime_overrides)
    policy = load_policy_config(project_root)

    configure_dspy_lm(
        provider=runtime.provider,
        model=runtime.model,
        base_url=runtime.base_url,
        provider_base_url=runtime.provider_base_url,
        api_key=runtime.api_key,
        api_key_env=runtime.api_key_env,
        provider_api_key=runtime.provider_api_key,
        provider_api_key_env=runtime.provider_api_key_env,
        provider_force_json_object=runtime.provider_force_json_object,
        strip_v1_model_prefixes=runtime.strip_v1_model_prefixes,
        temperature=runtime.temperature,
        seed=runtime.seed,
    )
    role_lms = _build_role_lms(runtime)
    program = PromptCompilerProgram(max_retries=runtime.max_retries, role_lms=role_lms)

    return CompilerService(
        profile_repo=FsProfileRepo(project_root / "profiles"),
        workflow_repo=FsWorkflowRepo(project_root / "workflows"),
        signature_repo=FsSignatureRepo(project_root / "signatures"),
        artifact_repo=FsArtifactRepo(project_root / "artifacts"),
        cache_repo=FsCacheRepo(project_root / "cache"),
        intent_engine=IntentEngine(program=program),
        profile_engine=ProfileEngine(program=program, policy=policy),
        signature_engine=SignatureEngine(),
        workflow_engine=WorkflowEngine(policy=policy),
        optimize_engine=OptimizeEngine(program=program),
        eval_engine=EvalEngine(
            program=program,
            runtime=runtime,
            calibration_repo=FsCalibrationRepo(project_root / "calibrations"),
            policy=policy,
        ),
        dspy_optimize_engine=DSPyOptimizeEngine(program=program, policy=policy),
        output_renderer=OutputRenderer(),
        runtime=runtime,
        policy=policy,
    )


def _build_role_lms(runtime: RuntimeConfig) -> dict[str, object]:
    roles = ("intent", "candidate", "judge", "refine", "profile")
    out: dict[str, object] = {}
    for role in roles:
        role_model = runtime.role_models.get(role)
        role_provider = runtime.role_providers.get(role, runtime.provider)
        role_temperature = runtime.role_temperatures.get(role, runtime.temperature)
        if not role_model:
            continue
        out[role] = build_dspy_lm(
            provider=role_provider,
            model=role_model,
            base_url=runtime.base_url,
            provider_base_url=runtime.provider_base_url,
            api_key=runtime.api_key,
            api_key_env=runtime.api_key_env,
            provider_api_key=runtime.provider_api_key,
            provider_api_key_env=runtime.provider_api_key_env,
            provider_force_json_object=runtime.provider_force_json_object,
            strip_v1_model_prefixes=runtime.strip_v1_model_prefixes,
            temperature=role_temperature,
            seed=runtime.seed,
        )
    return out


def _runtime_overrides(
    provider: Optional[str],
    model: Optional[str],
    optimizer: Optional[str],
    judge_count: Optional[int],
    eval_dataset_path: Optional[str],
    parallel_workers: Optional[int],
    rubric_version: Optional[str],
    base_url: Optional[str],
    api_key_env: Optional[str],
    seed: Optional[int],
) -> dict[str, object]:
    return {
        "provider": provider,
        "model": model,
        "optimizer": optimizer,
        "judge_count": judge_count,
        "eval_dataset_path": eval_dataset_path,
        "parallel_workers": parallel_workers,
        "rubric_version": rubric_version,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "seed": seed,
    }


def _run_compile(
    intent: str,
    context: str,
    context_file: Optional[Path],
    workflow: Optional[str],
    profile: Optional[str],
    output: Optional[OutputFormat],
    variants: Optional[int],
    max_iters: Optional[int],
    emit_tests: Optional[bool],
    show_metadata: Optional[bool],
    provider: Optional[str],
    model: Optional[str],
    optimizer: Optional[str],
    judge_count: Optional[int],
    eval_dataset_path: Optional[str],
    parallel_workers: Optional[int],
    rubric_version: Optional[str],
    base_url: Optional[str],
    api_key_env: Optional[str],
    seed: Optional[int],
    parent_artifact_id: Optional[str],
    project_root: Path,
) -> None:
    service = _build_service(
        project_root,
        _runtime_overrides(
            provider=provider,
            model=model,
            optimizer=optimizer,
            judge_count=judge_count,
            eval_dataset_path=eval_dataset_path,
            parallel_workers=parallel_workers,
            rubric_version=rubric_version,
            base_url=base_url,
            api_key_env=api_key_env,
            seed=seed,
        ),
    )

    raw_context = load_context_value(context, context_file)

    runtime = getattr(service, "runtime", None) or RuntimeConfig()
    effective_workflow = workflow or runtime.default_workflow_id
    effective_profile = profile or runtime.default_profile_id
    effective_output = output or runtime.default_output_format
    effective_variants = variants if variants is not None else runtime.default_variants
    effective_max_iters = max_iters if max_iters is not None else runtime.default_max_iters
    effective_emit_tests = emit_tests if emit_tests is not None else runtime.default_emit_tests
    effective_show_metadata = show_metadata if show_metadata is not None else runtime.default_show_metadata
    effective_seed = seed if seed is not None else runtime.seed

    req = CompileRequest(
        intent=intent,
        raw_context=raw_context,
        workflow_id=effective_workflow,
        profile_id=effective_profile,
        output_format=effective_output,
        variants=effective_variants,
        max_iters=effective_max_iters,
        emit_tests=effective_emit_tests,
        seed=effective_seed,
        parent_artifact_id=parent_artifact_id,
    )

    artifact, artifact_path = service.compile(req)
    typer.echo(artifact.prompt_final)
    typer.echo(f"\n[artifact] {artifact_path}")

    if effective_show_metadata:
        typer.echo(json.dumps(artifact.model_dump(mode="json"), indent=2))


@app.command()
def compile(
    intent: str = typer.Option(..., help="Vague intention from user"),
    context: str = typer.Option("", help="Raw context string"),
    context_file: Optional[Path] = typer.Option(None, help="Optional file path for raw context"),
    workflow: Optional[str] = typer.Option(None, help="Workflow id (defaults to config)"),
    profile: Optional[str] = typer.Option(None, help="Cognitive profile id (defaults to config)"),
    output: Optional[OutputFormat] = typer.Option(None, help="Output prompt format (defaults to config)"),
    variants: Optional[int] = typer.Option(None, min=1, max=8, help="Number of prompt variants (defaults to config)"),
    max_iters: Optional[int] = typer.Option(None, min=1, max=10, help="Optimization iterations (defaults to config)"),
    emit_tests: Optional[bool] = typer.Option(None, help="Emit simple prompt checks (defaults to config)"),
    show_metadata: Optional[bool] = typer.Option(None, help="Print metadata JSON (defaults to config)"),
    provider: Optional[str] = typer.Option(None, help="Provider name (defaults to config)"),
    model: Optional[str] = typer.Option(None, help="DSPy model id (defaults to config)"),
    optimizer: Optional[str] = typer.Option(None, help="none|mipro|bootstrap (defaults to config)"),
    judge_count: Optional[int] = typer.Option(None, min=1, max=3, help="Judge count (defaults to config)"),
    eval_dataset_path: Optional[str] = typer.Option(None, help="Path for DSPy optimizer/eval trainset"),
    parallel_workers: Optional[int] = typer.Option(None, min=1, max=32, help="Parallel workers (defaults to config)"),
    rubric_version: Optional[str] = typer.Option(None, help="Rubric version (defaults to config)"),
    base_url: Optional[str] = typer.Option(None, help="Override base URL for the selected provider"),
    api_key_env: Optional[str] = typer.Option(None, help="Environment variable containing API key"),
    seed: Optional[int] = typer.Option(None, help="Deterministic seed tag (defaults to config)"),
    parent_artifact_id: Optional[str] = typer.Option(None, help="Parent artifact id for lineage"),
    project_root: Path = typer.Option(Path(".promptc"), help="Storage root for profiles/workflows/artifacts"),
) -> None:
    _run_compile(
        intent=intent,
        context=context,
        context_file=context_file,
        workflow=workflow,
        profile=profile,
        output=output,
        variants=variants,
        max_iters=max_iters,
        emit_tests=emit_tests,
        show_metadata=show_metadata,
        provider=provider,
        model=model,
        optimizer=optimizer,
        judge_count=judge_count,
        eval_dataset_path=eval_dataset_path,
        parallel_workers=parallel_workers,
        rubric_version=rubric_version,
        base_url=base_url,
        api_key_env=api_key_env,
        seed=seed,
        parent_artifact_id=parent_artifact_id,
        project_root=project_root,
    )


def _workflow_command(default_workflow: str):
    def command(
        intent: str = typer.Option(..., help="Vague intention from user"),
        context: str = typer.Option("", help="Raw context string"),
        context_file: Optional[Path] = typer.Option(None, help="Optional file path for raw context"),
        profile: Optional[str] = typer.Option(None, help="Cognitive profile id (defaults to config)"),
        output: Optional[OutputFormat] = typer.Option(None, help="Output prompt format (defaults to config)"),
        variants: Optional[int] = typer.Option(None, min=1, max=8, help="Number of prompt variants (defaults to config)"),
        max_iters: Optional[int] = typer.Option(None, min=1, max=10, help="Optimization iterations (defaults to config)"),
        emit_tests: Optional[bool] = typer.Option(None, help="Emit simple prompt checks (defaults to config)"),
        show_metadata: Optional[bool] = typer.Option(None, help="Print metadata JSON (defaults to config)"),
        provider: Optional[str] = typer.Option(None, help="Provider name (defaults to config)"),
        model: Optional[str] = typer.Option(None, help="DSPy model id (defaults to config)"),
        optimizer: Optional[str] = typer.Option(None, help="none|mipro|bootstrap (defaults to config)"),
        judge_count: Optional[int] = typer.Option(None, min=1, max=3, help="Judge count (defaults to config)"),
        eval_dataset_path: Optional[str] = typer.Option(None, help="Path for DSPy optimizer/eval trainset"),
        parallel_workers: Optional[int] = typer.Option(None, min=1, max=32, help="Parallel workers (defaults to config)"),
        rubric_version: Optional[str] = typer.Option(None, help="Rubric version (defaults to config)"),
        base_url: Optional[str] = typer.Option(None, help="Override base URL for the selected provider"),
        api_key_env: Optional[str] = typer.Option(None, help="Environment variable containing API key"),
        seed: Optional[int] = typer.Option(None, help="Deterministic seed tag (defaults to config)"),
        parent_artifact_id: Optional[str] = typer.Option(None, help="Parent artifact id for lineage"),
        project_root: Path = typer.Option(Path(".promptc"), help="Storage root for profiles/workflows/artifacts"),
    ) -> None:
        _run_compile(
            intent=intent,
            context=context,
            context_file=context_file,
            workflow=default_workflow,
            profile=profile,
            output=output,
            variants=variants,
            max_iters=max_iters,
            emit_tests=emit_tests,
            show_metadata=show_metadata,
            provider=provider,
            model=model,
            optimizer=optimizer,
            judge_count=judge_count,
            eval_dataset_path=eval_dataset_path,
            parallel_workers=parallel_workers,
            rubric_version=rubric_version,
            base_url=base_url,
            api_key_env=api_key_env,
            seed=seed,
            parent_artifact_id=parent_artifact_id,
            project_root=project_root,
        )

    return command


app.command("explain")(_workflow_command("explain"))
app.command("summarize")(_workflow_command("summarize"))
app.command("analyze")(_workflow_command("analyze"))
app.command("decide")(_workflow_command("decide"))


@profile_app.command("list")
def profile_list(project_root: Path = typer.Option(Path(".promptc"))) -> None:
    _ensure_defaults(project_root)
    repo = FsProfileRepo(project_root / "profiles")
    for profile_id in repo.list_ids():
        versions = repo.list_versions(profile_id)
        suffix = f" (v{versions[-1]})" if versions else ""
        typer.echo(f"{profile_id}{suffix}")


@profile_app.command("create")
def profile_create(
    profile_id: str = typer.Option(..., help="new profile id"),
    description: str = typer.Option("", help="description"),
    intent: str = typer.Option(""),
    workflow_id: Optional[str] = typer.Option(None, help="Workflow id for autonomous generation"),
    provider: Optional[str] = typer.Option(None),
    model: Optional[str] = typer.Option(None),
    project_root: Path = typer.Option(Path(".promptc")),
) -> None:
    _ensure_defaults(project_root)
    if intent.strip():
        service = _build_service(project_root, {"provider": provider, "model": model})
        runtime = service.runtime
        effective_workflow_id = workflow_id or runtime.default_profile_suggest_workflow_id
        workflow = service.workflow_repo.get(effective_workflow_id)
        profile = service.profile_engine.suggest_profile(
            profile_id=profile_id,
            description=description,
            intent=intent,
            workflow=workflow,
        )
        service.profile_repo.put(profile)
        typer.echo(f"created profile: {profile.id}@{profile.version}")
        return

    repo = FsProfileRepo(project_root / "profiles")
    profile = service_profile_seed(profile_id=profile_id, description=description)
    repo.put(profile)
    typer.echo(f"created profile: {profile.id}@{profile.version}")


@profile_app.command("generate")
def profile_generate(
    profile_id: str = typer.Option(...),
    intent: str = typer.Option(...),
    workflow_id: Optional[str] = typer.Option(None, help="Workflow id (defaults to config)"),
    description: str = typer.Option(""),
    provider: Optional[str] = typer.Option(None),
    model: Optional[str] = typer.Option(None),
    project_root: Path = typer.Option(Path(".promptc")),
) -> None:
    service = _build_service(project_root, {"provider": provider, "model": model})
    runtime = service.runtime
    effective_workflow_id = workflow_id or runtime.default_profile_suggest_workflow_id
    workflow = service.workflow_repo.get(effective_workflow_id)
    profile = service.profile_engine.suggest_profile(
        profile_id=profile_id,
        description=description,
        intent=intent,
        workflow=workflow,
    )
    service.profile_repo.put(profile)
    typer.echo(yaml.safe_dump(profile.model_dump(mode="json"), sort_keys=False))


@profile_app.command("suggest")
def profile_suggest(
    profile_id: str = typer.Option(...),
    intent: str = typer.Option(...),
    workflow_id: Optional[str] = typer.Option(None, help="Workflow id (defaults to config)"),
    description: str = typer.Option(""),
    provider: Optional[str] = typer.Option(None),
    model: Optional[str] = typer.Option(None),
    project_root: Path = typer.Option(Path(".promptc")),
) -> None:
    service = _build_service(project_root, {"provider": provider, "model": model})
    runtime = service.runtime
    effective_workflow_id = workflow_id or runtime.default_profile_suggest_workflow_id
    workflow = service.workflow_repo.get(effective_workflow_id)
    profile = service.profile_engine.suggest_profile(
        profile_id=profile_id,
        description=description,
        intent=intent,
        workflow=workflow,
    )
    service.profile_repo.put(profile)
    typer.echo(yaml.safe_dump(profile.model_dump(mode="json"), sort_keys=False))


@profile_app.command("refine")
def profile_refine(
    profile_id: str = typer.Option(...),
    outcome_type: Optional[str] = typer.Option(None, help="Outcome type (defaults to config)"),
    workflow_id: Optional[str] = typer.Option(None, help="Workflow id (defaults to config)"),
    provider: Optional[str] = typer.Option(None),
    model: Optional[str] = typer.Option(None),
    rubric_version: Optional[str] = typer.Option(None),
    api_key_env: Optional[str] = typer.Option(None),
    base_url: Optional[str] = typer.Option(None),
    seed: Optional[int] = typer.Option(None),
    project_root: Path = typer.Option(Path(".promptc")),
) -> None:
    from promptc.models import OutcomeSpec, OutcomeType

    service = _build_service(
        project_root,
        _runtime_overrides(
            provider=provider,
            model=model,
            optimizer=None,
            judge_count=None,
            eval_dataset_path=None,
            parallel_workers=None,
            rubric_version=rubric_version,
            base_url=base_url,
            api_key_env=api_key_env,
            seed=seed,
        ),
    )
    runtime = service.runtime
    effective_outcome_type = outcome_type or runtime.default_profile_refine_outcome_type.value
    effective_workflow_id = workflow_id or runtime.default_profile_refine_workflow_id
    profile = service.profile_repo.get(profile_id)
    try:
        mapped_outcome = OutcomeType(effective_outcome_type)
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid outcome_type: {effective_outcome_type}") from exc
    pseudo_outcome = OutcomeSpec(
        outcome_type=mapped_outcome,
        objective="refine profile",
        success_criteria=[],
        assumptions=[],
        constraints=[],
        ambiguity_flags=[],
    )
    workflow = service.workflow_repo.get(effective_workflow_id)
    refined = service.profile_engine.refine_profile(profile, pseudo_outcome, ["manual refine command"], workflow=workflow)
    service.profile_repo.put(refined)
    typer.echo(f"refined profile: {refined.id}@{refined.version}")


@profile_app.command("history")
def profile_history(
    profile_id: str = typer.Option(...),
    project_root: Path = typer.Option(Path(".promptc")),
) -> None:
    _ensure_defaults(project_root)
    repo = FsProfileRepo(project_root / "profiles")
    versions = repo.list_versions(profile_id)
    if not versions:
        typer.echo("no versioned history")
        return
    for v in versions:
        typer.echo(f"{profile_id}@{v}")


@workflow_app.command("list")
def workflow_list(project_root: Path = typer.Option(Path(".promptc"))) -> None:
    _ensure_defaults(project_root)
    repo = FsWorkflowRepo(project_root / "workflows")
    for item in repo.list_ids():
        typer.echo(item)


@workflow_app.command("show")
def workflow_show(
    workflow_id: str = typer.Option(...),
    project_root: Path = typer.Option(Path(".promptc")),
) -> None:
    _ensure_defaults(project_root)
    repo = FsWorkflowRepo(project_root / "workflows")
    workflow = repo.get(workflow_id)
    typer.echo(yaml.safe_dump(workflow.model_dump(mode="json"), sort_keys=False))


@workflow_app.command("create")
def workflow_create(
    workflow_id: str = typer.Option(..., help="New workflow id"),
    from_workflow: Optional[str] = typer.Option("analyze", help="Template workflow id"),
    description: str = typer.Option("", help="Workflow description"),
    signature_template: Optional[str] = typer.Option(None, help="Override signature template"),
    project_root: Path = typer.Option(Path(".promptc")),
) -> None:
    _ensure_defaults(project_root)
    repo = FsWorkflowRepo(project_root / "workflows")
    if workflow_id in repo.list_ids():
        raise typer.BadParameter(f"Workflow already exists: {workflow_id}")

    template: WorkflowConfig | None = None
    if from_workflow:
        try:
            template = repo.get(from_workflow)
        except FileNotFoundError:
            raise typer.BadParameter(f"Template workflow not found: {from_workflow}") from None

    if template is None:
        workflow = service_workflow_seed(
            workflow_id=workflow_id,
            description=description or f"Custom workflow: {workflow_id}",
            signature_template=signature_template or "input_context, objective, constraints -> compiled_prompt, checks",
        )
    else:
        workflow = template.model_copy(
            update={
                "id": workflow_id,
                "description": description or template.description,
                "signature_template": signature_template or template.signature_template,
            }
        )
    repo.put(workflow)
    typer.echo(f"created workflow: {workflow.id}")


@workflow_app.command("validate")
def workflow_validate(
    workflow_id: str = typer.Option(...),
    project_root: Path = typer.Option(Path(".promptc")),
) -> None:
    _ensure_defaults(project_root)
    repo = FsWorkflowRepo(project_root / "workflows")
    workflow = repo.get(workflow_id)
    policy = load_policy_config(project_root)
    issues = WorkflowEngine(policy=policy).validate(workflow)
    if not issues:
        typer.echo("ok")
        return
    for issue in issues:
        typer.echo(f"- {issue}")


@artifact_app.command("list")
def artifact_list(
    limit: int = typer.Option(20, min=1, max=200),
    project_root: Path = typer.Option(Path(".promptc")),
) -> None:
    _ensure_defaults(project_root)
    repo = FsArtifactRepo(project_root / "artifacts")
    rows = repo.list_recent(limit=limit)
    for row in rows:
        parent = row.parent_artifact_id or "-"
        typer.echo(f"{row.artifact_id}\t{row.created_at.isoformat()}\tparent={parent}\tdepth={row.lineage_depth}")


@artifact_app.command("show")
def artifact_show(
    artifact_id: str = typer.Option(...),
    show_prompt: bool = typer.Option(True),
    project_root: Path = typer.Option(Path(".promptc")),
) -> None:
    _ensure_defaults(project_root)
    repo = FsArtifactRepo(project_root / "artifacts")
    artifact = repo.get(artifact_id)
    typer.echo(json.dumps(artifact.model_dump(mode="json"), indent=2))
    if show_prompt:
        typer.echo("\n--- prompt ---\n")
        typer.echo(artifact.prompt_final)


@artifact_app.command("diff")
def artifact_diff(
    artifact_a: str = typer.Option(...),
    artifact_b: str = typer.Option(...),
    project_root: Path = typer.Option(Path(".promptc")),
) -> None:
    import difflib

    _ensure_defaults(project_root)
    repo = FsArtifactRepo(project_root / "artifacts")
    a = repo.get(artifact_a).prompt_final.splitlines()
    b = repo.get(artifact_b).prompt_final.splitlines()
    for line in difflib.unified_diff(a, b, fromfile=artifact_a, tofile=artifact_b, lineterm=""):
        typer.echo(line)


@artifact_app.command("lineage")
def artifact_lineage(
    artifact_id: str = typer.Option(...),
    project_root: Path = typer.Option(Path(".promptc")),
) -> None:
    _ensure_defaults(project_root)
    repo = FsArtifactRepo(project_root / "artifacts")
    chain = repo.lineage(artifact_id)
    for row in chain:
        parent = row.parent_artifact_id or "-"
        typer.echo(f"{row.artifact_id}\tparent={parent}\tdepth={row.lineage_depth}\t{row.created_at.isoformat()}")


@eval_app.command("run")
def eval_run(
    dataset: Path = typer.Option(..., exists=True, dir_okay=False),
    max_iters: Optional[int] = typer.Option(None, min=1, max=10),
    provider: Optional[str] = typer.Option(None),
    model: Optional[str] = typer.Option(None),
    optimizer: Optional[str] = typer.Option(None),
    judge_count: Optional[int] = typer.Option(None, min=1, max=3),
    parallel_workers: Optional[int] = typer.Option(None, min=1, max=32),
    rubric_version: Optional[str] = typer.Option(None),
    seed: Optional[int] = typer.Option(None),
    project_root: Path = typer.Option(Path(".promptc")),
) -> None:
    service = _build_service(
        project_root,
        _runtime_overrides(
            provider=provider,
            model=model,
            optimizer=optimizer,
            judge_count=judge_count,
            eval_dataset_path=str(dataset),
            parallel_workers=parallel_workers,
            rubric_version=rubric_version,
            base_url=None,
            api_key_env=None,
            seed=seed,
        ),
    )
    effective_max_iters = max_iters if max_iters is not None else service.runtime.default_max_iters
    report = service.evaluate_dataset(dataset, max_iters=effective_max_iters)
    out = project_root / "eval_reports"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"eval-{report.created_at.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report.model_dump(mode="json"), indent=2))
    typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))
    typer.echo(f"[report] {path}")


@eval_app.command("calibration")
def eval_calibration(
    provider: Optional[str] = typer.Option(None),
    model: Optional[str] = typer.Option(None),
    rubric_version: Optional[str] = typer.Option(None),
    project_root: Path = typer.Option(Path(".promptc")),
) -> None:
    _ensure_defaults(project_root)
    runtime = load_runtime_config(project_root, {"provider": provider, "model": model})
    repo = FsCalibrationRepo(project_root / "calibrations")
    effective_rubric_version = rubric_version or runtime.rubric_version
    cal = repo.get(model=runtime.model, rubric_version=effective_rubric_version)
    if not cal:
        typer.echo("no calibration found")
        return
    typer.echo(json.dumps(cal.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    app()
