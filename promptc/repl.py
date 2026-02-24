"""Interactive TUI for promptc, built on prompt_toolkit.

Invoked by ``promptc`` (no sub-command). Features:
  - Permanent bottom toolbar with current state (profile / workflow / format / model)
  - Slash-command autocompletion
  - Persistent in-session history (up/down arrows)
  - Natural terminal scrollback — history is never overwritten
  - Ctrl-C cancels current input; Ctrl-D exits
"""
from __future__ import annotations

import difflib
import os
import traceback
from pathlib import Path
from typing import Any

import yaml
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style

from promptc.models import CompileRequest, OutputFormat

# ─── Known providers ───────────────────────────────────────────────────────────

_KNOWN_PROVIDERS: dict[str, tuple[str, str | None]] = {
    "openai":    ("openai/gpt-4o-mini",                    "OPENAI_API_KEY"),
    "anthropic": ("anthropic/claude-haiku-4-5-20251001",   "ANTHROPIC_API_KEY"),
    "groq":      ("groq/llama3-8b-8192",                   "GROQ_API_KEY"),
    "together":  ("together_ai/mistral-7b-instruct",       "TOGETHER_API_KEY"),
    "mistral":   ("mistral/mistral-small-latest",          "MISTRAL_API_KEY"),
    "cohere":    ("cohere/command-r",                      "COHERE_API_KEY"),
    "ollama":    ("ollama/llama3",                         None),
}

# ─── Scalar config keys safe to persist via /config set ───────────────────────

_SCALAR_CONFIG_KEYS = {
    "provider", "model", "base_url", "api_key_env", "optimizer",
    "temperature", "seed", "use_cache", "judge_count", "parallel_workers",
    "max_retries", "rubric_version", "default_workflow_id", "default_profile_id",
    "default_output_format", "default_variants", "default_max_iters",
    "default_emit_tests", "default_show_metadata",
}

# ─── Slash-command registry ────────────────────────────────────────────────────

_COMMANDS: dict[str, tuple[Any, str]] = {}  # name → (handler, one-line description)


def _cmd(name: str, description: str = "") -> Any:
    def decorator(fn: Any) -> Any:
        _COMMANDS[name] = (fn, description)
        return fn
    return decorator


# ─── Session state ─────────────────────────────────────────────────────────────

class _Session:
    """Mutable in-session state (profile, workflow, output format, overrides)."""

    def __init__(
        self,
        project_root: Path,
        profile_id: str,
        workflow_id: str,
        output_format: OutputFormat,
        overrides: dict[str, object],
    ) -> None:
        self.project_root = project_root
        self.profile_id = profile_id
        self.workflow_id = workflow_id
        self.output_format = output_format
        self.overrides = dict(overrides)
        self.context: str = ""  # optional context for next compile
        self._service: Any = None
        self._stale = True

    @property
    def service(self) -> Any:
        if self._service is None or self._stale:
            from promptc.cli import _build_service
            self._service = _build_service(self.project_root, self.overrides)
            self._stale = False
        return self._service

    def invalidate(self) -> None:
        self._stale = True

    def runtime(self) -> Any:
        from promptc.config import load_runtime_config
        return load_runtime_config(self.project_root, self.overrides)

    def toolbar_text(self) -> HTML:
        """Formatted text for the bottom toolbar."""
        rt = self.runtime()
        return HTML(
            f" <b>profile:</b>{self.profile_id}  "
            f"<b>workflow:</b>{self.workflow_id}  "
            f"<b>format:</b>{self.output_format.value}  "
            f"<b>model:</b>{rt.model} "
        )


# ─── Persistence helper ────────────────────────────────────────────────────────

def _persist(s: _Session, updates: dict[str, object]) -> None:
    from promptc.config import write_config
    write_config(s.project_root, updates)


# ─── Autocompleter ─────────────────────────────────────────────────────────────

class _SlashCompleter(Completer):
    """Completes /<command> and their first argument where applicable."""

    _SUB_ARGS: dict[str, list[str]] = {
        "profile":  ["list", "use", "show", "refine", "history"],
        "workflow": ["list", "use", "show", "create", "validate"],
        "format":   [f.value for f in OutputFormat],
        "artifacts":["list", "show", "diff", "lineage"],
        "cache":    ["clear"],
        "config":   ["show", "set"],
        "provider": list(_KNOWN_PROVIDERS.keys()),
        "context":  ["clear"],
    }

    def get_completions(self, document: Any, complete_event: Any):  # type: ignore[override]
        text = document.text_before_cursor
        if not text.startswith("/"):
            return

        parts = text[1:].split(" ", 1)
        if len(parts) == 1:
            # Complete the command name itself
            prefix = parts[0].lower()
            for name in sorted(_COMMANDS):
                if name.startswith(prefix):
                    _, desc = _COMMANDS[name]
                    yield Completion(
                        name[len(prefix):],
                        start_position=0,
                        display=f"/{name}",
                        display_meta=desc,
                    )
        elif len(parts) == 2:
            # Complete sub-arguments for known commands
            cmd = parts[0].lower()
            sub_prefix = parts[1].lower()
            for sub in self._SUB_ARGS.get(cmd, []):
                if sub.startswith(sub_prefix):
                    yield Completion(sub[len(sub_prefix):], start_position=0)


# ─── TUI style ─────────────────────────────────────────────────────────────────

_STYLE = Style.from_dict({
    "bottom-toolbar": "bg:#333333 #aaaaaa",
    "bottom-toolbar b": "bg:#333333 #ffffff bold",
    "prompt": "#00aa00 bold",
})


# ─── Command handlers ──────────────────────────────────────────────────────────

@_cmd("help", "Show all available slash commands")
def _cmd_help(args: str, s: _Session) -> None:
    print()
    width = max(len(n) for n in _COMMANDS) + 2
    for name, (_, desc) in sorted(_COMMANDS.items()):
        print(f"  /{name:<{width}} {desc}")
    print()
    print("  Everything else is treated as a prompt intent and compiled immediately.")
    print()


@_cmd("clear", "Clear the screen")
def _cmd_clear(args: str, s: _Session) -> None:
    print("\033[2J\033[H", end="", flush=True)


@_cmd("exit", "Exit promptc")
def _cmd_exit(args: str, s: _Session) -> None:
    raise SystemExit(0)


@_cmd("quit", "Exit promptc")
def _cmd_quit(args: str, s: _Session) -> None:
    raise SystemExit(0)


@_cmd("config", "Show or set config values.  /config [show | set <key> <value>]")
def _cmd_config(args: str, s: _Session) -> None:
    parts = args.strip().split(maxsplit=2)
    sub = parts[0].lower() if parts else "show"

    if sub in ("show", ""):
        _print_config(s)
    elif sub == "set":
        if len(parts) < 3:
            print("  Usage: /config set <key> <value>")
            return
        key, raw = parts[1], parts[2]
        value: object = raw
        if raw.lower() in ("true", "false"):
            value = raw.lower() == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    pass
        s.overrides[key] = value
        s.invalidate()
        if key in _SCALAR_CONFIG_KEYS:
            _persist(s, {key: value})
            print(f"  {key} = {value!r}  [saved]")
        else:
            print(f"  {key} = {value!r}  [session only]")
    else:
        _print_config(s)


def _print_config(s: _Session) -> None:
    rt = s.runtime()
    print()
    print(f"  profile:     {s.profile_id}")
    print(f"  workflow:    {s.workflow_id}")
    print(f"  format:      {s.output_format.value}")
    print(f"  provider:    {rt.provider}")
    print(f"  model:       {rt.model}")
    print(f"  base_url:    {rt.base_url or '(default)'}")
    print(f"  api_key_env: {rt.api_key_env or '(default)'}")
    print(f"  temperature: {rt.temperature}")
    print(f"  variants:    {rt.default_variants}")
    print(f"  iters:       {rt.default_max_iters}")
    print(f"  optimizer:   {rt.optimizer}")
    print(f"  cache:       {rt.use_cache}")
    print(f"  root:        {s.project_root}")
    print()


@_cmd("provider", "Set LLM provider.  /provider [<name> [<API_KEY_ENV>]]")
def _cmd_provider(args: str, s: _Session) -> None:
    parts = args.strip().split(maxsplit=1)

    if not parts or not parts[0]:
        # List known providers, show current
        rt = s.runtime()
        print()
        print(f"  current: {rt.provider}")
        print()
        print("  known providers:")
        for name, (default_model, key_env) in _KNOWN_PROVIDERS.items():
            marker = " *" if name == rt.provider else ""
            key_hint = f"  (env: {key_env})" if key_env else "  (no API key needed)"
            print(f"    {name:<12}{marker}  →  {default_model}{key_hint}")
        print()
        return

    name = parts[0].lower()
    api_key_env = parts[1].strip() if len(parts) > 1 else None

    updates: dict[str, object] = {"provider": name}

    if name in _KNOWN_PROVIDERS:
        default_model, default_key_env = _KNOWN_PROVIDERS[name]
        if api_key_env:
            updates["api_key_env"] = api_key_env
        elif default_key_env and not api_key_env:
            # Use the default env var name for known providers
            updates["api_key_env"] = default_key_env
        if name == "ollama":
            updates["base_url"] = "http://localhost:11434"
        s.overrides.update(updates)
        s.invalidate()
        _persist(s, updates)
        print(f"  provider → {name}  [saved]")
        print(f"  hint: use /model {default_model}")
        if name == "ollama":
            print("  base_url → http://localhost:11434  [saved]")
    else:
        updates = {"provider": name}
        if api_key_env:
            updates["api_key_env"] = api_key_env
        s.overrides.update(updates)
        s.invalidate()
        _persist(s, updates)
        print(f"  provider → {name}  [saved]")


@_cmd("profile", "Manage profiles.  /profile [list | use <id> | show <id> | refine <id> | history <id>]")
def _cmd_profile(args: str, s: _Session) -> None:
    parts = args.strip().split(maxsplit=1)
    sub = (parts[0].lower() if parts else "list") or "list"
    arg = parts[1].strip() if len(parts) > 1 else ""

    if sub == "list":
        for pid in s.service.profile_repo.list_ids():
            marker = " *" if pid == s.profile_id else ""
            print(f"  {pid}{marker}")
    elif sub == "use":
        if not arg:
            print("  Usage: /profile use <id>")
            return
        try:
            s.service.profile_repo.get(arg)
            s.profile_id = arg
            print(f"  profile → {arg}")
        except FileNotFoundError:
            print(f"  Profile not found: {arg!r}")
    elif sub == "show":
        pid = arg or s.profile_id
        try:
            prof = s.service.profile_repo.get(pid)
            print(yaml.safe_dump(prof.model_dump(mode="json"), sort_keys=False))
        except FileNotFoundError:
            print(f"  Profile not found: {pid!r}")
    elif sub == "history":
        pid = arg or s.profile_id
        try:
            from promptc.storage.fs import FsProfileRepo
            repo = FsProfileRepo(s.project_root / "profiles")
            versions = repo.list_versions(pid)
            if not versions:
                # Try service repo too
                versions = s.service.profile_repo.list_versions(pid) if hasattr(s.service.profile_repo, "list_versions") else []
            if not versions:
                print(f"  No versioned history for {pid!r}")
            else:
                print(f"  {pid}: versions {versions}")
        except Exception as exc:
            print(f"  error: {exc}")
    elif sub == "refine":
        pid = arg or s.profile_id
        try:
            profile = s.service.profile_repo.get(pid)
            workflow = s.service.workflow_repo.get(s.workflow_id)
            refined = s.service.profile_engine.refine_profile(
                profile=profile,
                outcome=_make_pseudo_outcome(pid, workflow),
                reasons=["manual refinement requested via TUI"],
                workflow=workflow,
            )
            s.service.profile_repo.put(refined)
            print(f"  profile {pid!r} refined → v{refined.version}  [saved]")
        except Exception as exc:
            print(f"  error: {exc}")
            if os.environ.get("PROMPTC_DEBUG"):
                traceback.print_exc()
    else:
        print("  Usage: /profile [list | use <id> | show <id> | refine <id> | history <id>]")


@_cmd("workflow", "Manage workflows.  /workflow [list | use <id> | show <id> | create <id> | validate <id>]")
def _cmd_workflow(args: str, s: _Session) -> None:
    parts = args.strip().split(maxsplit=1)
    sub = (parts[0].lower() if parts else "list") or "list"
    arg = parts[1].strip() if len(parts) > 1 else ""

    if sub == "list":
        for wid in s.service.workflow_repo.list_ids():
            marker = " *" if wid == s.workflow_id else ""
            print(f"  {wid}{marker}")
    elif sub == "use":
        if not arg:
            print("  Usage: /workflow use <id>")
            return
        try:
            s.service.workflow_repo.get(arg)
            s.workflow_id = arg
            print(f"  workflow → {arg}")
        except FileNotFoundError:
            print(f"  Workflow not found: {arg!r}")
    elif sub == "show":
        wid = arg or s.workflow_id
        try:
            wf = s.service.workflow_repo.get(wid)
            print(yaml.safe_dump(wf.model_dump(mode="json"), sort_keys=False))
        except FileNotFoundError:
            print(f"  Workflow not found: {wid!r}")
    elif sub == "create":
        if not arg:
            print("  Usage: /workflow create <id>")
            return
        try:
            template = s.service.workflow_repo.get("analyze")
            new_wf = template.model_copy(deep=True)
            new_wf.id = arg
            s.service.workflow_repo.put(new_wf)
            print(f"  workflow {arg!r} created from 'analyze' template  [saved]")
        except FileNotFoundError:
            print("  Template workflow 'analyze' not found — cannot clone")
        except Exception as exc:
            print(f"  error: {exc}")
    elif sub == "validate":
        wid = arg or s.workflow_id
        try:
            wf = s.service.workflow_repo.get(wid)
            issues = s.service.workflow_engine.validate(wf)
            if not issues:
                print(f"  {wid}: ok")
            else:
                print(f"  {wid}: {len(issues)} issue(s)")
                for issue in issues:
                    print(f"    - {issue}")
        except FileNotFoundError:
            print(f"  Workflow not found: {wid!r}")
    else:
        print("  Usage: /workflow [list | use <id> | show <id> | create <id> | validate <id>]")


@_cmd("format", "Set output format.  /format [plain | chatml | json_schema]")
def _cmd_format(args: str, s: _Session) -> None:
    fmt = args.strip().lower()
    if not fmt:
        print(f"  current: {s.output_format.value}")
        print(f"  options: {', '.join(f.value for f in OutputFormat)}")
        return
    try:
        s.output_format = OutputFormat(fmt)
        print(f"  format → {fmt}")
    except ValueError:
        print(f"  Unknown format: {fmt!r}. Options: {', '.join(f.value for f in OutputFormat)}")


@_cmd("model", "Show or set the LLM model.  /model [<provider>/<model>]")
def _cmd_model(args: str, s: _Session) -> None:
    m = args.strip()
    if not m:
        rt = s.runtime()
        print(f"  model: {rt.model}  provider: {rt.provider}")
        return
    s.overrides["model"] = m
    if "/" in m:
        provider_hint = m.split("/")[0]
        s.overrides["provider"] = provider_hint
        _persist(s, {"model": m, "provider": provider_hint})
    else:
        _persist(s, {"model": m})
    s.invalidate()
    print(f"  model → {m}  [saved]")


@_cmd("variants", "Set number of prompt variants.  /variants <n>")
def _cmd_variants(args: str, s: _Session) -> None:
    try:
        n = max(1, int(args.strip()))
    except (ValueError, TypeError):
        print("  Usage: /variants <n>")
        return
    s.overrides["default_variants"] = n
    s.invalidate()
    _persist(s, {"default_variants": n})
    print(f"  variants → {n}  [saved]")


@_cmd("iters", "Set max optimisation iterations.  /iters <n>")
def _cmd_iters(args: str, s: _Session) -> None:
    try:
        n = max(1, int(args.strip()))
    except (ValueError, TypeError):
        print("  Usage: /iters <n>")
        return
    s.overrides["default_max_iters"] = n
    s.invalidate()
    _persist(s, {"default_max_iters": n})
    print(f"  iters → {n}  [saved]")


@_cmd("context", "Set compile context.  /context [<text> | clear]")
def _cmd_context(args: str, s: _Session) -> None:
    text = args.strip()
    if not text:
        if s.context:
            print(f"  context: {s.context!r}")
        else:
            print("  context: (none)")
        return
    if text.lower() == "clear":
        s.context = ""
        print("  context cleared")
    else:
        s.context = text
        print(f"  context set  [session only]")


@_cmd("artifacts", "Browse compiled artifacts.  /artifacts [list | show <id> | diff <a> <b> | lineage <id>]")
def _cmd_artifacts(args: str, s: _Session) -> None:
    parts = args.strip().split(maxsplit=2)
    sub = (parts[0].lower() if parts else "list") or "list"
    arg = parts[1].strip() if len(parts) > 1 else ""
    arg2 = parts[2].strip() if len(parts) > 2 else ""

    if sub == "list":
        rows = s.service.artifact_repo.list_recent(20)
        if not rows:
            print("  (no artifacts yet)")
            return
        for row in rows:
            parent = f"  ← {row.parent_artifact_id[:8]}" if row.parent_artifact_id else ""
            ts = row.created_at.strftime("%Y-%m-%d %H:%M")
            print(f"  {row.artifact_id[:12]}  {ts}{parent}")
    elif sub == "show":
        if not arg:
            print("  Usage: /artifacts show <id>")
            return
        try:
            artifact = s.service.artifact_repo.get(arg)
            print()
            print(artifact.prompt_final)
            print()
        except FileNotFoundError:
            print(f"  Artifact not found: {arg!r}")
    elif sub == "diff":
        if not arg or not arg2:
            print("  Usage: /artifacts diff <id_a> <id_b>")
            return
        try:
            a = s.service.artifact_repo.get(arg)
            b = s.service.artifact_repo.get(arg2)
            a_lines = a.prompt_final.splitlines(keepends=True)
            b_lines = b.prompt_final.splitlines(keepends=True)
            diff = list(difflib.unified_diff(
                a_lines, b_lines,
                fromfile=arg[:12],
                tofile=arg2[:12],
            ))
            if not diff:
                print("  (no differences)")
            else:
                print()
                print("".join(diff))
        except FileNotFoundError as exc:
            print(f"  Artifact not found: {exc}")
    elif sub == "lineage":
        if not arg:
            print("  Usage: /artifacts lineage <id>")
            return
        try:
            chain = s.service.artifact_repo.lineage(arg)
            if not chain:
                print(f"  No lineage found for {arg!r}")
            else:
                for i, entry in enumerate(chain):
                    prefix = "  " + ("└─ " if i == len(chain) - 1 else "├─ ")
                    ts = entry.created_at.strftime("%Y-%m-%d %H:%M")
                    print(f"{prefix}{entry.artifact_id[:12]}  {ts}")
        except FileNotFoundError:
            print(f"  Artifact not found: {arg!r}")
    else:
        print("  Usage: /artifacts [list | show <id> | diff <a> <b> | lineage <id>]")


@_cmd("cache", "Manage the compile cache.  /cache [clear]")
def _cmd_cache(args: str, s: _Session) -> None:
    sub = args.strip().lower()
    if not sub or sub == "clear":
        s.service.cache_repo.clear()
        print("  cache cleared")
    else:
        print("  Usage: /cache [clear]")


# ─── Compile intent ────────────────────────────────────────────────────────────

def _compile_intent(intent: str, s: _Session) -> None:
    rt = s.runtime()
    req = CompileRequest(
        intent=intent,
        raw_context=s.context,
        workflow_id=s.workflow_id,
        profile_id=s.profile_id,
        output_format=s.output_format,
        variants=rt.default_variants,
        max_iters=rt.default_max_iters,
        emit_tests=rt.default_emit_tests,
        seed=rt.seed,
    )
    print("  compiling…", flush=True)
    artifact, artifact_path = s.service.compile(req)

    score = artifact.scores.get("final", 0.0)
    print()
    print(artifact.prompt_final)
    print()
    short = Path(artifact_path).name
    print(f"  [{artifact.artifact_id[:12]}] score={score:.3f}  {short}")
    print()


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _make_pseudo_outcome(profile_id: str, workflow: Any) -> Any:
    from promptc.engines.profile import outcome_from_workflow
    from promptc.models import OutcomeSpec
    return OutcomeSpec(
        outcome_type=outcome_from_workflow(workflow.id),
        objective=f"Refine profile {profile_id!r} for workflow {workflow.id!r}",
        success_criteria=[],
        assumptions=[],
        constraints=[],
        ambiguity_flags=[],
    )


# ─── Command dispatch ──────────────────────────────────────────────────────────

def _dispatch(line: str, s: _Session) -> None:
    stripped = line.lstrip("/")
    parts = stripped.split(maxsplit=1)
    name = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    entry = _COMMANDS.get(name)
    if entry is None:
        print(f"  Unknown command: /{name}  (try /help)")
        return

    handler, _ = entry
    try:
        handler(args, s)
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover
        print(f"  error in /{name}: {exc}")
        if os.environ.get("PROMPTC_DEBUG"):
            traceback.print_exc()


# ─── Public entry point ────────────────────────────────────────────────────────

def run_repl(
    project_root: Path,
    profile_id: str,
    workflow_id: str,
    output_format: OutputFormat,
    overrides: dict[str, object],
) -> None:
    """Start the interactive TUI. Blocks until the user exits (Ctrl-D / /exit)."""
    session = _Session(
        project_root=project_root,
        profile_id=profile_id,
        workflow_id=workflow_id,
        output_format=output_format,
        overrides=overrides,
    )

    history = InMemoryHistory()
    completer = _SlashCompleter()

    prompt_session: PromptSession[str] = PromptSession(
        history=history,
        completer=completer,
        auto_suggest=AutoSuggestFromHistory(),
        style=_STYLE,
        bottom_toolbar=session.toolbar_text,
        refresh_interval=2.0,  # toolbar refreshes so model/profile changes reflect live
        complete_while_typing=True,
    )

    print()
    print("  promptc  —  prompt compiler")
    print()
    print("  Type your intent and press Enter to compile.")
    print("  /help for commands  ·  Tab to autocomplete  ·  Ctrl-D to exit")
    print()

    while True:
        try:
            line = prompt_session.prompt("> ").strip()
        except KeyboardInterrupt:
            # Ctrl-C clears the current line; keep the session running.
            print()
            continue
        except EOFError:
            # Ctrl-D exits cleanly.
            print()
            break

        if not line:
            continue

        if line.startswith("/"):
            _dispatch(line, session)
            continue

        try:
            _compile_intent(line, session)
        except KeyboardInterrupt:
            print("\n  (cancelled)")
        except Exception as exc:
            print(f"\n  error: {exc}")
            if os.environ.get("PROMPTC_DEBUG"):
                traceback.print_exc()
