from __future__ import annotations

from pathlib import Path

import yaml

from promptc.models import PolicyConfig, RuntimeConfig

# ~/.promptc is the user-global root; per-project roots are usually ./.promptc
GLOBAL_ROOT: Path = Path.home() / ".promptc"


def load_runtime_config(
    project_root: Path, overrides: dict[str, object] | None = None
) -> RuntimeConfig:
    """Load RuntimeConfig with a four-layer precedence:

    1. Global defaults  (~/.promptc/config.yaml)
    2. Project config   (<project_root>/config.yaml)
    3. Local overrides  (<project_root>/config.local.yaml)  — gitignored
    4. CLI overrides    (the ``overrides`` dict)
    """
    base: dict[str, object] = {}

    for cfg_path in [
        GLOBAL_ROOT / "config.yaml",
        project_root / "config.yaml",
        project_root / "config.local.yaml",
    ]:
        layer = _load_yaml_dict(cfg_path)
        base.update(layer)

    if overrides:
        for k, v in overrides.items():
            if v is not None:
                base[k] = v

    return RuntimeConfig.model_validate(base)


def load_policy_config(project_root: Path) -> PolicyConfig:
    """Load policy, preferring project-local over global."""
    for path in [project_root / "policy.yaml", GLOBAL_ROOT / "policy.yaml"]:
        if path.exists():
            data = yaml.safe_load(path.read_text()) or {}
            if not isinstance(data, dict):
                raise ValueError(f"policy.yaml must be a mapping: {path}")
            return PolicyConfig.model_validate(data)
    raise FileNotFoundError(
        f"Missing policy config in {project_root} and {GLOBAL_ROOT}"
    )


def write_config(
    project_root: Path,
    updates: dict[str, object],
    scope: str = "project",
) -> Path:
    """Merge ``updates`` into the target config layer and write to disk."""
    target = GLOBAL_ROOT / "config.yaml" if scope == "global" else project_root / "config.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    current = _load_yaml_dict(target)
    current.update({k: v for k, v in updates.items() if v is not None})
    target.write_text(yaml.safe_dump(current, sort_keys=False))
    return target


def _load_yaml_dict(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    return data if isinstance(data, dict) else {}
