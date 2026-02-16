from __future__ import annotations

from pathlib import Path

import yaml

from promptc.models import PolicyConfig, RuntimeConfig


def load_runtime_config(
    project_root: Path, overrides: dict[str, object] | None = None
) -> RuntimeConfig:
    cfg_path = project_root / "config.yaml"
    base: dict[str, object] = {}
    if cfg_path.exists():
        data = yaml.safe_load(cfg_path.read_text()) or {}
        if isinstance(data, dict):
            base = data

    merged = dict(base)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                merged[k] = v
    return RuntimeConfig.model_validate(merged)


def load_policy_config(project_root: Path) -> PolicyConfig:
    path = project_root / "policy.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing policy config: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("policy.yaml must be a mapping")
    return PolicyConfig.model_validate(data)
