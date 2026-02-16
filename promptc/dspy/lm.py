from __future__ import annotations

import os
from urllib.parse import urlsplit, urlunsplit

try:
    import dspy
except Exception:  # pragma: no cover
    dspy = None

_JSON_ADAPTERS: dict[int, object] = {}


def configure_dspy_lm(
    provider: str,
    model: str,
    base_url: str | None = None,
    provider_base_url: dict[str, str] | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
    provider_api_key: dict[str, str] | None = None,
    provider_api_key_env: dict[str, str] | None = None,
    provider_force_json_object: dict[str, bool] | None = None,
    strip_v1_model_prefixes: list[str] | None = None,
    temperature: float | None = None,
    seed: int | None = None,
) -> object:
    if not dspy:
        raise RuntimeError("dspy-ai is not installed")

    lm = build_dspy_lm(
        provider=provider,
        model=model,
        base_url=base_url,
        provider_base_url=provider_base_url,
        api_key=api_key,
        api_key_env=api_key_env,
        provider_api_key=provider_api_key,
        provider_api_key_env=provider_api_key_env,
        provider_force_json_object=provider_force_json_object,
        strip_v1_model_prefixes=strip_v1_model_prefixes,
        temperature=temperature,
        seed=seed,
    )
    adapter = get_lm_json_adapter(lm)
    if adapter is not None:
        dspy.configure(lm=lm, adapter=adapter)
        return lm
    dspy.configure(lm=lm)
    return lm


def build_dspy_lm(
    provider: str,
    model: str,
    base_url: str | None = None,
    provider_base_url: dict[str, str] | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
    provider_api_key: dict[str, str] | None = None,
    provider_api_key_env: dict[str, str] | None = None,
    provider_force_json_object: dict[str, bool] | None = None,
    strip_v1_model_prefixes: list[str] | None = None,
    temperature: float | None = None,
    seed: int | None = None,
) -> object:
    if not dspy:
        raise RuntimeError("dspy-ai is not installed")

    provider_key = (provider or "").strip().lower()
    provider_base_url = provider_base_url or {}
    provider_api_key = provider_api_key or {}
    provider_api_key_env = provider_api_key_env or {}
    provider_force_json_object = provider_force_json_object or {}

    resolved_api_key = _resolve_api_key(
        provider=provider_key,
        api_key=api_key,
        api_key_env=api_key_env,
        provider_api_key=provider_api_key,
        provider_api_key_env=provider_api_key_env,
    )
    resolved_base_url = _resolve_base_url(
        provider=provider_key,
        model=model,
        base_url=base_url,
        provider_base_url=provider_base_url,
        strip_v1_model_prefixes=strip_v1_model_prefixes,
    )

    kwargs = {"model": model}
    if resolved_api_key:
        kwargs["api_key"] = resolved_api_key
    if resolved_base_url:
        kwargs["api_base"] = resolved_base_url
    if temperature is not None:
        kwargs["temperature"] = temperature
    if seed is not None:
        kwargs["seed"] = seed
    lm = dspy.LM(**kwargs)

    force_json_object = _should_force_json_object(
        provider=provider_key,
        provider_force_json_object=provider_force_json_object,
    )
    if force_json_object:
        # Some providers reject DSPy's structured-output schema. Force JSON mode when configured.
        class _JSONModeAdapter(dspy.JSONAdapter):
            def __call__(self, lm, lm_kwargs, signature, demos, inputs):
                from dspy.adapters.chat_adapter import ChatAdapter

                forced_kwargs = dict(lm_kwargs)
                forced_kwargs["response_format"] = {"type": "json_object"}
                return ChatAdapter.__call__(
                    self, lm, forced_kwargs, signature, demos, inputs
                )

            async def acall(self, lm, lm_kwargs, signature, demos, inputs):
                from dspy.adapters.chat_adapter import ChatAdapter

                forced_kwargs = dict(lm_kwargs)
                forced_kwargs["response_format"] = {"type": "json_object"}
                return await ChatAdapter.acall(
                    self, lm, forced_kwargs, signature, demos, inputs
                )

        _JSON_ADAPTERS[id(lm)] = _JSONModeAdapter()
    return lm


def get_lm_json_adapter(lm: object) -> object | None:
    return _JSON_ADAPTERS.get(id(lm))


def _resolve_api_key(
    provider: str,
    api_key: str | None,
    api_key_env: str | None,
    provider_api_key: dict[str, str],
    provider_api_key_env: dict[str, str],
) -> str | None:
    if api_key:
        return api_key

    provider_key = provider.strip().lower()
    if provider_key and provider_api_key.get(provider_key):
        return provider_api_key[provider_key]

    provider_env = provider_api_key_env.get(provider_key) if provider_key else None
    if provider_env:
        provider_env_value = os.getenv(provider_env)
        if provider_env_value:
            return provider_env_value

    if api_key_env:
        return os.getenv(api_key_env)
    return None


def _resolve_base_url(
    provider: str,
    model: str,
    base_url: str | None,
    provider_base_url: dict[str, str],
    strip_v1_model_prefixes: list[str] | None,
) -> str | None:
    provider_key = provider.strip().lower()
    resolved = base_url or (
        provider_base_url.get(provider_key) if provider_key else None
    )
    return _normalize_api_base(
        model=model,
        base_url=resolved,
        strip_v1_model_prefixes=strip_v1_model_prefixes,
    )


def _normalize_api_base(
    model: str, base_url: str | None, strip_v1_model_prefixes: list[str] | None
) -> str | None:
    if not base_url:
        return base_url

    prefixes = strip_v1_model_prefixes or []
    if any(model.startswith(prefix) for prefix in prefixes if prefix):
        parsed = urlsplit(base_url)
        path = parsed.path.rstrip("/")
        if path.endswith("/v1"):
            new_path = path[:-3] or ""
            return urlunsplit(
                (parsed.scheme, parsed.netloc, new_path, parsed.query, parsed.fragment)
            )
    return base_url


def _should_force_json_object(
    provider: str,
    provider_force_json_object: dict[str, bool],
) -> bool:
    provider_key = provider.strip().lower()
    return provider_force_json_object.get(provider_key, False)
