from __future__ import annotations

import promptc.dspy.lm as lm_config


class _FakeJSONAdapter:
    pass


class _FakeDSPY:
    JSONAdapter = _FakeJSONAdapter

    def __init__(self):
        self.lm_kwargs: list[dict[str, object]] = []
        self.configure_calls: list[dict[str, object]] = []

    def LM(self, **kwargs):
        self.lm_kwargs.append(kwargs)
        return {"kwargs": kwargs}

    def configure(self, **kwargs):
        self.configure_calls.append(kwargs)


def test_configure_dspy_lm_is_provider_agnostic(monkeypatch):
    fake = _FakeDSPY()
    monkeypatch.setattr(lm_config, "dspy", fake)

    lm_config.configure_dspy_lm(
        provider="groq",
        model="groq/llama-3.3-70b",
        provider_api_key={"groq": "test-key"},
    )

    assert fake.lm_kwargs == [{"model": "groq/llama-3.3-70b", "api_key": "test-key"}]
    assert len(fake.configure_calls) == 1
    assert "adapter" not in fake.configure_calls[0]


def test_configure_dspy_lm_uses_provider_specific_config(monkeypatch):
    fake = _FakeDSPY()
    monkeypatch.setattr(lm_config, "dspy", fake)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    lm_config.configure_dspy_lm(
        provider="openrouter",
        model="ollama/phi4",
        provider_base_url={"openrouter": "http://localhost:11434/v1"},
        provider_api_key_env={"openrouter": "OPENROUTER_API_KEY"},
        strip_v1_model_prefixes=["ollama/"],
    )

    assert fake.lm_kwargs == [{"model": "ollama/phi4", "api_key": "or-key", "api_base": "http://localhost:11434"}]


def test_configure_dspy_lm_has_no_implicit_fallbacks(monkeypatch):
    fake = _FakeDSPY()
    monkeypatch.setattr(lm_config, "dspy", fake)

    lm_config.configure_dspy_lm(
        provider="local",
        model="ollama/gpt-oss:120b",
        base_url="http://localhost:11434/v1",
    )

    assert fake.lm_kwargs == [{"model": "ollama/gpt-oss:120b", "api_base": "http://localhost:11434/v1"}]
    assert len(fake.configure_calls) == 1
    assert "adapter" not in fake.configure_calls[0]


def test_configure_dspy_lm_forces_json_mode_only_when_configured(monkeypatch):
    fake = _FakeDSPY()
    monkeypatch.setattr(lm_config, "dspy", fake)

    lm_config.configure_dspy_lm(
        provider="local",
        model="ollama/gpt-oss:120b",
        provider_force_json_object={"local": True},
    )

    assert len(fake.configure_calls) == 1
    assert "adapter" in fake.configure_calls[0]
