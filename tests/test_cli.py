from datetime import datetime
from pathlib import Path

from typer.testing import CliRunner

import promptc.cli as cli
from promptc.models import ConvergenceInfo, OutputFormat, PromptArtifact, RuntimeConfig


class _FakeService:
    def compile(self, req):
        art = PromptArtifact(
            artifact_id="fake1",
            created_at=datetime.utcnow(),
            intent=req.intent,
            workflow=req.workflow_id,
            profile=f"{req.profile_id}@1",
            output_format=req.output_format,
            prompt_final="PROMPT",
            convergence=ConvergenceInfo(iterations=1, stopped_reason="score_plateau", frontier_size=1),
            runtime=RuntimeConfig(),
        )
        return art, "/tmp/fake1.json"


def test_compile_cli_smoke(monkeypatch):
    runner = CliRunner()

    def _fake_build_service(project_root: Path, runtime_overrides: dict[str, object]):
        _ = project_root, runtime_overrides
        return _FakeService()

    monkeypatch.setattr(cli, "_build_service", _fake_build_service)

    result = runner.invoke(
        cli.app,
        [
            "compile",
            "--intent",
            "I want to analyze this",
            "--workflow",
            "analyze",
            "--profile",
            "expert",
            "--output",
            OutputFormat.PLAIN.value,
        ],
    )
    assert result.exit_code == 0
    assert "PROMPT" in result.stdout
