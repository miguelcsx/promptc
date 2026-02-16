from pathlib import Path

from promptc.engines.evaluate import EvalEngine
from promptc.engines.render import OutputRenderer
from promptc.models import CognitiveProfile, OutcomeSpec, OutcomeType, OutputFormat, PolicyConfig, PromptCandidate, RuntimeConfig, WorkflowConfig
from promptc.storage.fs import FsArtifactRepo, FsCacheRepo, FsCalibrationRepo


def _policy() -> PolicyConfig:
    return PolicyConfig.model_validate(
        {
            "search": {
                "improve_epsilon": 0.001,
                "plateau_patience": 2,
                "score_precision": 4,
                "weight_sum_tolerance": 0.001,
                "eval_weight_target": 1.0,
                "default_artifact_list_limit": 20,
                "max_artifact_list_limit": 200,
                "max_variants": 8,
                "max_iters": 10,
            },
            "profile": {"min_prompt_tokens": 250},
            "evaluation": {
                "variability_penalty_cap": 0.15,
                "variability_penalty_scale": 0.5,
                "score_floor": 0.0,
                "score_ceiling": 1.0,
                "round_digits": 4,
            },
            "optimizer_metric": {
                "non_empty_weight": 0.4,
                "required_terms_weight": 0.6,
                "output_contract_bonus": 0.3,
            },
        }
    )


class _FakeProgram:
    def judge(
        self,
        candidate_prompt: str,
        outcome_spec_json: str,
        workflow_json: str,
        workflow_rubric_json: str = "",
    ):
        _ = candidate_prompt, outcome_spec_json, workflow_json, workflow_rubric_json
        return {"score_quality": 0.8, "score_clarity": 0.7, "score_constraint_fit": 0.9}


def test_renderer_formats() -> None:
    r = OutputRenderer()
    base = "Goal:\n- Test"
    plain = r.render(base, OutputFormat.PLAIN)
    chatml = r.render(base, OutputFormat.CHATML)
    schema = r.render(base, OutputFormat.JSON_SCHEMA)

    assert plain.startswith("Goal")
    assert "<|start|>system" in chatml
    assert '"instruction"' in schema


def test_cache_clear(tmp_path: Path) -> None:
    cache = FsCacheRepo(tmp_path / "cache")
    cache.put("k", "v")
    assert cache.get("k") == "v"
    cache.clear()
    assert cache.get("k") is None


def test_artifact_index_roundtrip(tmp_path: Path) -> None:
    from datetime import datetime

    from promptc.models import PromptArtifact, RuntimeConfig

    repo = FsArtifactRepo(tmp_path / "artifacts")
    art = PromptArtifact(
        artifact_id="abc",
        created_at=datetime.utcnow(),
        intent="x",
        workflow="analyze",
        profile="expert@1",
        output_format=OutputFormat.PLAIN,
        prompt_final="hello",
        runtime=RuntimeConfig(),
    )
    repo.save(art)
    got = repo.get("abc")
    assert got.prompt_final == "hello"
    recents = repo.list_recent(5)
    assert recents and recents[0].artifact_id == "abc"


def test_artifact_lineage(tmp_path: Path) -> None:
    from datetime import datetime

    from promptc.models import PromptArtifact, RuntimeConfig

    repo = FsArtifactRepo(tmp_path / "artifacts")
    root = PromptArtifact(
        artifact_id="root",
        created_at=datetime.utcnow(),
        intent="x",
        workflow="analyze",
        profile="expert@1",
        output_format=OutputFormat.PLAIN,
        prompt_final="root",
        runtime=RuntimeConfig(),
    )
    child = root.model_copy(deep=True)
    child.artifact_id = "child"
    child.parent_artifact_id = "root"
    child.lineage_depth = 1
    repo.save(root)
    repo.save(child)

    chain = repo.lineage("child")
    assert [x.artifact_id for x in chain] == ["child", "root"]


def test_calibration_persistence(tmp_path: Path) -> None:
    runtime = RuntimeConfig(model="dummy-model", rubric_version="v1", judge_count=1)
    engine = EvalEngine(
        program=_FakeProgram(),
        runtime=runtime,
        calibration_repo=FsCalibrationRepo(tmp_path / "cal"),
        policy=_policy(),
    )
    candidate = PromptCandidate(prompt="# INSTRUCTION HIERARCHY\n# INPUT BOUNDARY\n# FAILURE POLICY\n# OUTPUT CONTRACT\n# FINAL SELF-CHECK\n")
    profile = CognitiveProfile(
        id="expert",
        context_patterns=[
            "trusted_instruction_source_only",
            "explicit_delimiter_blocks",
            "abstain_on_missing_context",
        ],
    )
    outcome = OutcomeSpec(outcome_type=OutcomeType.ANALYZE, objective="x")
    workflow = WorkflowConfig(
        id="analyze",
        signature_template="x->y",
        required_sections=["goal", "output_contract"],
        required_prompt_blocks=[
            "instruction_hierarchy",
            "input_boundary",
            "failure_policy",
            "output_contract",
            "final_self_check",
        ],
    )
    engine.score(candidate, outcome, profile, workflow)
    cal = FsCalibrationRepo(tmp_path / "cal").get("dummy-model", "v1")
    assert cal is not None
    assert cal.runs >= 1
