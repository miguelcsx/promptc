from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from promptc.models import ArtifactIndexEntry, CognitiveProfile, JudgeCalibration, PromptArtifact, SignatureSpec, WorkflowConfig
from promptc.storage.interfaces import ArtifactRepo, CacheRepo, CalibrationRepo, ProfileRepo, SignatureRepo, WorkflowRepo
from promptc.utils import now_utc


class FsProfileRepo(ProfileRepo):
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, profile_id: str) -> CognitiveProfile:
        if "@" in profile_id:
            base_id, version_text = profile_id.split("@", 1)
            version = int(version_text)
            path = self.root / f"{base_id}.v{version}.yaml"
            if not path.exists():
                raise FileNotFoundError(f"Profile version not found: {path}")
            data = yaml.safe_load(path.read_text()) or {}
            return CognitiveProfile.model_validate(data)

        version_paths = self._versioned_paths(profile_id)
        if version_paths:
            data = yaml.safe_load(version_paths[-1].read_text()) or {}
            return CognitiveProfile.model_validate(data)

        # Transitional fallback for non-versioned defaults.
        path = self.root / f"{profile_id}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Profile not found: {path}")
        data = yaml.safe_load(path.read_text()) or {}
        return CognitiveProfile.model_validate(data)

    def put(self, profile: CognitiveProfile) -> None:
        versions = self.list_versions(profile.id)
        latest = versions[-1] if versions else 0
        effective_version = profile.version if profile.version > latest else latest + 1
        profile.version = effective_version
        path = self.root / f"{profile.id}.v{effective_version}.yaml"
        path.write_text(yaml.safe_dump(profile.model_dump(mode="json"), sort_keys=False))

    def list_ids(self) -> list[str]:
        ids = set()
        for p in self.root.glob("*.v*.yaml"):
            stem = p.stem
            if ".v" in stem:
                ids.add(stem.rsplit(".v", 1)[0])
        for p in self.root.glob("*.yaml"):
            stem = p.stem
            if ".v" in stem:
                continue
            ids.add(stem)
        return sorted(ids)

    def list_versions(self, profile_id: str) -> list[int]:
        versions: list[int] = []
        for p in self._versioned_paths(profile_id):
            stem = p.stem
            _, v_text = stem.rsplit(".v", 1)
            try:
                versions.append(int(v_text))
            except ValueError:
                continue
        return sorted(versions)

    def _versioned_paths(self, profile_id: str) -> list[Path]:
        out: list[tuple[int, Path]] = []
        for p in self.root.glob(f"{profile_id}.v*.yaml"):
            stem = p.stem
            if ".v" not in stem:
                continue
            _, v_text = stem.rsplit(".v", 1)
            try:
                version = int(v_text)
            except ValueError:
                continue
            out.append((version, p))
        out.sort(key=lambda x: x[0])
        return [p for _, p in out]


class FsWorkflowRepo(WorkflowRepo):
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, workflow_id: str) -> WorkflowConfig:
        path = self.root / f"{workflow_id}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Workflow not found: {path}")
        data = yaml.safe_load(path.read_text()) or {}
        return WorkflowConfig.model_validate(data)

    def put(self, workflow: WorkflowConfig) -> None:
        path = self.root / f"{workflow.id}.yaml"
        path.write_text(yaml.safe_dump(workflow.model_dump(mode="json"), sort_keys=False))

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.yaml"))


class FsSignatureRepo(SignatureRepo):
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths_for(self, signature_id: str) -> list[Path]:
        return sorted(self.root.glob(f"{signature_id}.v*.json"))

    def get_latest(self, signature_id: str) -> SignatureSpec | None:
        paths = self._paths_for(signature_id)
        if not paths:
            return None
        data = json.loads(paths[-1].read_text())
        return SignatureSpec.model_validate(data)

    def put(self, spec: SignatureSpec) -> str:
        path = self.root / f"{spec.signature_id}.v{spec.version}.json"
        path.write_text(json.dumps(spec.model_dump(mode="json"), indent=2))
        return str(path)


class FsArtifactRepo(ArtifactRepo):
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.map_path = self.root / "index_map.json"
        self._index_cache: list[dict[str, object]] | None = None
        self._map_cache: dict[str, str] | None = None
        if not self.index_path.exists():
            self.index_path.write_text("[]")
        if not self.map_path.exists():
            self.map_path.write_text("{}")

    def _load_index(self) -> list[dict[str, object]]:
        if self._index_cache is None:
            self._index_cache = json.loads(self.index_path.read_text())
        return self._index_cache

    def _save_index(self, rows: list[dict[str, object]]) -> None:
        self.index_path.write_text(json.dumps(rows, indent=2))
        self._index_cache = rows

    def _load_map(self) -> dict[str, str]:
        if self._map_cache is None:
            self._map_cache = json.loads(self.map_path.read_text())
        return self._map_cache

    def _save_map(self, m: dict[str, str]) -> None:
        self.map_path.write_text(json.dumps(m, indent=2))
        self._map_cache = m

    def save(self, artifact: PromptArtifact) -> str:
        day_dir = self.root / datetime.utcnow().strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{artifact.artifact_id}.json"
        out_path = day_dir / filename
        out_path.write_text(json.dumps(artifact.model_dump(mode="json"), indent=2))

        index = self._load_index()
        index.append(
            ArtifactIndexEntry(
                artifact_id=artifact.artifact_id,
                path=str(out_path),
                created_at=artifact.created_at,
                parent_artifact_id=artifact.parent_artifact_id,
                lineage_depth=artifact.lineage_depth,
            ).model_dump(mode="json")
        )
        self._save_index(index)

        index_map = self._load_map()
        index_map[artifact.artifact_id] = str(out_path)
        self._save_map(index_map)
        return str(out_path)

    def _lookup_path(self, artifact_id: str) -> Path:
        index_map = self._load_map()
        path = index_map.get(artifact_id)
        if path:
            return Path(path)

        # Backfill map for old entries once.
        for row in reversed(self._load_index()):
            if row.get("artifact_id") == artifact_id:
                resolved = str(row["path"])
                index_map[artifact_id] = resolved
                self._save_map(index_map)
                return Path(resolved)
        raise FileNotFoundError(f"Artifact not found in index: {artifact_id}")

    def get(self, artifact_id: str) -> PromptArtifact:
        return self.get_by_path(str(self._lookup_path(artifact_id)))

    def get_by_path(self, artifact_path: str) -> PromptArtifact:
        data = json.loads(Path(artifact_path).read_text())
        return PromptArtifact.model_validate(data)

    def list_recent(self, limit: int = 20) -> list[ArtifactIndexEntry]:
        rows = self._load_index()
        selected = rows[-max(1, limit) :]
        return [ArtifactIndexEntry.model_validate(r) for r in reversed(selected)]

    def lineage(self, artifact_id: str) -> list[ArtifactIndexEntry]:
        rows = [ArtifactIndexEntry.model_validate(r) for r in self._load_index()]
        by_id = {row.artifact_id: row for row in rows}

        chain: list[ArtifactIndexEntry] = []
        current = by_id.get(artifact_id)
        while current:
            chain.append(current)
            if not current.parent_artifact_id:
                break
            current = by_id.get(current.parent_artifact_id)
        return chain


class FsCacheRepo(CacheRepo):
    def __init__(self, root: Path, max_entries: int = 512):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "cache.json"
        self.max_entries = max(32, max_entries)
        if not self.path.exists():
            self.path.write_text("{}")

    def _load(self) -> dict[str, Any]:
        data = json.loads(self.path.read_text())
        if not isinstance(data, dict):
            return {}
        return data

    def _save(self, data: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(data, indent=2))

    def get(self, key: str) -> str | None:
        value = self._load().get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            path = value.get("path")
            if isinstance(path, str) and path:
                return path
        return None

    def put(self, key: str, artifact_path: str) -> None:
        data = self._load()
        data[key] = {"path": artifact_path, "created_at": now_utc().isoformat()}
        data = self._prune(data)
        self._save(data)

    def clear(self) -> None:
        self._save({})

    def _prune(self, data: dict[str, Any]) -> dict[str, Any]:
        # Drop corrupted entries and paths that no longer exist.
        normalized: dict[str, Any] = {}
        for key, value in data.items():
            path = ""
            created_at = ""
            if isinstance(value, str):
                path = value
            elif isinstance(value, dict):
                raw_path = value.get("path")
                raw_created = value.get("created_at")
                if isinstance(raw_path, str):
                    path = raw_path
                if isinstance(raw_created, str):
                    created_at = raw_created
            if not path:
                continue
            normalized[key] = {"path": path, "created_at": created_at}

        if len(normalized) <= self.max_entries:
            return normalized

        ranked = sorted(
            normalized.items(),
            key=lambda item: str(item[1].get("created_at", "")),
        )
        to_remove = len(normalized) - self.max_entries
        for cache_key, _ in ranked[:to_remove]:
            normalized.pop(cache_key, None)
        return normalized


class FsLayeredProfileRepo(ProfileRepo):
    """Searches a prioritised list of roots; writes go to the first root."""

    def __init__(self, roots: list[Path]) -> None:
        self._repos = [FsProfileRepo(r) for r in roots]

    def get(self, profile_id: str) -> CognitiveProfile:
        for repo in self._repos:
            try:
                return repo.get(profile_id)
            except FileNotFoundError:
                continue
        raise FileNotFoundError(f"Profile not found: {profile_id!r}")

    def put(self, profile: CognitiveProfile) -> None:
        self._repos[0].put(profile)

    def list_ids(self) -> list[str]:
        ids: set[str] = set()
        for repo in self._repos:
            ids.update(repo.list_ids())
        return sorted(ids)

    def list_versions(self, profile_id: str) -> list[int]:
        for repo in self._repos:
            versions = repo.list_versions(profile_id)
            if versions:
                return versions
        return []


class FsLayeredWorkflowRepo(WorkflowRepo):
    """Searches a prioritised list of roots; writes go to the first root."""

    def __init__(self, roots: list[Path]) -> None:
        self._repos = [FsWorkflowRepo(r) for r in roots]

    def get(self, workflow_id: str) -> WorkflowConfig:
        for repo in self._repos:
            try:
                return repo.get(workflow_id)
            except FileNotFoundError:
                continue
        raise FileNotFoundError(f"Workflow not found: {workflow_id!r}")

    def put(self, workflow: WorkflowConfig) -> None:
        self._repos[0].put(workflow)

    def list_ids(self) -> list[str]:
        ids: set[str] = set()
        for repo in self._repos:
            ids.update(repo.list_ids())
        return sorted(ids)


class FsCalibrationRepo(CalibrationRepo):
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, model: str, rubric_version: str) -> Path:
        safe_model = model.replace("/", "_")
        return self.root / f"{safe_model}.{rubric_version}.json"

    def get(self, model: str, rubric_version: str) -> JudgeCalibration | None:
        path = self._path(model, rubric_version)
        if not path.exists():
            return None
        raw = path.read_text().strip()
        if not raw:
            return None
        data = json.loads(raw)
        return JudgeCalibration.model_validate(data)

    def put(self, calibration: JudgeCalibration) -> None:
        path = self._path(calibration.model, calibration.rubric_version)
        calibration.updated_at = now_utc()
        path.write_text(json.dumps(calibration.model_dump(mode="json"), indent=2))
