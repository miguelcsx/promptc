from __future__ import annotations

from abc import ABC, abstractmethod

from promptc.models import (
    ArtifactIndexEntry,
    CognitiveProfile,
    JudgeCalibration,
    PromptArtifact,
    SignatureSpec,
    WorkflowConfig,
)


class ProfileRepo(ABC):
    @abstractmethod
    def get(self, profile_id: str) -> CognitiveProfile:
        raise NotImplementedError

    @abstractmethod
    def put(self, profile: CognitiveProfile) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_ids(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def list_versions(self, profile_id: str) -> list[int]:
        raise NotImplementedError


class WorkflowRepo(ABC):
    @abstractmethod
    def get(self, workflow_id: str) -> WorkflowConfig:
        raise NotImplementedError

    @abstractmethod
    def put(self, workflow: WorkflowConfig) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_ids(self) -> list[str]:
        raise NotImplementedError


class SignatureRepo(ABC):
    @abstractmethod
    def get_latest(self, signature_id: str) -> SignatureSpec | None:
        raise NotImplementedError

    @abstractmethod
    def put(self, spec: SignatureSpec) -> str:
        raise NotImplementedError


class ArtifactRepo(ABC):
    @abstractmethod
    def save(self, artifact: PromptArtifact) -> str:
        raise NotImplementedError

    @abstractmethod
    def get(self, artifact_id: str) -> PromptArtifact:
        raise NotImplementedError

    @abstractmethod
    def get_by_path(self, artifact_path: str) -> PromptArtifact:
        raise NotImplementedError

    @abstractmethod
    def list_recent(self, limit: int = 20) -> list[ArtifactIndexEntry]:
        raise NotImplementedError

    @abstractmethod
    def lineage(self, artifact_id: str) -> list[ArtifactIndexEntry]:
        raise NotImplementedError


class CacheRepo(ABC):
    @abstractmethod
    def get(self, key: str) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def put(self, key: str, artifact_path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


class CalibrationRepo(ABC):
    @abstractmethod
    def get(self, model: str, rubric_version: str) -> JudgeCalibration | None:
        raise NotImplementedError

    @abstractmethod
    def put(self, calibration: JudgeCalibration) -> None:
        raise NotImplementedError
