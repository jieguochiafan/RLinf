import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

SegmentType = Literal["fixed_horizon", "episode"]


def build_inference_request_id(env_rank: int, stage_id: int, local_step: int) -> str:
    return f"{env_rank}:{stage_id}:{local_step}:{uuid.uuid4().hex}"


def build_inference_response_key(env_rank: int, stage_id: int, request_id: str) -> str:
    return f"action:{env_rank}:{stage_id}:{request_id}"


@dataclass(kw_only=True)
class InferenceRequest:
    request_id: str
    env_rank: int
    stage_id: int
    env_ids: list[int]
    obs: dict[str, Any]
    created_at: float = field(default_factory=time.perf_counter)
    policy_version_hint: int | None = None

    @property
    def response_key(self) -> str:
        return build_inference_response_key(
            self.env_rank,
            self.stage_id,
            self.request_id,
        )


@dataclass(kw_only=True)
class InferenceResponse:
    request_id: str
    rollout_rank: int
    actions: Any
    rollout_result: Any
    policy_version: int
    timing: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    @property
    def has_error(self) -> bool:
        return self.error is not None


@dataclass(kw_only=True)
class AsyncTrajectoryEnvelope:
    env_rank: int
    stage_id: int
    segment_type: SegmentType
    auto_reset: bool
    trajectory: Any
    started_at: float
    completed_at: float
    last_policy_version: int | None = None

    def __post_init__(self) -> None:
        if self.segment_type not in ("fixed_horizon", "episode"):
            raise ValueError(f"Unsupported segment_type: {self.segment_type}")
        if self.completed_at < self.started_at:
            raise ValueError(
                "completed_at must be greater than or equal to started_at: "
                f"{self.started_at=} {self.completed_at=}"
            )

    @property
    def duration_s(self) -> float:
        return self.completed_at - self.started_at

    def timing_payload(self) -> dict[str, Any]:
        return {
            "env_rank": self.env_rank,
            "stage_id": self.stage_id,
            "segment_type": self.segment_type,
            "auto_reset": self.auto_reset,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": self.duration_s,
            "last_policy_version": self.last_policy_version,
        }
