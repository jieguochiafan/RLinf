from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from rlinf.scheduler.resource_pool.bindings import WorkerResourceBinding
from toolkits.resource_orchestration.types import CandidatePair


def build_mps_plan_payload(
    base_bindings: Iterable[WorkerResourceBinding],
    candidate: CandidatePair,
) -> dict[str, list[dict]]:
    """Build a JSON-compatible MPS plan payload for a candidate."""
    bindings = [
        _build_binding_payload(binding, candidate) for binding in base_bindings
    ]
    for item in bindings:
        WorkerResourceBinding.from_json(json.dumps(item))
    return {"bindings": bindings}


def write_mps_plan(
    output_path: str | Path,
    base_bindings: Iterable[WorkerResourceBinding],
    candidate: CandidatePair,
) -> None:
    """Write a resource binding MPS plan JSON file."""
    payload = build_mps_plan_payload(base_bindings, candidate)
    Path(output_path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_binding_payload(
    binding: WorkerResourceBinding,
    candidate: CandidatePair,
) -> dict:
    payload = json.loads(binding.to_json())
    if binding.component not in ("actor", "rollout") or binding.gpu is None:
        return payload
    if binding.gpu.mode != "mps":
        raise ValueError(
            f"{binding.component} binding rank {binding.rank} must use mps GPU mode"
        )
    payload["gpu"]["sm_percent"] = (
        candidate.actor_sm if binding.component == "actor" else candidate.rollout_sm
    )
    return payload
