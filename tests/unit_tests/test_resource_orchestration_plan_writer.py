import json

import pytest

from rlinf.scheduler.resource_pool.bindings import (
    CpuBinding,
    GpuBinding,
    WorkerResourceBinding,
)
from toolkits.resource_orchestration.plan_writer import (
    build_mps_plan_payload,
    write_mps_plan,
)
from toolkits.resource_orchestration.types import CandidatePair


def _binding(
    component: str,
    *,
    rank: int,
    cpu: CpuBinding | None,
    gpu: GpuBinding | None,
) -> WorkerResourceBinding:
    return WorkerResourceBinding(
        component=component,
        rank=rank,
        cluster_node_rank=2,
        node_group_label="train",
        cpu=cpu,
        gpu=gpu,
    )


def _base_bindings(actor_cpu: CpuBinding) -> dict[str, list[WorkerResourceBinding]]:
    return {
        "actor": [
            _binding(
                "actor",
                rank=0,
                cpu=actor_cpu,
                gpu=GpuBinding(
                    mode="mps",
                    sm_percent=50,
                    visible_devices=("0",),
                    parent_gpu=0,
                ),
            )
        ],
        "rollout": [
            _binding(
                "rollout",
                rank=1,
                cpu=CpuBinding(process_cpu_cores=(2, 3)),
                gpu=GpuBinding(
                    mode="mps",
                    sm_percent=50,
                    visible_devices=("0",),
                    parent_gpu=0,
                ),
            )
        ],
        "env": [
            _binding(
                "env",
                rank=2,
                cpu=CpuBinding(process_cpu_cores=(4, 5)),
                gpu=None,
            )
        ],
    }


def test_write_mps_plan_updates_actor_and_rollout_sm_and_preserves_bindings(
    tmp_path,
) -> None:
    actor_cpu = CpuBinding(
        process_cpu_cores=(0, 1),
        env_cpu_core_groups=((0,), (1,)),
    )
    base_bindings = _base_bindings(actor_cpu)
    output_path = tmp_path / "nested" / "plans" / "mps_plan.json"

    write_mps_plan(output_path, base_bindings, CandidatePair(actor_sm=30, rollout_sm=70))

    payload = json.loads(output_path.read_text())
    parsed = [
        WorkerResourceBinding.from_json(json.dumps(item))
        for item in payload["bindings"]
    ]
    assert payload == build_mps_plan_payload(
        base_bindings, CandidatePair(actor_sm=30, rollout_sm=70)
    )
    assert parsed[0].gpu is not None
    assert parsed[0].gpu.sm_percent == 30
    assert parsed[0].cpu == actor_cpu
    assert parsed[0].rank == 0
    assert parsed[0].cluster_node_rank == 2
    assert parsed[0].node_group_label == "train"
    assert parsed[0].gpu.visible_devices == ("0",)
    assert parsed[0].gpu.parent_gpu == 0
    assert parsed[1].gpu is not None
    assert parsed[1].gpu.sm_percent == 70
    assert parsed[2].gpu is None


@pytest.mark.parametrize("component", ["actor", "rollout"])
def test_build_mps_plan_payload_rejects_non_mps_actor_rollout_gpu(
    component: str,
) -> None:
    base_bindings = {
        component: [
            _binding(
                component,
                rank=0,
                cpu=None,
                gpu=GpuBinding(
                    mode="mig",
                    sm_percent=50,
                    visible_devices=("MIG-0",),
                    mig_device_uuid="MIG-0",
                    parent_gpu=0,
                ),
            )
        ]
    }

    with pytest.raises(ValueError, match="mps"):
        build_mps_plan_payload(
            base_bindings, CandidatePair(actor_sm=30, rollout_sm=70)
        )
