from rlinf.data.embodied_async import (
    AsyncTrajectoryEnvelope,
    InferenceResponse,
    build_inference_request_id,
    build_inference_response_key,
)


def test_inference_request_id_and_response_key_are_stable():
    request_id = build_inference_request_id(env_rank=2, stage_id=1, local_step=7)

    assert request_id.startswith("2:1:7:")
    assert build_inference_response_key(2, 1, request_id) == (
        f"action:2:1:{request_id}"
    )


def test_inference_response_carries_rollout_result_and_error_exclusively():
    response = InferenceResponse(
        request_id="2:1:7:test",
        rollout_rank=0,
        actions=None,
        rollout_result=None,
        policy_version=3,
        timing={"queue_wait_s": 0.01},
        error="boom",
    )

    assert response.has_error
    assert response.error == "boom"


def test_trajectory_envelope_segment_type_validation():
    envelope = AsyncTrajectoryEnvelope(
        env_rank=0,
        stage_id=0,
        segment_type="fixed_horizon",
        auto_reset=False,
        trajectory=object(),
        completed_at=1.0,
        last_policy_version=4,
    )

    assert envelope.segment_type == "fixed_horizon"
