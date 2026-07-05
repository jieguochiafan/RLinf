from rlinf.workers.rollout.hf.async_batching import DynamicBatchState


def test_target_batch_size_triggers_flush():
    state = DynamicBatchState(target_batch_size=4, max_wait_time_s=1.0)
    state.mark_first_request(now=10.0)

    assert state.should_flush(queue_size=4, now=10.1)


def test_max_wait_time_triggers_flush_before_target_size():
    state = DynamicBatchState(target_batch_size=8, max_wait_time_s=0.05)
    state.mark_first_request(now=10.0)

    assert state.should_flush(queue_size=2, now=10.06)


def test_empty_queue_never_flushes():
    state = DynamicBatchState(target_batch_size=1, max_wait_time_s=0.0)
    state.mark_first_request(now=10.0)

    assert not state.should_flush(queue_size=0, now=11.0)
