import asyncio
import logging
import sys
from pathlib import Path

import pytest
import torch
from torch.futures import Future

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rlinf.scheduler.collective.async_work import AsyncFuncWork
from rlinf.scheduler.collective.multi_channel_pg import MultiChannelProcessGroup


def test_async_func_work_propagates_exceptions_to_waiters() -> None:
    def fail() -> None:
        raise RuntimeError("boom")

    work = AsyncFuncWork(fail)
    work(None)

    assert work.done()
    with pytest.raises(RuntimeError, match="boom"):
        work.wait()


def test_async_func_work_propagates_completed_exceptions_to_async_waiters() -> None:
    def fail() -> None:
        raise RuntimeError("async boom")

    work = AsyncFuncWork(fail)
    work(None)

    assert work.done()
    with pytest.raises(RuntimeError, match="async boom"):
        asyncio.run(work.async_wait())


def test_async_func_work_does_not_run_callback_after_upstream_failure() -> None:
    callback_ran = False

    def callback() -> None:
        nonlocal callback_ran
        callback_ran = True

    upstream = Future()
    work = AsyncFuncWork(callback)
    upstream.then(work)
    upstream.set_exception(RuntimeError("upstream boom"))

    assert work.done()
    with pytest.raises(RuntimeError, match="upstream boom"):
        work.wait()
    assert not callback_ran


def test_async_func_work_propagates_chained_work_failures() -> None:
    def fail() -> None:
        raise RuntimeError("chained boom")

    inner = AsyncFuncWork(fail)
    outer = AsyncFuncWork(lambda: inner)

    outer(None)
    inner(None)

    assert outer.done()
    with pytest.raises(RuntimeError, match="chained boom"):
        outer.wait()


def test_async_func_work_propagates_chained_callback_failures() -> None:
    def fail_after_inner_success() -> None:
        raise RuntimeError("chained callback boom")

    inner = AsyncFuncWork(lambda: None)
    inner.then(fail_after_inner_success)
    outer = AsyncFuncWork(lambda: inner)

    outer(None)
    inner(None)

    assert outer.done()
    with pytest.raises(RuntimeError, match="chained callback boom"):
        outer.wait()


def test_multi_channel_broadcast_reraises_process_group_failures(monkeypatch) -> None:
    group = MultiChannelProcessGroup.__new__(MultiChannelProcessGroup)
    group._cur_rank = 0
    group._logger = logging.getLogger("test")

    monkeypatch.setattr(
        "rlinf.scheduler.collective.multi_channel_pg._check_single_tensor",
        lambda tensor, name: None,
        raising=False,
    )
    monkeypatch.setattr(
        "torch.distributed.distributed_c10d._rank_not_in_group",
        lambda pg: False,
    )
    monkeypatch.setattr(
        "torch.distributed.distributed_c10d.get_group_rank",
        lambda pg, rank: rank,
    )
    monkeypatch.setattr(
        "torch.distributed._get_process_group_name",
        lambda pg: "test_pg",
    )

    class BrokenGroup:
        def broadcast(self, tensors, opts):
            raise RuntimeError("gloo failed")

    with pytest.raises(RuntimeError, match="Broadcast failed on ProcessGroup test_pg"):
        group._broadcast(
            torch.zeros(1),
            src=0,
            group=BrokenGroup(),
            async_op=False,
        )
