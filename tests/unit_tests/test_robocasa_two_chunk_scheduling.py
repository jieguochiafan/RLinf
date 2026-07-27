from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "logs"
    / "latency_balance_eval"
    / "figures"
    / "simulate_robocasa_two_chunk_scheduling.py"
)
SPEC = importlib.util.spec_from_file_location(
    "simulate_robocasa_two_chunk_scheduling", SCRIPT_PATH
)
sim = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = sim
SPEC.loader.exec_module(sim)


def _profile(local_env: int, durations: tuple[float, ...]):
    return sim.EnvProfile(
        local_env=local_env,
        global_env=local_env,
        step_durations_s=durations,
    )


def test_baseline_and_dynamic_strategies_match_expected_makespans() -> None:
    first_chunk = [
        _profile(0, (8.0, 1.0)),
        _profile(1, (1.0, 2.0)),
    ]
    second_chunk = [
        _profile(10, (1.0, 2.0)),
        _profile(11, (8.0, 1.0)),
    ]

    baseline = sim.simulate_baseline_chunk_barrier(first_chunk, second_chunk)
    dynamic = sim.simulate_dynamic_online_step_latency(first_chunk, second_chunk)

    assert baseline.makespan_s == 20.0
    assert [
        event.start_s for event in baseline.events if event.chunk_id == 1
    ] == [0.0, 0.0, 8.0, 8.0]
    assert dynamic.makespan_s == 13.0


def test_run_simulation_only_returns_baseline_and_dynamic() -> None:
    profiles = [
        _profile(0, (10.0,)),
        _profile(1, (4.0,)),
        _profile(2, (3.0,)),
    ]

    _second_chunk, results = sim.run_simulation(profiles, shuffle_seed=1)

    assert [result.name for result in results] == [
        "baseline_step_barrier",
        "dynamic_online_step_latency",
    ]


def test_load_baseline_uses_two_sequential_chunks_from_log_bounds(tmp_path: Path) -> None:
    path = tmp_path / "env_rank_0.jsonl"
    rows = [
        {"event": "start", "chunk_step": 0, "wall_ns": 1_000_000_000},
        {
            "event": "robocasa_env_step",
            "chunk_step": 0,
            "local_env": 0,
            "global_env": 0,
            "vector_step": 0,
            "wall_start_ns": 1_010_000_000,
            "wall_end_ns": 1_200_000_000,
        },
        {"event": "end", "chunk_step": 0, "wall_ns": 1_500_000_000},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    result = sim.load_baseline_step_barrier_result(
        path,
        chunk_step=0,
        second_chunk_order=[0],
    )

    assert result.name == "baseline_step_barrier"
    assert result.makespan_s == 1.0
    assert [event.chunk_id for event in result.events] == [1, 2]
    assert result.events[0].start_s == 0.01
    assert result.events[0].end_s == 0.2
    assert result.events[1].start_s == 0.51
    assert result.events[1].end_s == 0.7


def test_dynamic_step_scheduler_updates_latency_without_probe_barrier() -> None:
    first_chunk = [
        _profile(0, (10.0,)),
        _profile(1, (4.0,)),
    ]
    second_chunk = [
        _profile(10, (8.0, 2.0)),
        _profile(11, (1.0, 6.0)),
    ]

    dynamic = sim.simulate_dynamic_probe_lpt(first_chunk, second_chunk)

    assert dynamic.name == "dynamic_online_step_latency"
    assert dynamic.makespan_s == 17.0
    second_events = [event for event in dynamic.events if event.chunk_id == 2]
    assert [event.phase for event in second_events] == ["step"] * 4
    assert [event.start_s for event in second_events] == [4.0, 10.0, 11.0, 12.0]
    assert [(event.local_env, event.step_index) for event in second_events] == [
        (10, 0),
        (11, 0),
        (11, 1),
        (10, 1),
    ]


def test_dynamic_first_chunk_is_split_into_step_events() -> None:
    first_chunk = [
        _profile(0, (2.0, 3.0)),
        _profile(1, (1.0, 4.0)),
    ]
    second_chunk = [_profile(10, (1.0,)), _profile(11, (1.0,))]

    dynamic = sim.simulate_dynamic_online_step_latency(first_chunk, second_chunk)
    first_events = [event for event in dynamic.events if event.chunk_id == 1]

    assert [(event.local_env, event.step_index) for event in first_events] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]


def test_display_env_id_offsets_second_chunk() -> None:
    assert sim.display_env_id(sim.ScheduleEvent("", 1, 7, None, 7, 0.0, 1.0, "")) == 7
    assert sim.display_env_id(sim.ScheduleEvent("", 2, 7, None, 7, 0.0, 1.0, "")) == 15


def test_load_env_profiles_reads_step_timings_by_chunk(tmp_path: Path) -> None:
    path = tmp_path / "env_rank_0.jsonl"
    rows = [
        {
            "event": "robocasa_env_step",
            "chunk_step": 3,
            "local_env": 1,
            "global_env": 8,
            "chunk_action_index": 1,
            "wall_start_ns": 1_000_000_000,
            "wall_end_ns": 1_300_000_000,
        },
        {
            "event": "robocasa_env_step",
            "chunk_step": 3,
            "local_env": 1,
            "global_env": 8,
            "chunk_action_index": 0,
            "wall_start_ns": 2_000_000_000,
            "wall_end_ns": 2_100_000_000,
        },
        {
            "event": "robocasa_env_step",
            "chunk_step": 4,
            "local_env": 1,
            "global_env": 8,
            "chunk_action_index": 0,
            "wall_start_ns": 3_000_000_000,
            "wall_end_ns": 3_900_000_000,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    profiles = sim.load_env_profiles(path, chunk_step=3)

    assert len(profiles) == 1
    assert profiles[0].local_env == 1
    assert profiles[0].global_env == 8
    assert profiles[0].step_durations_s == (0.1, 0.3)


def test_plot_gantt_writes_output_file(tmp_path: Path) -> None:
    first_chunk = [_profile(0, (2.0,)), _profile(1, (1.0,))]
    second_chunk = [_profile(10, (1.5,)), _profile(11, (0.5,))]
    results = [
        sim.simulate_baseline_chunk_barrier(first_chunk, second_chunk),
        sim.simulate_dynamic_online_step_latency(first_chunk, second_chunk),
    ]
    out_path = tmp_path / "gantt.png"

    sim.plot_gantt(results, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
