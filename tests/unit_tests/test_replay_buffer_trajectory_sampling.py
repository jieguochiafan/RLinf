import torch

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer


def _make_traj(offset: int) -> Trajectory:
    t, b, c = 3, 2, 1
    rewards = torch.arange(
        offset, offset + t * b * c, dtype=torch.float32
    ).reshape(t, b, c)
    dones = torch.zeros(t + 1, b, c, dtype=torch.bool)
    dones[-1] = True
    return Trajectory(
        max_episode_length=12,
        model_weights_id=f"w{offset}",
        rewards=rewards,
        dones=dones,
        terminations=dones.clone(),
        truncations=torch.zeros_like(dones),
        prev_logprobs=torch.zeros(t, b, c),
        prev_values=torch.zeros(t + 1, b, c),
        versions=torch.full((t, b, c), float(offset)),
        forward_inputs={"action": torch.zeros(t, b, c)},
    )


def test_sample_trajectory_batch_preserves_time_and_batch_dims():
    buffer = TrajectoryReplayBuffer(seed=0, enable_cache=True, sample_window_size=4)
    buffer.add_trajectories([_make_traj(0), _make_traj(10)])

    batch = buffer.sample_trajectory_batch(num_trajectories=2)

    assert batch["rewards"].shape == (3, 4, 1)
    assert batch["dones"].shape == (4, 4, 1)
    assert batch["forward_inputs"]["action"].shape == (3, 4, 1)
    assert batch["versions"].shape == (3, 4, 1)


def test_sample_trajectory_batch_defaults_missing_loss_mask_to_valid_steps():
    buffer = TrajectoryReplayBuffer(seed=0, enable_cache=True, sample_window_size=4)
    buffer.add_trajectories([_make_traj(0), _make_traj(10)])

    batch = buffer.sample_trajectory_batch(num_trajectories=2)

    assert batch["loss_mask"].shape == batch["rewards"].shape
    assert batch["loss_mask"].dtype == torch.bool
    assert batch["loss_mask"].all()


def test_sample_trajectory_batch_pads_step_aligned_done_fields():
    trajectory = _make_traj(0)
    trajectory.dones = trajectory.dones[1:]
    trajectory.terminations = trajectory.terminations[1:]
    trajectory.truncations = trajectory.truncations[1:]
    buffer = TrajectoryReplayBuffer(seed=0, enable_cache=True, sample_window_size=4)
    buffer.add_trajectories([trajectory])

    batch = buffer.sample_trajectory_batch(num_trajectories=1)

    assert batch["rewards"].shape == (3, 2, 1)
    assert batch["dones"].shape == (4, 2, 1)
    assert batch["terminations"].shape == (4, 2, 1)
    assert batch["truncations"].shape == (4, 2, 1)
    assert not batch["dones"][0].any()


def test_sample_trajectory_batch_pads_step_aligned_prev_values():
    trajectory = _make_traj(0)
    trajectory.prev_values = trajectory.prev_values[:-1]
    buffer = TrajectoryReplayBuffer(seed=0, enable_cache=True, sample_window_size=4)
    buffer.add_trajectories([trajectory])

    batch = buffer.sample_trajectory_batch(num_trajectories=1)

    assert batch["rewards"].shape == (3, 2, 1)
    assert batch["prev_values"].shape == (4, 2, 1)
    torch.testing.assert_close(batch["prev_values"][-1], trajectory.prev_values[-1])


def test_sample_trajectory_batch_pads_variable_length_trajectories():
    short = _make_traj(0)
    long = _make_traj(10)
    long.rewards = torch.ones(5, 2, 1)
    long.dones = torch.zeros(6, 2, 1, dtype=torch.bool)
    long.dones[-1] = True
    long.terminations = long.dones.clone()
    long.truncations = torch.zeros_like(long.dones)
    long.prev_logprobs = torch.zeros(5, 2, 1)
    long.prev_values = torch.zeros(6, 2, 1)
    long.versions = torch.ones(5, 2, 1)
    long.forward_inputs = {"action": torch.ones(5, 2, 1)}

    buffer = TrajectoryReplayBuffer(seed=0, enable_cache=True, sample_window_size=4)
    buffer.add_trajectories([short, long])

    batch = buffer.sample_trajectory_batch(num_trajectories=2)

    assert batch["rewards"].shape == (5, 4, 1)
    assert batch["dones"].shape == (6, 4, 1)
    assert batch["prev_values"].shape == (6, 4, 1)
    assert batch["forward_inputs"]["action"].shape == (5, 4, 1)
    assert batch["loss_mask"].shape == (5, 4, 1)
    assert batch["loss_mask"][:3, :2].all()
    assert not batch["loss_mask"][3:, :2].any()
    torch.testing.assert_close(batch["rewards"][3:, :2], torch.zeros(2, 2, 1))
