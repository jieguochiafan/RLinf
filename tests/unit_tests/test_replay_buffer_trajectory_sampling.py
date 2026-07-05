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
