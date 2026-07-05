import torch

from rlinf.algorithms.losses import compute_gipo_actor_loss
from rlinf.algorithms.registry import policy_loss


def test_gipo_loss_is_finite_for_extreme_log_ratios():
    logprobs = torch.tensor([10.0, -10.0, 0.1], requires_grad=True)
    old_logprobs = torch.zeros(3)
    advantages = torch.ones(3)
    mask = torch.tensor([True, True, True])

    loss, metrics = compute_gipo_actor_loss(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        loss_mask=mask,
        gipo_sigma=1.0,
        gipo_rho_min=0.0067,
        gipo_rho_max=148.0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logprobs.grad).all()
    assert metrics["actor/gipo_weight_min"] >= 0


def test_policy_loss_registry_has_gipo_actor_critic():
    loss, metrics = policy_loss(
        task_type="embodied",
        loss_type="gipo_actor_critic",
        logprob_type="chunk_level",
        reward_type="chunk_level",
        single_action_dim=1,
        logprobs=torch.zeros(2, 1, requires_grad=True),
        old_logprobs=torch.zeros(2, 1),
        advantages=torch.ones(2, 1),
        returns=torch.ones(2, 1),
        values=torch.zeros(2, 1),
        prev_values=torch.zeros(2, 1),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        value_clip=0.2,
        huber_delta=10.0,
        gipo_sigma=1.0,
        gipo_rho_min=0.0067,
        gipo_rho_max=148.0,
        max_episode_steps=None,
        loss_mask=None,
        loss_mask_sum=None,
    )

    assert torch.isfinite(loss)
    assert "actor/gipo_weight_mean" in metrics
    assert "critic/value_loss" in metrics
