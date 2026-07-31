# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bootstrap value / reward shaping, extracted out of the env worker.

In the training repo the env worker mixed three concerns into
``compute_bootstrap_rewards``: blending an external reward model into the env reward,
and adding ``gamma * V(s_final)`` on truncated steps. Both are optional and
algorithm-flavoured, so they live here as a small, testable plugin.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from omegaconf import DictConfig, OmegaConf

__all__ = [
    "BootstrapRewardShaper",
    "estimate_bootstrap_values",
]


@dataclass
class BootstrapRewardShaper:
    """Blends reward-model output into env rewards and bootstraps truncated steps.

    Attributes:
        enabled: When ``False``, bootstrap values are ignored (reward-model blending
            still applies).
        gamma: Discount applied to the bootstrapped final value.
        bootstrap_type: ``"standard"`` bootstraps only truncated episodes;
            ``"done"`` bootstraps every terminal step.
        auto_reset: Whether the env auto-resets. Without auto-reset there is no
            truncation boundary inside a chunk, so no bootstrapping happens.
        reward_weight: Weight of the external reward-model output.
        env_reward_weight: Weight of the env's own reward when a reward model is used.
    """

    enabled: bool = True
    gamma: float = 1.0
    bootstrap_type: str = "standard"
    auto_reset: bool = True
    reward_weight: float = 1.0
    env_reward_weight: float = 0.0

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "BootstrapRewardShaper":
        """Build from ``rollout.postprocess.bootstrap`` + ``reward`` + ``env.train``."""
        auto_reset = bool(OmegaConf.select(cfg, "env.train.auto_reset", default=True))
        return cls(
            enabled=bool(
                OmegaConf.select(
                    cfg, "rollout.postprocess.bootstrap.enabled", default=True
                )
            ),
            gamma=float(
                OmegaConf.select(
                    cfg, "rollout.postprocess.bootstrap.gamma", default=1.0
                )
            ),
            bootstrap_type=str(
                OmegaConf.select(
                    cfg, "rollout.postprocess.bootstrap.type", default="standard"
                )
            ),
            auto_reset=auto_reset,
            reward_weight=float(
                OmegaConf.select(cfg, "reward.reward_weight", default=1.0)
            ),
            env_reward_weight=float(
                OmegaConf.select(cfg, "reward.env_reward_weight", default=0.0)
            ),
        )

    def shape(
        self,
        *,
        rewards: Optional[torch.Tensor],
        dones: Optional[torch.Tensor],
        truncations: Optional[torch.Tensor],
        bootstrap_values: Optional[torch.Tensor],
        reward_model_output: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Return shaped chunk rewards.

        Args:
            rewards: ``[num_envs, chunk_len]`` env rewards, or ``None``.
            dones: ``[num_envs, chunk_len]`` done flags.
            truncations: ``[num_envs, chunk_len]`` truncation flags.
            bootstrap_values: ``[num_envs, 1]`` value estimates of the final obs.
            reward_model_output: Optional external reward-model output.

        Returns:
            A new reward tensor, or ``None`` when ``rewards`` is ``None``.
        """
        if rewards is None:
            return None

        if reward_model_output is not None:
            reward_model_output = reward_model_output.to(rewards.dtype)
            rewards = (
                self.env_reward_weight * rewards
                + self.reward_weight * reward_model_output
            )

        shaped = rewards.clone()
        if (
            not self.enabled
            or bootstrap_values is None
            or not self.auto_reset
            or dones is None
        ):
            return shaped

        if self.bootstrap_type == "standard":
            if truncations is None:
                return shaped
            last_step_terminal = truncations[:, -1]
        else:
            last_step_terminal = dones[:, -1]

        if not last_step_terminal.any():
            return shaped

        final_values = torch.zeros_like(shaped[:, -1], dtype=torch.float32)
        final_values[last_step_terminal] = (
            bootstrap_values[last_step_terminal].reshape(-1).to(torch.float32)
        )
        shaped[:, -1] += self.gamma * final_values
        return shaped


def estimate_bootstrap_values(
    model: Any,
    predict_fn: Callable[[dict[str, Any]], tuple[torch.Tensor, dict[str, Any]]],
    final_obs: Optional[dict[str, Any]],
) -> Optional[torch.Tensor]:
    """Estimate ``V(s_final)`` for the terminal observation of a chunk.

    Args:
        model: The rollout policy; must expose ``value_head`` or ``q_head`` for a
            value estimate to exist.
        predict_fn: Callable running a forward pass on ``final_obs`` and returning
            ``(actions, result)``, where ``result`` may hold ``prev_values``.
        final_obs: Terminal observation batch, or ``None``.

    Returns:
        ``[num_envs, 1]`` CPU tensor of values, or ``None`` when unavailable.
    """
    if final_obs is None:
        return None
    if not (hasattr(model, "value_head") or hasattr(model, "q_head")):
        return None
    with torch.no_grad():
        actions, result = predict_fn(final_obs)
        prev_values = result.get("prev_values") if isinstance(result, dict) else None
        if prev_values is not None:
            final_values = prev_values
        else:
            final_values = torch.zeros_like(actions[:, :1], dtype=torch.float32)
    return final_values[:, :1].cpu().contiguous()
