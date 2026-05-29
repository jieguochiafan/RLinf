from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

TrainingProfileRunner = Callable[..., Mapping[str, float]]

TRAINING_PROFILE_RUNNER: TrainingProfileRunner | None = None


def run_training_profile(
    cfg: Any,
    actor_sm: int,
    warmup_steps: int,
    measure_steps: int,
    rollout_chunk_count: int,
) -> dict[str, float]:
    """Profile actor training throughput.

    A test or deployment can override the default backend by setting
    TRAINING_PROFILE_RUNNER. Without an injected runner, this profiles the
    built-in MLP embodied policy on a synthetic PPO-style batch.
    """
    runner = TRAINING_PROFILE_RUNNER or _run_default_training_profile
    metrics = runner(
        cfg=cfg,
        actor_sm=actor_sm,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        rollout_chunk_count=rollout_chunk_count,
    )
    return _validate_training_profile_metrics(metrics)


def _validate_training_profile_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    try:
        actor_chunk_steps_per_sec = float(metrics["actor_chunk_steps_per_sec"])
    except KeyError as exc:
        raise ValueError(
            "training profile metrics must include actor_chunk_steps_per_sec"
        ) from exc

    if (
        not math.isfinite(actor_chunk_steps_per_sec)
        or actor_chunk_steps_per_sec <= 0.0
    ):
        raise ValueError(
            "actor_chunk_steps_per_sec must be finite and positive"
        )

    return {"actor_chunk_steps_per_sec": actor_chunk_steps_per_sec}


def _run_default_training_profile(
    cfg: Any,
    actor_sm: int,
    warmup_steps: int,
    measure_steps: int,
    rollout_chunk_count: int,
) -> dict[str, float]:
    """Profile the default single-process actor training backend."""
    model_type = str(_select(cfg, "actor.model.model_type", default=""))
    if model_type != "mlp_policy":
        raise RuntimeError(
            "default training profile backend currently supports "
            "actor.model.model_type == 'mlp_policy' only; got "
            f"{model_type!r}. Complex VLA models such as openpi, openvla, and "
            "gr00t need toolkits.training_eval.run.TRAINING_PROFILE_RUNNER "
            "injection or a dedicated training profile backend."
        )

    if rollout_chunk_count <= 0:
        raise ValueError(
            f"rollout_chunk_count must be positive, got {rollout_chunk_count}"
        )
    if measure_steps <= 0:
        raise ValueError(f"measure_steps must be positive, got {measure_steps}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")

    with _temporary_mps_percentage(actor_sm):
        torch = _import_torch()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _build_mlp_policy(cfg).to(device=device, dtype=torch.float32)
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(_select(cfg, "actor.optim.lr", default=1.0e-3)),
            betas=(
                float(_select(cfg, "actor.optim.adam_beta1", default=0.9)),
                float(_select(cfg, "actor.optim.adam_beta2", default=0.999)),
            ),
            eps=float(_select(cfg, "actor.optim.adam_eps", default=1.0e-8)),
            weight_decay=float(_select(cfg, "actor.optim.weight_decay", default=0.0)),
        )
        batch = _make_mlp_training_batch(cfg, rollout_chunk_count, device)

        for _ in range(warmup_steps):
            _run_profile_iteration(cfg, model, optimizer, batch, rollout_chunk_count)
        _synchronize(device)

        start = time.perf_counter()
        for _ in range(measure_steps):
            _run_profile_iteration(cfg, model, optimizer, batch, rollout_chunk_count)
        _synchronize(device)
        elapsed_s = time.perf_counter() - start

    actor_chunk_steps_per_sec = (rollout_chunk_count * measure_steps) / elapsed_s
    return {"actor_chunk_steps_per_sec": actor_chunk_steps_per_sec}


def _build_mlp_policy(cfg: Any) -> torch.nn.Module:
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.mlp_policy import get_model

    torch = _import_torch()
    loss_type = str(_select(cfg, "algorithm.loss_type", default="actor_critic"))
    model_cfg = OmegaConf.create(
        {
            "obs_dim": _select(cfg, "actor.model.obs_dim"),
            "action_dim": _select(cfg, "actor.model.action_dim"),
            "num_action_chunks": _select(
                cfg, "actor.model.num_action_chunks", default=1
            ),
            "add_value_head": _select(
                cfg,
                "actor.model.add_value_head",
                default=loss_type in {"actor_critic", "decoupled_actor_critic"},
            ),
            "add_q_head": _select(cfg, "actor.model.add_q_head", default=False),
            "q_head_type": _select(cfg, "actor.model.q_head_type", default="default"),
        }
    )
    return get_model(model_cfg, torch_dtype=torch.float32)


def _make_mlp_training_batch(
    cfg: Any,
    rollout_chunk_count: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    torch = _import_torch()
    obs_dim = int(_select(cfg, "actor.model.obs_dim"))
    action_dim = int(_select(cfg, "actor.model.action_dim"))
    batch_shape = (rollout_chunk_count,)
    return {
        "states": torch.randn(*batch_shape, obs_dim, device=device),
        "action": torch.randn(*batch_shape, action_dim, device=device),
        "old_logprobs": torch.zeros(
            *batch_shape, action_dim, device=device, dtype=torch.float32
        ),
        "advantages": torch.randn(*batch_shape, device=device, dtype=torch.float32),
        "returns": torch.randn(*batch_shape, device=device, dtype=torch.float32),
        "prev_values": torch.zeros(*batch_shape, device=device, dtype=torch.float32),
        "loss_mask": torch.ones(*batch_shape, device=device, dtype=torch.bool),
    }


def _run_profile_iteration(
    cfg: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    rollout_chunk_count: int,
) -> None:
    update_epoch = int(_select(cfg, "algorithm.update_epoch", default=1))
    global_batch_size = int(
        _select(cfg, "actor.global_batch_size", default=rollout_chunk_count)
    )
    micro_batch_size = int(
        _select(cfg, "actor.micro_batch_size", default=global_batch_size)
    )
    if update_epoch <= 0:
        raise ValueError(f"algorithm.update_epoch must be positive, got {update_epoch}")
    if global_batch_size <= 0:
        raise ValueError(
            f"actor.global_batch_size must be positive, got {global_batch_size}"
        )
    if micro_batch_size <= 0:
        raise ValueError(
            f"actor.micro_batch_size must be positive, got {micro_batch_size}"
        )
    if global_batch_size % micro_batch_size != 0:
        raise ValueError(
            "actor.global_batch_size must be divisible by "
            "actor.micro_batch_size for default training profiling"
        )
    if rollout_chunk_count % global_batch_size != 0:
        raise ValueError(
            "rollout_chunk_count must be divisible by actor.global_batch_size "
            "for default training profiling"
        )

    for _ in range(update_epoch):
        for global_start in range(0, rollout_chunk_count, global_batch_size):
            global_end = min(global_start + global_batch_size, rollout_chunk_count)
            optimizer.zero_grad(set_to_none=True)
            micro_count = math.ceil((global_end - global_start) / micro_batch_size)
            for micro_start in range(global_start, global_end, micro_batch_size):
                micro_end = min(micro_start + micro_batch_size, global_end)
                micro_batch = {
                    key: value[micro_start:micro_end] for key, value in batch.items()
                }
                loss = _compute_mlp_policy_loss(cfg, model, micro_batch)
                (loss / micro_count).backward()
            optimizer.step()


def _compute_mlp_policy_loss(
    cfg: Any,
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    from rlinf.algorithms.registry import policy_loss

    loss_type = str(_select(cfg, "algorithm.loss_type", default="actor_critic"))
    out = model(
        forward_inputs={"states": batch["states"], "action": batch["action"]},
        compute_logprobs=True,
        compute_entropy=False,
        compute_values=loss_type in {"actor_critic", "decoupled_actor_critic"},
    )
    loss, _ = policy_loss(
        task_type=str(_select(cfg, "runner.task_type", default="embodied")),
        loss_type=loss_type,
        logprob_type=_select(cfg, "algorithm.logprob_type", default="action_level"),
        reward_type=_select(cfg, "algorithm.reward_type", default="action_level"),
        single_action_dim=int(_select(cfg, "actor.model.action_dim")),
        logprobs=out["logprobs"].float(),
        values=out.get("values", None),
        old_logprobs=batch["old_logprobs"],
        advantages=batch["advantages"],
        returns=batch["returns"],
        prev_values=batch["prev_values"],
        clip_ratio_c=float(_select(cfg, "algorithm.clip_ratio_c", default=3.0)),
        clip_ratio_low=float(_select(cfg, "algorithm.clip_ratio_low", default=0.2)),
        clip_ratio_high=float(_select(cfg, "algorithm.clip_ratio_high", default=0.2)),
        value_clip=float(_select(cfg, "algorithm.value_clip", default=1.0)),
        huber_delta=float(_select(cfg, "algorithm.huber_delta", default=10.0)),
        loss_mask=batch["loss_mask"],
        loss_mask_sum=batch["loss_mask"].sum(),
        max_episode_steps=int(
            _select(cfg, "env.train.max_episode_steps", default=batch["states"].size(0))
        ),
    )
    return loss


def _select(cfg: Any, path: str, default: Any = None) -> Any:
    from omegaconf import DictConfig, OmegaConf

    if isinstance(cfg, DictConfig):
        value = OmegaConf.select(cfg, path, default=default)
    else:
        value = cfg
        for part in path.split("."):
            if isinstance(value, Mapping):
                value = value.get(part, default)
            else:
                value = getattr(value, part, default)
            if value is default:
                break
    if value is None and default is None:
        raise ValueError(f"missing required config value: {path}")
    return value


def _import_torch():
    import torch

    return torch


@contextmanager
def _temporary_mps_percentage(actor_sm: int):
    env_name = "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
    original = os.environ.get(env_name)
    if actor_sm > 0:
        os.environ[env_name] = str(actor_sm)
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = original


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch = _import_torch()
        torch.cuda.synchronize(device)
