from __future__ import annotations

from dataclasses import dataclass
from typing import Any

RESET_MODE_FULL = "full"
RESET_MODE_STATE = "state"
RESET_MODE_TASK_AWARE = "task_aware"
VALID_RESET_MODES = {
    RESET_MODE_FULL,
    RESET_MODE_STATE,
    RESET_MODE_TASK_AWARE,
}
RESET_SAMPLING_RANDOM = "random"
RESET_SAMPLING_TASK_AFFINE_FIXED = "task_affine_fixed"
VALID_RESET_SAMPLING_STRATEGIES = {
    RESET_SAMPLING_RANDOM,
    RESET_SAMPLING_TASK_AFFINE_FIXED,
}


class ResetModeError(ValueError):
    """Raised when a requested reset mode cannot be honored."""


@dataclass(frozen=True)
class LiberoResetDecision:
    full_reset: bool
    fallback_used: bool = False


def _cfg_contains(cfg: Any, key: str) -> bool:
    return hasattr(cfg, "__contains__") and key in cfg


def get_reset_mode(env_cfg: Any) -> str:
    mode = str(env_cfg.get("reset_mode", RESET_MODE_FULL)).lower()
    if mode not in VALID_RESET_MODES:
        raise ValueError(
            f"reset_mode must be one of {sorted(VALID_RESET_MODES)}, got {mode!r}"
        )
    return mode


def reset_full_on_state_mismatch(env_cfg: Any) -> bool:
    return bool(env_cfg.get("reset_full_on_state_mismatch", True))


def reset_optimization_enabled(env_cfg: Any) -> bool:
    return bool(env_cfg.get("reset_optimization_enabled", False))


def validate_env_reset_mode_cfg(env_cfg: Any, path: str) -> None:
    if _cfg_contains(env_cfg, "reset_mode"):
        try:
            get_reset_mode(env_cfg)
        except ValueError as error:
            raise ValueError(f"{path}.reset_mode is invalid: {error}") from error
    if _cfg_contains(env_cfg, "reset_full_on_state_mismatch"):
        value = env_cfg.get("reset_full_on_state_mismatch")
        if not isinstance(value, bool):
            raise ValueError(
                f"{path}.reset_full_on_state_mismatch must be a boolean, "
                f"got {type(value).__name__}"
            )
    if _cfg_contains(env_cfg, "reset_optimization_enabled"):
        value = env_cfg.get("reset_optimization_enabled")
        if not isinstance(value, bool):
            raise ValueError(
                f"{path}.reset_optimization_enabled must be a boolean, "
                f"got {type(value).__name__}"
            )
    if _cfg_contains(env_cfg, "reset_sampling_strategy"):
        value = str(env_cfg.get("reset_sampling_strategy"))
        if value not in VALID_RESET_SAMPLING_STRATEGIES:
            raise ValueError(
                f"{path}.reset_sampling_strategy must be one of "
                f"{sorted(VALID_RESET_SAMPLING_STRATEGIES)}, got {value!r}"
            )


def robocasa_hard_reset_from_cfg(env_cfg: Any) -> bool:
    if _cfg_contains(env_cfg, "reset_mode"):
        if not reset_optimization_enabled(env_cfg):
            return True
        return get_reset_mode(env_cfg) == RESET_MODE_FULL
    return bool(env_cfg.get("hard_reset", True))


def libero_should_full_reset(
    *,
    reset_mode: str,
    task_changed: bool,
    is_eval: bool,
    bddl_may_change_without_task_change: bool,
    fallback_on_state_mismatch: bool,
) -> LiberoResetDecision:
    mode = str(reset_mode).lower()
    if mode not in VALID_RESET_MODES:
        raise ValueError(
            f"reset_mode must be one of {sorted(VALID_RESET_MODES)}, got {mode!r}"
        )

    asset_mismatch = task_changed or bddl_may_change_without_task_change
    if mode == RESET_MODE_FULL:
        return LiberoResetDecision(full_reset=task_changed or not is_eval)
    if mode == RESET_MODE_TASK_AWARE:
        return LiberoResetDecision(full_reset=asset_mismatch)
    if asset_mismatch:
        if fallback_on_state_mismatch:
            return LiberoResetDecision(full_reset=True, fallback_used=True)
        raise ResetModeError(
            "reset_mode='state' requires state reset, but target task or BDDL "
            "differs from the current environment"
        )
    return LiberoResetDecision(full_reset=False)
