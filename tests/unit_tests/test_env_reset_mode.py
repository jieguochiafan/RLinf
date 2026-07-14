import pytest
from omegaconf import OmegaConf

from rlinf.envs.reset_mode import (
    RESET_MODE_FULL,
    RESET_MODE_STATE,
    RESET_MODE_TASK_AWARE,
    ResetModeError,
    get_reset_mode,
    libero_should_full_reset,
    reset_full_on_state_mismatch,
    robocasa_hard_reset_from_cfg,
    validate_env_reset_mode_cfg,
)


def test_get_reset_mode_defaults_to_full_without_mutating_cfg():
    cfg = OmegaConf.create({})

    assert get_reset_mode(cfg) == RESET_MODE_FULL
    assert "reset_mode" not in cfg


@pytest.mark.parametrize(
    "mode", [RESET_MODE_FULL, RESET_MODE_STATE, RESET_MODE_TASK_AWARE]
)
def test_validate_env_reset_mode_accepts_known_modes(mode):
    cfg = OmegaConf.create({"reset_mode": mode})

    validate_env_reset_mode_cfg(cfg, "env.train")


def test_validate_env_reset_mode_rejects_unknown_mode():
    cfg = OmegaConf.create({"reset_mode": "fast"})

    with pytest.raises(ValueError, match="env.train.reset_mode"):
        validate_env_reset_mode_cfg(cfg, "env.train")


def test_reset_full_on_state_mismatch_defaults_true():
    assert reset_full_on_state_mismatch(OmegaConf.create({})) is True
    assert (
        reset_full_on_state_mismatch(
            OmegaConf.create({"reset_full_on_state_mismatch": False})
        )
        is False
    )


@pytest.mark.parametrize(
    ("cfg_dict", "expected"),
    [
        ({"reset_mode": "full"}, True),
        ({"reset_mode": "state"}, False),
        ({"reset_mode": "task_aware"}, False),
        ({"hard_reset": False}, False),
        ({}, True),
    ],
)
def test_robocasa_hard_reset_from_cfg(cfg_dict, expected):
    assert robocasa_hard_reset_from_cfg(OmegaConf.create(cfg_dict)) is expected


def test_robocasa_reset_mode_overrides_hard_reset_compat_key():
    cfg = OmegaConf.create({"reset_mode": "state", "hard_reset": True})

    assert robocasa_hard_reset_from_cfg(cfg) is False


@pytest.mark.parametrize(
    (
        "mode",
        "task_changed",
        "is_eval",
        "bddl_may_change",
        "expected_full",
        "expected_fallback",
    ),
    [
        ("full", False, False, False, True, False),
        ("full", False, True, False, False, False),
        ("task_aware", False, False, False, False, False),
        ("task_aware", True, False, False, True, False),
        ("task_aware", False, False, True, True, False),
        ("state", False, False, False, False, False),
        ("state", True, False, False, True, True),
    ],
)
def test_libero_should_full_reset_decision(
    mode,
    task_changed,
    is_eval,
    bddl_may_change,
    expected_full,
    expected_fallback,
):
    result = libero_should_full_reset(
        reset_mode=mode,
        task_changed=task_changed,
        is_eval=is_eval,
        bddl_may_change_without_task_change=bddl_may_change,
        fallback_on_state_mismatch=True,
    )

    assert result.full_reset is expected_full
    assert result.fallback_used is expected_fallback


def test_libero_state_mode_can_fail_fast_on_task_change():
    with pytest.raises(ResetModeError, match="requires state reset"):
        libero_should_full_reset(
            reset_mode="state",
            task_changed=True,
            is_eval=False,
            bddl_may_change_without_task_change=False,
            fallback_on_state_mismatch=False,
        )
