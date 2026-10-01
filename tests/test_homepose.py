"""Re-zeroing must change the numbers on the dials and nothing else."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from robotic_arm.homepose import UNSHIFTED_KEYFRAMES, apply_home_zero, upright_pose
from robotic_arm.mjcf import ASSET_DIR
from robotic_arm.reference import BASELINE_MJCF


@pytest.fixture(scope="module")
def rezeroed(baseline):
    spec = mujoco.MjSpec.from_file(str(BASELINE_MJCF))
    spec.meshdir = str(ASSET_DIR).replace("\\", "/")
    probe = spec.compile()
    pose = upright_pose(probe)
    apply_home_zero(spec, pose, {n: int(probe.joint(n).qposadr[0]) for n in pose})
    return spec.compile()


def _servo_targets_match_pose(model, key: int) -> list[str]:
    """Actuators whose keyframe ctrl does not hold the keyframe's own qpos."""
    wrong = []
    for a in range(model.nu):
        joint = model.actuator_trnid[a, 0]
        q = model.key_qpos[key, model.jnt_qposadr[joint]]
        if not np.isclose(model.key_ctrl[key, a], q):
            wrong.append(f"{model.actuator(a).name}: ctrl {model.key_ctrl[key, a]:.3f} vs qpos {q:.3f}")
    return wrong


def test_stock_keyframes_hold_their_pose(baseline):
    """The invariant the re-zero has to preserve: every key's servos aim at its qpos."""
    for key in range(baseline.nkey):
        assert _servo_targets_match_pose(baseline, key) == [], baseline.key(key).name


def test_rezero_shifts_keyframe_ctrl_with_qpos(rezeroed):
    """Shifting qpos alone leaves the servos aimed at the old numbers, so a
    loaded key starts in one pose and immediately drives to another."""
    for key in range(rezeroed.nkey):
        if rezeroed.key(key).name in UNSHIFTED_KEYFRAMES:
            continue
        assert _servo_targets_match_pose(rezeroed, key) == [], rezeroed.key(key).name
