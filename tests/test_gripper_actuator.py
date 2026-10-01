"""The gripper is one motor, so the twin gives it one actuator."""

from __future__ import annotations

import mujoco
import pytest

from robotic_arm.mjcf import ASSET_DIR, GRIPPER_ACTUATOR, single_gripper_actuator
from robotic_arm.reference import BASELINE_MJCF


@pytest.fixture(scope="module")
def merged(baseline):
    spec = mujoco.MjSpec.from_file(str(BASELINE_MJCF))
    spec.meshdir = str(ASSET_DIR).replace("\\", "/")
    return single_gripper_actuator(spec).compile()


def test_one_actuator_drives_the_gripper(baseline, merged):
    assert merged.nu == baseline.nu - 1
    names = [merged.actuator(i).name for i in range(merged.nu)]
    assert names.count(GRIPPER_ACTUATOR) == 1
    assert not any("right" in n for n in names)


def test_keyframes_lose_the_deleted_entry(merged):
    assert merged.key_ctrl.shape == (merged.nkey, merged.nu)


def test_both_fingers_still_open_together(merged):
    """The equality constraint, not a second servo, moves the right finger."""
    data = mujoco.MjData(merged)
    data.ctrl[merged.actuator(GRIPPER_ACTUATOR).id] = 0.03
    for _ in range(1000):
        mujoco.mj_step(merged, data)
    left = data.qpos[merged.joint("joint_left").qposadr[0]]
    right = data.qpos[merged.joint("joint_right").qposadr[0]]
    assert left == pytest.approx(0.03, abs=2e-3)
    assert right == pytest.approx(left, abs=1e-3)
