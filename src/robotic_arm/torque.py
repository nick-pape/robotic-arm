"""Gravity-torque analysis against the actuator budget.

The spec's load model (``spec/rebot-arm-b601-rs-clone.md`` section 2) was built
on an estimated mass distribution and says so explicitly: "The self-weight
numbers below are derived from the 6.7 kg total and actuator masses
[estimate]. Replace them with the URDF inertials."

This module does that replacement. It reads the real inertials from the
vendored model and reports what the actuators are actually asked for.

Units: metres, kilograms, newton-metres.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from robotic_arm.actuators import JOINT_ACTUATOR, for_joint, get
from robotic_arm.reference import ARM_JOINTS, load_baseline

GRAVITY = 9.81


def _rs06():
    """The J1-J3 actuator; J2 sets the budget."""
    return get("RS06")


@dataclass(frozen=True)
class TorquePose:
    """Gravity torque on every arm joint at one configuration."""

    qpos: np.ndarray
    torques: np.ndarray  # N*m, one per ARM_JOINTS entry
    tcp: np.ndarray  # gripper_end world position, m

    @property
    def reach(self) -> float:
        """Radial distance from the J1 axis to the tool point, m."""
        return float(np.hypot(self.tcp[0], self.tcp[1]))


def gravity_torques(model: mujoco.MjModel, qpos: np.ndarray) -> TorquePose:
    """Static gravity torque at one configuration.

    Uses ``qfrc_bias``, which at zero velocity is exactly the gravity term
    g(q) -- the same quantity the stock controller feeds forward.
    """
    data = mujoco.MjData(model)
    data.qpos[: len(qpos)] = qpos
    mujoco.mj_forward(model, data)
    return TorquePose(
        qpos=np.array(qpos, dtype=float),
        torques=np.abs(data.qfrc_bias[: len(ARM_JOINTS)].copy()),
        tcp=data.body("gripper_end").xpos.copy(),
    )


def worst_case_j2(model: mujoco.MjModel | None = None, samples: int = 121) -> TorquePose:
    """Find the shoulder configuration that loads J2 hardest under self-weight.

    J2 is the binding joint: it carries the whole distal chain on the longest
    moment arm. Sweeping J2 x J3 is sufficient because the wrist joints move
    too little mass to shift the shoulder moment appreciably.
    """
    model = model if model is not None else load_baseline()
    j2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint2")
    j3 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint3")

    best: TorquePose | None = None
    for q2 in np.linspace(*model.jnt_range[j2], samples):
        for q3 in np.linspace(*model.jnt_range[j3], samples):
            qpos = np.zeros(model.nq)
            qpos[1], qpos[2] = q2, q3
            pose = gravity_torques(model, qpos)
            if best is None or pose.torques[1] > best.torques[1]:
                best = pose
    assert best is not None
    return best


def payload_torque_j2(pose: TorquePose, payload_kg: float) -> float:
    """J2 torque including a payload at the tool point.

    The payload acts on the same moment arm as the tool, so its contribution is
    `m * g * reach`. Added to self-weight rather than re-simulated, so callers
    can sweep payload cheaply.
    """
    return float(pose.torques[1] + payload_kg * GRAVITY * pose.reach)


def report(model: mujoco.MjModel | None = None) -> str:
    """Human-readable budget summary, recomputed from real inertials."""
    model = model if model is not None else load_baseline()
    pose = worst_case_j2(model)
    lines = [
        f"total model mass        {model.body_mass.sum():.4f} kg",
        f"worst-case J2 pose      q2={pose.qpos[1]:.3f} q3={pose.qpos[2]:.3f} "
        f"rad, reach {pose.reach:.3f} m",
        "",
        f"{'joint':8s} {'actuator':9s} {'tau':>8s} {'rated':>8s} {'derated':>8s} {'peak':>8s}",
    ]
    for i, joint in enumerate(ARM_JOINTS):
        act = for_joint(joint)
        lines.append(
            f"{joint:8s} {act.name:9s} {pose.torques[i]:8.3f} {act.rated_nm:8.1f} "
            f"{act.derated_nm():8.1f} {act.peak_nm:8.1f}"
        )
    j2 = pose.torques[1]
    lines += [
        "",
        f"J2 self-weight only: {j2:.2f} N*m = {j2 / _rs06().rated_nm:.0%} of rated, "
        f"{j2 / _rs06().derated_nm():.0%} of derated, {j2 / _rs06().peak_nm:.0%} of peak",
        f"spec estimated 10.7 N*m -> real is {j2 / 10.7 - 1:+.0%}",
        "",
        "payload at worst-case reach:",
    ]
    for payload in (0.0, 1.0, 2.5, 5.0):
        tau = payload_torque_j2(pose, payload)
        verdict = "peak-only" if tau <= _rs06().peak_nm else "INFEASIBLE"
        if tau <= _rs06().derated_nm():
            verdict = "continuous"
        lines.append(f"  {payload:4.1f} kg -> {tau:7.2f} N*m  {verdict}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())


def j2_moment_arm(data, model) -> float:
    """Horizontal distance from the J2 axis to the tool, in metres.

    This, not TCP reach, is what sets the shoulder moment: the forearm can fold
    back so the tool sits close to the base while the upper arm stays
    horizontal. Filtering a workspace by reach therefore barely reduces J2
    load, which is why "70% of the workspace" cannot mean 70% of reach.
    """
    return abs(data.body("gripper_end").xpos[0] - data.body("link2").xpos[0])


def max_payload(
    model: mujoco.MjModel,
    limit_nm: float,
    max_arm: float | None = None,
    samples: int = 45,
    tolerance: float = 0.05,
    ceiling: float = 8.0,
) -> float:
    """Heaviest payload keeping worst-case J2 torque within `limit_nm`.

    `max_arm` restricts the sweep to poses whose J2 moment arm is at most that
    far out, which is how a reduced working envelope is expressed.

    Which `limit_nm` you pass decides what question is being asked, and the
    answers differ enormously:

    * **peak torque** (36 N*m) -- what the arm can lift at all. This is the
      criterion that reproduces a manufacturer payload rating.
    * **rated torque** (11 N*m) -- what it could hold indefinitely with the
      specified heat sink. Even the stock arm scores 0 kg here, because
      self-weight alone exceeds it at extension; that is a sign the criterion
      is not what payload ratings mean, not a sign the arm is deficient.
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_end")
    original = float(model.body_mass[bid])
    original_com = np.array(model.body_ipos[bid], dtype=float)
    data = mujoco.MjData(model)
    j2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint2")
    j3 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint3")

    def worst(payload: float) -> float:
        # Place the load at the grasp point, not at the gripper's COM: the two
        # are 113 mm apart, and applying it at the COM understates the moment.
        mass, com = with_payload(model, payload)
        model.body_mass[bid] = mass
        model.body_ipos[bid] = com
        out = 0.0
        for q2 in np.linspace(*model.jnt_range[j2], samples):
            for q3 in np.linspace(*model.jnt_range[j3], samples):
                data.qpos[:] = 0
                data.qpos[1], data.qpos[2] = q2, q3
                mujoco.mj_forward(model, data)
                if max_arm is not None and j2_moment_arm(data, model) > max_arm:
                    continue
                out = max(out, abs(float(data.qfrc_bias[1])))
        return out

    try:
        if worst(0.0) > limit_nm:
            return 0.0
        low, high = 0.0, ceiling
        while high - low > tolerance:
            mid = (low + high) / 2
            if worst(mid) <= limit_nm:
                low = mid
            else:
                high = mid
        return low
    finally:
        model.body_mass[bid] = original
        model.body_ipos[bid] = original_com


def max_moment_arm(model: mujoco.MjModel, samples: int = 45) -> float:
    """Largest J2 moment arm the arm can reach, for scaling an envelope."""
    data = mujoco.MjData(model)
    j2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint2")
    j3 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint3")
    out = 0.0
    for q2 in np.linspace(*model.jnt_range[j2], samples):
        for q3 in np.linspace(*model.jnt_range[j3], samples):
            data.qpos[:] = 0
            data.qpos[1], data.qpos[2] = q2, q3
            mujoco.mj_forward(model, data)
            out = max(out, j2_moment_arm(data, model))
    return out


def grasp_point(model: mujoco.MjModel) -> tuple[int, np.ndarray]:
    """(body id, point in that body's frame) where a payload actually hangs.

    A grasped object sits between the fingers, not at the gripper's centre of
    mass. That distinction is worth 113 mm of moment arm: `gripper_end`'s COM
    is that far behind its tool origin, so adding payload as a bare mass bump
    -- which leaves the COM untouched -- applies the load well inboard of where
    it really acts and understates the shoulder moment.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    tool = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_end")
    fingers = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("gripper_left", "gripper_right")
    ]
    fingers = [f for f in fingers if f >= 0]
    if not fingers:
        return tool, np.zeros(3)

    world = np.mean([data.xpos[f] for f in fingers], axis=0)
    rotation = data.xmat[tool].reshape(3, 3)
    return tool, rotation.T @ (world - data.xpos[tool])


def with_payload(model: mujoco.MjModel, payload: float) -> tuple[float, np.ndarray]:
    """Combined mass and COM of the tool body carrying `payload`, in body frame.

    Returned rather than applied, so callers can restore the original state.
    The payload is treated as a point mass at the grasp point: its own inertia
    about its centre is negligible beside the m*d^2 term at this lever arm.
    """
    tool, point = grasp_point(model)
    base_mass = float(model.body_mass[tool])
    base_com = np.array(model.body_ipos[tool], dtype=float)
    if payload <= 0:
        return base_mass, base_com
    total = base_mass + payload
    return total, (base_mass * base_com + payload * point) / total
