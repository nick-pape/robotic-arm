"""The J2 gravity balancer.

M2 showed the shoulder needs 15.36 N*m to hold the arm's own weight at full
reach, against an 11 N*m rated actuator.

Read that correctly: it does not mean the stock arm is broken. Stock bolts J2
to an aluminium flange that conducts heat into the sheet-metal link, and the
firmware gates on temperature, so that pose is time-limited rather than
impossible -- see `thermal.py`, and Seeed's own "stay within about 70% of the
workspace" warning.

The balancer exists because a *printed* link removes that heat path. The spec
assumes 60-70% of rated is available continuously without it (~7.7 N*m), and
against that budget the unbalanced clone cannot hold itself at reach at all.

The element modelled is a zero-free-length (ZFL) spring: force proportional to
current length, F = k*L, with no free length to subtract. That is what makes
exact static balance possible, because for a spring anchored a distance `a`
from the pivot and attached `b` along the arm, the torque is k*a*b*sin(theta) --
the same shape as the gravity moment m*g*r*sin(theta). They cancel identically
when k*a*b = m*g*r, at every angle rather than at one.

Real extension springs approximate ZFL with initial tension or by routing over
an idler with the spring body parked in the base.

The important limitation, which the spec also notes: this cancels the *upper
arm's* moment exactly, but J3 folding changes the distal moment, so only the
constant part can be cancelled. `residual_torque()` measures what is actually
left rather than assuming the ideal.

MJCF uses `springlength="0"` to express ZFL: force is -k*(L - 0) = -k*L.

Units: metres, newtons, N/m, N*m.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from robotic_arm.reference import ARM_JOINTS

#: Cancellation the spring is sized for, N*m.
#:
#: Re-optimised against the PRINTED twin. The original 8.25 was found by
#: sweeping the stock arm, whose self-weight is 15.36 N*m; the twin is lighter
#: at 13.08, so that spring over-cancels and leaves 8.61 N*m at the worst pose
#: -- past the 7.70 N*m continuous budget. 6.5 leaves 6.90.
#:
#: This is why it must live in one place. It was duplicated across three
#: scripts and two test modules, so re-tuning the arm meant remembering all
#: five, and the P2 test that should have caught the mismatch was pointed at
#: the stock model instead.
DEFAULT_CANCEL_NM = 6.5

#: Sites added to the model by the balancer.
ANCHOR_SITE = "j2_balancer_anchor"
ARM_SITE = "j2_balancer_arm"
TENDON_NAME = "j2_balancer"


@dataclass(frozen=True)
class Balancer:
    """A zero-free-length spring across J2.

    Geometry is given in design terms rather than raw frame offsets, because
    the frames are not intuitive here: the J2 axis points along world Y, so an
    anchor offset along Y would sit *on* the axis and produce no torque at all.
    Site positions are derived from these scalars in `add_to_spec`.

    * `anchor_height` -- how far above the J2 pivot the fixed anchor sits (a).
    * `arm_offset` -- how far along the upper arm the spring attaches (b).

    Both are parameters rather than settled design choices; where the anchor can
    physically go depends on the base design, which is not decided.
    """

    stiffness: float  # k, N/m
    anchor_height: float = 0.06  # a, m, above the J2 pivot
    arm_offset: float = 0.10  # b, m, along the arm from the pivot
    damping: float = 0.4  # small, for numerical calm rather than physics
    parent_body: str = "link1"
    arm_body: str = "link2"

    @property
    def cancel_nm(self) -> float:
        """Peak torque this spring can cancel: k*a*b."""
        return self.stiffness * self.anchor_height * self.arm_offset

    @property
    def peak_force(self) -> float:
        """Largest spring force, at maximum extension L = a + b.

        This load goes into the anchor and the J1 bearing stack, so the anchor
        must be metal. It scales directly with how much torque is cancelled.
        """
        return self.stiffness * (self.anchor_height + self.arm_offset)

    @classmethod
    def sized_for(
        cls,
        target_nm: float,
        anchor_height: float = 0.06,
        arm_offset: float = 0.10,
        **kwargs,
    ) -> Balancer:
        """Size a spring to cancel `target_nm` of gravity moment.

        Solves k*a*b = target, the exact-balance condition.
        """
        if anchor_height <= 0 or arm_offset <= 0:
            raise ValueError(
                f"anchor_height and arm_offset must be positive (got "
                f"{anchor_height}, {arm_offset}); a spring through the pivot "
                f"produces no torque"
            )
        return cls(
            stiffness=target_nm / (anchor_height * arm_offset),
            anchor_height=anchor_height,
            arm_offset=arm_offset,
            **kwargs,
        )


def add_to_spec(spec: mujoco.MjSpec, balancer: Balancer) -> mujoco.MjSpec:
    """Add the balancer's sites and spring tendon to a model spec.

    This is a deliberate addition beyond the stock model, unlike the inertial
    swaps: it adds two sites and one tendon. It moves no frames, so the
    kinematic half of requirement F2 still holds.
    """
    parent = spec.body(balancer.parent_body)
    arm = spec.body(balancer.arm_body)

    # The anchor is fixed to the parent, directly above the J2 pivot. The pivot
    # sits at the moving body's origin, whose position is expressed in the
    # parent's frame -- so offsetting from there puts the anchor above the
    # pivot rather than above the parent's own origin.
    pivot_in_parent = np.asarray(arm.pos, dtype=float)
    anchor = pivot_in_parent + np.array([0.0, 0.0, balancer.anchor_height])

    # The upper arm extends along its own -X (link3 hangs off at -0.236 X), so
    # attaching at -b puts the spring on the arm, perpendicular to the J2 axis.
    attach = np.array([-balancer.arm_offset, 0.0, 0.0])

    parent.add_site(name=ANCHOR_SITE, pos=anchor)
    arm.add_site(name=ARM_SITE, pos=attach)

    tendon = spec.add_tendon(name=TENDON_NAME)
    # stiffness is polynomial-capable in recent MuJoCo; index 0 is the linear
    # term, which is the ordinary spring constant.
    stiffness = np.zeros_like(tendon.stiffness)
    stiffness[0] = balancer.stiffness
    tendon.stiffness = stiffness

    damping = np.zeros_like(tendon.damping)
    damping[0] = balancer.damping
    tendon.damping = damping

    # springlength 0 is what makes this zero-free-length: F = -k*(L - 0).
    tendon.springlength = np.array([0.0, 0.0])
    tendon.width = 0.0025
    tendon.rgba = np.array([0.9, 0.2, 0.2, 1.0])

    tendon.wrap_site(ANCHOR_SITE)
    tendon.wrap_site(ARM_SITE)
    return spec


def static_torques(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    """Net static joint torque an actuator must supply, including any spring.

    MuJoCo's equation of motion puts gravity in `qfrc_bias` and passive spring
    forces in `qfrc_passive`, so what the motor actually has to hold is the
    difference. Using this everywhere means the analysis and the simulation
    cannot disagree about the spring.
    """
    data = mujoco.MjData(model)
    data.qpos[: len(qpos)] = qpos
    mujoco.mj_forward(model, data)
    net = data.qfrc_bias[: len(ARM_JOINTS)] - data.qfrc_passive[: len(ARM_JOINTS)]
    return np.abs(net.copy())


def residual_torque(
    model: mujoco.MjModel, joint_index: int = 1, samples: int = 41
) -> tuple[float, np.ndarray]:
    """Worst-case remaining torque on a joint after the spring.

    Sweeps J2 x J3, because J3 folding is exactly what the spring cannot
    cancel. Returns the worst torque and the pose that produced it.
    """
    j2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint2")
    j3 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint3")

    worst, worst_q = 0.0, np.zeros(model.nq)
    for q2 in np.linspace(*model.jnt_range[j2], samples):
        for q3 in np.linspace(*model.jnt_range[j3], samples):
            qpos = np.zeros(model.nq)
            qpos[1], qpos[2] = q2, q3
            tau = static_torques(model, qpos)[joint_index]
            if tau > worst:
                worst, worst_q = float(tau), qpos.copy()
    return worst, worst_q


def optimise_cancellation(
    build,
    bounds: tuple[float, float] = (4.0, 14.0),
    step: float = 0.25,
    samples: int = 41,
) -> tuple[Balancer, float]:
    """Find the cancellation target that minimises worst-case residual J2 torque.

    There is a genuine optimum rather than "cancel everything". Over-springing
    makes the worst case *worse*: at folded poses the gravity moment is small
    and an oversized spring drives the joint the other way. The best spring
    cancels roughly the workspace-average moment, not the peak.

    `build` takes a Balancer and returns a compiled MjModel, so the caller owns
    how the model is assembled.
    """
    best: tuple[Balancer, float] | None = None
    target = bounds[0]
    while target <= bounds[1] + 1e-9:
        candidate = Balancer.sized_for(target)
        worst, _ = residual_torque(build(candidate), samples=samples)
        if best is None or worst < best[1]:
            best = (candidate, worst)
        target += step
    assert best is not None
    return best


def max_continuous_payload(
    model: mujoco.MjModel,
    limit_nm: float,
    payload_body: str = "gripper_end",
    samples: int = 31,
    tolerance: float = 0.02,
    ceiling: float = 6.0,
) -> float:
    """Heaviest payload whose worst-case J2 torque stays within `limit_nm`.

    The payload is added as mass on the tool body and the workspace re-swept,
    rather than assumed to act on a fixed moment arm: the pose that loads J2
    hardest with a payload is not the same pose that does so under self-weight.

    Worst-case torque is monotonic in payload mass, so this bisects rather than
    scanning -- each evaluation is a full workspace sweep and is not cheap.

    Mutates the model's mass in place and restores it, which is far cheaper
    than regenerating the MJCF per candidate payload.
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, payload_body)
    if bid < 0:
        raise KeyError(f"no body named {payload_body!r}")

    original = float(model.body_mass[bid])

    def worst_with(payload: float) -> float:
        model.body_mass[bid] = original + payload
        return residual_torque(model, samples=samples)[0]

    try:
        if worst_with(0.0) > limit_nm:
            return 0.0
        if worst_with(ceiling) <= limit_nm:
            return ceiling

        low, high = 0.0, ceiling
        while high - low > tolerance:
            mid = (low + high) / 2
            if worst_with(mid) <= limit_nm:
                low = mid
            else:
                high = mid
        return low
    finally:
        model.body_mass[bid] = original
