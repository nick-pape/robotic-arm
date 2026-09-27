"""How far each joint can actually turn, measured rather than inherited.

The stock ranges came from the URDF and describe the *stock* arm's geometry:
flat forks and beams that fold past each other. The printed arm is barrels
and tees, so those numbers describe nothing here. They are wrong in both
directions at once -- J2 was allowed only 180 degrees where the printed arm
clears 342, and J3's inherited upper limit put the forearm through the
shoulder, which a sweep catches immediately.

So the ranges are swept from the geometry: rotate one joint away from the
home pose until a non-adjacent pair of links comes within `CLEARANCE` of
touching, and stop at the last angle that was clear.

**Adjacent links are excluded, and have to be.** Two links that share a joint
are concentric barrels meeting at a seam; they are in contact at every angle
by construction, so including them would report zero range everywhere. The
model's own `<exclude>` list already encodes exactly this, and the sweep
inherits it.

**This is a per-joint sweep, and per-joint sweeps are optimistic.** Each joint
is moved with the others at home, so a combination of two joints can still
collide inside these limits. That is the normal state of affairs for a serial
arm -- UR ship +/-360 degrees on every joint and leave the rest to runtime
checking -- and it is why the S1 clearance sweep exists separately. These
limits are a guard rail, not a proof.
"""

from __future__ import annotations

import numpy as np

#: Required gap before two links count as colliding, in metres. This is
#: requirement S1's 3 mm, applied as a geom margin so MuJoCo reports a contact
#: while the parts are still apart rather than once they have merged.
CLEARANCE = 0.003

#: Most total travel any joint is given, in radians -- one full turn. Nothing
#: mechanical here enforces it; it is a deliberate ceiling, since past a full
#: turn a joint only repeats poses it already had while the cable run does not.
TOTAL_CAP = 2 * np.pi

#: Sweep resolution. The reported limit is always a sample that was measured
#: clear, so the error is one-sided: the true boundary is up to this much
#: further out, never nearer. The 3 mm clearance covers the difference.
STEP = np.radians(1.0)


def collision_free_ranges(
    model,
    clearance: float = CLEARANCE,
    total_cap: float = TOTAL_CAP,
    step: float = STEP,
) -> dict[str, tuple[float, float]]:
    """Largest clear arc each joint can turn through, centred on home.

    `model` is consumed as scratch: its geom margins are overwritten. Pass a
    throwaway compile, not a model anything else holds.

    Home is `qpos = 0`, which after `homepose` is the upright pose -- **not**
    `qpos0`, which is where every joint's rotation is zero and which on this
    arm is the stock sprawl. Sweeping from `qpos0` reports zero range for five
    of the six joints, because the sprawl self-collides before the sweep even
    starts.
    """
    import mujoco

    model.geom_margin[:] = clearance
    data = mujoco.MjData(model)
    half = total_cap / 2.0

    def clear(address: int, angle: float) -> bool:
        data.qpos[:] = 0.0
        data.qpos[address] = angle
        mujoco.mj_forward(model, data)
        return data.ncon == 0

    ranges: dict[str, tuple[float, float]] = {}
    for index in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        if name is None or not name.startswith("joint") or name[-1] not in "123456":
            continue
        address = int(model.jnt_qposadr[index])
        bounds = []
        for direction in (1.0, -1.0):
            reached = 0.0
            angle = 0.0
            while abs(angle) < half - 1e-9:
                angle = direction * min(abs(angle) + step, half)
                if not clear(address, angle):
                    break
                reached = angle
            bounds.append(reached)
        high, low = bounds
        ranges[name] = (low, high)
    return ranges


def apply_joint_ranges(spec, ranges: dict[str, tuple[float, float]]):
    """Write swept ranges onto `spec`'s joints and their position actuators.

    Both, always. A `ctrlrange` left at the old limits silently keeps the
    extra travel unreachable through the actuators, which is the form this
    mistake takes when it is made.
    """
    for name, (low, high) in ranges.items():
        joint = spec.joint(name)
        joint.range = [low, high]
        joint.limited = True

    for actuator in spec.actuators:
        if actuator.target in ranges:
            actuator.ctrlrange = list(ranges[actuator.target])

    return spec
