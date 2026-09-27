"""Where the arm stands when every joint reads zero.

Stock zero is a sprawl: the upper arm lies horizontally along -x, the forearm
doubles back, and the tool points out sideways at roughly elbow height. That
is an artefact of how the URDF's link frames happened to be authored, not a
pose anyone would choose, and it makes every commanded angle hard to read --
"joint 3 at -2.9" tells you nothing about where the arm is.

This module re-zeros the arm so that all-zeros is a pose worth naming: the
arm standing straight up, the wrist turned to point left, and the gripper
fingers lying level with the ground.

It does that with the joint `ref` attribute, **not** by rotating body frames.
MuJoCo applies `qpos - ref` as the joint's rotation, so `ref` shifts the
coordinate without moving a single frame: the kinematics, the segment lengths
and the reach are all bit-identical afterwards, and only the numbers on the
dials change. Rotating frames instead would have meant re-deriving every
joint origin, which is exactly the class of change the project keeps out of
the generator.

Three things have to move together with `ref`, and missing any one of them
silently changes what the arm can do rather than what it reads:

* **joint ranges**, which MuJoCo applies to `qpos`, not to the rotation. Left
  alone, re-zeroing would quietly rotate the reachable arc of every joint.
* **actuator ctrlranges**, for the same reason -- these command `qpos`.
* **keyframes**, whose stored `qpos` would otherwise mean a different pose.

`home` is the exception: it is deliberately *not* shifted, because all-zeros
is now the pose it is supposed to name.

**`qpos0` is not home.** `qpos0` equals `ref`, which is where every joint's
*rotation* is zero -- on this arm, the old stock sprawl. Home is `qpos = 0`.
The two were the same thing before this module existed and are not now, and
that distinction has already produced three separate wrong answers in this
project: a joint-range sweep that reported zero travel on five of six joints,
an interference check that reported four collisions at "home", and a payload
comparison skewed by 0.3 kg. Code that wants "the arm as designed" wants
zeros; code that wants "every joint neutral on whatever model this is" wants
`qpos0`.
"""

from __future__ import annotations

import numpy as np

#: Solved by maximising tool height -- see `upright_pose`. J1 and J6 cannot
#: raise the arm (one turns about the vertical, the other rolls about the tool
#: axis) and J5 is set by convention below, so none of them belong here.
UPRIGHT_JOINTS = ("joint2", "joint3", "joint4")

#: Which way the arm faces at home, as a J1 rotation. A convention, not a
#: derived quantity: nothing in the geometry prefers one heading over another.
BASE_YAW = -0.981

#: J5 at home, turning the wrist to point left while the arm itself stays
#: vertical. Also a convention. It is applied *after* `upright_pose` rather
#: than inside it, so that the body of the arm is still solved for true
#: vertical and only the wrist is turned.
WRIST_YAW = -1.63

#: Which way link4's barrel -- the "green" run, which lies along the J5 axis
#: -- points at home. "vertical" stands it in line with the forearm,
#: "horizontal" lays it across.
#:
#: It has to be said explicitly because nothing else constrains it. J4 is
#: distal to every origin that fixes the arm's posture, so the height solve
#: that defines home does not care what J4 does and left it at 23 degrees off
#: either axis -- an angle with no meaning, which is exactly what it looked
#: like.
GREEN_RUN = "vertical"

#: Keyframe left at all-zeros, because zero is now the pose it names.
UNSHIFTED_KEYFRAMES = ("home",)


def upright_pose(model) -> dict[str, float]:
    """Joint **rotations** that stand the arm straight up.

    Defined as the configuration that lifts the tool highest. That is not a
    proxy for "vertical" but the same thing: a serial arm is at its tallest
    exactly when every segment points along +z, because any bend shortens the
    projection of the segments below it.

    Solved on a coarse grid and then polished, rather than from a single seed.
    The objective has local maxima -- folding the elbow the wrong way is one --
    and a seeded search that lands in one would produce a plausible-looking
    pose that is not actually upright.
    """
    import itertools

    import mujoco
    from scipy.optimize import minimize

    data = mujoco.MjData(model)
    address = [int(model.joint(n).qposadr[0]) for n in UPRIGHT_JOINTS]
    # Everything here is in **rotation**, not `qpos`: rotation is
    # `qpos - ref`, and `ref` is `qpos0` for a hinge. Working in qpos would
    # make the answer depend on whether the model passed in had already been
    # re-zeroed, which is exactly the question this function must not care
    # about -- and did not, until it was handed a re-zeroed model and returned
    # a pose half in one convention and half in the other.
    bounds = [
        (
            float(model.joint(n).range[0]) - float(model.qpos0[adr]),
            float(model.joint(n).range[1]) - float(model.qpos0[adr]),
        )
        for n, adr in zip(UPRIGHT_JOINTS, address)
    ]

    def negative_height(q) -> float:
        data.qpos[:] = model.qpos0
        for adr, rotation in zip(address, q):
            data.qpos[adr] = rotation + model.qpos0[adr]
        mujoco.mj_forward(model, data)
        return -float(data.body("gripper_end").xpos[2])

    grid = [np.linspace(lo, hi, 5) for lo, hi in bounds]
    seeds = sorted(itertools.product(*grid), key=negative_height)[:8]

    best = None
    for seed in seeds:
        found = minimize(negative_height, seed, bounds=bounds, method="L-BFGS-B")
        if best is None or found.fun < best.fun:
            best = found

    pose = {name: float(q) for name, q in zip(UPRIGHT_JOINTS, best.x)}
    # J4 does not move link2, link3 or link4's origin -- it only turns the
    # wrist -- so maximising height leaves it effectively unconstrained, and
    # the value it happened to land on left link4's barrel tilted 23 degrees
    # out of horizontal. Set it by the criterion that actually matters here.
    pose["joint4"] = align_wrist_axis(model, pose)
    pose["joint1"] = BASE_YAW
    pose["joint5"] = WRIST_YAW
    pose["joint6"] = level_gripper(model, pose)
    return pose


def align_wrist_axis(model, pose: dict[str, float]) -> float:
    """J4 rotation that squares link4's barrel to the world.

    link4 is a tee whose run lies along the J5 axis, so "green is level" and
    "the J5 axis is horizontal" are the same statement. J4 is the last joint
    before it and the only one that tilts it: a joint cannot rotate its own
    axis, and J1 turns about the vertical, which cannot change a direction's
    z-component.

    Free to choose, because J4 is distal to every origin that defines the
    arm's posture -- link2, link3 and link4 all stay exactly where the height
    solve put them, so the arm is still standing straight up afterwards.

    `GREEN_RUN` picks which square orientation. Horizontal has two solutions
    half a turn apart, and this takes the one still pointing the way the
    tilted axis did, so squaring it does not also swing the wrist round to
    face the other way. Vertical is unambiguous up to sign, and the sign only
    decides which end of the barrel is up.
    """
    import mujoco
    from scipy.optimize import brentq

    data = mujoco.MjData(model)
    joint = model.joint("joint5")
    body = int(model.jnt_bodyid[joint.id])
    axis = np.asarray(model.jnt_axis[joint.id], dtype=float)
    roll = int(model.joint("joint4").qposadr[0])
    others = {
        int(model.joint(n).qposadr[0]): q
        for n, q in pose.items()
        if n != "joint4"
    }

    def wrist_axis(angle: float) -> np.ndarray:
        data.qpos[:] = model.qpos0
        for adr, rotation in others.items():
            data.qpos[adr] = rotation + model.qpos0[adr]
        data.qpos[roll] = angle + model.qpos0[roll]
        mujoco.mj_forward(model, data)
        return data.xmat[body].reshape(3, 3) @ axis

    grid = np.radians(np.arange(-180.0, 180.1, 5.0))

    if GREEN_RUN == "vertical":
        # Vertical is a turning point of the z-component, not a crossing, so
        # it is found by maximising |z| rather than by root-finding.
        coarse = max(grid, key=lambda a: abs(wrist_axis(a)[2]))
        fine = np.linspace(coarse - np.radians(5.0), coarse + np.radians(5.0), 401)
        return float(max(fine, key=lambda a: abs(wrist_axis(a)[2])))

    was = wrist_axis(pose["joint4"])
    heading = was[:2] / np.linalg.norm(was[:2])
    values = [wrist_axis(a)[2] for a in grid]
    roots = [
        brentq(lambda a: wrist_axis(a)[2], a, b)
        for a, b, u, v in zip(grid, grid[1:], values, values[1:])
        if u == 0.0 or u * v < 0.0
    ]
    if not roots:
        return pose["joint4"]
    return float(max(roots, key=lambda r: wrist_axis(r)[:2] @ heading))


def level_gripper(model, pose: dict[str, float]) -> float:
    """J6 rotation that lays the gripper fingers level with the ground.

    "Level" is measured on the axis the fingers open along: the arm is level
    when that axis is horizontal. Measured with the fingers **open**, because
    closed they sit on top of each other and the vector between them is pure
    numerical noise -- a first attempt measured it closed and found no level
    angle anywhere in the full turn.

    Depends on J1 and J5, so it can only be solved once those are fixed. With
    the tool axis vertical, every J6 would qualify; it is the wrist turn that
    makes the answer meaningful.
    """
    import mujoco
    from scipy.optimize import brentq

    data = mujoco.MjData(model)
    left, right = model.body("gripper_left").id, model.body("gripper_right").id
    fingers = {
        int(model.joint(n).qposadr[0]): float(model.joint(n).range[1])
        for n in ("joint_left", "joint_right")
    }
    others = {int(model.joint(n).qposadr[0]): q for n, q in pose.items()}
    roll = int(model.joint("joint6").qposadr[0])

    def tilt(angle: float) -> float:
        # `pose` and `angle` are rotations; qpos is rotation + ref (= qpos0).
        data.qpos[:] = model.qpos0
        for adr, rotation in others.items():
            data.qpos[adr] = rotation + model.qpos0[adr]
        for adr, opened in fingers.items():
            data.qpos[adr] = opened
        data.qpos[roll] = angle + model.qpos0[roll]
        mujoco.mj_forward(model, data)
        span = data.xpos[right] - data.xpos[left]
        return float(span[2] / np.linalg.norm(span))

    # Scan for sign changes across the full turn and take the root needing the
    # smallest turn. Roots come in pairs 180 degrees apart -- the same axis
    # with the two fingers swapped -- so either is level and the nearer one is
    # the smaller change to what the joint already reads.
    grid = np.radians(np.arange(-180.0, 180.1, 5.0))
    values = [tilt(a) for a in grid]
    roots = [
        brentq(tilt, a, b)
        for a, b, u, v in zip(grid, grid[1:], values, values[1:])
        if u == 0.0 or u * v < 0.0
    ]
    if not roots:
        return 0.0
    return float(min(roots, key=abs))


def apply_home_zero(spec, pose: dict[str, float], address: dict[str, int]):
    """Re-zero `spec` so all-zeros is `pose`, preserving what the arm can do.

    `address` maps joint name to its `qpos` index, which only a compiled model
    knows, so the caller passes it in rather than this compiling a second time.
    """
    for name, q in pose.items():
        joint = spec.joint(name)
        joint.ref = -q
        low, high = (float(v) for v in joint.range)
        joint.range = [low - q, high - q]

    for actuator in spec.actuators:
        if actuator.target in pose:
            q = pose[actuator.target]
            low, high = (float(v) for v in actuator.ctrlrange)
            actuator.ctrlrange = [low - q, high - q]

    for key in spec.keys:
        if key.name in UNSHIFTED_KEYFRAMES:
            continue
        qpos = list(key.qpos)
        for name, q in pose.items():
            qpos[address[name]] -= q
        key.qpos = qpos

    return spec
