"""The UR-style link archetype: rotor flange, tube, stator housing.

A cobot link is the same shape at every joint on the arm::

    (=)  ================  [  M  ]
     |          tube          |
     |                        stator housing: encloses this link's
     |                        child motor and bolts to its outer ring
     rotor flange: caps the parent's housing and
     bolts to the previous joint's inner ring

Both actuators here are pancake motors with **two concentric bolt rings on one
face** -- an inner rotor ring on a raised hub and an outer stator ring on a
fixed flange. A joint is therefore two links bolted to the *same* face at
different radii, and the visible seam between them is where the rotor hub
meets the housing mouth.

Earlier revisions of this project reproduced the *stock* arm's structure
instead: a fork of two flat plates straddling exposed motors, held apart by a
standoff collar. That is a legitimate machine-tool design and a poor fit for
printed parts, and it is not what a UR-style arm looks like. The stock plates
also turn out not to bolt to the motors at all -- their O54 circle is the
plate-to-plate standoff, so stock geometry is no guide to the motor interface.
The vendor STEP is: RS06 rotor O24.02 x 6, stator O82 x 8; RS00 rotor O27 x 6,
stator O50 x 6.

Units: millimetres.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from robotic_arm.design import RULES, STYLE
from robotic_arm.parts.cobot import Drum

#: Axial gap left at the joint seam, so the rotating half never rubs the fixed
#: half. The motor's own hub protrusion is consumed by seating the housing on
#: the stator face, so this is on top of it.
SEAM_GAP = 0.6


def housing_diameter(actuator, clearance: float | None = None,
                     wall: float | None = None) -> float:
    """Outside diameter of the cylinder that encloses this actuator.

    Set by the motor, not by taste: the barrel has to clear the stator flange
    with a running gap and still carry a full structural wall. An RS06 is O87
    across its flange, so no housing that actually encloses it can be slimmer
    than about O94 -- wider than the stock arm's O67 rings, which get away
    with it by leaving the motor in open air.
    """
    clearance = STYLE.joint_gap if clearance is None else clearance
    wall = RULES.structural_wall_thickness if wall is None else wall
    return actuator.stator_outer_diameter + 2 * clearance + 2 * wall


def housing_bore(actuator, clearance: float | None = None,
                 wall: float | None = None) -> float:
    """The opening in the housing's mouth, through which the joint works.

    Wide enough to clear the rotor hub turning in it and to admit a driver to
    the rotor bolts of the link that caps it, and narrow enough to leave the
    housing's own stator bolts a seat.
    """
    from robotic_arm.assembly import DRIVER_DIAMETER

    clearance = STYLE.joint_gap if clearance is None else clearance
    wall = RULES.structural_wall_thickness if wall is None else wall
    bore = max(
        actuator.hub_diameter + 2 * clearance,
        actuator.rotor_circle.bcd + DRIVER_DIAMETER["M3"] + 2 * wall,
    )
    headroom = actuator.stator_circle.bcd - M3_HEAD - 2 * clearance
    if bore > headroom:
        raise ValueError(
            f"{actuator.name}: a O{bore:.1f} bore leaves no seat for its "
            f"O{actuator.stator_circle.bcd:.0f} stator bolts"
        )
    return bore


#: ISO 4762 socket-cap head across-corners for M3.
M3_HEAD = 5.5


def stator_housing(axis, joint_plane, actuator, length: float,
                   wall: float | None = None) -> Drum:
    """The cylinder that encloses a motor, open at the joint plane.

    `axis` points from the joint plane into the body of the link that holds
    it -- the motor lies on that side.

    The mouth cap **bears on the stator face**, which sits one hub-protrusion
    behind the rotor hub face, and it has to sit on the *child's* side of that
    face because that is where its bolts go in. So the cap spans
    `[hub_protrusion - wall, hub_protrusion]`, straddling the joint plane
    slightly, and its bore clears the rotor hub turning inside it.

    It used to be recessed to `hub_protrusion + SEAM_GAP`, which put the whole
    2 mm cap annulus *inside the motor*: 2,551 mm3 of clash at J5 and J6, and
    5,900 mm3 at J1 where the RS06 is fatter. It also left a 2.1 mm annular
    slot open at the seam -- about 620 mm2 of side window, with the motor
    visible through it -- where a UR5e shows a hairline.
    """
    wall = RULES.structural_wall_thickness if wall is None else wall
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    mouth = np.asarray(joint_plane, dtype=float) + axis * (
        actuator.hub_protrusion - wall
    )
    return Drum(
        centre=mouth + axis * length / 2,
        axis=axis,
        diameter=housing_diameter(actuator),
        length=length,
    )


def flange_relief_depth(actuator, wall: float | None = None) -> float:
    """How far a rotor flange must be set back outside the hub it bears on.

    The flange contacts the rotor hub face over the hub's own diameter. The
    housing's mouth cap now reaches `wall - hub_protrusion` past that face on
    the flange's side, so everything outside the hub has to clear it by a seam
    gap or the two halves of the joint grind together.
    """
    wall = RULES.structural_wall_thickness if wall is None else wall
    return max(0.0, wall - actuator.hub_protrusion) + SEAM_GAP


def stub_diameter(actuator, wall: float | None = None) -> float:
    """Diameter of a flange that is a *stub* rather than a visible cap.

    Just enough to carry the rotor bolt circle with a driver and a wall, and
    to stay under the hub it bears on so it turns freely inside the housing
    it enters.
    """
    from robotic_arm.assembly import DRIVER_DIAMETER

    wall = RULES.structural_wall_thickness if wall is None else wall
    return max(
        actuator.rotor_circle.bcd + DRIVER_DIAMETER["M3"] + 2 * wall,
        actuator.hub_diameter - 2 * STYLE.joint_gap,
    )


def rotor_flange(axis, joint_plane, actuator, length: float,
                 diameter: float | None = None) -> Drum:
    """The part that bolts to a rotor and turns with it.

    By default it matches the housing it caps, so the joint reads as one
    continuous cylinder with a fine seam. Pass `diameter` for the other case:
    a **stub** that disappears inside the neighbouring barrel rather than
    forming a visible half of the joint.

    The shoulder needs the stub. Its vertical barrel is the primary body and
    the piece reaching across to M2 is a small spigot sticking out of its
    side, not a second full-diameter barrel -- making it full diameter gave
    the shoulder two competing cylinders instead of one.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    plane = np.asarray(joint_plane, dtype=float)
    return Drum(
        centre=plane + axis * length / 2,
        axis=axis,
        diameter=housing_diameter(actuator) if diameter is None else diameter,
        length=length,
    )


def motor_depth(actuator) -> float:
    """How far the motor reaches back from its face, along its own axis."""
    return float(actuator.bbox_mm[2])


def default_housing_length(actuator, wall: float | None = None) -> float:
    """Just deep enough to enclose the motor and close behind it."""
    wall = RULES.structural_wall_thickness if wall is None else wall
    return motor_depth(actuator) - actuator.hub_protrusion + 2 * wall


def _joint_pose(joint: str):
    """(anchor, world axis, model, data) for a joint at the zero pose."""
    import mujoco

    from robotic_arm.reference import load_baseline

    model = load_baseline()
    data = mujoco.MjData(model)
    data.qpos[:] = 0
    mujoco.mj_forward(model, data)
    jid = model.joint(joint).id
    axis = data.xaxis[jid] / np.linalg.norm(data.xaxis[jid])
    return data.xanchor[jid].copy(), axis, model, data


def motor_side(joint: str) -> np.ndarray:
    """World direction from a joint plane toward the motor that drives it.

    The housing has to enclose that motor, so this is the direction it must
    reach. Measured from the stock motor meshes rather than inferred: a link's
    centre of mass is a poor witness here because every link on this arm
    extends *perpendicular* to its own joint axis, so its COM barely projects
    onto that axis at all and the sign is noise. The motor is a 50 mm slug
    sitting squarely on one side, so it is not ambiguous.
    """
    import mujoco

    anchor, axis, model, data = _joint_pose(joint)

    # Only motors carried by the two bodies this joint connects. Without
    # this, a neighbouring motor is scored too, and at the wrist -- where J5
    # and J6 are 60 mm apart on *intersecting* axes -- motor_6 outvoted
    # motor_5 and put J5's motor on the wrong side of its own joint. That is
    # what made link4 and link5 occupy the same space. The same contamination
    # had already been found and fixed once in `mounts.joint_roles`; it was
    # never fixed here.
    child_body = int(model.jnt_bodyid[model.joint(joint).id])
    adjacent = {child_body, int(model.body_parentid[child_body])}

    reach = []
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if model.geom_bodyid[gid] not in adjacent:
            continue
        mesh_id = model.geom_dataid[gid]
        if not model.mesh(mesh_id).name.startswith("motor"):
            continue
        first, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        vertices = model.mesh_vert[first : first + count].astype(float)
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[gid])
        vertices = vertices @ rotation.reshape(3, 3).T + model.geom_pos[gid]
        bid = model.geom_bodyid[gid]
        vertices = vertices @ data.xmat[bid].reshape(3, 3).T + data.xpos[bid]

        delta = vertices - anchor
        along = delta @ axis
        radial = np.linalg.norm(delta - np.outer(along, axis), axis=1)
        # One mesh can hold two motors (motor_2_3 does), so select the slug
        # actually centred on this axis rather than the whole mesh.
        near = (radial < 0.050) & (np.abs(along) < 0.080)
        if near.sum() <= 50:
            continue
        # A joint's own motor bears against its joint plane, so its material
        # reaches along ~ 0. A neighbour's does not. Adjacency alone cannot
        # separate them at the wrist: motor_6 sits on link5, which is one of
        # J5's own two bodies, and it was outvoting motor_5 -- putting J5's
        # motor below its joint when the real one is above it, which is what
        # drove link4 and link5 into each other.
        if float(np.abs(along[near]).min()) > 0.005:
            continue
        reach.append(float(np.mean(along[near])))
    if not reach:
        # No motor is modelled on either of this joint's own bodies, which is
        # the case at J1. Point into whichever body holds the stator: that is
        # where the motor must live, and it is the same rule the modelled
        # joints obey. Reading the nearest motor mesh instead picks up
        # motor_2_3 sitting above J1 and puts the base's own motor on the
        # wrong side of its joint, sending link1's flange down into the base.
        child_name = model.body(child_body).name
        holder = child_body if carries_own_housing(child_name) else int(
            model.body_parentid[child_body]
        )
        toward = float((data.xipos[holder] - anchor) @ axis)
        if abs(toward) > 1e-6:
            return axis * (1.0 if toward > 0 else -1.0)

        # Last resort: whatever motor geometry sits on this axis at all.
        for gid in range(model.ngeom):
            if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh_id = model.geom_dataid[gid]
            if not model.mesh(mesh_id).name.startswith("motor"):
                continue
            first, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
            vertices = model.mesh_vert[first : first + count].astype(float)
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[gid])
            vertices = vertices @ rotation.reshape(3, 3).T + model.geom_pos[gid]
            bid = model.geom_bodyid[gid]
            vertices = vertices @ data.xmat[bid].reshape(3, 3).T + data.xpos[bid]
            delta = vertices - anchor
            along = delta @ axis
            radial = np.linalg.norm(delta - np.outer(along, axis), axis=1)
            near = (radial < 0.050) & (np.abs(along) < 0.080)
            if near.sum() > 50:
                reach.append(float(np.mean(along[near])))
    if not reach:
        raise KeyError(f"no motor mesh found on {joint}")
    return axis * (1.0 if max(reach, key=abs) >= 0 else -1.0)


#: Where this design deliberately differs from stock about which link carries
#: a joint's housing. "parent" puts the barrel on the link nearer the base.
#:
#: Empty, and worth keeping so as a place to record deviations.
#:
#: J2 was briefly overridden to "parent", on a reading that the shoulder
#: should be a full-width barrel on the J1-rotating link. Looking at where a
#: UR5e's end caps actually are settles it the other way: the J1 motor lives
#: in the **base**, the J2 motor lives in the **upper arm**, and there is no
#: distinct shoulder part between them at all. So link1 is only a short
#: connector carrying the J1 rotor through to the J2 rotor, and its being
#: barely visible is correct rather than a defect. That also happens to be
#: what `mounts` measures from stock, so no override is needed.
HOUSING_OWNER: dict[str, str] = {"joint1": "child"}

#: How far a motor sits along its own joint axis, from the joint origin,
#: measured toward the motor side. Zero means the rotor hub face lands exactly
#: on the joint origin, which is the datum convention everywhere else.
#:
#: Sliding a motor along its own rotation axis changes no kinematics at all --
#: the axis and the joint frame are untouched, so reach and segment lengths
#: are unaffected. It is purely where the hardware sits on that axis.
#:
#: J1 needs it. Its motor belongs in link1, but a 50.5 mm RS06 seated at the
#: joint origin would push link1's barrel up to z=128 and straight through
#: link2's J2 barrel, which reaches down to z=98. Seating it 35 mm lower drops
#: the barrel to z=93 and it clears, leaving the base as the small flare a
#: UR5e has rather than a full-height column.
MOTOR_PLANE_OFFSET = {"joint1": -35.0}

#: Joints whose motor is mounted facing the opposite way from stock.
#:
#: J2. Stock runs link2's barrel in -y, straight across link1's vertical axis
#: at y=-1, so it occupies z>102.6 there and link1 can never reach up to its
#: own J2 flange -- the shoulder cannot close into the sideways T a UR5e has.
#: Facing the motor +y instead puts link2's barrel clear of that axis and
#: brings link1's flange back over it, where the two can fuse. The joint
#: frame, axis and origin are untouched, so reach and segment lengths are
#: unaffected; only which side of the plane the hardware sits on changes.
#:
#: J5. Stock puts M5 on the far side of the joint from link4, so link4's
#: barrel grows away from its own branch while link5's flange grows *back*
#: along it -- the two then occupy the same space, coaxially, and link4's
#: branch runs straight through link5's flange. Facing M5 the other way puts
#: link4's barrel on the branch side, where it merges with the branch, and
#: sends link5 clear to the far side.
MOTOR_SIDE_FLIP: set[str] = {"joint2"}


def motor_plane(body: str, joint_index: int) -> np.ndarray:
    """Where the rotor hub face sits, in `body`'s frame.

    The joint origin unless `MOTOR_PLANE_OFFSET` moves it along the axis.
    """
    from robotic_arm.linkframes import link_frame

    joint = f"joint{joint_index}"
    frame = link_frame(body)
    index = int(body.removeprefix("link")) if body.startswith("link") else None
    origin = (
        np.zeros(3)
        if joint_index == index
        else np.asarray(frame.child_origin, dtype=float)
    )
    offset = MOTOR_PLANE_OFFSET.get(joint, 0.0)
    if offset == 0.0:
        return origin
    side = _in_frame(body, np.asarray(joint_motor_side(joint), dtype=float))
    return origin + side * offset


def carries_own_housing(body: str) -> bool:
    """Does this link hold the stator of its *own* joint?

    Measured, not assumed -- see `robotic_arm.mounts`. True for link2, which
    carries the stators of both J2 and J3; that is why the stock model ships
    one `motor_2_3` mesh belonging to it.
    """
    from robotic_arm.mounts import own_joint_role

    index = int(body.removeprefix("link"))
    override = HOUSING_OWNER.get(f"joint{index}")
    if override is not None:
        return override == "child"

    try:
        return own_joint_role(body) == "stator"
    except KeyError:
        # J1's roles are ambiguous in the stock meshes; the base holds that
        # stator and link1 turns on it, like every other first joint.
        return False


def child_carries_housing(joint: str) -> bool:
    """Does the child link hold the stator of `joint`? (`o[n] = +1`)"""
    override = HOUSING_OWNER.get(joint)
    if override is not None:
        return override == "child"

    import mujoco

    from robotic_arm.reference import load_baseline

    model = load_baseline()
    child = model.body(int(model.jnt_bodyid[model.joint(joint).id])).name
    return carries_own_housing(child)


def stock_child_carries_housing(joint: str) -> bool:
    """What stock does, ignoring any override -- used to flip the motor side."""
    import mujoco

    from robotic_arm.reference import load_baseline

    model = load_baseline()
    child = model.body(int(model.jnt_bodyid[model.joint(joint).id])).name
    return carries_own_housing(child)


@lru_cache(maxsize=None)
def joint_motor_side(joint: str) -> tuple[float, float, float]:
    """World direction from a joint plane toward its motor, as built here.

    Where two consecutive joints are **parallel** -- J2, J3 and J4 all turn
    about the same axis on this arm -- the link between them carries a barrel
    at each end on parallel axes. Both of that link's ends must reach the same
    way or the tube runs a barrel-length across the link and comes out
    diagonal. A UR5e's forearm is straight because both its barrels lean the
    same way; the arm zigzags at the joints instead.

    A housing reaches *toward* its motor and a flange *away* from it, so which
    way an end reaches depends on whether this link holds that joint's stator.
    With `o[n] = +1` when the child link carries joint n's housing:

        side[n+1] = -o[n] * o[n+1] * side[n]        (parallel axes only)

    An earlier version hardcoded `o = -1`, i.e. "the parent always holds the
    stator". That is true at J3, J4 and J5 and false at J2, where stock puts
    link1 on the rotor -- the arrangement a UR5e also uses, with the motor in
    the rotating half bolting back to the fixed column.
    """
    index = int(joint.removeprefix("joint"))
    _, axis, _, _ = _joint_pose(joint)
    if index > 1:
        previous = f"joint{index - 1}"
        _, previous_axis, _, _ = _joint_pose(previous)
        if abs(float(axis @ previous_axis)) > 0.99:
            o_prev = 1.0 if child_carries_housing(previous) else -1.0
            o_this = 1.0 if child_carries_housing(joint) else -1.0
            sign = -o_prev * o_this
            return tuple(
                sign * np.asarray(joint_motor_side(previous), dtype=float)
            )
    side = motor_side(joint)
    if joint in MOTOR_SIDE_FLIP:
        side = -side
    override = HOUSING_OWNER.get(joint)
    if override is not None and (override == "child") != stock_child_carries_housing(
        joint
    ):
        # The motor lives with whichever link holds the stator, so moving the
        # housing moves the motor to the other side of the joint plane.
        side = -side
    return tuple(side)


def _in_frame(body: str, world_direction: np.ndarray) -> np.ndarray:
    import mujoco

    from robotic_arm.reference import load_baseline

    model = load_baseline()
    data = mujoco.MjData(model)
    data.qpos[:] = 0
    mujoco.mj_forward(model, data)
    bid = model.body(body).id
    return data.xmat[bid].reshape(3, 3).T @ world_direction


def flange_direction(body: str) -> np.ndarray:
    """Which way a link's rotor flange reaches, in the link's own frame.

    Exactly opposite the motor its joint drives -- so, by construction, the
    opposite side of the joint plane from the housing that encloses that
    motor. Deriving the two ends independently is what put link3's and
    link4's flanges 14 mm inside the housings they were meant to cap.
    """
    index = int(body.removeprefix("link"))
    return _in_frame(body, -np.asarray(joint_motor_side(f"joint{index}")))


def housing_direction(body: str, child_index: int) -> np.ndarray:
    """Which way a link's stator housing reaches, in the link's own frame."""
    return _in_frame(body, np.asarray(joint_motor_side(f"joint{child_index}")))


def end_kinds(body: str) -> tuple[str, str]:
    """("housing" | "flange") for this link's own end and its child end.

    A joint has exactly two sides, so whichever link holds the stator builds a
    housing there and the other builds a flange. Measured per joint rather
    than assumed: link2 comes out with a housing at *both* ends, because stock
    gives it the stators of J2 and J3, and link1 with a flange at both, which
    is the plain vertical column a UR5e has at the shoulder.
    """
    from robotic_arm.linkframes import link_frame

    own = "housing" if carries_own_housing(body) else "flange"
    child = None
    frame = link_frame(body)
    if frame.child_name and frame.child_name.startswith("link"):
        if int(frame.child_name.removeprefix("link")) <= 6:
            # If the child holds its own stator, this link takes the rotor.
            child = "flange" if carries_own_housing(frame.child_name) else "housing"
    return own, child


def end_direction(body: str, joint_index: int, kind: str) -> np.ndarray:
    """Which way an end reaches, in the link's own frame.

    A housing reaches toward its motor so it can enclose it; a flange reaches
    away, because the housing capping it occupies that side.
    """
    side = np.asarray(joint_motor_side(f"joint{joint_index}"), dtype=float)
    return _in_frame(body, side if kind == "housing" else -side)


def _build_end(kind, direction, plane, actuator, flange_length, housing_length,
               flange_diameter=None):
    if kind == "housing":
        length = (
            default_housing_length(actuator)
            if housing_length is None
            else housing_length
        )
        return stator_housing(direction, plane, actuator, length)
    return rotor_flange(direction, plane, actuator, flange_length, flange_diameter)


def link_ends(body: str, flange_length: float, housing_length: float | None = None,
               flange_diameter: float | None = None):
    """(near end, far end, own actuator, carried actuator) for one link.

    "near" is the end at this link's own joint, "far" the end at its child's.
    Either may be a housing or a flange -- see `end_kinds`. This used to
    hardcode flange-at-own / housing-at-child, which cannot express link2 or
    link1 at all.
    """
    from robotic_arm.actuators import for_joint
    from robotic_arm.linkframes import link_frame

    frame = link_frame(body)
    index = int(body.removeprefix("link"))
    own = for_joint(f"joint{index}")
    own_kind, child_kind = end_kinds(body)

    near = _build_end(
        own_kind,
        end_direction(body, index, own_kind),
        motor_plane(body, index),
        own,
        flange_length,
        housing_length,
        flange_diameter,
    )

    far = None
    carried = None
    if child_kind is not None:
        child_index = int(frame.child_name.removeprefix("link"))
        carried = for_joint(f"joint{child_index}")
        far = _build_end(
            child_kind,
            end_direction(body, child_index, child_kind),
            motor_plane(body, child_index),
            carried,
            flange_length,
            housing_length,
            flange_diameter,
        )
    return near, far, own, carried


def _blend_junctions(shape, radius: float):
    """Fillet the tube-to-barrel intersection curves, largest radius that takes.

    Those curves are BSPLINEs, not circles. OCCT refuses a fillet that would
    run past the end of a face, and how much room there is depends on the tube
    diameter and its rake, so the radius is tried and backed off rather than
    assumed. Returns the shape unchanged if none of them take -- a missing
    blend is cosmetic, a failed build is not.
    """
    from build123d import fillet

    junctions = shape.edges().filter_by(
        lambda e: e.geom_type.name in ("BSPLINE", "ELLIPSE")
    )
    if not junctions:
        return shape
    attempt = float(radius)
    while attempt >= 1.0:
        try:
            return fillet(junctions, radius=attempt)
        except Exception:  # noqa: BLE001 - see docstring
            attempt -= 1.0
    return shape


def build_ur_link(
    body: str,
    flange_length: float,
    tube_stations: tuple[float, ...],
    tube_diameters: tuple[float, ...],
    housing_length: float | None = None,
    wall: float | None = None,
    tube_end_offset: float = 0.0,
    tube_offsets: tuple[float, ...] | None = None,
    flange_diameter: float | None = None,
    tube_waypoints: tuple[tuple[float, float, float], ...] | None = None,
):
    """Assemble one link: rotor flange, tube, stator housing."""
    from robotic_arm.assembly import DRIVER_DIAMETER
    from robotic_arm.linkframes import link_frame
    from robotic_arm.parts.cobot import (
        _oriented_cylinder,
        bolt_ring,
        break_edges,
        lofted_tube,
    )

    wall = RULES.structural_wall_thickness if wall is None else wall
    near, far, own, carried = link_ends(
        body, flange_length, housing_length, flange_diameter
    )
    if far is None:
        return None
    own_kind, child_kind = end_kinds(body)

    n_axis = np.asarray(near.axis, float) / np.linalg.norm(near.axis)
    f_axis = np.asarray(far.axis, float) / np.linalg.norm(far.axis)
    start = np.asarray(near.centre, float)
    finish = np.asarray(far.centre, float) + f_axis * tube_end_offset
    points = [start + (finish - start) * t for t in tube_stations]

    # Bow the tube off that straight line, one offset per station along the
    # far barrel's axis. Without this the path is always the chord between the
    # two barrel centres, so the only way to clear a neighbour is to make the
    # tube thinner -- which is the opposite of what a link joining two barrels
    # at right angles needs. link4 has to arc over link5's flange rim; a
    # straight chord either clips it or has to shrink to a stick.
    # An explicit path, in the link's own frame, replacing the chord between
    # the barrel centres. A straight chord can only meet both barrels square
    # when it runs along their common perpendicular; where the joints are
    # offset in all three axes it cannot, and the tube leaves and arrives at a
    # rake. Naming the corners instead lets a link be an elbow -- each
    # segment perpendicular to the barrel it meets, which is what "no weird
    # angles" actually requires.
    if tube_waypoints is not None:
        if len(tube_waypoints) != len(tube_diameters):
            raise ValueError(
                f"{body}: {len(tube_waypoints)} waypoints for "
                f"{len(tube_diameters)} diameters"
            )
        points = [np.asarray(w, dtype=float) for w in tube_waypoints]
    elif tube_offsets is not None:
        if len(tube_offsets) != len(tube_stations):
            raise ValueError(
                f"{body}: {len(tube_offsets)} tube offsets for "
                f"{len(tube_stations)} stations"
            )
        points = [p + f_axis * o for p, o in zip(points, tube_offsets)]

    run = finish - start
    run = run / np.linalg.norm(run)

    ends = (
        (own_kind, near, own, n_axis, tube_diameters[0], "own"),
        (child_kind, far, carried, f_axis, tube_diameters[-1], "child"),
    )

    # Do the barrels already meet? If so the link is just the two of them
    # fused, and every tube check below is moot.
    barrels = near.solid() + far.solid()
    joined = len(barrels.solids()) == 1

    for kind, drum, actuator, axis, diameter, label in ends:
        if joined:
            break
        if diameter > drum.diameter - wall:
            raise ValueError(
                f"{body}: tube is O{diameter:.0f} at the {label} end, which "
                f"is only O{drum.diameter:.0f}; its cavity would cut the wall"
            )
        # A tube meeting a barrel across its axis needs the barrel at least as
        # long as the tube is wide, or the tube's end section pokes out past
        # the barrel and is left behind as a loose sliver.
        span = diameter * float(np.sqrt(max(0.0, 1.0 - (run @ axis) ** 2)))
        if span > drum.length - 4 * wall:
            raise ValueError(
                f"{body}: a O{diameter:.0f} tube meets the {label} end across "
                f"{span:.0f} mm of a {drum.length:.0f} mm barrel; lengthen the "
                f"barrel or narrow the tube"
            )

    # The tube's end station must also stay inside the barrel it aims into.
    # Checking only the crossing span missed `tube_end_offset` walking the
    # tube clean out of the far cap -- 11.4 mm of stub on link4, with its
    # cavity punching an open O24 hole into the link.
    along_last = float((points[-1] - np.asarray(far.centre, float)) @ f_axis)
    reach = abs(along_last) + tube_diameters[-1] * float(
        np.sqrt(max(0.0, 1.0 - (run @ f_axis) ** 2))
    ) / 2
    if not joined and reach > far.length / 2 - wall:
        raise ValueError(
            f"{body}: the tube reaches {reach:.1f} mm from the far barrel's "
            f"centre but only {far.length / 2 - wall:.1f} mm is available; "
            f"reduce tube_end_offset ({tube_end_offset:.0f})"
        )

    # Only run a tube when the barrels do not already meet. Where they
    # intersect, a UR-style link is simply the two barrels fused with a fillet
    # -- a sideways T at the shoulder, an elbow at the wrist -- and a tube
    # between them is surplus geometry. link5's barrels overlap by 35 mm, so
    # its tube was always buried and invisible; link1's are 1.3 mm apart, and
    # its tube had to be bowed 40 mm sideways to dodge link2, which put a
    # visible C-shaped loop on the shoulder joining two things already touching.
    if joined:
        outer = barrels
    elif tube_waypoints is not None:
        # Named waypoints mean an **elbow**: straight runs meeting at corners.
        # Lofting through them instead sweeps one smooth surface and the part
        # comes out a banana -- which is what happened the first time link4
        # was given corners. A UR link is dead straight between joints and
        # bends only at them.
        from robotic_arm.parts.cobot import tube as straight_tube

        from build123d import Pos, Sphere

        run = barrels
        for a, b, da, db in zip(
            points, points[1:], tube_diameters, tube_diameters[1:]
        ):
            run = run + straight_tube(a, b, max(da, db))
        # Fill each corner. Two cylinders meeting at a waypoint each end in a
        # flat face there, so the outside of the bend is left open -- a notch
        # with the bore showing. A ball of the same diameter closes it, which
        # is how a cast elbow is actually shaped.
        for corner, diameter in zip(points[1:-1], tube_diameters[1:-1]):
            run = run + Pos(*corner) * Sphere(radius=diameter / 2)
        outer = run
    else:
        outer = barrels + lofted_tube(points, list(tube_diameters))

    # Blend where the tube meets each barrel. This has to happen on the solid
    # union, before the cavities are cut: afterwards the junction curve is
    # interrupted by the shell openings and OCCT refuses it.
    outer = _blend_junctions(outer, STYLE.shoulder_fillet)

    # Cavities, one per end. A flange's mating cap is thicker than a wall by
    # exactly the relief cut into it below, so what survives the relief is
    # still a full wall; at plain `wall` a 2.2 mm relief cut through a 2.0 mm
    # cap and freed the rim as a second solid.
    if joined:
        inner = None
    elif tube_waypoints is not None:
        from robotic_arm.parts.cobot import tube as straight_tube

        from build123d import Pos, Sphere

        inner = None
        for a, b, da, db in zip(
            points, points[1:], tube_diameters, tube_diameters[1:]
        ):
            bore = straight_tube(a, b, max(da, db) - 2 * wall)
            inner = bore if inner is None else inner + bore
        for corner, diameter in zip(points[1:-1], tube_diameters[1:-1]):
            ball = Pos(*corner) * Sphere(radius=diameter / 2 - wall)
            inner = ball if inner is None else inner + ball
    else:
        inner = lofted_tube(points, [d - 2 * wall for d in tube_diameters])
    for kind, drum, actuator, axis, _, _ in ends:
        if kind == "housing":
            cavity = drum.solid(drum.diameter - 2 * wall, drum.length - 2 * wall)
            inner = cavity if inner is None else inner + cavity
            continue
        else:
            relief = flange_relief_depth(actuator, wall)
            cavity = _oriented_cylinder(
                np.asarray(drum.centre, float) + axis * relief / 2,
                axis,
                (drum.diameter - 2 * wall) / 2,
                drum.length - 2 * wall - relief,
            )
            inner = cavity if inner is None else inner + cavity
    part = outer - inner

    driver = DRIVER_DIAMETER["M3"]
    for kind, drum, actuator, axis, _, _ in ends:
        # The joint-plane end of this barrel, whichever kind it is.
        face = np.asarray(drum.centre, float) - axis * drum.length / 2
        if kind == "housing":
            # Opening for the joint to work through, and the ring its stator
            # bolts into. Driven from the open mouth before the next link is
            # offered up, so it needs no access port -- that is the order a
            # cobot is assembled in.
            part -= _oriented_cylinder(
                face, axis, housing_bore(actuator) / 2, 4 * wall
            )
            part -= bolt_ring(
                centre=face + axis * wall / 2,
                axis=axis,
                bcd=actuator.stator_circle.bcd,
                count=actuator.stator_circle.count,
                hole_diameter=RULES.m3_clearance,
                depth=wall * 3,
            )
        else:
            # Relieve outside the hub this flange bears on, so it clears the
            # housing cap reaching onto its side of the stator face.
            relief = flange_relief_depth(actuator, wall)
            if relief > 0.0:
                part -= (
                    _oriented_cylinder(
                        face + axis * relief / 2, axis,
                        (drum.diameter + 2.0) / 2, relief,
                    )
                    - _oriented_cylinder(
                        face + axis * relief / 2, axis,
                        actuator.hub_diameter / 2, relief + 2.0,
                    )
                )
            part -= bolt_ring(
                centre=face + axis * wall / 2,
                axis=axis,
                bcd=actuator.rotor_circle.bcd,
                count=actuator.rotor_circle.count,
                hole_diameter=RULES.m3_clearance,
                depth=wall * 3,
            )
            # Service ports through the far cap so a driver can still reach
            # those heads now that the barrel is closed.
            part -= bolt_ring(
                centre=np.asarray(drum.centre, float)
                + axis * (drum.length / 2 - wall / 2),
                axis=axis,
                bcd=actuator.rotor_circle.bcd,
                count=actuator.rotor_circle.count,
                hole_diameter=driver,
                depth=wall * 3,
            )

    # Clear the next link's barrel. Where the child holds the shared joint's
    # housing, this link's far end is a flange and that housing is coaxial
    # with the joint -- so its swept volume is the cylinder itself, the same
    # at every angle, and relieving for it is exact rather than a fudge.
    #
    # link1 needs it: its own barrel is O94 about the J1 axis and reaches
    # 47 mm in +y, which is where link2's barrel sits once the shoulder
    # closes into a T.
    if child_kind == "flange" and carried is not None:
        clearance = STYLE.joint_gap
        child_barrel = stator_housing(
            -f_axis,
            np.asarray(link_frame(body).child_origin, dtype=float),
            carried,
            default_housing_length(carried),
        )
        part -= _oriented_cylinder(
            np.asarray(child_barrel.centre, float),
            np.asarray(child_barrel.axis, float),
            child_barrel.diameter / 2 + clearance,
            child_barrel.length + 2 * clearance,
        )

    part = break_edges(part)
    if len(part.solids()) != 1:
        sizes = sorted((float(s.volume) for s in part.solids()), reverse=True)
        raise ValueError(
            f"{body}: built {len(part.solids())} solids "
            f"({', '.join(f'{v:,.0f}' for v in sizes)} mm^3); a link is one "
            f"printed piece"
        )
    return part




def housed_actuators(body: str) -> list:
    """Every vendor actuator this body encloses, placed as this design mounts it.

    A body holds a motor wherever it holds that joint's **stator**, which can
    be its own joint, its child's, or both -- link2 carries M2 and M3. The
    earlier version returned only the child's, so M1 was drawn on base_link
    even after link1 took it over, and it ignored `MOTOR_PLANE_OFFSET`, so it
    appeared 35 mm above where it actually sits: a black band sticking out of
    the shoulder barrel that looked like a modelling error in the part.

    The stock `motor_*` meshes are no use here. They are motor *assemblies* --
    `motor_2_3` holds two -- reaching 59 to 90 mm from their joint planes and
    at least 50 mm in radius, where an RS00 is O57 and 51 mm deep. Drawn
    against housings sized from the vendor STEP they look like motors bursting
    out of their covers.
    """
    from build123d import Location, Plane, Vector, import_step

    from robotic_arm.actuators import for_joint
    from robotic_arm.linkframes import link_frame
    from robotic_arm.reference import RS00_STEP, RS06_STEP, require

    frame = link_frame(body)
    held = []
    if body.startswith("link"):
        index = int(body.removeprefix("link"))
        if carries_own_housing(body):
            held.append(index)
    child = frame.child_name
    if child and child.startswith("link"):
        child_index = int(child.removeprefix("link"))
        if child_index <= 6 and not carries_own_housing(child):
            held.append(child_index)

    out = []
    for joint_index in held:
        actuator = for_joint(f"joint{joint_index}")
        step = RS06_STEP if actuator.name == "RS06" else RS00_STEP
        solid = import_step(str(require(step)))
        hub_face = solid.bounding_box().min.Z
        axis = end_direction(body, joint_index, "housing")
        plane = motor_plane(body, joint_index)
        placed = Plane(origin=Vector(*plane), z_dir=Vector(*axis))
        out.append(placed * Location((0.0, 0.0, -hub_face)) * solid)
    return out


def housed_actuator(body: str):
    """The first actuator this body encloses, or None."""
    held = housed_actuators(body)
    return held[0] if held else None
