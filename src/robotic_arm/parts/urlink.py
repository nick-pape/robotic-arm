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


def stator_housing(axis, joint_plane, actuator, length: float) -> Drum:
    """The cylinder that encloses a motor, open at the joint plane.

    `axis` points from the joint plane into the body of the link that holds
    it -- the motor lies on that side. The mouth is recessed by the hub
    protrusion plus a seam gap, so the flange capping it never touches.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    mouth = np.asarray(joint_plane, dtype=float) + axis * (
        actuator.hub_protrusion + SEAM_GAP
    )
    return Drum(
        centre=mouth + axis * length / 2,
        axis=axis,
        diameter=housing_diameter(actuator),
        length=length,
    )


def rotor_flange(axis, joint_plane, actuator, length: float) -> Drum:
    """The disc that caps the parent's housing and turns with the rotor.

    Its outside diameter matches that housing exactly, so the joint reads as
    one continuous cylinder with a fine seam -- the detail that makes a cobot
    look like a cobot rather than like a bracket next to a motor.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    plane = np.asarray(joint_plane, dtype=float)
    return Drum(
        centre=plane + axis * length / 2,
        axis=axis,
        diameter=housing_diameter(actuator),
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
    reach = []
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
        # One mesh can hold two motors (motor_2_3 does), so select the slug
        # actually centred on this axis rather than the whole mesh.
        near = (radial < 0.050) & (np.abs(along) < 0.080)
        if near.sum() > 50:
            reach.append(float(np.mean(along[near])))
    if not reach:
        raise KeyError(f"no motor mesh found on {joint}")
    return axis * (1.0 if max(reach, key=abs) >= 0 else -1.0)


def designed_motor_side(joint: str) -> np.ndarray:
    """Where this design puts the motor, which is not always where stock does.

    The UR convention is uniform: at every joint the **parent** holds the
    stator and the child's flange bolts to the rotor, so the motor always sits
    on the parent's side. The stock arm follows that everywhere except J2,
    where link1 takes the rotor and link2 carries the stators of both J2 and
    J3 (see `robotic_arm.mounts`). Reproducing that here would send link1's
    housing forward over link2 instead of back over its own motor.

    So J2's motor is deliberately re-mounted the other way round. That is a
    real assembly difference from stock, not a modelling convenience -- it
    changes which part the J2 housing belongs to. It does not touch the joint
    frame, so reach and segment lengths are unaffected.
    """
    from robotic_arm.mounts import joint_roles

    side = motor_side(joint)
    roles = joint_roles().get(joint, {})
    # If stock puts the *parent* on the rotor, the joint is inverted and the
    # motor sits on the child's side; flip it back to the parent's.
    import mujoco

    from robotic_arm.reference import load_baseline

    model = load_baseline()
    child = int(model.jnt_bodyid[model.joint(joint).id])
    parent = model.body(int(model.body_parentid[child])).name
    return -side if roles.get(parent) == "rotor" else side


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
    return _in_frame(body, -designed_motor_side(f"joint{index}"))


def housing_direction(body: str, child_index: int) -> np.ndarray:
    """Which way a link's stator housing reaches, in the link's own frame."""
    return _in_frame(body, designed_motor_side(f"joint{child_index}"))


def link_ends(body: str, flange_length: float, housing_length: float | None = None):
    """(flange, housing, own actuator, carried actuator) for one link."""
    from robotic_arm.actuators import for_joint
    from robotic_arm.linkframes import link_frame

    frame = link_frame(body)
    index = int(body.removeprefix("link"))
    own = for_joint(f"joint{index}")

    flange = rotor_flange(
        flange_direction(body), np.zeros(3), own, flange_length
    )

    housing = None
    carried = None
    if frame.child_name and frame.child_name.startswith("link"):
        child_index = int(frame.child_name.removeprefix("link"))
        if child_index <= 6:
            carried = for_joint(f"joint{child_index}")
            # The motor lies on this link's side of the child joint plane.
            into = housing_direction(body, child_index)
            length = (
                default_housing_length(carried)
                if housing_length is None
                else housing_length
            )
            housing = stator_housing(into, frame.child_origin, carried, length)
    return flange, housing, own, carried


def build_ur_link(
    body: str,
    flange_length: float,
    tube_stations: tuple[float, ...],
    tube_diameters: tuple[float, ...],
    housing_length: float | None = None,
    wall: float | None = None,
    tube_end_offset: float = 0.0,
):
    """Assemble one link: rotor flange, tube, stator housing."""
    from robotic_arm.assembly import DRIVER_DIAMETER
    from robotic_arm.parts.cobot import (
        _oriented_cylinder,
        bolt_ring,
        break_edges,
        lofted_tube,
    )

    wall = RULES.structural_wall_thickness if wall is None else wall
    flange, housing, own, carried = link_ends(body, flange_length, housing_length)
    if housing is None:
        return None

    f_axis = np.asarray(flange.axis, float) / np.linalg.norm(flange.axis)
    h_axis = np.asarray(housing.axis, float) / np.linalg.norm(housing.axis)
    start = np.asarray(flange.centre, float)
    end = np.asarray(housing.centre, float) + h_axis * tube_end_offset
    points = [start + (end - start) * f for f in tube_stations]

    for label, drum, diameter, axis in (
        ("flange", flange, tube_diameters[0], f_axis),
        ("housing", housing, tube_diameters[-1], h_axis),
    ):
        if diameter > drum.diameter - wall:
            raise ValueError(
                f"{body}: tube is O{diameter:.0f} at the {label}, which is "
                f"only O{drum.diameter:.0f}; its cavity would cut the wall"
            )
        # A tube meeting a barrel across its axis needs the barrel to be at
        # least as long as the tube is wide, or the tube's end section pokes
        # out past the barrel and is left behind as a loose sliver. The same
        # mistake, in the same shape, as a fat tube in a short housing.
        run = end - start
        run = run / np.linalg.norm(run)
        span = diameter * float(np.sqrt(max(0.0, 1.0 - (run @ axis) ** 2)))
        if span > drum.length:
            raise ValueError(
                f"{body}: a O{diameter:.0f} tube meets the {label} across "
                f"{span:.0f} mm of a {drum.length:.0f} mm barrel; lengthen "
                f"the {label} or narrow the tube"
            )

    outer = flange.solid() + housing.solid() + lofted_tube(points, list(tube_diameters))

    # Cavities. The flange keeps only its joint-plane cap -- the far end opens
    # into the link so a driver can reach the rotor bolts -- while the housing
    # keeps both, the mouth cap being the seat its stator bolts thread into.
    flange_cavity = _oriented_cylinder(
        start + f_axis * wall, f_axis, (flange.diameter - 2 * wall) / 2,
        flange.length,
    )
    inner = (
        flange_cavity
        + housing.solid(housing.diameter - 2 * wall, housing.length - 2 * wall)
        + lofted_tube(points, [d - 2 * wall for d in tube_diameters])
    )
    part = outer - inner

    # The joint opening, through the housing's mouth cap.
    mouth = np.asarray(housing.centre, float) - h_axis * housing.length / 2
    part -= _oriented_cylinder(
        mouth, h_axis, housing_bore(carried) / 2, 4 * wall
    )

    # Stator ring, into the mouth cap. Driven from the open mouth before the
    # next link is offered up, so it needs no access port at all -- that is
    # the assembly order a cobot is built in.
    part -= bolt_ring(
        centre=mouth + h_axis * wall / 2,
        axis=h_axis,
        bcd=carried.stator_circle.bcd,
        count=carried.stator_circle.count,
        hole_diameter=RULES.m3_clearance,
        depth=wall * 3,
    )
    # Rotor ring, through the flange's cap; heads inside the link.
    part -= bolt_ring(
        centre=start - f_axis * flange.length / 2 + f_axis * wall / 2,
        axis=f_axis,
        bcd=own.rotor_circle.bcd,
        count=own.rotor_circle.count,
        hole_diameter=RULES.m3_clearance,
        depth=wall * 3,
    )

    # No decorative seam groove. The joint itself now leaves a real gap
    # between the flange and the housing it caps, which is the seam a cobot
    # actually shows -- and a 2 mm groove cut in a 2 mm barrel wall goes
    # straight through it, freeing a thin rim as a second solid.

    part = break_edges(part)
    if len(part.solids()) != 1:
        sizes = sorted((float(s.volume) for s in part.solids()), reverse=True)
        raise ValueError(
            f"{body}: built {len(part.solids())} solids "
            f"({', '.join(f'{v:,.0f}' for v in sizes)} mm^3); a link is one "
            f"printed piece"
        )
    return part
