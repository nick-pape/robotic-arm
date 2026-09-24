"""link3 -- the forearm, between J3 and J4.

The first long link, and the first with **parallel** joint axes: J3 and J4 both
turn about Z, so this is not a two-perpendicular-drums wrist but a straight
tube with a drum at each end. That is the elbow-to-wrist link on any cobot, and
the shape falls out of the kinematics rather than being imposed on it.

The tube tapers. Stock narrows to about O44 at midspan and flares at both ends,
which is where the bending moment is lowest, and a printed part has the same
reason to do it. Measuring that profile first also keeps the part inside the
space the stock arm occupies, which is what stops it colliding where stock does
not.

link3 carries `motor_4`, the RS00 that drives the wrist, so its child drum is a
housing. Its own parent boss is not: `motor_3` sits on link2.

Stock is 1252 g across a 294 x 129 x 87 mm envelope.
"""

from __future__ import annotations

import numpy as np
from build123d import Part

from robotic_arm.actuators import RS00, RS06
from robotic_arm.design import RULES, STYLE
from robotic_arm.linkframes import actuator_centre, link_frame
from robotic_arm.materials import PC_CF
from robotic_arm.parts.cobot import (
    CollisionCylinder,
    Drum,
    access_ports,
    bolt_ring,
    cable_channel,
    boss_centre,
    housing_over,
    boss_mount_face,
    break_edges,
    lofted_tube,
    mating_face_opening,
    mount_face_ring,
    solid_mount_end,
    output_interface,
    mount_boss_diameter,
    seam_groove,
    shell,
)

MATERIAL = PC_CF
BODY = "link3"

#: J3 boss. Bolts to the RS06 output; stock leaves room for a full drum of
#: about O40 here, so the mount boss fits with margin.
PARENT_LENGTH = 46.0
PARENT_PROTRUSION = -STYLE.joint_gap / 2

#: J4 drum, housing the RS00 that drives the wrist.
CHILD_CLEARANCE = 3.0
CHILD_LENGTH = 54.0

#: Tube profile, as fractions along the span and diameters at those stations.
#: Held under the measured stock cross-section everywhere: O71 available at 20%
#: along, O53 at 35%, O44 at midspan, O48 at 80%.
#:
#: The first station sits at the origin, inside the parent boss, on purpose.
#: Starting it partway out left the tube merely touching the boss tangentially,
#: which OCCT will not fuse -- the part came out as two disconnected solids.
#:
#: The taper is shallow and, past midspan, monotonically outward: the tube
#: grows into the O67 wrist housing rather than pinching to O46 and jumping.
#: An arm should not get thinner than the section that follows it.
#:
#: Two earlier profiles were wrong. One flared to O52 a fifth of the way along
#: and read as a bone rather than a forearm. The next held O46 to the end and
#: left the forearm visibly slimmer than the wrist hanging off it.
#: Elbow boss. Swept against clearance with link2 in place: O64 still costs
#: nothing, so O60 leaves margin. The bolt circle alone would give O36, but
#: that met link2's O92 housing as a cliff rather than a step.
BOSS_DIAMETER = 60.0

#:
#: The first station is sized to clear the actuator, not for looks. A tube of
#: O56 centred on the boss axis reaches 4 mm past the mount face on its
#: upper surface -- straight into the motor's stator flange. Checked against
#: RobStride's own STEP, O42 is the largest that clears.
TUBE_STATIONS = (0.0, 0.5, 0.8, 0.95)
TUBE_DIAMETERS = (42.0, 46.0, 50.0, 60.0)

CABLE_CHANNEL_DIAMETER = 12.0


def child_diameter() -> float:
    rs00 = RS00()
    body_diameter = max(rs00.bbox_mm[0], rs00.bbox_mm[1]) if rs00.bbox_mm else 57.0
    return body_diameter + 2 * (RULES.structural_wall_thickness + CHILD_CLEARANCE)


def _mount_circles():
    """(J3 output circle on the RS06, J4 mount circle on the RS00)."""
    return RS06().output_circle, RS00().output_circle


def _drums() -> tuple[Drum, Drum]:
    """Parent boss and child housing, shared by the solid and its collision proxy."""
    frame = link_frame(BODY)
    parent_circle, _ = _mount_circles()

    parent = Drum(
        centre=boss_centre(frame, PARENT_LENGTH, PARENT_PROTRUSION),
        axis=frame.parent_axis,
        diameter=mount_boss_diameter(
            parent_circle.bcd, RULES.m3_clearance, floor=BOSS_DIAMETER
        ),
        length=PARENT_LENGTH,
    )
    # The housing encloses the actuator *body*, not its full installed
    # envelope. The motor meshes measure about 82 mm across -- body plus
    # connectors and output boss -- and sizing drums to that gave O92 housings
    # that exceeded the stock envelope and produced 1676 clearance
    # regressions. Stock does not enclose them either: its links are open
    # brackets with the motors visibly exposed, which is why they show in the
    # renders. A fully closed wrist would need more room than this arm has.
    # Centre the housing on the motor it encloses, not on the joint origin:
    # the motor sits 26-30 mm off that plane on these links, and placing the
    # drum at the joint left it visibly beside the motor rather than round it.
    motor_at = actuator_centre(BODY, near=frame.child_origin)
    if motor_at is not None:
        placed = housing_over(frame, motor_at, CHILD_LENGTH)
        child = Drum(
            centre=placed.centre,
            axis=placed.axis,
            diameter=child_diameter(),
            length=placed.length,
        )
    else:
        child = Drum(
            centre=frame.child_origin,
            axis=frame.child_axis,
            diameter=child_diameter(),
            length=CHILD_LENGTH,
        )
    return parent, child


def _tube_path() -> tuple[list[np.ndarray], list[float]]:
    """Stations along the tube, and its diameter at each.

    The tube runs between the two drum *centres*, not between the joint
    origins. Those differ: the motors sit 26 mm off the joint plane, so a tube
    drawn between origins floats above the housing it is supposed to meet --
    on link2 that made the part 87 mm tall against a 66 mm stock envelope.
    """
    parent, child = _drums()
    start = np.asarray(parent.centre, dtype=float)
    end = np.asarray(child.centre, dtype=float)
    return (
        [start + (end - start) * fraction for fraction in TUBE_STATIONS],
        list(TUBE_DIAMETERS),
    )


def collision_primitives() -> list[CollisionCylinder]:
    """Collision proxy: the two drums plus two cylinders tracing the taper.

    The taper is approximated by two straight cylinders at the *larger* of each
    pair of diameters, which is conservative -- it can only over-report a
    collision, never miss one.
    """
    parent, child = _drums()
    points, diameters = _tube_path()
    proxies = [CollisionCylinder.from_drum(parent), CollisionCylinder.from_drum(child)]
    for start, end, d_start, d_end in zip(
        points, points[1:], diameters, diameters[1:]
    ):
        proxies.append(
            CollisionCylinder.from_span(start, end, max(d_start, d_end))
        )
    return proxies


def _channel_offset(drum: Drum) -> np.ndarray:
    """Radial offset putting the cable channel beside a drum -- clear of its
    bolt circle and inside its outer wall."""
    axis = np.asarray(drum.axis, dtype=float)
    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(seed @ axis)) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    radial = np.cross(axis, seed)
    radial /= np.linalg.norm(radial)
    return radial * (
        drum.diameter / 2
        - RULES.structural_wall_thickness
        - CABLE_CHANNEL_DIAMETER / 2
        - 1.0
    )


def build_forearm() -> Part:
    """Return the forearm as a solid, in link3's frame."""
    frame = link_frame(BODY)
    parent, child = _drums()
    points, diameters = _tube_path()
    wall = RULES.structural_wall_thickness

    outer = parent.solid() + child.solid() + lofted_tube(points, diameters)
    inner = (
        parent.solid(parent.diameter - 2 * wall, parent.length - 2 * wall)
        + child.solid(child.diameter - 2 * wall, child.length - 2 * wall)
        + lofted_tube(points, [d - 2 * wall for d in diameters])
    )
    # Keep the cavity clear of the mount end so its cap survives the relief.
    inner -= solid_mount_end(
        parent,
        boss_mount_face(frame, PARENT_LENGTH, PARENT_PROTRUSION),
        RS06().hub_protrusion + STYLE.joint_gap + RULES.structural_wall_thickness,
    )
    part = shell(outer, inner)

    # No central bore: neither actuator is a hollow-shaft motor, so a
    # through-bore here would imply a cable route that does not exist --
    # and on link2 and link3 it cut into the mounting bolt circle.
    # The harness runs beside the actuator instead.
    part -= cable_channel(
        parent.centre + _channel_offset(parent),
        child.centre + _channel_offset(child),
        CABLE_CHANNEL_DIAMETER,
    )

    parent_circle, child_circle = _mount_circles()
    part -= mount_face_ring(
        parent, towards=-parent.axis * 1000.0, circle=parent_circle,
        hole_diameter=RULES.m3_clearance, depth=RULES.structural_wall_thickness * 3,
    )
    part -= mount_face_ring(
        child, towards=frame.child_origin, circle=child_circle,
        hole_diameter=RULES.m3_clearance, depth=RULES.structural_wall_thickness * 3,
    )

    # Driver access to the parent mount. The tube leaves the boss directly
    # over four of the six bolts, so a hex key reaches none of them from
    # either side; without ports the part cannot be fastened at all.
    from robotic_arm.assembly import DRIVER_DIAMETER

    part -= access_ports(
        parent,
        boss_mount_face(frame, PARENT_LENGTH, PARENT_PROTRUSION),
        parent_circle,
        DRIVER_DIAMETER["M3"],
        depth=PARENT_LENGTH * 2,
    )

    # Seat the mount face on the actuator's output hub, relieved clear of
    # the stator beside it. Without this the boss lands on a fixed face and
    # the joint binds.
    relief = output_interface(
        parent, boss_mount_face(frame, PARENT_LENGTH, PARENT_PROTRUSION), RS06()
    )
    if relief is not None:
        part -= relief

    # Open the mating face. The child link's boss enters here, as does the
    # actuator output; a shelled drum caps both ends, and the closed cap is
    # what the child boss was punching through.
    from robotic_arm.parts import child_interface_diameter

    opening = child_interface_diameter(BODY)
    if opening is not None:
        part -= mating_face_opening(
            child, towards=frame.child_origin, diameter=opening + 2 * STYLE.joint_gap
        )

    part -= seam_groove(parent, offset_along_axis=-parent.length / 2 + 5.0)
    part -= seam_groove(child, offset_along_axis=child.length / 2 - 5.0)

    return break_edges(part)


if __name__ == "__main__":
    from robotic_arm.linkframes import stock_mass
    from robotic_arm.massprops import mass_properties
    from robotic_arm.parts import effective_material

    part = build_forearm()
    props = mass_properties(part, effective_material(part, MATERIAL))
    bb = part.bounding_box()
    print(f"volume {part.volume:,.0f} mm^3")
    print(f"mass   {props.mass * 1000:.1f} g  (stock {stock_mass(BODY) * 1000:.0f} g)")
    print(f"extent {[round(v, 1) for v in (bb.size.X, bb.size.Y, bb.size.Z)]} mm")
    print(f"stock  {link_frame(BODY).stock_extent.round(1)} mm")
