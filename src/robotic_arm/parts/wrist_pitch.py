"""link4 -- the forearm-to-wrist link, between J4 and J5.

The longest of the three wrist parts: J5 sits 104 mm from J4, so unlike link5
this one has room for a visible tube between its drums. That waist is what
makes the drums read as joints rather than as one continuous lump, and it is
the most recognisable feature of the form language.

J4 and J5 cross at a right angle, so again a drum on each axis.

Stock is 460 g across a 156 x 131 x 87 mm envelope, most of it the enclosed
J5 actuator rather than structure.
"""

from __future__ import annotations

import numpy as np
from build123d import Part

from robotic_arm.actuators import RS00
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
    mating_face_opening,
    mount_face_ring,
    boss_mount_face,
    break_edges,
    mount_boss_diameter,
    seam_groove,
    shelled_body,
)

MATERIAL = PC_CF
BODY = "link4"

#: J4 drum. A mounting boss, not a housing: the motor that drives J4 sits on
#: link3, so nothing needs enclosing here. Two earlier revisions got this
#: wrong -- O76 "because J4 carries the wrist", then O67 "to match the
#: actuator" -- and the clearance sweep caught both fouling link2, because a
#: full drum of that size fills a quadrant the stock arm leaves open.
PARENT_LENGTH = 48.0

#: How far the J4 drum may stand proud of the joint plane, towards link3.
#: Stock reaches z = +12.5 mm here. A drum centred on the joint plane would
#: reach +27 and foul link2 and the base through part of the workspace -- the
#: clearance sweep caught exactly that.
PARENT_PROTRUSION = -STYLE.joint_gap / 2

#: J5 drum, enclosing the actuator that drives link5.
#: Boss diameter, set for proportion rather than strength: it keeps the
#: wrist reading as a continuous stack instead of drums on thin necks.
#: Swept against clearance -- O54 is free, O60 costs 9 poses in 6000.
BOSS_DIAMETER = 54.0

CHILD_CLEARANCE = 3.0
CHILD_LENGTH = 56.0

#: The waist. Slim enough to read as a tube between two joints, thick enough
#: to carry the bending load of the wrist hanging off the end.
TUBE_DIAMETER = 52.0

CABLE_CHANNEL_DIAMETER = 12.0


def child_diameter() -> float:
    rs00 = RS00()
    body_diameter = max(rs00.bbox_mm[0], rs00.bbox_mm[1]) if rs00.bbox_mm else 57.0
    return body_diameter + 2 * (RULES.structural_wall_thickness + CHILD_CLEARANCE)


def _drums() -> tuple[Drum, Drum]:
    """The two joint drums. Shared by the solid and its collision proxy so the
    two cannot drift apart."""
    frame = link_frame(BODY)
    mount = RS00().output_circle
    parent = Drum(
        centre=boss_centre(frame, PARENT_LENGTH, PARENT_PROTRUSION),
        axis=frame.parent_axis,
        diameter=mount_boss_diameter(
            mount.bcd, RULES.m3_clearance, floor=BOSS_DIAMETER
        ),
        length=PARENT_LENGTH,
    )
    # Pull the child drum back along its own axis so its far face lands on the
    # J5 origin, which is where link5 mounts.
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


def collision_primitives() -> list[CollisionCylinder]:
    """Collision proxy: one cylinder per drum, plus the connecting tube."""
    parent, child = _drums()
    return [
        CollisionCylinder.from_drum(parent),
        CollisionCylinder.from_drum(child),
        CollisionCylinder.from_span(parent.centre, child.centre, TUBE_DIAMETER),
    ]


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


def build_wrist_pitch() -> Part:
    """Return the forearm-to-wrist link as a solid, in link4's frame."""
    frame = link_frame(BODY)
    parent, child = _drums()

    part = shelled_body(parent, child, TUBE_DIAMETER)

    # No central bore: neither actuator is a hollow-shaft motor, so a
    # through-bore here would imply a cable route that does not exist --
    # and on link2 and link3 it cut into the mounting bolt circle.
    # The harness runs beside the actuator instead.
    part -= cable_channel(
        parent.centre + _channel_offset(parent),
        child.centre + _channel_offset(child),
        CABLE_CHANNEL_DIAMETER,
    )

    mount = RS00().output_circle

    # Onto the J4 actuator output.
    part -= bolt_ring(
        centre=np.array([0.0, 0.0, -PARENT_LENGTH / 2]),
        axis=parent.axis,
        bcd=mount.bcd,
        count=mount.count,
        hole_diameter=RULES.m3_clearance,
        depth=40.0,
    )
    # The J5 actuator bolts into the far face of the child drum.
    part -= bolt_ring(
        centre=frame.child_origin,
        axis=frame.child_axis,
        bcd=mount.bcd,
        count=mount.count,
        hole_diameter=RULES.m3_clearance,
        depth=30.0,
    )

    # Driver access to the parent mount. Without ports two of the six bolts
    # cannot be reached from either side.
    from robotic_arm.assembly import DRIVER_DIAMETER

    part -= access_ports(
        parent,
        boss_mount_face(frame, PARENT_LENGTH, PARENT_PROTRUSION),
        mount,
        DRIVER_DIAMETER["M3"],
        depth=PARENT_LENGTH * 2,
    )

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

    part = build_wrist_pitch()
    props = mass_properties(part, effective_material(part, MATERIAL))
    bb = part.bounding_box()
    print(f"volume {part.volume:,.0f} mm^3")
    print(f"mass   {props.mass * 1000:.1f} g  (stock {stock_mass(BODY) * 1000:.0f} g)")
    print(f"extent {[round(v, 1) for v in (bb.size.X, bb.size.Y, bb.size.Z)]} mm")
    print(f"stock  {link_frame(BODY).stock_extent.round(1)} mm")
