"""link5 -- the wrist-roll housing, between J5 and J6.

Two perpendicular drums joined by a short tube, which is what a cobot wrist
looks like and is not a stylistic imposition: J5 and J6 genuinely cross at a
right angle (`linkframes` confirms it), so a drum on each axis is the honest
shape for the geometry.

The J5 drum houses this link's own bearing; the J6 drum encloses the RS00 that
drives the tool flange, which is why it is sized to that actuator's body rather
than to anything arbitrary.

Stock is 201 g across a 88 x 82 x 104 mm envelope, most of which is the
enclosed actuator rather than structure.
"""

from __future__ import annotations

import numpy as np
from build123d import Part

from robotic_arm.actuators import RS00
from robotic_arm.design import RULES
from robotic_arm.linkframes import link_frame
from robotic_arm.materials import PC_CF
from robotic_arm.parts.cobot import Drum, bolt_ring, break_edges, seam_groove, shelled_body

MATERIAL = PC_CF
BODY = "link5"

#: J5 drum. Diameter follows the stock envelope's narrow axis so the wrist
#: keeps its proportions; length spans the bearing and its seat.
PARENT_DIAMETER = 64.0
PARENT_LENGTH = 42.0

#: J6 drum, sized to enclose the RS00 that drives the tool flange. The
#: actuator body is O57, so the shell clears it by a wall plus a running gap.
CHILD_CLEARANCE = 3.0
CHILD_LENGTH = 52.0

#: Connecting tube, slimmer than both drums -- the waist is what makes the
#: drums read as joints rather than as one lumpy mass.
TUBE_DIAMETER = 46.0

#: Cable pass-through along the J5 axis.
BORE_DIAMETER = 20.0


def child_diameter() -> float:
    """Outer diameter of the J6 drum, from the actuator it has to enclose."""
    rs00 = RS00()
    body_diameter = max(rs00.bbox_mm[0], rs00.bbox_mm[1]) if rs00.bbox_mm else 57.0
    return body_diameter + 2 * (RULES.structural_wall_thickness + CHILD_CLEARANCE)


def build_wrist_roll() -> Part:
    """Return the wrist-roll housing as a solid, in link5's frame."""
    frame = link_frame(BODY)

    parent = Drum(
        centre=np.array([0.0, 0.0, PARENT_LENGTH / 2 - 8.0]),
        axis=frame.parent_axis,
        diameter=PARENT_DIAMETER,
        length=PARENT_LENGTH,
    )
    child = Drum(
        centre=frame.child_origin - frame.child_axis * (CHILD_LENGTH / 2 - 4.0),
        axis=frame.child_axis,
        diameter=child_diameter(),
        length=CHILD_LENGTH,
    )

    part = shelled_body(parent, child, TUBE_DIAMETER, tube_from=6.0)

    # Cable route along the J5 axis, and out through the J6 drum.
    part -= Drum(
        centre=parent.centre, axis=parent.axis, diameter=BORE_DIAMETER, length=200.0
    ).solid()

    # Mounting pattern onto the J5 actuator output, from measured geometry.
    mount = {round(c.bcd, 2): c for c in RS00().bolt_circles}[27.0]
    part -= bolt_ring(
        centre=np.array([0.0, 0.0, -8.0]),
        axis=parent.axis,
        bcd=mount.bcd,
        count=mount.count,
        hole_diameter=RULES.m3_clearance,
        depth=40.0,
    )

    # The J6 actuator bolts into the far face of the child drum.
    part -= bolt_ring(
        centre=frame.child_origin,
        axis=frame.child_axis,
        bcd=mount.bcd,
        count=mount.count,
        hole_diameter=RULES.m3_clearance,
        depth=30.0,
    )

    # Seams at both rotating interfaces.
    part -= seam_groove(parent, offset_along_axis=-PARENT_LENGTH / 2 + 5.0)
    part -= seam_groove(child, offset_along_axis=CHILD_LENGTH / 2 - 5.0)

    return break_edges(part)


if __name__ == "__main__":
    from robotic_arm.linkframes import stock_mass
    from robotic_arm.massprops import mass_properties
    from robotic_arm.parts import effective_material

    part = build_wrist_roll()
    props = mass_properties(part, effective_material(part, MATERIAL))
    bb = part.bounding_box()
    print(f"volume {part.volume:,.0f} mm^3")
    print(f"mass   {props.mass * 1000:.1f} g  (stock {stock_mass(BODY) * 1000:.0f} g)")
    print(f"extent {[round(v, 1) for v in (bb.size.X, bb.size.Y, bb.size.Z)]} mm")
    print(f"stock  {link_frame(BODY).stock_extent.round(1)} mm")
