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
from robotic_arm.design import RULES
from robotic_arm.linkframes import link_frame
from robotic_arm.materials import PC_CF
from robotic_arm.parts.cobot import (
    CollisionCylinder,
    Drum,
    bolt_ring,
    break_edges,
    lofted_tube,
    mount_boss_diameter,
    seam_groove,
    shell,
)

MATERIAL = PC_CF
BODY = "link3"

#: J3 boss. Bolts to the RS06 output; stock leaves room for a full drum of
#: about O40 here, so the mount boss fits with margin.
PARENT_LENGTH = 46.0
PARENT_PROTRUSION = 9.0

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
TUBE_STATIONS = (0.0, 0.5, 0.8, 0.95)
TUBE_DIAMETERS = (40.0, 42.0, 47.0, 60.0)

BORE_DIAMETER = 24.0


def child_diameter() -> float:
    rs00 = RS00()
    body_diameter = max(rs00.bbox_mm[0], rs00.bbox_mm[1]) if rs00.bbox_mm else 57.0
    return body_diameter + 2 * (RULES.structural_wall_thickness + CHILD_CLEARANCE)


def _mount_circles():
    """(J3 output circle on the RS06, J4 mount circle on the RS00)."""
    rs06 = {round(c.bcd, 2): c for c in RS06().bolt_circles}
    rs00 = {round(c.bcd, 2): c for c in RS00().bolt_circles}
    return rs06[24.02], rs00[27.0]


def _drums() -> tuple[Drum, Drum]:
    """Parent boss and child housing, shared by the solid and its collision proxy."""
    frame = link_frame(BODY)
    parent_circle, _ = _mount_circles()

    parent = Drum(
        centre=np.array([0.0, 0.0, PARENT_PROTRUSION - PARENT_LENGTH / 2]),
        axis=frame.parent_axis,
        diameter=mount_boss_diameter(parent_circle.bcd, RULES.m3_clearance),
        length=PARENT_LENGTH,
    )
    child = Drum(
        centre=frame.child_origin,
        axis=frame.child_axis,
        diameter=child_diameter(),
        length=CHILD_LENGTH,
    )
    return parent, child


def _tube_path() -> tuple[list[np.ndarray], list[float]]:
    """Stations along the span, and the tube diameter at each."""
    frame = link_frame(BODY)
    return (
        [frame.child_origin * fraction for fraction in TUBE_STATIONS],
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
    part = shell(outer, inner)

    # Cable route through the J3 boss, into the hollow tube.
    part -= Drum(
        centre=parent.centre, axis=parent.axis, diameter=BORE_DIAMETER, length=120.0
    ).solid()

    parent_circle, child_circle = _mount_circles()
    part -= bolt_ring(
        centre=np.array([0.0, 0.0, PARENT_PROTRUSION - PARENT_LENGTH]),
        axis=parent.axis,
        bcd=parent_circle.bcd,
        count=parent_circle.count,
        hole_diameter=RULES.m3_clearance,
        depth=40.0,
    )
    part -= bolt_ring(
        centre=frame.child_origin + frame.child_axis * (CHILD_LENGTH / 2),
        axis=child.axis,
        bcd=child_circle.bcd,
        count=child_circle.count,
        hole_diameter=RULES.m3_clearance,
        depth=40.0,
    )

    part -= seam_groove(parent, offset_along_axis=-PARENT_LENGTH / 2 + 5.0)
    part -= seam_groove(child, offset_along_axis=CHILD_LENGTH / 2 - 5.0)

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
