"""link2 -- the upper arm, between J2 and J3.

The largest link, and the one the gravity balancer attaches to. Like link3 its
joint axes are parallel, so it is a tapered tube with a drum at each end.

Two things set it apart from the forearm:

* It carries the **RS06** that drives J3, not an RS00, so the child housing is
  O92 rather than O67 -- the biggest section on the arm.
* Stock here is not a tube at all. The cross-section is 65 mm tall the whole
  way but narrows to 26 mm wide at midspan: a flat structural beam, two plates
  rather than a shell. A round section of any useful diameter is therefore
  *wider* than stock at midspan, so the clearance sweep decides whether that
  costs anything rather than the stock silhouette ruling it out.

Stock is 1552 g across a 327 x 88 x 66 mm envelope.
"""

from __future__ import annotations

import numpy as np
from build123d import Part

from robotic_arm.actuators import RS06
from robotic_arm.design import RULES, STYLE
from robotic_arm.linkframes import actuator_centre, link_frame
from robotic_arm.materials import PC_CF
from robotic_arm.parts.cobot import (
    CollisionCylinder,
    Drum,
    bolt_ring,
    cable_channel,
    boss_centre,
    housing_over,
    boss_mount_face,
    break_edges,
    lofted_tube,
    mating_face_opening,
    mount_face_ring,
    mount_boss_diameter,
    seam_groove,
    shell,
)

MATERIAL = PC_CF
BODY = "link2"

#: J2 boss. Stock leaves room for a full drum of about O88 on this axis, so
#: unlike the wrist this end can be generous -- which it should be, since it is
#: the shoulder end of the longest link and the biggest section of the arm.
BOSS_DIAMETER = 76.0
PARENT_LENGTH = 48.0
PARENT_PROTRUSION = -STYLE.joint_gap / 2

#: J3 housing, enclosing the RS06.
CHILD_CLEARANCE = 3.0
CHILD_LENGTH = 58.0

#: Tube profile. Waisted at midspan where the bending moment is lowest, and
#: growing into the child housing rather than pinching below it.
TUBE_STATIONS = (0.0, 0.5, 0.92)
TUBE_DIAMETERS = (60.0, 52.0, 64.0)

CABLE_CHANNEL_DIAMETER = 12.0


def child_diameter() -> float:
    """Outer diameter of the J3 housing, from the RS06 it encloses."""
    rs06 = RS06()
    body_diameter = min(rs06.bbox_mm[0], rs06.bbox_mm[1]) if rs06.bbox_mm else 82.0
    return body_diameter + 2 * (RULES.structural_wall_thickness + CHILD_CLEARANCE)


def _mount_circle():
    """The RS06 output circle this link bolts to at both ends."""
    return RS06().output_circle


def _drums() -> tuple[Drum, Drum]:
    """Parent boss and child housing, shared by the solid and its collision proxy."""
    frame = link_frame(BODY)
    circle = _mount_circle()
    parent = Drum(
        centre=boss_centre(frame, PARENT_LENGTH, PARENT_PROTRUSION),
        axis=frame.parent_axis,
        diameter=mount_boss_diameter(
            circle.bcd, RULES.m3_clearance, floor=BOSS_DIAMETER
        ),
        length=PARENT_LENGTH,
    )
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
    parent, child = _drums()
    points, diameters = _tube_path()
    proxies = [CollisionCylinder.from_drum(parent), CollisionCylinder.from_drum(child)]
    for start, end, d_start, d_end in zip(points, points[1:], diameters, diameters[1:]):
        proxies.append(CollisionCylinder.from_span(start, end, max(d_start, d_end)))
    return proxies


def _channel_offset(drum: Drum) -> np.ndarray:
    """Radial offset placing the cable channel beside a drum, clear of its
    bolt circle and inside its outer wall."""
    axis = np.asarray(drum.axis, dtype=float)
    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(seed @ axis)) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    radial = np.cross(axis, seed)
    radial /= np.linalg.norm(radial)
    return radial * (drum.diameter / 2 - RULES.structural_wall_thickness
                     - CABLE_CHANNEL_DIAMETER / 2 - 1.0)


def build_upper_arm() -> Part:
    """Return the upper arm as a solid, in link2's frame."""
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

    # No central bore: neither actuator is a hollow-shaft motor (both STEP
    # files show only a O4 central feature), so a through-bore would imply a
    # cable route that does not exist -- and on link2 and link3 it cut into
    # the mounting bolt circle, leaving -2.7 mm and -1.7 mm of material. The
    # harness runs beside the actuator, as the stock arm's clipped XT30 daisy
    # chain does.
    part -= cable_channel(
        parent.centre + _channel_offset(parent),
        child.centre + _channel_offset(child),
        CABLE_CHANNEL_DIAMETER,
    )

    part -= mount_face_ring(
        parent, towards=-parent.axis * 1000.0, circle=RS06().output_circle,
        hole_diameter=RULES.m3_clearance, depth=RULES.structural_wall_thickness * 3,
    )
    part -= mount_face_ring(
        child, towards=frame.child_origin, circle=RS06().output_circle,
        hole_diameter=RULES.m3_clearance, depth=RULES.structural_wall_thickness * 3,
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

    part = build_upper_arm()
    props = mass_properties(part, effective_material(part, MATERIAL))
    bb = part.bounding_box()
    print(f"solids {len(part.solids())}  volume {part.volume:,.0f} mm^3")
    print(f"mass   {props.mass * 1000:.1f} g  (stock {stock_mass(BODY) * 1000:.0f} g)")
    print(f"extent {[round(v, 1) for v in (bb.size.X, bb.size.Y, bb.size.Z)]} mm")
    print(f"stock  {link_frame(BODY).stock_extent.round(1)} mm")
