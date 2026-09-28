"""link3 -- the forearm, between J3 and J4.

Built by `urlink.build_ur_link`, the UR-style archetype every link on this arm
shares::

    (=)  ================  [  M  ]
     rotor flange          stator housing

The flange caps the previous joint's housing and bolts to its **rotor** ring;
the housing encloses this link's own child motor and bolts to its **stator**
ring. Both rings are on one face of a pancake actuator, so a joint is two
links bolted to the same face at different radii, and the seam between flange
and housing is the only thing visible from outside.

This link carries the RS00 that drives the wrist.
"""

from __future__ import annotations

from build123d import Part

from robotic_arm.materials import PC_CF
from robotic_arm.parts.urlink import build_ur_link, link_ends

MATERIAL = PC_CF
BODY = "link3"

#: How far the rotor flange reaches into the link. The housing depth is not a
#: constant: it is derived from the motor it has to enclose.
FLANGE_LENGTH = 76.0
#:
#: 76 mm: the diameter of the J4 canister, used as the common height for
#: every piece that is free to take it. The four -- J2's and J3's motor
#: halves (this barrel serves both), J3's empty side and J4's motor half --
#: are the only halves on the arm not pinned by tee geometry or by the
#: gripper face, so they are the ones that can be made to agree.
#:
#: Matches link1's branch, which is what makes J3 the same length as J2:
#: both are this flange plus the shared link2 barrel.

#: The J4 barrel. Longer than the 55 mm the RS00 needs, because the tube is
#: squared to the barrels rather than raked between their centres: this link's
#: two barrels sit 9.6 mm apart along their shared axis, and meeting both at
#: the same station moves the tube's end 4.8 mm off centre, which a 55 mm
#: barrel cannot contain. Widening the barrel is the trade worth making --
#: a segment that is visibly 2 degrees off straight is worse than a barrel a
#: centimetre longer.
BARREL_LENGTH = 76.0
#:
#: 76 mm: the diameter of the J4 canister, used as the common height for
#: every piece that is free to take it. The four -- J2's and J3's motor
#: halves (this barrel serves both), J3's empty side and J4's motor half --
#: are the only halves on the arm not pinned by tee geometry or by the
#: gripper face, so they are the ones that can be made to agree.
#:
#: Sized so every RS00 joint presents the same 118 mm canister -- housing
#: plus the flange that caps it. They were 112, 122 and 84, which is what
#: made the wrist look unplanned next to itself. 118 is the floor, not a
#: preference: link4's barrel has to be at least as long as its branch is
#: wide (O76) or the tee stops reading as one, so J5 cannot come in under
#: 76 + 42.

#: Tube profile between the two ends, as fractions of the run and diameters at
#: each station.
TUBE_STATIONS = (0.0, 0.5, 1.0)
TUBE_DIAMETERS = (52.0, 48.0, 44.0)

#: Slides the tube's target deeper into the housing. Aiming at the housing
#: centre is right for an in-line joint and wrong for a perpendicular one,
#: where it drives the tube through the joint bore.
TUBE_END_OFFSET = 0.0


def _drums():
    """(rotor flange, stator housing) for this link."""
    flange, housing, _, _ = link_ends(BODY, FLANGE_LENGTH, BARREL_LENGTH)
    return flange, housing


def collision_primitives() -> list:
    """Collision proxy: one cylinder per end, plus the tube between."""
    import numpy as np

    from robotic_arm.parts.cobot import CollisionCylinder

    flange, housing = _drums()
    proxies = [
        CollisionCylinder.from_drum(flange),
        CollisionCylinder.from_drum(housing),
    ]
    axis = np.asarray(housing.axis, float)
    start = np.asarray(flange.centre, float)
    end = np.asarray(housing.centre, float) + axis / np.linalg.norm(axis) * TUBE_END_OFFSET
    points = [start + (end - start) * f for f in TUBE_STATIONS]
    for a, b, da, db in zip(points, points[1:], TUBE_DIAMETERS, TUBE_DIAMETERS[1:]):
        proxies.append(CollisionCylinder.from_span(a, b, max(da, db)))
    return proxies


def build_forearm() -> Part:
    """Return link3 as a solid, in its own frame."""
    # Three printed pieces -- casting, tube, casting -- combined here because
    # bolted together they are one rigid body. See `parts.span` for why the
    # tube is separate and how it attaches.
    from robotic_arm.parts.span import build_span_link

    return build_span_link(BODY)


if __name__ == "__main__":
    from robotic_arm.linkframes import link_frame, stock_mass
    from robotic_arm.massprops import mass_properties
    from robotic_arm.parts import effective_material

    part = build_forearm()
    props = mass_properties(part, effective_material(part, MATERIAL))
    bb = part.bounding_box()
    print(f"solids {len(part.solids())}  volume {part.volume:,.0f} mm^3")
    print(f"mass   {props.mass * 1000:.1f} g  (stock {stock_mass(BODY) * 1000:.0f} g)")
    print(f"extent {[round(v, 1) for v in (bb.size.X, bb.size.Y, bb.size.Z)]} mm")
    print(f"stock  {link_frame(BODY).stock_extent.round(1)} mm")
