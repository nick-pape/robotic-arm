"""link5 -- the wrist-roll housing, J5 to J6.

Built by `urlink.build_ur_link`, the UR-style archetype every link on this arm
shares::

    (=)  ================  [  M  ]
     rotor flange          stator housing

The flange caps the previous joint's housing and bolts to its **rotor** ring;
the housing encloses this link's own child motor and bolts to its **stator**
ring. Both rings are on one face of a pancake actuator, so a joint is two
links bolted to the same face at different radii, and the seam between flange
and housing is the only thing visible from outside.

This link carries the RS00 that drives the tool flange.
"""

from __future__ import annotations

from build123d import Part

from robotic_arm.materials import PC_CF
from robotic_arm.parts.urlink import build_ur_link, link_ends

MATERIAL = PC_CF
BODY = "link5"

#: How far the rotor flange reaches into the link. The housing depth is not a
#: constant: it is derived from the motor it has to enclose.
FLANGE_LENGTH = 42.0

#: Tube profile between the two ends, as fractions of the run and diameters at
#: each station.
TUBE_STATIONS = (0.0, 0.5, 1.0)
TUBE_DIAMETERS = (38.0, 36.0, 38.0)

#: Slides the tube's target deeper into the housing. Aiming at the housing
#: centre is right for an in-line joint and wrong for a perpendicular one,
#: where it drives the tube through the joint bore.
TUBE_END_OFFSET = 0.0


def _drums():
    """(rotor flange, stator housing) for this link."""
    flange, housing, _, _ = link_ends(BODY, FLANGE_LENGTH)
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


def build_wrist_roll() -> Part:
    """Return link5 as a solid, in its own frame."""
    return build_ur_link(
        BODY,
        FLANGE_LENGTH,
        TUBE_STATIONS,
        TUBE_DIAMETERS,
        tube_end_offset=TUBE_END_OFFSET,
    )


if __name__ == "__main__":
    from robotic_arm.linkframes import link_frame, stock_mass
    from robotic_arm.massprops import mass_properties
    from robotic_arm.parts import effective_material

    part = build_wrist_roll()
    props = mass_properties(part, effective_material(part, MATERIAL))
    bb = part.bounding_box()
    print(f"solids {len(part.solids())}  volume {part.volume:,.0f} mm^3")
    print(f"mass   {props.mass * 1000:.1f} g  (stock {stock_mass(BODY) * 1000:.0f} g)")
    print(f"extent {[round(v, 1) for v in (bb.size.X, bb.size.Y, bb.size.Z)]} mm")
    print(f"stock  {link_frame(BODY).stock_extent.round(1)} mm")
