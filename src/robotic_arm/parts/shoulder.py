"""link1 -- the shoulder, between J1 and J2.

Built by `urlink.build_ur_link`, the UR-style archetype every link on this arm
shares::

    (=)  ================  [  M  ]
     rotor flange          stator housing

The shoulder is the most compact link on the arm: J1 and J2 are only 78 mm
apart and their axes are perpendicular, so two O94 barrels for two RS06s
overlap heavily and merge into a single casting rather than being joined by
any real length of tube. That is exactly what a UR5e's shoulder looks like,
and why it is a short fat elbow rather than a tube.

This link carries the RS06 that drives J2 -- and it carries it on the
*opposite* side from the stock arm. Stock mounts J2 backwards, putting link1
on the rotor and making link2 carry two stators; this design follows the
uniform UR convention instead. See `urlink.designed_motor_side`.
"""

from __future__ import annotations

from build123d import Part

from robotic_arm.materials import PC_CF
from robotic_arm.parts.urlink import build_ur_link, link_ends

MATERIAL = PC_CF
BODY = "link1"

#: Long enough that the two barrels actually intersect. They are 78 mm apart
#: and each is O94, so anything shorter leaves them merely touching, which
#: builds as two solids rather than one part.
FLANGE_LENGTH = 48.0

TUBE_STATIONS = (0.0, 0.5, 1.0)
TUBE_DIAMETERS = (44.0, 42.0, 44.0)
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


def build_shoulder() -> Part:
    """Return link1 as a solid, in its own frame."""
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

    part = build_shoulder()
    props = mass_properties(part, effective_material(part, MATERIAL))
    bb = part.bounding_box()
    print(f"solids {len(part.solids())}  volume {part.volume:,.0f} mm^3")
    print(f"mass   {props.mass * 1000:.1f} g  (stock {stock_mass(BODY) * 1000:.0f} g)")
    print(f"extent {[round(v, 1) for v in (bb.size.X, bb.size.Y, bb.size.Z)]} mm")
