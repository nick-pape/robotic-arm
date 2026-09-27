"""link1 -- the shoulder column, between J1 and J2.

A tall vertical barrel holding the J1 motor, with a short pad on its side that
J2's rotor bolts to. The barrel is the part; the pad is a detail on it.

That arrangement is measured, not styled. UR publish CAD for the e-Series, and
in it the base-joint barrel carries a flat boss the full barrel diameter
standing only ~1.9 mm proud of its side, against which the next barrel's
output face butts -- separated by the black seam ring, with essentially zero
overlap. The shoulder is an **L of two barrels that meet**, not a T of two
that fuse, and the vertical barrel slightly overshoots the horizontal one's
crown.

link1 owns the J1 motor. That is also measured: UR put the base-joint module
in the piece *above* it, so the module turns and its output bolts back down
into the fixed pedestal, leaving the base a small flare. `mounts` finds the
same arrangement in this arm's own stock geometry.
"""

from __future__ import annotations

from build123d import Part

from robotic_arm.materials import PC_CF
from robotic_arm.parts.urlink import build_ur_link, link_ends

MATERIAL = PC_CF
BODY = "link1"

#: The branch of the tee, reaching across to M2.
#:
#: Stops at the barrel's centreline, which is how a plumbing tee is actually
#: made -- the branch bore meets the run bore and goes no further. Run past
#: it and the branch emerges from the far side, turning the part into a plus
#: instead of a tee: at 74 mm the tip reached y = -46.8 against a barrel
#: surface at -47, so it crossed the whole barrel.
FLANGE_LENGTH = 28.0

#: The vertical barrel, which is the whole part. Long enough that the pad's
#: circumference sits within its height, so from any angle the shoulder reads
#: as one cylinder with a detail on its flank. Past ~152 mm it starts to foul
#: link2's tube.
BARREL_LENGTH = 152.0

#: No tube: the pad sits directly on the barrel, so the two ends already meet
#: and `build_ur_link` omits one. These remain because the builder still takes
#: a profile, and to keep `collision_primitives` honest if that ever changes.
TUBE_STATIONS = (0.0, 0.5, 1.0)
TUBE_DIAMETERS = (38.0, 38.0, 38.0)
TUBE_OFFSETS = (0.0, 0.0, 0.0)


def _drums():
    """(barrel, pad) for this link."""
    barrel, pad, _, _ = link_ends(BODY, FLANGE_LENGTH, BARREL_LENGTH)
    return barrel, pad


def collision_primitives() -> list:
    """Collision proxy: one cylinder per end."""
    from robotic_arm.parts.cobot import CollisionCylinder

    barrel, pad = _drums()
    return [
        CollisionCylinder.from_drum(barrel),
        CollisionCylinder.from_drum(pad),
    ]


def build_shoulder() -> Part:
    """Return link1 as a solid, in its own frame."""
    return build_ur_link(
        BODY,
        FLANGE_LENGTH,
        TUBE_STATIONS,
        TUBE_DIAMETERS,
        housing_length=BARREL_LENGTH,
        tube_offsets=TUBE_OFFSETS,
    )


if __name__ == "__main__":
    from robotic_arm.linkframes import stock_mass
    from robotic_arm.massprops import mass_properties
    from robotic_arm.parts import effective_material

    part = build_shoulder()
    props = mass_properties(part, effective_material(part, MATERIAL))
    bb = part.bounding_box()
    print(f"solids {len(part.solids())}  volume {part.volume:,.0f} mm^3")
    print(f"mass   {props.mass * 1000:.1f} g  (stock {stock_mass(BODY) * 1000:.0f} g)")
    print(f"extent {[round(v, 1) for v in (bb.size.X, bb.size.Y, bb.size.Z)]} mm")
