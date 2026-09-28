"""link4 -- the first wrist section, J4 to J5.

A **tee**, matching the UR e-Series wrist: the J5 barrel is the run and the
J4 rotor flange is a branch off its cylindrical side.

That shape is only available because the J4 and J5 axes **intersect**. Stock
leaves them 87 mm apart -- the reBot is planar and the UR is not -- so a link
spanning them could only ever be a rod raked across two skew axes, meeting
neither square. `JOINT_SHIFT` removes the 87 mm; see `linkframes`.

Removing it costs reach, and this link buys that back, which is why the run
is long. The two directions are not interchangeable:

* the **branch** lies along the J4 axis. J2, J3 and J4 are all parallel, so
  no joint on the arm can turn a vector along that axis toward the radial
  direction -- it only ever adds in quadrature against a ~700 mm lever.
  Measured, 112 mm of extra branch bought 33 mm of reach. So the branch is
  held at 42 mm, the shortest that still reads as a tee.
* the **run** lies along the J5 axis, perpendicular to those three, so it
  rotates into the radial direction and pays about 0.63 mm of reach per mm.

At a 177 mm run the arm reaches 785.9 mm against stock's 783.9 -- parity to
0.3% -- while J4->J5 grows to 181.9 mm from stock's 104.0. That segment is
the one deliberate departure; every other segment matches stock to 0.08 mm.

The barrel only has to **contain** the branch, not centre it, so it is
177 + 40 mm rather than twice the run. That 40 mm overshoot past the branch
axis is what makes the corner an elbow instead of a tangency.

link4 bolts to the J4 rotor at the branch and carries the J5 motor in the run.
"""

from __future__ import annotations

from build123d import Part

from robotic_arm.materials import PC_CF
from robotic_arm.parts.urlink import build_ur_link, link_ends

MATERIAL = PC_CF
BODY = "link4"

#: The branch, out-of-plane and therefore worthless for reach: the shortest
#: that still reads as a tee. 32 mm of it is buried in the run, leaving a
#: 10 mm stub proud of the barrel.
FLANGE_LENGTH = 59.75
#:
#: The same as link1's and link3's branches: these are the three pieces of
#: the arm that carry no motor, and they now read as one family. The branch
#: still stops at the run's centreline -- `JOINT_SHIFT` moves the J5 axis out
#: with it, or the branch would cross the barrel and make a plus.

#: The run: the 177 mm that carries J5 out to reach parity, plus 40 mm of
#: overshoot past the branch axis so the corner blends. The RS00 needs only
#: 55 mm of this; the rest is structure.
BARREL_LENGTH = 76.0
#:
#: Sized so every joint on the arm presents the same 160.75 mm canister -- housing
#: plus the flange that caps it. They were 112, 122 and 84, which is what
#: made the wrist look unplanned next to itself. 118 is the floor, not a
#: preference: link4's barrel has to be at least as long as its branch is
#: wide (O76) or the tee stops reading as one, so J5 cannot come in under
#: 76 + 42.

#: link4 is a **tee**: the J4 and J5 barrels touch and fuse, so there is no
#: tube between them at all and `build_ur_link` omits one. These remain
#: because the builder still takes a profile.
TUBE_STATIONS = (0.0, 0.5, 1.0)
TUBE_DIAMETERS = (44.0, 44.0, 44.0)


def _drums():
    """(rotor flange, stator housing) for this link."""
    flange, housing, _, _ = link_ends(BODY, FLANGE_LENGTH, BARREL_LENGTH)
    return flange, housing


def collision_primitives() -> list:
    """Collision proxy: one cylinder per barrel, plus one per elbow segment."""
    import numpy as np

    from robotic_arm.parts.cobot import CollisionCylinder

    flange, housing = _drums()
    proxies = [
        CollisionCylinder.from_drum(flange),
        CollisionCylinder.from_drum(housing),
    ]
    start = np.asarray(flange.centre, float)
    end = np.asarray(housing.centre, float)
    points = [start + (end - start) * f for f in TUBE_STATIONS]
    for a, b, da, db in zip(points, points[1:], TUBE_DIAMETERS, TUBE_DIAMETERS[1:]):
        proxies.append(CollisionCylinder.from_span(a, b, max(da, db)))
    return proxies


def build_wrist_pitch() -> Part:
    """Return link4 as a solid, in its own frame."""
    return build_ur_link(
        BODY,
        FLANGE_LENGTH,
        TUBE_STATIONS,
        TUBE_DIAMETERS,
        housing_length=BARREL_LENGTH,
    )


if __name__ == "__main__":
    from robotic_arm.linkframes import link_frame, stock_mass
    from robotic_arm.massprops import mass_properties
    from robotic_arm.parts import effective_material

    part = build_wrist_pitch()
    props = mass_properties(part, effective_material(part, MATERIAL))
    bb = part.bounding_box()
    print(f"solids {len(part.solids())}  volume {part.volume:,.0f} mm^3")
    print(f"mass   {props.mass * 1000:.1f} g  (stock {stock_mass(BODY) * 1000:.0f} g)")
    print(f"extent {[round(v, 1) for v in (bb.size.X, bb.size.Y, bb.size.Z)]} mm")
    print(f"stock  {link_frame(BODY).stock_extent.round(1)} mm")
