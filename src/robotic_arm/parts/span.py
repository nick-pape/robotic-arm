"""The long tubes between joints, printed as their own parts.

On a UR these are the drawn aluminium spans between joint castings, and the
patent is explicit about why they are tubes at all: "thin-walled tubes, which
constitutes a preferred choice due to their optimal rigidity/weight ratio"
(EP3045273A1). It attaches them at a "connecting portion" on the joint
housing without saying how they are fixed.

Here they are printed like everything else, and they are separate parts for
two reasons. The first is that link2 and link3 are 381 and 359 mm as single
pieces -- the only two parts on the arm that will not fit a 256 mm bed, and
the tube is all of the excess. Split at the tube, every printed piece fits.
The second is that a tube prints far better on its own axis than a whole link
does lying down.

**How it attaches.** The casting carries a splined spigot; the tube slides
over it and two bolts go in perpendicular to that motion. The load path is
the point:

* **torque** is carried by the twelve flats, not the fasteners. Round and
  bolted, 36 N*m at J2 puts ~360 N of shear on a O4 hole through a 3 mm
  printed wall -- 30 MPa against PC-CF's ~70, and loaded across the layers,
  which is where a print is weakest. Spread over the flats the same torque is
  single-digit MPa.
* **bending** is carried by the slide fit over a 60 mm engagement, not the
  fasteners.
* the **bolts** stop the tube sliding off and do nothing else. Perpendicular
  to the slide, so they are in shear rather than pulling out of printed
  threads.
* a **shoulder** at the barrel takes the axial butt.

The tube is printed standing on its axis, which is what makes it round and
lets it fit the bed. That puts mid-span bending across the layers, so the
wall is 3.0 mm rather than the 2.0 used on the shells.
"""

from __future__ import annotations

import numpy as np
from build123d import Part

from robotic_arm.materials import PC_CF
from robotic_arm.parts.urlink import (
    SPAN_BOLT_COUNT,
    SPAN_BOLT_THREAD,
    SPAN_ENGAGEMENT,
    SPAN_SETBACK,
    SPAN_WALL,
    _perpendicular,
    polygon_prism,
    span_gap,
)

MATERIAL = PC_CF

#: Links built as casting + tube + casting.
SPAN_BODIES = ("link2", "link3")


def _profile(body: str):
    """(module, flange length, barrel length) for a span link."""
    import importlib

    module = importlib.import_module(
        {"link2": "robotic_arm.parts.upper_arm",
         "link3": "robotic_arm.parts.forearm"}[body]
    )
    return (
        module,
        module.FLANGE_LENGTH,
        getattr(module, "BARREL_LENGTH", None),
    )


def build_span_tube(body: str) -> Part:
    """The tube for one span link, in that link's own frame."""
    from robotic_arm.design import RULES
    from robotic_arm.parts.cobot import _oriented_cylinder, break_edges

    module, flange_length, barrel_length = _profile(body)
    start, end, axis = span_gap(body, flange_length, barrel_length)
    # Stand off both barrels so the tube clears the spigot's root blend.
    start = start + axis * SPAN_SETBACK
    end = end - axis * SPAN_SETBACK
    length = float(np.linalg.norm(end - start))
    outer_diameter = max(module.TUBE_DIAMETERS)
    bore = outer_diameter / 2 - SPAN_WALL

    tube = _oriented_cylinder(
        (start + end) / 2, axis, outer_diameter / 2, length
    )
    # Splined bore the whole way. The flats only do work over the engagement,
    # but carrying them through costs nothing, prints the same, and leaves no
    # step inside for a cable to catch on.
    tube -= polygon_prism(start - axis * 1.0, axis, bore, length + 2.0)

    hole, _ = RULES.boss_for(SPAN_BOLT_THREAD)
    for origin, sign in ((start, 1.0), (end, -1.0)):
        seat = origin + axis * sign * (SPAN_ENGAGEMENT * 0.6)
        for index in range(SPAN_BOLT_COUNT):
            radial = _perpendicular(axis, index, SPAN_BOLT_COUNT)
            tube -= _oriented_cylinder(
                seat, radial, hole / 2, outer_diameter + 2.0
            )

    return break_edges(tube)


def build_span_pieces(body: str) -> dict[str, Part]:
    """The three printed pieces of a span link, in that link's own frame.

    Keyed "near", "tube", "far" -- near being the casting at this link's own
    joint. Ordered by distance from the link origin rather than by whatever
    order the boolean happened to leave them in, so the keys mean the same
    thing on every rebuild and the colours do not swap between renders.
    """
    from robotic_arm.parts.urlink import build_ur_link

    module, flange_length, barrel_length = _profile(body)
    castings = build_ur_link(
        body,
        flange_length,
        module.TUBE_STATIONS,
        module.TUBE_DIAMETERS,
        housing_length=barrel_length,
        tube_end_offset=getattr(module, "TUBE_END_OFFSET", 0.0),
        span=True,
    )
    near, far = sorted(
        castings,
        key=lambda piece: float(
            np.linalg.norm(
                np.array(
                    [
                        piece.bounding_box().center().X,
                        piece.bounding_box().center().Y,
                        piece.bounding_box().center().Z,
                    ]
                )
            )
        ),
    )
    return {"near": near, "tube": build_span_tube(body), "far": far}


def build_span_link(body: str) -> Part:
    """The whole span link as one solid, for mass and collision.

    It is three printed pieces but one rigid body: bolted together, they move
    as a unit, so the physics wants them combined and only the print files and
    the render want them apart.
    """
    pieces = build_span_pieces(body)
    return pieces["near"] + pieces["tube"] + pieces["far"]


if __name__ == "__main__":
    from robotic_arm.massprops import mass_properties
    from robotic_arm.parts import effective_material

    for body in SPAN_BODIES:
        part = build_span_tube(body)
        props = mass_properties(part, effective_material(part, MATERIAL))
        box = part.bounding_box()
        longest = max(box.size.X, box.size.Y, box.size.Z)
        print(
            f"{body}: {len(part.solids())} solid  volume {part.volume:,.0f} mm^3  "
            f"mass {props.mass * 1000:.1f} g  longest {longest:.1f} mm "
            f"({'fits' if longest <= 256 else 'TOO BIG'})"
        )
