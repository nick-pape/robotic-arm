"""The cap over the back of each joint, and the way a motor gets in.

Every UR joint shows one of these: a disc on the face opposite the motor's
output, sitting slightly proud of the barrel. It is not decoration. A pancake
actuator is wider than anything its own joint can pass -- an RS00 is O57 and
its mouth bore is O37.6 -- so the motor has to be fitted from behind, and
something has to close the hole afterwards.

Without it the arm is not buildable. Sweeping each actuator along its own
axis through the housing that holds it, a motor fouled 6,041 mm3 on the way
out through the mouth and up to 10,874 mm3 on the way out through the back:
both ends were solid. Every housing now opens at the back to
`urlink.service_opening`, and this is what goes over it.

**Mounting.** A flat lid flush on the back plate, held by three M2.5 screws
into heat-set inserts. A spigot on the lid's underside locates it in the
opening, so the screws are not what centres it.

There is nothing to screw into on a 2.0 mm wall, and the fix went through two
shapes. First a flared collar on the last 10 mm of each barrel, which worked
and looked like a collar. Now the barrel simply carries that diameter for its
whole length -- O103.5 on an RS06 joint, O76 on an RS00 -- and the inserts sit
in a land inside the back plate. Same material, one diameter instead of two,
and nothing to see from outside.

The screws are M2.5 rather than M3 because this cap carries no load at all,
and its boss is what sets the barrel diameter.

The lid is a separate printed piece but is bolted solid to its link, so its
mass belongs to that link's inertial while its geometry belongs in its own
STL. `placed` gives it in the link's own frame for both purposes.
"""

from __future__ import annotations

import numpy as np
from build123d import (
    Align,
    BuildPart,
    BuildSketch,
    Circle,
    Cylinder,
    Mode,
    Part,
    Plane,
    PolarLocations,
    extrude,
)

from robotic_arm.design import RULES, STYLE
from robotic_arm.materials import PC_CF

MATERIAL = PC_CF

#: How thick the lid is. It spans an O87.5 hole on the RS06 joints with no
#: support behind it, so it is a structural disc, not a sticker.
THICKNESS = 4.0

#: How far the locating spigot reaches into the opening. Short: it centres
#: the lid, it does not carry load.
SPIGOT_DEPTH = 3.0

#: Counterbore sinking each M3 head below the outer face, so the back of the
#: joint finishes flat.
COUNTERBORE_DIAMETER = 5.5
COUNTERBORE_DEPTH = 2.0


def build_service_cap(actuator) -> Part:
    """One cap, in a frame with the collar face at z=0 and the lid at +z.

    The spigot hangs below z=0 into the opening; the lid sits proud of the
    barrel, which is where a UR's caps sit too.
    """
    from robotic_arm.parts.cobot import break_edges
    from robotic_arm.parts.urlink import (
        SERVICE_BOLT_COUNT,
        service_bcd,
        service_collar_diameter,
        service_opening,
    )

    outer = service_collar_diameter(actuator)
    opening = service_opening(actuator)
    spigot = opening - 2 * STYLE.joint_gap

    with BuildPart() as cap:
        Cylinder(
            radius=outer / 2,
            height=THICKNESS,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        # Locating spigot, reaching down into the opening.
        with BuildSketch(Plane.XY):
            Circle(spigot / 2)
        extrude(amount=-SPIGOT_DEPTH)

        # Clearance holes for the screws that hold it down, counterbored so
        # the heads finish below the outer face.
        with BuildSketch(Plane.XY.offset(THICKNESS)):
            with PolarLocations(service_bcd(actuator) / 2, SERVICE_BOLT_COUNT):
                Circle(RULES.m25_clearance / 2)
        extrude(amount=-THICKNESS, mode=Mode.SUBTRACT)

        with BuildSketch(Plane.XY.offset(THICKNESS)):
            with PolarLocations(service_bcd(actuator) / 2, SERVICE_BOLT_COUNT):
                Circle(COUNTERBORE_DIAMETER / 2)
        extrude(amount=-COUNTERBORE_DEPTH, mode=Mode.SUBTRACT)

    return break_edges(cap.part)


def placed(body: str) -> list[Part]:
    """Every service cap this link carries, in the link's own frame."""
    from build123d import Location, Plane as BuildPlane, Vector

    from robotic_arm.parts.urlink import housed_joints, housing_drum, service_face

    out = []
    for joint_index, actuator in housed_joints(body):
        drum = housing_drum(body, joint_index)
        face = service_face(drum)
        axis = np.asarray(drum.axis, dtype=float)
        seat = BuildPlane(origin=Vector(*face), z_dir=Vector(*axis))
        out.append(seat * build_service_cap(actuator))
    return out


if __name__ == "__main__":
    from robotic_arm.actuators import RS00, RS06
    from robotic_arm.massprops import mass_properties
    from robotic_arm.parts import effective_material

    for actuator in (RS06(), RS00()):
        part = build_service_cap(actuator)
        props = mass_properties(part, effective_material(part, MATERIAL))
        bb = part.bounding_box()
        print(
            f"{actuator.name}: volume {part.volume:,.0f} mm^3  "
            f"mass {props.mass * 1000:.1f} g  "
            f"O{bb.size.X:.1f} x {bb.size.Z:.1f} deep"
        )
