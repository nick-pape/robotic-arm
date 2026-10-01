"""link6 -- the wrist-roll output, i.e. the tool flange.

The most distal link, and the first replaced. In cobot terms this is the tool
flange: a shallow disc that bolts to the wrist actuator's output on one face
and presents a standard tool pattern on the other, with a central bore for the
cable run.

Stock is a 57 mm disc, 10 mm deep, 0.100 kg. This holds the diameter so the
gripper still mates and the proportions carry over, and goes slightly deeper to
house M6 inserts.

Two interfaces, deliberately on different bolt circles:

* **Motor side** -- the O27 six-hole circle measured from RobStride's own STEP.
  Clearance holes only: the bolts thread into the actuator's tapped holes, so
  the printed part just passes them through.
* **Tool side** -- 4 x M5 heat-set inserts on a O44 circle, cobot-style but
  not claiming a standard: ISO 9409-1's spigot-and-bolt pairings do not fit a
  57 mm flange, and an earlier revision that claimed ISO 9409-1-40-4-M6 left
  only 0.25 mm between the inserts and the spigot wall.

Tool location is on the O18 central bore rather than a spigot recess. A recess
large enough to clear the M3 bolt heads (O33.5) would leave barely a
millimetre of wall to the inserts, so the heads are counterbored individually
instead -- simpler, stronger, and it keeps the whole tool face flat.

Frame: link6's origin is on the J6 axis, +Z pointing out towards the tool.
"""

from __future__ import annotations

from build123d import (
    Align,
    Axis,
    BuildPart,
    BuildSketch,
    Circle,
    Cylinder,
    GeomType,
    Mode,
    Part,
    Plane,
    PolarLocations,
    extrude,
    fillet,
)

from robotic_arm.actuators import RS00
from robotic_arm.design import RULES, STYLE
from robotic_arm.materials import PC_CF

#: Bulk material. Effective density is derived per-part from geometry, since a
#: thin flange is nearly all perimeter and a fixed infill estimate would be
#: badly wrong here.
MATERIAL = PC_CF

#: Stock diameter, held so the gripper still mates.
OUTER_DIAMETER = 57.0
#: Deeper than stock's 10 mm, to take an 8 mm M6 insert under the spigot recess.
THICKNESS = 39.0
#:
#: 27 mm longer than stock's 12, which is the number that stops the gripper
#: hitting the forearm. With the wrist folded back -- J5 at +/-3.14 and J6
#: anywhere near +/-75 degrees -- the gripper reached **20.9 mm inside**
#: link3. Standing it further off the wrist is what clears it:
#:
#:     extension   worst gripper clearance
#:        0 mm            -20.92 mm
#:       10 mm            -11.23
#:       20 mm             -1.27
#:       24 mm             +2.72   (under S1's 3 mm)
#:       27 mm             +5.36   <- this
#:
#: `JOINT_SHIFT["gripper_end"]` carries the gripper out with it; the stock
#: frame does not know the flange grew, so lengthening the flange alone would
#: simply drive it into the gripper.
#:
#: Worth knowing what this does *not* fix: at fully folded poses such as
#: J4 = J5 = -180 the gripper still reaches link3, and no amount of standoff
#: helps there -- 60 mm of it still left 42 mm of interference. That one is a
#: combined-pose problem for the joint limits, not for this part.

#: The collar that actually caps link5's housing. It matches that barrel's
#: outside diameter, so the joint reads as one continuous cylinder with a
#: hairline seam rather than a recessed puck.
#:
#: Without it this part was a plain O57 disc against a O64 barrel mouth,
#: leaving a 3.5 mm annular ledge of open housing rim exposed all the way
#: round -- 665 mm2 of link5's mouth face uncovered, plus a 1 mm axial gap of
#: which 0.6 mm was bare motor. That is the dark band at the wrist.
COLLAR_DIAMETER = 64.0
COLLAR_LENGTH = 8.0

#: Cable pass-through. The wrist carries gripper power and CAN through the
#: hollow shaft, so this clears a connector, not just wires.
#:
#: Was O18, which left only 1.25 mm between the bore and the motor
#: counterbores -- below the 2.0 mm structural wall, and the thinnest section
#: in the part.
BORE_DIAMETER = 16.5

TOOL_BCD = 44.0
TOOL_BOLT_COUNT = 4
TOOL_THREAD = "M5"
TOOL_INSERT_DEPTH = 7.0

#: Counterbore sinking each M3 motor-bolt head below the tool face. Deep
#: enough that the bolt spans only the collar, so a stock M3 x 10 reaches the
#: rotor rather than needing a special length for the whole 20 mm stack.
MOTOR_CBORE_DIAMETER = 6.5
MOTOR_CBORE_DEPTH = 10.0


def build_tool_flange() -> Part:
    """Return the tool flange as a solid, in link6's frame."""
    motor = _motor_mount_circle()

    from robotic_arm.parts.urlink import flange_relief_depth

    relief = flange_relief_depth(RS00())
    total = COLLAR_LENGTH + THICKNESS

    with BuildPart() as flange:
        # The collar that caps link5's barrel, then the tool boss on top of
        # it. A cobot flange is a disc and should read as one -- no webs or
        # lightening, which at this depth would cost more stiffness than the
        # grams they save.
        Cylinder(
            radius=COLLAR_DIAMETER / 2,
            height=COLLAR_LENGTH,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        with BuildSketch(Plane.XY.offset(COLLAR_LENGTH)):
            Circle(OUTER_DIAMETER / 2)
        extrude(amount=THICKNESS)

        # Relieve everything outside the rotor hub this bears on, so the
        # collar clears link5's mouth cap -- which now reaches onto this side
        # of the stator face so its own bolts can go in.
        with BuildSketch(Plane.XY):
            Circle(COLLAR_DIAMETER / 2 + 1.0)
            Circle(RS00().hub_diameter / 2, mode=Mode.SUBTRACT)
        extrude(amount=relief, mode=Mode.SUBTRACT)

        # Central cable bore, through everything. Doubles as the tool pilot.
        with BuildSketch(Plane.XY):
            Circle(BORE_DIAMETER / 2)
        extrude(amount=total, mode=Mode.SUBTRACT)

        # Motor-side clearance holes on the measured RS00 output circle, each
        # counterbored so its head sits below the tool mating face.
        with BuildSketch(Plane.XY):
            with PolarLocations(motor.bcd / 2, motor.count):
                Circle(RULES.m3_clearance / 2)
        extrude(amount=total, mode=Mode.SUBTRACT)

        with BuildSketch(Plane.XY.offset(total)):
            with PolarLocations(motor.bcd / 2, motor.count):
                Circle(MOTOR_CBORE_DIAMETER / 2)
        extrude(amount=-MOTOR_CBORE_DEPTH, mode=Mode.SUBTRACT)

        # Tool-side inserts, on their own bolt circle clear of the motor
        # counterbores at every angle.
        insert_hole, _ = RULES.boss_for(TOOL_THREAD)
        with BuildSketch(Plane.XY.offset(total)):
            with PolarLocations(TOOL_BCD / 2, TOOL_BOLT_COUNT):
                Circle(insert_hole / 2)
        extrude(amount=-TOOL_INSERT_DEPTH, mode=Mode.SUBTRACT)

        # Break the two outer rim edges. Selected by radius, not by height:
        # grouping by Z also catches the spigot lip and the insert openings,
        # and filleting those fails.
        rim = [
            e
            for e in flange.edges().filter_by(GeomType.CIRCLE)
            if abs(e.radius - OUTER_DIAMETER / 2) < 1e-6
            or abs(e.radius - COLLAR_DIAMETER / 2) < 1e-6
        ]
        fillet(rim, radius=STYLE.edge_break)

    return flange.part


def collision_primitives() -> list:
    """Collision proxy: the flange is a plain disc, so one cylinder covers it."""
    import numpy as np

    from robotic_arm.parts.cobot import CollisionCylinder

    return [
        CollisionCylinder(
            centre=np.array([0.0, 0.0, (COLLAR_LENGTH + THICKNESS) / 2]),
            axis=np.array([0.0, 0.0, 1.0]),
            radius=COLLAR_DIAMETER / 2,
            length=COLLAR_LENGTH + THICKNESS,
        )
    ]


def _motor_mount_circle():
    """The RS00 output circle this flange bolts to.

    The six-hole circle at O27, which the vendor STEP confirms and which is the
    one published figure for this actuator that is actually correct.
    """
    return RS00().output_circle


def interface_clearance() -> dict[str, float]:
    """Smallest wall left between every pair of features, in mm.

    Negative means they intersect. Worth computing rather than eyeballing:
    two successive revisions of this part had features overlapping, and both
    were only obvious once rendered. Checking the motor counterbore rather than
    the clearance hole is the point -- the counterbore is the larger feature and
    the one that actually collides.
    """
    import math

    motor = _motor_mount_circle()
    insert_hole, _ = RULES.boss_for(TOOL_THREAD)
    r_motor, r_tool = motor.bcd / 2, TOOL_BCD / 2

    worst = float("inf")
    for i in range(motor.count):
        a = 2 * math.pi * i / motor.count
        for j in range(TOOL_BOLT_COUNT):
            b = 2 * math.pi * j / TOOL_BOLT_COUNT
            centres = math.dist(
                (r_motor * math.cos(a), r_motor * math.sin(a)),
                (r_tool * math.cos(b), r_tool * math.sin(b)),
            )
            worst = min(
                worst,
                centres - MOTOR_CBORE_DIAMETER / 2 - insert_hole / 2,
            )

    return {
        "counterbore_to_insert": worst,
        "insert_to_rim": OUTER_DIAMETER / 2 - (r_tool + insert_hole / 2)
        - STYLE.edge_break,
        "bore_to_counterbore": (r_motor - MOTOR_CBORE_DIAMETER / 2)
        - BORE_DIAMETER / 2,
        "remaining_floor_under_counterbore": COLLAR_LENGTH + THICKNESS
        - MOTOR_CBORE_DEPTH,
    }


if __name__ == "__main__":
    from robotic_arm.massprops import mass_properties
    from robotic_arm.parts import effective_material

    part = build_tool_flange()
    props = mass_properties(part, effective_material(part, MATERIAL))
    print(f"volume    {part.volume:,.0f} mm^3")
    print(f"mass      {props.mass * 1000:.1f} g  (stock link6 is 100 g)")
    for name, gap in interface_clearance().items():
        flag = "  <-- TIGHT" if gap < 1.0 else ""
        print(f"  {name:<34}{gap:6.2f} mm{flag}")
