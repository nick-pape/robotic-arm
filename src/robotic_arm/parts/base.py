"""base_link -- the pedestal that carries the J1 motor.

The one part with no rotor flange: nothing drives it, so it is a stator
housing on a mounting plate. That is the whole of a UR5e's base -- a cylinder
the diameter of its shoulder joint, standing on a bolt-down flange.

    [ M ]
      |     stator housing, enclosing the J1 RS06
    =====   mounting plate

The housing rises to meet J1's plane at the top, so the shoulder's rotor
flange caps it and the seam sits where the base stops turning.
"""

from __future__ import annotations

import numpy as np
from build123d import Box, Cone, Part, Pos

from robotic_arm.design import RULES
from robotic_arm.materials import PC_CF
from robotic_arm.parts.urlink import (
    MOTOR_PLANE_OFFSET,
    flange_relief_depth,
    housing_bore,
)

MATERIAL = PC_CF
BODY = "base_link"

#: Footprint, matching the stock base so it drops onto the same fixture.
PLATE_X = 140.0
PLATE_Y = 200.0
PLATE_THICKNESS = 8.0

#: Bolt-down holes, inset from the corners.
MOUNT_INSET = 18.0
MOUNT_HOLE = 6.6  # M6 clearance

#: The flare from plate to column. A UR5e's base is a truncated cone, not a
#: plain cylinder, and the difference is not decoration: with a straight
#: column the grey reads as another arm segment competing with the shoulder
#: above it, which is why link1 looked like a thin washer squeezed between
#: two cylinders. Flaring the base makes it read as a base.
#:
#: The draft is about 21 degrees from vertical, so it prints without support.
#: The footprint and bolt pattern are left at stock so the arm still drops
#: onto the same fixture.
CONE_BASE_DIAMETER = 124.0
CONE_TOP = 30.0


def _seat():
    """(rotor flange drum, J1 actuator, direction) for the base.

    base_link holds no motor. The J1 RS06 lives in **link1**, which sheathes
    it, and the base is only what a UR5e's base is: a small flare presenting
    that joint's rotor. So this is a flange, not a housing.

    It sits at the motor's own plane rather than the joint origin --
    `urlink.MOTOR_PLANE_OFFSET` seats J1's motor 35 mm low so link1's barrel
    clears link2's, which is what lets the base be a flare instead of a
    full-height column. Sliding a motor along its own rotation axis changes no
    kinematics.
    """
    from robotic_arm.actuators import for_joint
    from robotic_arm.linkframes import link_frame
    from robotic_arm.parts.urlink import (
        housing_diameter,
        joint_motor_side,
        rotor_flange,
    )

    frame = link_frame(BODY)
    plane = np.asarray(frame.child_origin, dtype=float)
    actuator = for_joint("joint1")
    side = np.asarray(joint_motor_side("joint1"), dtype=float)
    plane = plane + side * MOTOR_PLANE_OFFSET["joint1"]
    # The flange reaches away from the motor, down toward the plate.
    away = -side
    # Height of the seat above the top of the flare. Projected onto `side`
    # (which points up, toward the motor), not onto `away`, or it comes out
    # negative and the drum fails to construct.
    length = float(plane @ side) - CONE_TOP
    return rotor_flange(away, plane, actuator, length), actuator, away


def _housing():
    """Kept as the name the rest of the project reaches for."""
    return _seat()


def collision_primitives() -> list:
    from robotic_arm.parts.cobot import CollisionCylinder

    housing, _, _ = _housing()
    return [CollisionCylinder.from_drum(housing)]


def build_base() -> Part:
    """Return base_link as a solid, in its own frame."""
    from robotic_arm.parts.cobot import _oriented_cylinder, bolt_ring, break_edges

    wall = RULES.structural_wall_thickness
    seat, actuator, away = _seat()

    plate = Pos(0, 0, PLATE_THICKNESS / 2) * Box(PLATE_X, PLATE_Y, PLATE_THICKNESS)

    # Flared skirt from the plate up to the seat, hollow so it costs almost
    # nothing: a solid cone here would outweigh the whole shell.
    rise = CONE_TOP - PLATE_THICKNESS
    flare = Pos(0, 0, PLATE_THICKNESS + rise / 2) * Cone(
        bottom_radius=CONE_BASE_DIAMETER / 2,
        top_radius=seat.diameter / 2,
        height=rise,
    )
    hollow = Pos(0, 0, PLATE_THICKNESS + rise / 2) * Cone(
        bottom_radius=CONE_BASE_DIAMETER / 2 - wall,
        top_radius=seat.diameter / 2 - wall,
        height=rise,
    )

    relief = flange_relief_depth(actuator, wall)
    part = (seat.solid() + plate + flare) - (
        _oriented_cylinder(
            np.asarray(seat.centre, float) + away * relief / 2,
            away,
            (seat.diameter - 2 * wall) / 2,
            seat.length - 2 * wall - relief,
        )
        + hollow
    )

    # Bolt to the J1 rotor, and relieve outside the hub it bears on so the
    # base clears link1's mouth cap.
    face = np.asarray(seat.centre, float) - away * seat.length / 2
    part -= (
        _oriented_cylinder(
            face + away * relief / 2, away, (seat.diameter + 2.0) / 2, relief
        )
        - _oriented_cylinder(
            face + away * relief / 2, away, actuator.hub_diameter / 2, relief + 2.0
        )
    )
    part -= bolt_ring(
        centre=face + away * wall / 2,
        axis=away,
        bcd=actuator.rotor_circle.bcd,
        count=actuator.rotor_circle.count,
        hole_diameter=RULES.m3_clearance,
        depth=wall * 3,
    )

    for x in (-1, 1):
        for y in (-1, 1):
            part -= _oriented_cylinder(
                (x * (PLATE_X / 2 - MOUNT_INSET), y * (PLATE_Y / 2 - MOUNT_INSET),
                 PLATE_THICKNESS / 2),
                (0, 0, 1), MOUNT_HOLE / 2, PLATE_THICKNESS * 3,
            )

    part = break_edges(part)
    if len(part.solids()) != 1:
        raise ValueError(f"{BODY}: built {len(part.solids())} solids")
    return part


if __name__ == "__main__":
    from robotic_arm.linkframes import stock_mass
    from robotic_arm.massprops import mass_properties
    from robotic_arm.parts import effective_material

    part = build_base()
    props = mass_properties(part, effective_material(part, MATERIAL))
    bb = part.bounding_box()
    print(f"solids {len(part.solids())}  volume {part.volume:,.0f} mm^3")
    print(f"mass   {props.mass * 1000:.1f} g  (stock {stock_mass(BODY) * 1000:.0f} g)")
    print(f"extent {[round(v, 1) for v in (bb.size.X, bb.size.Y, bb.size.Z)]} mm")
