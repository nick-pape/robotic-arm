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
from build123d import Box, Part, Pos

from robotic_arm.design import RULES
from robotic_arm.materials import PC_CF
from robotic_arm.parts.urlink import housing_bore, stator_housing

MATERIAL = PC_CF
BODY = "base_link"

#: Footprint, matching the stock base so it drops onto the same fixture.
PLATE_X = 140.0
PLATE_Y = 200.0
PLATE_THICKNESS = 8.0

#: Bolt-down holes, inset from the corners.
MOUNT_INSET = 18.0
MOUNT_HOLE = 6.6  # M6 clearance


def _housing():
    from robotic_arm.actuators import for_joint
    from robotic_arm.linkframes import link_frame

    frame = link_frame(BODY)
    plane = np.asarray(frame.child_origin, dtype=float)
    # Into the base, which is the only direction there is.
    axis = -np.asarray(frame.child_axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    actuator = for_joint("joint1")
    # Reach from the joint plane right down to the plate, so the column is
    # continuous rather than a barrel perched on a stalk.
    length = float(plane @ -axis) - PLATE_THICKNESS
    return stator_housing(axis, plane, actuator, length), actuator, axis


def collision_primitives() -> list:
    from robotic_arm.parts.cobot import CollisionCylinder

    housing, _, _ = _housing()
    return [CollisionCylinder.from_drum(housing)]


def build_base() -> Part:
    """Return base_link as a solid, in its own frame."""
    from robotic_arm.parts.cobot import _oriented_cylinder, bolt_ring, break_edges

    wall = RULES.structural_wall_thickness
    housing, actuator, axis = _housing()

    plate = Pos(0, 0, PLATE_THICKNESS / 2) * Box(PLATE_X, PLATE_Y, PLATE_THICKNESS)
    part = (housing.solid() + plate) - housing.solid(
        housing.diameter - 2 * wall, housing.length - 2 * wall
    )

    # The joint opening and the ring the J1 stator bolts into.
    mouth = np.asarray(housing.centre, float) - axis * housing.length / 2
    part -= _oriented_cylinder(mouth, axis, housing_bore(actuator) / 2, 4 * wall)
    part -= bolt_ring(
        centre=mouth + axis * wall / 2,
        axis=axis,
        bcd=actuator.stator_circle.bcd,
        count=actuator.stator_circle.count,
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
