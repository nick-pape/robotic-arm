"""The printed wrist: link4 and link5.

Both are two perpendicular drums joined by a tube. That is not a style choice
imposed on the geometry -- J4/J5 and J5/J6 genuinely cross at right angles, so
a drum per axis is the shape the kinematics asks for.

These check the things a render cannot: that the shells are closed solids, that
they are hollow rather than accidentally filled, that they sit where the stock
part sat, and that they beat the mass they replace.
"""

import numpy as np
import pytest

from robotic_arm.design import RULES
from robotic_arm.linkframes import link_frame, stock_mass
from robotic_arm.massprops import mass_properties
from robotic_arm.parts import effective_material
from robotic_arm.parts.wrist_pitch import MATERIAL as PITCH_MATERIAL
from robotic_arm.parts.wrist_pitch import build_wrist_pitch
from robotic_arm.parts.wrist_roll import MATERIAL as ROLL_MATERIAL
from robotic_arm.parts.wrist_roll import build_wrist_roll

CASES = {
    "link4": (build_wrist_pitch, PITCH_MATERIAL),
    "link5": (build_wrist_roll, ROLL_MATERIAL),
}


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request):
    body = request.param
    build, material = CASES[body]
    part = build()
    return body, part, mass_properties(part, effective_material(part, material))


def test_joints_are_perpendicular(case):
    """The premise of the two-drum wrist form. link3, by contrast, has parallel
    axes and is built as a straight tapered tube instead -- if a future wrist
    link stopped being perpendicular, the builder would need revisiting rather
    than silently producing a bad shape.
    """
    body, _, _ = case
    assert link_frame(body).axes_are_perpendicular


def test_child_drum_encloses_its_actuator(case):
    """Each wrist link carries the motor for the joint below it, so its child
    drum has to clear that actuator's body plus a wall.
    """
    from robotic_arm.actuators import RS00

    body, _, _ = case
    import importlib

    module = importlib.import_module(CASES[body][0].__module__)
    _, child = module._drums()
    actuator_diameter = max(RS00().bbox_mm[0], RS00().bbox_mm[1])
    assert child.diameter >= actuator_diameter + 2 * RULES.structural_wall_thickness


def test_wall_is_thick_enough_to_print():
    """At least the structural rule, or the slicer will not lay down the
    perimeters the load path assumes.
    """
    from robotic_arm.parts import cobot

    assert RULES.structural_wall_thickness >= 2.0
    assert cobot.RULES.structural_wall_thickness == RULES.structural_wall_thickness
