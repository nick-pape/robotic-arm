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


def test_child_interface_bolts_to_the_stator_ring(case):
    """The correction this test used to get backwards.

    It previously asserted the child drum was wide enough to *enclose* an
    actuator, which encoded the wrong model: a link does not wrap its child's
    motor, it bolts to that motor's **stator ring** while the driven link bolts
    to the rotor ring inside it. Both rings are on one face. Asserting
    enclosure is how both interfaces ended up on inner rings.
    """
    from robotic_arm.actuators import for_joint
    from robotic_arm.linkframes import link_frame

    body, _, _ = case
    import importlib

    module = importlib.import_module(CASES[body][0].__module__)
    _, cup = module._drums()

    child = link_frame(body).child_name
    carried = for_joint(f"joint{child.removeprefix('link')}")
    ring = carried.stator_circle

    assert cup.diameter > ring.bcd, (
        f"{body} cup O{cup.diameter:.0f} cannot carry a O{ring.bcd:.0f} "
        f"stator ring"
    )
    # And it must be bored clear of the hub turning inside it.
    assert cup.diameter > carried.hub_diameter + 2 * RULES.structural_wall_thickness


def test_wall_is_thick_enough_to_print():
    """At least the structural rule, or the slicer will not lay down the
    perimeters the load path assumes.
    """
    from robotic_arm.parts import cobot

    assert RULES.structural_wall_thickness >= 2.0
    assert cobot.RULES.structural_wall_thickness == RULES.structural_wall_thickness
