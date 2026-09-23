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


def test_is_a_single_closed_solid(case):
    """A failed boolean would leave loose bodies or an open shell behind."""
    _, part, _ = case
    assert len(part.solids()) == 1
    assert part.is_valid


def test_p1_lighter_than_the_part_it_replaces(case):
    """Spec requirement P1, per part."""
    body, _, props = case
    assert props.mass < stock_mass(body), (
        f"{body} printed at {props.mass * 1000:.0f} g against "
        f"{stock_mass(body) * 1000:.0f} g stock"
    )


def test_is_actually_hollow(case):
    """A shell that silently filled in would pass a mass budget on the wrong
    physics and print for hours. Compare against the solid envelope.
    """
    _, part, _ = case
    bb = part.bounding_box().size
    envelope = bb.X * bb.Y * bb.Z
    assert part.volume < 0.35 * envelope, "shell looks solid, not hollow"


def test_spans_the_joint_it_bridges(case):
    """The child joint must fall inside the part, or the next link has nothing
    to bolt to.
    """
    body, part, _ = case
    frame = link_frame(body)
    bb = part.bounding_box()
    child = frame.child_origin
    for axis, lo, hi in (
        (child[0], bb.min.X, bb.max.X),
        (child[1], bb.min.Y, bb.max.Y),
        (child[2], bb.min.Z, bb.max.Z),
    ):
        assert lo - 1.0 <= axis <= hi + 1.0


def test_envelope_is_close_to_stock(case):
    """Growing well beyond the stock envelope would foul the neighbouring
    links through the joint ranges, which S1 would catch much later.
    """
    body, part, _ = case
    bb = part.bounding_box().size
    stock = link_frame(body).stock_extent
    for ours, theirs in zip((bb.X, bb.Y, bb.Z), stock):
        assert ours <= theirs * 1.12, f"{body} is {ours:.0f} mm against {theirs:.0f}"


def test_wall_is_thick_enough_to_print(case):
    """The shell wall must be at least the structural rule, or the slicer will
    not lay down the perimeters the load path assumes.
    """
    from robotic_arm.parts import cobot

    assert RULES.structural_wall_thickness >= 2.0
    # The builder defaults to the rule, so this guards the rule itself being
    # weakened rather than a per-part override.
    assert cobot.RULES.structural_wall_thickness == RULES.structural_wall_thickness


def test_joints_are_perpendicular(case):
    """The premise of the two-drum form. If a future link breaks it, the
    builder needs revisiting rather than silently producing a bad shape.
    """
    body, _, _ = case
    assert link_frame(body).axes_are_perpendicular


def test_centre_of_mass_is_inside_the_part(case):
    """A COM outside the bounding box means the inertia is nonsense."""
    _, part, props = case
    bb = part.bounding_box()
    com_mm = props.com * 1000
    assert bb.min.X <= com_mm[0] <= bb.max.X
    assert bb.min.Y <= com_mm[1] <= bb.max.Y
    assert bb.min.Z <= com_mm[2] <= bb.max.Z
    assert props.satisfies_triangle_inequality()


def test_whole_wrist_saves_meaningful_mass():
    """The wrist is the worst place to carry mass -- it sits on the longest
    moment arm from J2, so savings here are worth more than anywhere else.
    """
    total_new = sum(
        mass_properties(build(), effective_material(build(), material)).mass
        for build, material in CASES.values()
    )
    total_stock = sum(stock_mass(body) for body in CASES)
    assert total_new < total_stock * 0.5
