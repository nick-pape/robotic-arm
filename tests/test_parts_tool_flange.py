"""The tool flange (link6), the first printed part.

Most of these are interface checks rather than shape checks. Two successive
revisions of this part had features overlapping each other -- first the tool
inserts into the motor holes, then the inserts into the spigot recess -- and
neither was visible in the numbers, only in a render. `interface_clearance()`
turns that class of error into a test.
"""

import pytest

from robotic_arm.actuators import RS00
from robotic_arm.design import RULES
from robotic_arm.parts import effective_material
from robotic_arm.parts.tool_flange import (
    BORE_DIAMETER,
    MATERIAL,
    OUTER_DIAMETER,
    THICKNESS,
    TOOL_BOLT_COUNT,
    build_tool_flange,
    interface_clearance,
)

#: Stock link6 mass, from the URDF inertials.
STOCK_MASS_KG = 0.100


@pytest.fixture(scope="module")
def flange():
    return build_tool_flange()


@pytest.fixture(scope="module")
def props(flange):
    from robotic_arm.massprops import mass_properties

    return mass_properties(flange, effective_material(flange, MATERIAL))


def test_is_a_single_solid(flange):
    """A boolean that silently failed would leave loose bodies behind."""
    assert len(flange.solids()) == 1


def test_holds_the_stock_diameter(flange):
    """The gripper mates on this face, so the envelope is not ours to change."""
    size = flange.bounding_box().size
    assert size.X == pytest.approx(OUTER_DIAMETER, abs=0.01)
    assert size.Y == pytest.approx(OUTER_DIAMETER, abs=0.01)
    assert size.Z == pytest.approx(THICKNESS, abs=0.01)


def test_p1_lighter_than_the_part_it_replaces(props):
    """Spec requirement P1, for this part."""
    assert props.mass < STOCK_MASS_KG, (
        f"printed flange is {props.mass * 1000:.1f} g against a "
        f"{STOCK_MASS_KG * 1000:.0f} g stock part"
    )


def test_every_interface_clears(flange):
    """No two features may intersect, and none may leave a sliver of wall.

    1 mm is roughly three perimeters at a 0.4 nozzle -- below that a wall is
    not reliably printable.
    """
    for name, gap in interface_clearance().items():
        assert gap >= 1.0, f"{name} is only {gap:.2f} mm"


def test_motor_pattern_follows_the_measured_actuator():
    """The flange bolts to the RS00 output, so its pattern must come from the
    measured geometry rather than from a number typed in here.
    """
    from robotic_arm.parts.tool_flange import _motor_mount_circle

    circle = _motor_mount_circle()
    assert circle in RS00().bolt_circles
    assert circle.count == 6
    assert circle.bcd == pytest.approx(27.0, abs=0.01)


def test_bore_clears_a_connector(flange):
    """The wrist carries gripper power and CAN through the hollow shaft, so the
    bore has to pass a connector, not just loose wires.
    """
    assert BORE_DIAMETER >= 16.0


def test_hole_count_is_what_was_designed(flange):
    """Catches a boolean that dropped features without changing the envelope."""
    recognise_holes = pytest.importorskip(
        "quiddity", reason="feature recognition lives in the optional cad extra"
    ).recognise_holes

    holes = recognise_holes(flange)
    # 6 counterbored motor holes + 4 tool inserts, plus the central bore.
    assert len(holes) >= 6 + TOOL_BOLT_COUNT


def test_counterbore_leaves_a_solid_floor():
    """The motor counterbores must not break through into nothing."""
    gaps = interface_clearance()
    assert gaps["remaining_floor_under_counterbore"] > 2 * RULES.structural_wall_thickness
