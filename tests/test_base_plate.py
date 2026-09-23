"""Dimensional checks on the base plate.

These guard the things a render will not catch: that holes are actually
through-holes, and that the outline matches the parameters.
"""

import pytest

from robotic_arm.params import BASE, SERVO
from robotic_arm.parts import build_base_plate


@pytest.fixture(scope="module")
def plate():
    return build_base_plate()


def test_outline_matches_params(plate):
    size = plate.bounding_box().size
    assert size.X == pytest.approx(BASE.length, abs=0.01)
    assert size.Y == pytest.approx(BASE.width, abs=0.01)
    assert size.Z == pytest.approx(BASE.thickness, abs=0.01)


def test_is_a_single_solid(plate):
    assert len(plate.solids()) == 1


def test_holes_are_subtracted(plate):
    """Volume must be less than the solid slab by roughly the hole volumes."""
    slab = BASE.length * BASE.width * BASE.thickness
    holes = (
        4 * 3.14159 * (BASE.mount_hole_dia / 2) ** 2 * BASE.thickness
        + 3.14159 * (SERVO.horn_dia / 2) ** 2 * BASE.thickness
    )
    # Corner rounding and the top fillet remove a little more material.
    assert plate.volume < slab - holes * 0.95
