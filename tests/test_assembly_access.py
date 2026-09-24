"""Can the printed structure actually be assembled?

A shell can be a valid solid, the right mass, inside its envelope, clear of
its neighbours, and still impossible to build. Nothing else in this suite asks
whether a hex key can reach a screw -- and when this check was first written,
four of six parent-mount bolts on link2 and link3 could not be reached from
either direction.
"""

import pytest

from robotic_arm.assembly import assembly_order, mount_rings, reachable_directions
from robotic_arm.parts import REGISTRY


@pytest.mark.parametrize("body", sorted(b for b in REGISTRY if mount_rings(b)))
def test_every_mount_ring_can_be_reached(body):
    """Each mount must have at least one working approach direction.

    Both is better, but one is enough: the assembly order fixes which way a
    part is offered up, so a ring reachable from one side is fastenable.
    """
    for label, directions in reachable_directions(body).items():
        assert directions, (
            f"{body} {label} mount cannot be reached by a driver from either "
            f"direction; the part cannot be fastened"
        )


def test_assembly_order_is_parent_before_child():
    """Every link bolts to the actuator its parent carries, so the parent has
    to be on the bench first.
    """
    order = assembly_order()
    assert order == sorted(order), "expected proximal-to-distal order"
    assert order[0] == "link2"


def test_driver_needs_more_room_than_its_screw():
    """A check that passed a screw-sized corridor would prove nothing."""
    from robotic_arm.assembly import DRIVER_DIAMETER
    from robotic_arm.design import RULES

    assert DRIVER_DIAMETER["M3"] > RULES.m3_clearance * 1.5
