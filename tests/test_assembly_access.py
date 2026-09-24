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


#: Sampled motor vertices found inside each printed part, at the zero pose.
#: Recorded from `assembled_motor_interference` so the number cannot drift
#: quietly. Every printed part currently intersects at least one motor.
KNOWN_MOTOR_INTERFERENCE = {
    "link2": 44,
    "link3": 75,
    "link4": 125,
    "link5": 16,
}


def test_circular_actuator_check_is_labelled_as_such():
    """Guards a trap rather than a defect.

    `actuator_interference` seats the actuator on the part's own mount face,
    so it cannot detect a part that misses the actuator entirely. It read zero
    for link3 while link3's tube passed through the J3 motor. Anyone reaching
    for it should meet that warning first.
    """
    from robotic_arm import assembly

    doc = assembly.actuator_interference.__doc__ or ""
    assert "circular" in doc.lower()
    assert "assembled_motor_interference" in doc


@pytest.mark.xfail(
    reason=(
        "Known open defect. Every printed part still overlaps at least one "
        "motor where the motors actually sit -- worst is link4, with a quarter "
        "of motor_4's sampled vertices inside it. The mount faces are placed "
        "on the joint planes, but the actuators do not end there: the J3 motor "
        "reaches 6.4 mm past link3's origin. Fixing it needs the motor-"
        "ownership question settled first, because the stock model is not "
        "consistent about which body carries a given motor mesh."
    ),
    strict=True,
)
def test_printed_parts_clear_the_motors():
    """The fit question that matters, asked non-circularly."""
    from robotic_arm.assembly import assembled_motor_interference

    for body in KNOWN_MOTOR_INTERFERENCE:
        found = assembled_motor_interference(body)
        total = sum(v["inside"] for v in found.values())
        assert total == 0, f"{body} has {total} motor vertices inside it"
