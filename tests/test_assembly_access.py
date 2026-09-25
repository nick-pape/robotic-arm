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
#: A ceiling, not a record.
#:
#: **These numbers are measured against the *stock* motor positions**, and the
#: UR archetype deliberately re-mounts J2 the other way round (stock puts
#: link1 on the J2 rotor and makes link2 carry two stators; this design uses
#: the uniform UR convention -- see `robotic_arm.parts.urlink`). So anything
#: these report about `motor_2_3` and link1 or link2 is comparing this design
#: against a motor that is no longer mounted where the reference puts it. The
#: J4/J5/J6 figures are like-for-like; the J2 ones are not, and link1's 255 is
#: mostly that mismatch rather than a part fouling a motor.
KNOWN_MOTOR_INTERFERENCE = {
    "link1": 255,
    "link2": 28,
    "link3": 47,
    "link4": 47,
    "link5": 43,
    "link6": 7,
}


@pytest.mark.parametrize("body", sorted(KNOWN_MOTOR_INTERFERENCE))
def test_motor_interference_does_not_regress(body):
    """A ratchet on the open defect below.

    `test_printed_parts_clear_the_motors` is the goal and is still expected to
    fail. This keeps the gap from widening, and fails loudly if a number
    improves so the ceiling gets lowered rather than quietly beaten.
    """
    from robotic_arm.assembly import assembled_motor_interference

    found = assembled_motor_interference(body)
    inside = sum(v["inside"] for v in found.values())
    ceiling = KNOWN_MOTOR_INTERFERENCE[body]
    assert inside <= ceiling, (
        f"{body} motor interference regressed: {inside} > {ceiling}"
    )
    assert inside == ceiling, (
        f"{body} improved to {inside} (was {ceiling}); lower the ceiling"
    )


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
        "motor mesh at the zero pose. The motor-ownership question that used "
        "to block this is now settled and measured (`robotic_arm.mounts`): "
        "stock inverts J2, and this design does not. What remains is two "
        "different things the count cannot separate -- real interference at "
        "J4/J5/J6, where the housings enclose motors that are 51 mm deep and "
        "the barrels are not yet quite deep enough, and a bookkeeping "
        "mismatch at J2, where the reference motor is mounted on the opposite "
        "link from this design's. Splitting the two is the next step."
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
