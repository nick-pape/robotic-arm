"""Which side of each joint holds the rotor -- pinned against stock geometry.

This table was assumed for the whole life of the project and never checked.
The assumption ("a link is driven by the joint below it, so its own end bolts
to that joint's rotor") is right four times out of five and wrong at J2, where
it puts a turning boss where the stock arm has a fixed ring.
"""

import pytest

from robotic_arm.mounts import child_joint_role, joint_roles, own_joint_role

#: Measured from the vendored MJCF at the zero pose.
EXPECTED = {
    "joint2": {"link1": "rotor", "link2": "stator"},
    "joint3": {"link3": "rotor", "link2": "stator"},
    "joint4": {"link4": "rotor", "link3": "stator"},
    "joint5": {"link5": "rotor", "link4": "stator"},
}


@pytest.mark.parametrize("joint", sorted(EXPECTED))
def test_measured_roles_match_the_stock_arm(joint):
    assert joint_roles()[joint] == EXPECTED[joint]


def test_j2_is_the_exception_and_link2_holds_two_stators():
    """The finding itself, stated so it cannot quietly revert.

    link2 carries the stators of *both* J2 and J3 -- which is why the vendored
    model ships one `motor_2_3` mesh belonging to link2, straddling two
    housings, with link1 and link3 each reaching in with a rotor hub. Stock
    link2 therefore has no driven boss at all.
    """
    assert own_joint_role("link2") == "stator"
    assert child_joint_role("link2") == "stator"
    assert [own_joint_role(f"link{i}") for i in (3, 4, 5)] == ["rotor"] * 3


def test_hub_and_bore_signatures_do_not_overlap():
    """The threshold is not doing fine judgement, and should not start to."""
    from robotic_arm.mounts import HUB_RADIUS_LIMIT

    assert 10.0 < HUB_RADIUS_LIMIT < 18.0


def test_every_joint_has_exactly_one_of_each_side():
    """A joint with two rotors or two stators would be a measurement fault."""
    for joint, roles in joint_roles().items():
        if joint not in EXPECTED:
            continue
        assert sorted(roles.values()) == ["rotor", "stator"], f"{joint}: {roles}"
