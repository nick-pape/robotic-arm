"""Checks that apply to every printed part, whatever its shape.

Parametrised over the registry, so a new part inherits them the moment it is
registered rather than needing its own copy. Shape-specific expectations live
in the per-part test modules.
"""

import pytest

from robotic_arm.linkframes import link_frame, stock_mass
from robotic_arm.massprops import mass_properties
from robotic_arm.parts import REGISTRY, effective_material


def _is_shell_built(body: str) -> bool:
    """Whether the part is a hollow shell rather than a solid."""
    import importlib

    module = importlib.import_module(REGISTRY[body][0].__module__)
    return hasattr(module, "_drums")


@pytest.fixture(scope="module", params=sorted(REGISTRY))
def part_case(request):
    body = request.param
    build, material = REGISTRY[body]
    solid = build()
    return body, solid, mass_properties(solid, effective_material(solid, material))


def test_is_one_valid_solid(part_case):
    """Two solids means a union silently failed to fuse -- which happened on
    link3, where the tube only touched the parent boss tangentially.
    """
    body, solid, _ = part_case
    assert len(solid.solids()) == 1, f"{body} is {len(solid.solids())} disconnected pieces"
    assert solid.is_valid


def test_p1_beats_the_stock_part(part_case):
    body, _, props = part_case
    assert props.mass < stock_mass(body)


def test_is_hollow(part_case):
    """A shell that silently filled would pass a mass budget on the wrong
    physics and print for hours.

    Only applies to shell-built parts. A tool flange is a 12 mm disc and is
    solid on purpose: hollowing it would cost more stiffness than the grams it
    saves.
    """
    body, solid, _ = part_case
    if not _is_shell_built(body):
        pytest.skip("solid part by design, not a shell")
    size = solid.bounding_box().size
    assert solid.volume < 0.35 * (size.X * size.Y * size.Z)


def test_reaches_its_child_joint(part_case):
    """The child joint must fall inside the part, or the next link has nothing
    to bolt to.
    """
    body, solid, _ = part_case
    frame = link_frame(body)
    if frame.child_name is None:
        pytest.skip("tip link has no child joint")
    # Some children are separate assemblies mounted well clear of the link --
    # the gripper sits 166 mm off link6's flange. Stock does not span that
    # either, so requiring our part to is the wrong test.
    if frame.span > float(max(link_frame(body).stock_extent)):
        pytest.skip(f"{frame.child_name} is a separate assembly, not spanned by stock")
    box = solid.bounding_box()
    child = frame.child_origin
    # The housing mouth is deliberately recessed from the joint plane by the
    # rotor's hub protrusion plus a seam gap, so the part stops a couple of
    # millimetres short of the joint origin by design.
    from robotic_arm.parts.urlink import SEAM_GAP

    slack = 1.0 + SEAM_GAP + 2.0
    assert box.min.X - slack <= child[0] <= box.max.X + slack
    assert box.min.Y - slack <= child[1] <= box.max.Y + slack
    assert box.min.Z - slack <= child[2] <= box.max.Z + slack


def test_stays_within_the_stock_envelope(part_case):
    """Growing past stock risks occupying space stock leaves open, which shows
    up later as a self-collision stock does not have.

    The bound is not stock's own extent, because stock keeps its links slim by
    leaving the motors in open air between two fork plates. A cobot encloses
    them, and an RS06 is O87 across its stator flange, so no housing that
    actually contains one fits inside stock's O67 rings. Enclosing the motors
    was the goal, so the floor here is what the motors demand and the
    allowance above it is one housing radius -- enough for a barrel to stand
    proud of where stock's bare motor already was, not enough to hide a part
    that has quietly doubled.

    This is a cheap proxy either way. The question it stands in for is asked
    directly, over the whole joint range, in `test_s1_clearance.py`.
    """
    import importlib

    from robotic_arm.actuators import for_joint
    from robotic_arm.parts import carried_actuator
    from robotic_arm.parts.urlink import housing_diameter

    body, solid, _ = part_case
    size = solid.bounding_box().size
    stock = link_frame(body).stock_extent

    module = importlib.import_module(REGISTRY[body][0].__module__)
    floor = 0.0
    if hasattr(module, "FLANGE_LENGTH"):
        floor = housing_diameter(for_joint(f"joint{body.removeprefix('link')}"))
    carried = carried_actuator(body)
    if carried is not None and (hasattr(module, "FLANGE_LENGTH") or body == "base_link"):
        floor = max(floor, housing_diameter(carried[0]))

    for ours, theirs in zip((size.X, size.Y, size.Z), stock):
        # A proportional limit alone is too tight on a thin feature: link6 is
        # 12 mm against stock's 9.5, deliberately, to house M5 inserts.
        allowed = max(float(theirs) * 1.12 + 3.0, max(float(theirs), floor) + floor / 2)
        assert ours <= allowed, (
            f"{body}: {ours:.0f} mm against stock {theirs:.0f} "
            f"(allowed {allowed:.0f}, motor floor {floor:.0f})"
        )


def test_inertia_is_physically_realisable(part_case):
    body, solid, props = part_case
    box = solid.bounding_box()
    com = props.com * 1000
    assert box.min.X <= com[0] <= box.max.X
    assert box.min.Y <= com[1] <= box.max.Y
    assert box.min.Z <= com[2] <= box.max.Z
    assert props.satisfies_triangle_inequality()


def test_rotor_flange_matches_the_housing_it_caps(part_case):
    """The inverse of what this test used to assert.

    It previously required a link's own end to be *clearly smaller* than its
    child housing, on the reasoning that "a link's own parent drum has nothing
    to enclose". That was true of the old design and is the opposite of a
    cobot: the rotor flange caps the parent's housing, so it matches that
    housing's diameter exactly and the joint reads as one continuous cylinder.
    Sizing it small is what made the arm look like brackets beside motors.
    """
    from robotic_arm.actuators import for_joint
    from robotic_arm.parts.urlink import housing_diameter

    body, _, _ = part_case
    import importlib

    module = importlib.import_module(REGISTRY[body][0].__module__)
    drums = getattr(module, "_drums", None)
    if drums is None:
        pytest.skip("part is not built from a flange and a housing")
    flange, _ = drums()
    index = int(body.removeprefix("link"))
    expected = housing_diameter(for_joint(f"joint{index}"))
    assert flange.diameter == pytest.approx(expected, abs=0.01), (
        f"{body} flange O{flange.diameter:.0f} does not match the O{expected:.0f} "
        f"housing it caps"
    )


def test_flange_and_housing_sit_on_opposite_sides_of_their_joints(part_case):
    """A link's two ends must reach in opposite directions from their own
    joint planes, or one buries itself in the part it is supposed to meet.

    Deriving each end's direction independently put link3's and link4's
    flanges 14 mm inside the housings they capped, and the assembled render
    could not show it -- from outside, a buried flange and a missing flange
    look identical.
    """
    import mujoco
    import numpy as np

    from robotic_arm.parts.urlink import designed_motor_side, link_ends
    from robotic_arm.reference import load_baseline

    body, _, _ = part_case
    import importlib

    module = importlib.import_module(REGISTRY[body][0].__module__)
    if not hasattr(module, "FLANGE_LENGTH"):
        pytest.skip("part is not built from a flange and a housing")
    flange, housing, _, _ = link_ends(body, module.FLANGE_LENGTH)
    if housing is None:
        pytest.skip("tip link carries no motor")

    model = load_baseline()
    data = mujoco.MjData(model)
    data.qpos[:] = 0
    mujoco.mj_forward(model, data)
    rotation = data.xmat[model.body(body).id].reshape(3, 3)

    index = int(body.removeprefix("link"))
    own = designed_motor_side(f"joint{index}")
    flange_world = rotation @ (
        np.asarray(flange.axis, float) / np.linalg.norm(flange.axis)
    )
    assert float(flange_world @ own) < 0, (
        f"{body}'s flange reaches toward its own motor instead of away from it"
    )


def test_whole_printed_assembly_saves_mass():
    """Mass removed distal of J2 is worth more than anywhere else: it rides the
    longest moment arm from the joint that actually binds.
    """
    total_new = 0.0
    for body, (build, material) in REGISTRY.items():
        solid = build()
        total_new += mass_properties(solid, effective_material(solid, material)).mass
    total_stock = sum(stock_mass(body) for body in REGISTRY)
    assert total_new < total_stock * 0.4


# `test_parent_boss_sits_on_the_body_side_of_its_joint` lived here. It required
# a link's own end to overlap the space its stock part occupies, which was the
# right check while that end was a boss reaching into the link. Under the UR
# archetype the rotor flange deliberately reaches the other way -- out across
# the joint plane, to cap the parent's housing -- so the old assertion is not
# stale but backwards. What it guarded against, an end placed on the wrong
# side of its own joint, is now covered by
# `test_flange_and_housing_sit_on_opposite_sides_of_their_joints`, which tests
# the relationship that matters rather than a proxy for it.
