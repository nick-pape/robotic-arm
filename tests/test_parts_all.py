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
    assert box.min.X - 1.0 <= child[0] <= box.max.X + 1.0
    assert box.min.Y - 1.0 <= child[1] <= box.max.Y + 1.0
    assert box.min.Z - 1.0 <= child[2] <= box.max.Z + 1.0


def test_stays_within_the_stock_envelope(part_case):
    """Growing past stock risks occupying space stock leaves open, which shows
    up later as a self-collision stock does not have.
    """
    body, solid, _ = part_case
    size = solid.bounding_box().size
    stock = link_frame(body).stock_extent
    for ours, theirs in zip((size.X, size.Y, size.Z), stock):
        # A proportional limit alone is too tight on a thin feature: link6 is
        # 12 mm against stock's 9.5, deliberately, to house M5 inserts. A few
        # millimetres on a small dimension is not the kind of growth that
        # causes collisions.
        assert ours <= theirs * 1.12 + 3.0, (
            f"{body}: {ours:.0f} mm against {theirs:.0f}"
        )


def test_inertia_is_physically_realisable(part_case):
    body, solid, props = part_case
    box = solid.bounding_box()
    com = props.com * 1000
    assert box.min.X <= com[0] <= box.max.X
    assert box.min.Y <= com[1] <= box.max.Y
    assert box.min.Z <= com[2] <= box.max.Z
    assert props.satisfies_triangle_inequality()


def test_parent_boss_is_not_an_actuator_housing(part_case):
    """A joint's motor mounts on its parent link, so a link's own parent drum
    has nothing to enclose. Sizing it to an actuator makes it fat enough to
    fill space stock leaves open -- the cause of the only S1 regression found.
    """
    body, _, _ = part_case
    import importlib

    module = importlib.import_module(REGISTRY[body][0].__module__)
    drums = getattr(module, "_drums", None)
    if drums is None:
        pytest.skip("part is not built from drums")
    parent, child = drums()
    assert parent.diameter < child.diameter * 0.9, (
        f"{body} parent boss O{parent.diameter:.0f} is not clearly smaller than "
        f"its O{child.diameter:.0f} actuator housing"
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


def test_parent_boss_sits_on_the_body_side_of_its_joint(part_case):
    """A boss must overlap the space its own stock part occupies.

    Parent axes on this arm are not consistently signed -- link2's points the
    opposite way to link3's, link4's and link5's -- so writing a boss offset
    against the raw axis places it on the wrong side for some links. That put
    link2's shoulder boss 31 mm clear of its own joint and left a visible gap
    in the render, while every number the suite checked stayed plausible.
    """
    body, _, _ = part_case
    if not _is_shell_built(body):
        pytest.skip("solid part has no parent boss")

    import importlib

    module = importlib.import_module(REGISTRY[body][0].__module__)
    parent, _ = module._drums()
    frame = link_frame(body)

    axis = frame.parent_axis
    boss_at = float(parent.centre @ axis)
    half = parent.length / 2
    stock_at = float(frame.stock_centre @ axis)
    stock_half = float(abs(frame.stock_extent @ axis)) / 2

    overlap = min(boss_at + half, stock_at + stock_half) - max(
        boss_at - half, stock_at - stock_half
    )
    assert overlap > 0.6 * parent.length, (
        f"{body} boss spans {boss_at - half:+.0f}..{boss_at + half:+.0f} along its "
        f"axis but stock spans {stock_at - stock_half:+.0f}..{stock_at + stock_half:+.0f}"
    )
