"""S1 -- self-collision clearance.

S1 as written asks for 3 mm of clearance through all joint ranges. Measured
against the stock model, that is not achievable and never was: a 6-DOF arm with
these joint limits can fold into itself, and a random sweep finds stock
penetrating by more than 50 mm. The real machine is simply never commanded into
those poses.

So S1 gets the same treatment as P2: stated as parity against stock. The
question that matters is not "can this arm self-collide" -- it can, and so can
the one you can buy -- but "does the printed clone collide in poses the stock
arm handles". That is a regression, and it is checkable.
"""

import pytest

from robotic_arm.collision import (
    MIN_CLEARANCE_MM,
    closest_approaches,
    compare_clearance,
)
from robotic_arm.mjcf import generate_twin, load
from robotic_arm.reference import load_baseline

#: Absorbs the difference in collision fidelity: the twin uses a few cylinders
#: per link where stock uses a convex decomposition, and a cylinder bulges
#: where the real shell is waisted.
PROXY_TOLERANCE_MM = 1.0


@pytest.fixture(scope="module")
def stock():
    return load_baseline()


@pytest.fixture(scope="module")
def twin(tmp_path_factory):
    return load(generate_twin(out=tmp_path_factory.mktemp("s1") / "model.xml"))


def test_stock_arm_also_fails_s1_as_literally_written(stock):
    """The evidence that S1 needs restating, kept as a test so the reasoning
    cannot quietly be lost.
    """
    approaches = closest_approaches(stock, samples=800)
    below = [a for a in approaches.values() if a.distance_mm < MIN_CLEARANCE_MM]
    assert below, "expected the stock arm to self-collide somewhere in joint space"
    assert min(a.distance_mm for a in below) < -10.0


def test_s1_no_clearance_regression_against_stock(stock, twin):
    """The requirement that actually means something.

    Zero poses where stock stays clear but the printed twin does not.
    """
    result = compare_clearance(
        stock, twin, samples=2000, tolerance_mm=PROXY_TOLERANCE_MM
    )
    assert not result.regressions, (
        f"{len(result.regressions)} of {result.samples} poses are clear on stock "
        f"but collide on the twin; worst twin clearance "
        f"{min(b for _, b, _ in result.regressions):+.2f} mm"
    )


def test_printed_parts_use_primitive_collision_geometry(twin):
    """MuJoCo treats a mesh geom as its convex hull, and these shells are
    L-shaped -- one hull round a two-drum wrist would fill the space between
    the drums and invent collisions. Primitives avoid that.
    """
    import mujoco

    for body in ("link4", "link5", "link6"):
        bid = mujoco.mj_name2id(twin, mujoco.mjtObj.mjOBJ_BODY, body)
        collision = [
            g
            for g in range(twin.ngeom)
            if twin.geom_bodyid[g] == bid and twin.geom_group[g] == 3
        ]
        assert collision, f"{body} has no collision geometry at all"
        assert all(
            twin.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER for g in collision
        ), f"{body} still carries stock mesh hulls"


def test_parent_drums_are_bosses_not_housings():
    """The error that caused the only real S1 regression found so far.

    A joint's motor is mounted on the *parent* link, so a link's own parent
    drum has nothing to enclose. Sizing it to an actuator makes it fat enough
    to fill a quadrant the stock arm leaves open.
    """
    from robotic_arm.parts.wrist_pitch import _drums as pitch_drums
    from robotic_arm.parts.wrist_roll import _drums as roll_drums

    for drums in (pitch_drums(), roll_drums()):
        parent, child = drums
        assert parent.diameter < child.diameter * 0.9, (
            f"parent drum {parent.diameter:.1f} mm is not clearly slimmer than "
            f"the actuator housing {child.diameter:.1f} mm"
        )
