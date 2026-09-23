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


#: Worst approach per adjacent seam on the stock arm, mm. Negative because the
#: stock arm's own links interpenetrate at folded limits -- "do they overlap"
#: answers yes for the shipping product, so the test has to be parity.
STOCK_SEAM_BASELINE = {
    ("link2", "link3"): -16.2,
    ("link3", "link4"): -15.6,
    ("link4", "link5"): -7.6,
    ("link5", "link6"): -2.0,
}

#: How much worse than stock a printed seam may be, mm.
SEAM_REGRESSION_ALLOWANCE = 5.0


def test_adjacent_seams_are_checked_at_all(stock, twin):
    """Guards the blind spot itself.

    link2/link3, link3/link4 and link4/link5 are all in the upstream MJCF's
    contact exclusion list, so the clearance sweep can never report them --
    correct for simulating stock, useless for validating a new joint seam.
    This asserts the direct query sees them.
    """
    from robotic_arm.collision import adjacent_pair_clearance

    import numpy as np

    qpos = np.zeros(stock.nq)
    for pair in STOCK_SEAM_BASELINE:
        assert np.isfinite(adjacent_pair_clearance(twin, *pair, qpos))


@pytest.mark.xfail(
    reason=(
        "Known open defect. The printed seams are worse than the stock ones "
        "they replace: link2/link3 by 36 mm, link3/link4 by 32 mm, "
        "link4/link5 by 14 mm. Both arms interpenetrate at folded joint "
        "limits -- stock's link2/link3 by 16 mm -- so this is a parity "
        "regression, not overlap per se. The cause is that the printed drums "
        "are round and fatter at the seams than stock's flat beams, and the "
        "collision proxy cylinders are themselves conservative. Fixing it "
        "needs slimmer seam geometry or reduced joint limits, which is a "
        "design decision."
    ),
    strict=True,
)
def test_printed_seams_no_worse_than_stock(stock, twin):
    """Spec S1, stated per adjacent pair rather than over the whole model."""
    from robotic_arm.collision import adjacent_seam_parity

    worst = adjacent_seam_parity(stock, twin, samples=200)
    for pair, (stock_gap, twin_gap) in worst.items():
        assert twin_gap >= stock_gap - SEAM_REGRESSION_ALLOWANCE, (
            f"{pair[0]}/{pair[1]}: twin {twin_gap:.1f} mm vs stock "
            f"{stock_gap:.1f} mm"
        )
