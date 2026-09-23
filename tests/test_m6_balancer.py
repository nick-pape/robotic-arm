"""M6 -- the J2 gravity balancer, and spec requirement P2.

M2 showed the shoulder needs 15.36 N*m to hold the arm's own weight at full
reach. Against the *printed clone's* ~7.7 N*m continuous budget -- derated
because a printed link has no aluminium heat path -- the balancer is what makes
that pose sustainable. The stock arm, which keeps the metal heat path and gates
on temperature in firmware, manages the same pose as time-limited instead.

The same spring model feeds both the MJCF tendon and the torque analysis, via
`static_torques`, so the simulation and the analysis cannot disagree about it.
"""

import numpy as np
import pytest

from robotic_arm.actuators import RS06
from robotic_arm.balancer import (
    ANCHOR_SITE,
    ARM_SITE,
    TENDON_NAME,
    Balancer,
    max_continuous_payload,
    residual_torque,
)
from robotic_arm.mjcf import frame_differences, generate, load
from robotic_arm.reference import load_baseline

#: Found by sweeping cancellation against worst-case residual; see
#: `optimise_cancellation`. Notably this is the figure the spec recommended,
#: which its own (wrong) self-weight estimate did not actually support.
OPTIMAL_CANCEL_NM = 8.25


@pytest.fixture(scope="module")
def balanced(tmp_path_factory):
    out = tmp_path_factory.mktemp("bal") / "model.xml"
    return load(generate(out=out, balancer=Balancer.sized_for(OPTIMAL_CANCEL_NM)))


@pytest.fixture(scope="module")
def unbalanced(tmp_path_factory):
    out = tmp_path_factory.mktemp("plain") / "model.xml"
    return load(generate(out=out))


def test_exact_balance_condition():
    """k*a*b = target is the whole sizing rule; check the algebra round-trips."""
    b = Balancer.sized_for(8.25, anchor_height=0.06, arm_offset=0.10)
    assert b.cancel_nm == pytest.approx(8.25, rel=1e-12)
    assert b.stiffness == pytest.approx(8.25 / (0.06 * 0.10), rel=1e-12)


def test_spring_through_the_pivot_is_rejected():
    """A zero moment arm produces no torque; that is a design error, not a
    degenerate-but-valid spring.
    """
    with pytest.raises(ValueError, match="produces no torque"):
        Balancer.sized_for(8.0, anchor_height=0.0)


def test_peak_force_scales_with_cancellation():
    """The anchor load is what forces a metal anchor, so it must be visible."""
    small = Balancer.sized_for(8.0)
    large = Balancer.sized_for(16.0)
    assert large.peak_force == pytest.approx(2 * small.peak_force, rel=1e-12)
    assert small.peak_force > 100, "expected a structurally significant load"


def test_balancer_appears_in_the_model(balanced):
    """Sites and tendon must survive the XML round-trip."""
    import mujoco

    assert balanced.ntendon >= 1
    names = [
        mujoco.mj_id2name(balanced, mujoco.mjtObj.mjOBJ_TENDON, i)
        for i in range(balanced.ntendon)
    ]
    assert TENDON_NAME in names
    sites = [
        mujoco.mj_id2name(balanced, mujoco.mjtObj.mjOBJ_SITE, i)
        for i in range(balanced.nsite)
    ]
    assert ANCHOR_SITE in sites and ARM_SITE in sites


def test_spring_is_zero_free_length(balanced):
    """springlength 0 is what makes F = k*L and exact balance possible."""
    assert np.allclose(balanced.tendon_lengthspring, 0.0)


def test_balancer_does_not_move_any_frame(balanced):
    """It adds sites and a tendon, but must not disturb the kinematics."""
    assert frame_differences(balanced, load_baseline()) == []


def test_balancer_substantially_reduces_j2_torque(balanced, unbalanced):
    """The headline: worst-case self-weight torque roughly halves."""
    before, _ = residual_torque(unbalanced, samples=31)
    after, _ = residual_torque(balanced, samples=31)
    assert before > 15.0
    assert after < before * 0.55


def test_self_weight_comes_within_the_continuous_budget(balanced):
    """Unbalanced, the arm cannot hold itself continuously at reach. Balanced,
    it can -- which is the balancer's actual job.
    """
    after, _ = residual_torque(balanced, samples=41)
    assert after < RS06().derated_nm()


def test_over_springing_makes_the_worst_case_worse(tmp_path):
    """Cancelling more is not better. At folded poses the gravity moment is
    small, so an oversized spring drives the joint the other way. This is why
    the sizing is an optimisation rather than "cancel the peak".
    """
    optimal = load(
        generate(
            out=tmp_path / "opt.xml",
            balancer=Balancer.sized_for(OPTIMAL_CANCEL_NM),
        )
    )
    excessive = load(
        generate(out=tmp_path / "big.xml", balancer=Balancer.sized_for(15.36))
    )
    best, _ = residual_torque(optimal, samples=31)
    worse, _ = residual_torque(excessive, samples=31)
    assert worse > best


def test_balancer_improves_intermittent_payload(balanced, unbalanced):
    """Within peak torque, the balancer buys real capability."""
    rs06 = RS06()
    before = max_continuous_payload(unbalanced, rs06.peak_nm, samples=21)
    after = max_continuous_payload(balanced, rs06.peak_nm, samples=21)
    assert after > before + 0.5
