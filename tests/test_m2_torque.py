"""M2 -- the torque budget, recomputed from real inertials.

The spec's load model was built on an estimated mass distribution and flags
itself as such. These tests pin what the actual inertials say, because the
answer turned out to differ materially from the estimate.
"""

import numpy as np
import pytest

from robotic_arm.torque import RS06, payload_torque_j2, worst_case_j2


@pytest.fixture(scope="module")
def worst(baseline):
    # Coarser than the module default; the optimum is broad and this keeps the
    # suite fast without moving the answer appreciably.
    return worst_case_j2(baseline, samples=41)


def test_j2_is_the_binding_joint(worst):
    """The spec's central claim: J2 carries the most, so it sets the budget."""
    assert worst.torques[1] == worst.torques.max()


def test_worst_case_is_near_full_reach(worst):
    """Sanity: the worst pose should be the arm extended, ~754 mm nominal."""
    assert 0.70 < worst.reach < 0.80


def test_self_weight_exceeds_spec_estimate(worst):
    """The spec estimated 10.7 N*m of J2 self-weight from an assumed 3.3 kg
    distal mass. The real inertials give substantially more, because the true
    distal mass is ~4.37 kg. Recorded so the gap cannot quietly reappear.
    """
    assert worst.torques[1] > 14.0
    assert worst.torques[1] / 10.7 > 1.3


def test_self_weight_alone_exceeds_rated_torque(worst):
    """Unbalanced, the arm cannot hold itself at full reach continuously --
    before any payload. This is what forces the J2 gravity balancer.
    """
    assert worst.torques[1] > RS06.rated_nm
    assert worst.torques[1] > RS06.derated_nm()
    assert worst.torques[1] < RS06.peak_nm, "still inside peak, so briefly reachable"


def test_five_kg_payload_is_infeasible(worst):
    """Seeed advertise 5 kg max. At full reach that exceeds even peak torque."""
    assert payload_torque_j2(worst, 5.0) > RS06.peak_nm


def test_rated_payload_needs_peak_torque(worst):
    """The 2.5 kg rated payload is reachable only intermittently at full
    reach, consistent with the spec's reading of it as a 70%-workspace figure.
    """
    tau = payload_torque_j2(worst, 2.5)
    assert RS06.derated_nm() < tau < RS06.peak_nm


@pytest.mark.xfail(
    reason="P2 requires the J2 gravity balancer, designed in M6",
    strict=True,
)
def test_p2_j2_within_derated_limit(worst):
    """Spec requirement P2: J2 torque at full reach, minus spring, within 0.7x
    derated rated torque. Currently fails by design -- no balancer exists yet.
    Flips to passing when M6 lands, which is the point.
    """
    assert payload_torque_j2(worst, 1.0) <= RS06.derated_nm()
