"""M7 -- thermal and current limits.

These tests exist to keep the M2 torque result from being read as "the stock
arm does not work". It does work: it is a shipping product. What the numbers
say is that extended poses are *time-limited* by heat rather than impossible,
which is what Seeed's own workspace warning and the firmware's temperature
gating are both about.
"""

import math

import pytest

from robotic_arm.actuators import RS00, RS06
from robotic_arm.thermal import (
    EMERGENCY_DISABLE_C,
    SLOW_RETURN_C,
    WARN_C,
    copper_loss_w,
    current_apk,
    current_arms,
    sustainability,
    within_peak_current,
    within_rated_current,
)


def test_torque_constant_reproduces_published_rated_torque():
    """The strongest available check on the winding model: Kt and the rated
    current must independently reproduce the published rated torque.
    """
    rs06 = RS06()
    implied = rs06.rated_current_apk / math.sqrt(2) * 1.09
    assert implied == pytest.approx(rs06.rated_nm, rel=0.01)


def test_copper_loss_at_rated_matches_the_spec_estimate():
    """The spec computes ~35 W at 11 N*m. Independent agreement is worth
    keeping, since it validates the loss model rather than just the arithmetic.
    """
    assert copper_loss_w(RS06(), 11.0) == pytest.approx(35.0, rel=0.1)


def test_holding_full_extension_is_time_limited_not_impossible():
    """The correct reading of the M2 result.

    15.36 N*m is above the continuous current rating but far below peak, so
    the pose is reachable and holdable -- for a while, bounded by heat.
    """
    rs06 = RS06()
    tau = 15.36
    assert not within_rated_current(rs06, tau)
    assert within_peak_current(rs06, tau)
    assert sustainability(rs06, tau) == "time-limited"
    assert current_apk(rs06, tau) < rs06.peak_current_apk / 2


def test_balanced_residual_is_genuinely_continuous():
    """What the balancer buys, in the units that actually bind: the balanced
    residual draws well under rated current, so it is thermally sustainable.
    """
    rs06 = RS06()
    assert within_rated_current(rs06, 7.13)
    assert sustainability(rs06, 7.13) == "continuous"
    assert copper_loss_w(rs06, 7.13) < copper_loss_w(rs06, 15.36) / 4


def test_loss_scales_with_torque_squared():
    """I^2R: doubling torque quadruples the heat. This is why a modest torque
    reduction from the balancer buys a large thermal margin.
    """
    rs06 = RS06()
    assert copper_loss_w(rs06, 14.0) == pytest.approx(
        4 * copper_loss_w(rs06, 7.0), rel=1e-9
    )


def test_excessive_torque_is_infeasible():
    rs06 = RS06()
    assert sustainability(rs06, 60.0) == "infeasible"


def test_firmware_thresholds_are_ordered():
    """Stock gates at 80 warn / 100 slow-return / 140 emergency disable."""
    assert WARN_C < SLOW_RETURN_C < EMERGENCY_DISABLE_C


def test_rs00_has_its_own_winding():
    """The wrist actuator is a different motor; its limits must not silently
    inherit RS06's.
    """
    assert current_arms(RS00(), 5.0) != current_arms(RS06(), 5.0)


def test_unknown_winding_is_rejected():
    from dataclasses import replace

    from robotic_arm import thermal

    with pytest.raises(KeyError, match="no winding data"):
        thermal.winding(replace(RS06(), name="RS99"))
