"""P2 -- regression guards against the stock arm.

The point of these requirements is not "is this arm good enough" in the
abstract. The clone reuses stock actuators, kinematics and software, so the
only question that matters is whether swapping metal for plastic cost us
anything. P2a and P2b are therefore stated as parity against the stock model,
the same way P1 states mass.

Stating them as parity also makes them self-calibrating: re-measure the stock
baseline and the targets move with it, instead of going stale against a number
someone wrote down once.

P2c is the one deliberate absolute, because it covers the one place the clone
is worse off by construction -- a printed link has no aluminium heat path, so
the clone needs a balancer to hold itself where stock does not.

Until printed parts exist the clone *is* the stock model, so P2a and P2b pass
trivially. That is the intended behaviour: they are tripwires armed ahead of
the change they guard, not measurements of work already done.
"""

import pytest

from robotic_arm.actuators import RS06
from robotic_arm.balancer import DEFAULT_CANCEL_NM, Balancer, residual_torque
from robotic_arm.mjcf import generate, generate_twin, load
from robotic_arm.torque import max_moment_arm, max_payload, worst_case_j2

#: How much worse than stock the clone may be before it counts as a regression.
TORQUE_REGRESSION_LIMIT = 1.05  # P2a: at most 5% more J2 torque
CAPABILITY_RETENTION = 0.90  # P2b: at least 90% of stock payload capacity

#: Seeed qualify their payload figures as holding within about 70% of the
#: workspace, measured on the J2 moment arm rather than on tool reach.
WORKING_ENVELOPE = 0.70


@pytest.fixture(scope="module")
def stock(tmp_path_factory):
    return load(generate(out=tmp_path_factory.mktemp("stock") / "model.xml"))


@pytest.fixture(scope="module")
def clone(tmp_path_factory):
    """The clone as actually built: printed parts plus the J2 balancer.

    This used `generate()` rather than `generate_twin()`, dating from when no
    parts existed. Five were registered afterwards and the fixture was never
    updated, so every P2 check here ran against the stock model and passed
    without touching the printed design at all -- the exact regression these
    tests exist to catch.
    """
    return load(
        generate_twin(
            out=tmp_path_factory.mktemp("clone") / "model.xml",
            balancer=Balancer.sized_for(DEFAULT_CANCEL_NM),
            visuals=False,
        )
    )


@pytest.fixture(scope="module")
def envelope(stock):
    return WORKING_ENVELOPE * max_moment_arm(stock)


def test_p2a_torque_parity(stock, clone):
    """The clone must not load J2 appreciably harder than stock does.

    Compared on gravity torque alone, without the balancer's help, so a heavier
    printed link cannot hide behind a stronger spring.
    """
    stock_worst = worst_case_j2(stock, samples=31).torques[1]
    clone_worst = worst_case_j2(clone, samples=31).torques[1]
    assert clone_worst <= stock_worst * TORQUE_REGRESSION_LIMIT, (
        f"clone loads J2 at {clone_worst:.2f} N*m vs stock {stock_worst:.2f}, "
        f"a {clone_worst / stock_worst - 1:.1%} regression"
    )


def test_p2b_capability_parity(stock, clone, envelope):
    """The clone must retain most of stock's payload capacity.

    This is the requirement a user would actually notice, so it is stated on
    payload rather than on torque.
    """
    stock_capacity = max_payload(stock, RS06().peak_nm, max_arm=envelope, samples=31)
    clone_capacity = max_payload(clone, RS06().peak_nm, max_arm=envelope, samples=31)
    assert clone_capacity >= stock_capacity * CAPABILITY_RETENTION, (
        f"clone carries {clone_capacity:.2f} kg vs stock {stock_capacity:.2f}, "
        f"retaining only {clone_capacity / stock_capacity:.0%}"
    )


def test_p2c_thermal_floor(clone):
    """The one absolute: the balanced clone must hold its own weight.

    Not parity, because stock does not need to pass it -- stock keeps the
    aluminium heat path that makes its rated torque real. The clone loses that,
    so it has to buy the margin back with the balancer.
    """
    worst, _ = residual_torque(clone, samples=41)
    budget = RS06().derated_nm()
    assert worst <= budget, (
        f"balanced self-weight {worst:.2f} N*m exceeds the {budget:.2f} N*m "
        f"continuous budget; the arm cannot hold itself indefinitely"
    )


def test_stock_baseline_matches_recorded_reference(stock, envelope):
    """Pin the numbers P2a and P2b calibrate against.

    If the stock baseline moves, the parity targets move silently with it, so
    the baseline itself deserves a guard.
    """
    assert worst_case_j2(stock, samples=31).torques[1] == pytest.approx(15.36, abs=0.3)
    capacity = max_payload(stock, RS06().peak_nm, max_arm=envelope, samples=31)
    assert capacity == pytest.approx(4.97, abs=0.6)


def test_parity_limits_are_actually_binding():
    """A tripwire set to infinity guards nothing. Keep the thresholds tight
    enough that a real regression would trip them.
    """
    assert 1.0 < TORQUE_REGRESSION_LIMIT <= 1.10
    assert 0.85 <= CAPABILITY_RETENTION < 1.0
