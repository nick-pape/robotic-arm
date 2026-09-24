"""Validating the model against the shipping product's advertised specs.

This is the most load-bearing test in the suite. Everything else checks the
model against itself or against closed-form maths; this checks it against a
product that demonstrably works and has published numbers.

It also settles what requirement P2 actually means. P2 asks for a payload the
arm can hold *continuously at the worst pose in the whole workspace* within a
derated thermal budget. That is a far stricter question than a manufacturer
payload rating, and the decisive evidence is that **the stock arm fails P2
too**, with zero payload. P2 failing is therefore not a clone regression.
"""

import pytest

from robotic_arm.actuators import RS06
from robotic_arm.mjcf import generate, load
from robotic_arm.torque import max_moment_arm, max_payload, worst_case_j2

#: Seeed advertise 2.5 kg rated and 5 kg maximum, both qualified as being
#: within about 70% of the workspace.
ADVERTISED_RATED_KG = 2.5
ADVERTISED_MAX_KG = 5.0
ADVERTISED_ENVELOPE = 0.70


@pytest.fixture(scope="module")
def stock(tmp_path_factory):
    return load(generate(out=tmp_path_factory.mktemp("stock") / "model.xml"))


@pytest.fixture(scope="module")
def envelope(stock):
    return ADVERTISED_ENVELOPE * max_moment_arm(stock)


def test_model_reproduces_advertised_maximum_payload(stock, envelope):
    """Within peak torque and the stated 70% envelope, the model should land
    near Seeed's advertised 5 kg maximum.

    Compared on the **total at the tool**, gripper included, because that is
    how industrial payload ratings are normally quoted. The workpiece figure
    alone is about 4.0 kg with the 0.8 kg gripper already on the arm.

    This agreement is weaker evidence than it first appeared. An earlier
    version of this test matched 5 kg almost exactly, but only because the
    payload was being applied at the gripper's centre of mass -- 113 mm
    inboard of where a grasped object actually hangs -- which understated the
    shoulder moment. With the load at the grasp point the total is 4.8 kg.
    Close, but the quoting convention is an assumption, so treat this as
    consistency rather than validation.
    """
    gripper = sum(
        float(stock.body_mass[stock.body(name).id])
        for name in ("gripper_end", "gripper_left", "gripper_right")
    )
    workpiece = max_payload(stock, RS06().peak_nm, max_arm=envelope, samples=31)
    assert workpiece + gripper == pytest.approx(ADVERTISED_MAX_KG, abs=0.8)


def test_advertised_rated_payload_has_margin(stock, envelope):
    """2.5 kg rated should sit comfortably inside the same envelope, not at
    the edge of feasibility.
    """
    capacity = max_payload(stock, RS06().peak_nm, max_arm=envelope, samples=31)
    assert capacity > ADVERTISED_RATED_KG * 1.2


def test_restricting_the_envelope_buys_real_capacity(stock):
    """The 70% qualifier on the advertised figures is doing real work."""
    full = max_moment_arm(stock)
    inside = max_payload(stock, RS06().peak_nm, max_arm=0.70 * full, samples=31)
    everywhere = max_payload(stock, RS06().peak_nm, max_arm=None, samples=31)
    assert inside > everywhere + 1.0


def test_stock_arm_also_fails_p2(stock):
    """The finding that reframes P2.

    P2 asks for continuous payload at the worst pose within a derated budget.
    The stock arm cannot meet it either -- its own self-weight already exceeds
    that budget -- so P2 failing on the clone says nothing about the clone. It
    says P2 is a stricter criterion than any payload rating uses.
    """
    derated = RS06().derated_nm()
    assert worst_case_j2(stock, samples=31).torques[1] > derated
    assert max_payload(stock, derated, samples=21) == 0.0


def test_continuous_rated_criterion_is_not_how_payload_is_rated(stock, envelope):
    """Even at full rated torque with the heat sink, and inside the advertised
    envelope, the continuous-hold criterion yields no payload at all. A
    criterion that scores a working product at zero is the wrong criterion.
    """
    assert max_payload(stock, RS06().rated_nm, max_arm=envelope, samples=21) == 0.0


def test_moment_arm_is_not_tool_reach(stock):
    """Why 'within 70% of the workspace' cannot mean 70% of reach: the forearm
    folds, so the tool can be close to the base while the upper arm is fully
    horizontal and J2 is loaded hardest.
    """
    import mujoco
    import numpy as np

    from robotic_arm.torque import j2_moment_arm

    data = mujoco.MjData(stock)
    found = False
    for q2 in np.linspace(*stock.jnt_range[1], 25):
        for q3 in np.linspace(*stock.jnt_range[2], 25):
            data.qpos[:] = 0
            data.qpos[1], data.qpos[2] = q2, q3
            mujoco.mj_forward(stock, data)
            tcp = data.body("gripper_end").xpos
            reach = float(np.hypot(tcp[0], tcp[1]))
            if reach < 0.4 and j2_moment_arm(data, stock) > 0.35:
                found = True
    assert found, "expected a folded pose with a short reach but a long J2 arm"
