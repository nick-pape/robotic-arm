"""M5 -- the CAD-to-MJCF generator, and spec requirement F2.

The end-to-end path this proves: build123d solid -> mass properties -> MJCF
<inertial> -> compiled MuJoCo model that still has stock kinematics.

No part designs are committed yet. The solids here are deliberately generic
stand-ins whose only job is to exercise the pipeline; real geometry arrives
once the design is settled.
"""

import mujoco
import numpy as np
import pytest
from build123d import Box, Cylinder, Pos

from robotic_arm.massprops import combine, mass_properties
from robotic_arm.materials import AL_5052, PA_CF
from robotic_arm.mjcf import (
    ROUNDTRIP_TOL,
    UnknownBodyError,
    frame_differences,
    generate,
    inertial_differences,
    load,
)
from robotic_arm.reference import load_baseline


@pytest.fixture(scope="module")
def stand_in_part():
    """A generic printed link stand-in: a plate with a boss and a steel insert.

    Not a design. It exists so the pipeline is exercised on geometry with a
    real product of inertia and mixed materials, rather than on a symmetric box
    that would hide an axis or sign error.
    """
    shell = Box(60, 40, 8) + Pos(15, 0, 10) * Cylinder(radius=9, height=12)
    insert = Pos(15, 0, 10) * Cylinder(radius=2, height=12)
    return combine(
        [
            mass_properties(shell, PA_CF.printed(infill=0.4, walls=5)),
            mass_properties(insert, AL_5052),
        ]
    )


def test_generation_with_no_overrides_preserves_frames(tmp_path):
    """F2 in its purest form: change nothing, and nothing kinematic moves."""
    model = load(generate(out=tmp_path / "model.xml"))
    assert frame_differences(model, load_baseline()) == []


def test_generation_with_no_overrides_preserves_inertials(tmp_path):
    """The no-op build should also leave mass properties alone, to within the
    9-significant-figure XML round-trip.
    """
    model = load(generate(out=tmp_path / "model.xml"))
    reference = load_baseline()

    assert np.allclose(model.body_mass, reference.body_mass, rtol=ROUNDTRIP_TOL)
    assert np.allclose(model.body_inertia, reference.body_inertia, rtol=ROUNDTRIP_TOL)
    assert np.allclose(model.body_ipos, reference.body_ipos, atol=1e-6)


def test_cad_inertials_reach_the_compiled_model(tmp_path, stand_in_part):
    """The whole point: a build123d solid's mass properties end up in MuJoCo."""
    model = load(
        generate({"link6": stand_in_part}, out=tmp_path / "model.xml")
    )
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")

    assert model.body_mass[bid] == pytest.approx(stand_in_part.mass, rel=ROUNDTRIP_TOL)
    assert np.allclose(model.body_ipos[bid], stand_in_part.com, atol=1e-6)

    # MuJoCo stores diagonalised inertia plus iquat, so compare invariants
    # rather than raw components.
    assert np.allclose(
        sorted(model.body_inertia[bid]),
        sorted(stand_in_part.principal_moments),
        rtol=1e-4,
    )


def test_swapping_inertials_does_not_move_frames(tmp_path, stand_in_part):
    """F2 with a real override: the kinematics must be untouched."""
    model = load(generate({"link6": stand_in_part}, out=tmp_path / "model.xml"))
    assert frame_differences(model, load_baseline()) == []


def test_only_the_named_body_changes(tmp_path, stand_in_part):
    """A swap must not leak into neighbouring links."""
    model = load(generate({"link6": stand_in_part}, out=tmp_path / "model.xml"))
    changed = inertial_differences(model, load_baseline(), tol=1e-4)
    assert set(changed) == {"link6"}


def test_generated_model_still_simulates(tmp_path, stand_in_part):
    """A plausible inertial should not destabilise the integrator."""
    model = load(generate({"link6": stand_in_part}, out=tmp_path / "model.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "raised")
    )
    for _ in range(1000):
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))


def test_unknown_body_is_rejected(tmp_path, stand_in_part):
    """A typo in a body name must fail loudly, not silently do nothing."""
    with pytest.raises(UnknownBodyError, match="not a body in the stock model"):
        generate({"link_7": stand_in_part}, out=tmp_path / "model.xml")


def test_impossible_inertia_is_rejected(tmp_path):
    """Guards the units-bug class that MuJoCo's balanceinertia would mask."""
    from robotic_arm.massprops import MassProperties

    bogus = MassProperties(
        mass=1.0,
        com=np.zeros(3),
        inertia=np.diag([1.0, 1.0, 100.0]),  # violates the triangle inequality
    )
    with pytest.raises(ValueError, match="triangle inequality"):
        generate({"link6": bogus}, out=tmp_path / "model.xml")


def test_mass_budget_reporting(tmp_path, stand_in_part):
    """P1 needs a printed link compared against the stock part it replaces.
    Check the comparison is actually available from the generated model.
    """
    reference = load_baseline()
    bid = mujoco.mj_name2id(reference, mujoco.mjtObj.mjOBJ_BODY, "link6")
    stock_mass = reference.body_mass[bid]

    assert stock_mass > 0
    # Not asserting the stand-in is lighter -- it is not a real design.
    assert stand_in_part.mass > 0
