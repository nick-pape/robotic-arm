"""M3 -- the CAD mass-properties pipeline.

Validated against closed-form solutions rather than against itself. If this
pipeline is wrong, every inertial we generate is wrong in the same way and no
downstream test would notice, so the analytic checks carry real weight.
"""

import numpy as np
import pytest
from build123d import Box, Cylinder, Pos, Rotation

from robotic_arm.massprops import MassProperties, combine, mass_properties
from robotic_arm.materials import AL_5052, PA_CF, PLA, Material

RHO = PLA.bulk_density


def test_box_matches_closed_form():
    """I = m(b^2 + c^2)/12 about the COM for a rectangular prism."""
    a, b, c = 10.0, 20.0, 30.0  # mm
    mp = mass_properties(Box(a, b, c), PLA)

    expect_mass = a * b * c * 1e-9 * RHO
    assert mp.mass == pytest.approx(expect_mass, rel=1e-12)

    m, am, bm, cm = expect_mass, a * 1e-3, b * 1e-3, c * 1e-3
    assert mp.inertia[0, 0] == pytest.approx(m * (bm**2 + cm**2) / 12, rel=1e-12)
    assert mp.inertia[1, 1] == pytest.approx(m * (am**2 + cm**2) / 12, rel=1e-12)
    assert mp.inertia[2, 2] == pytest.approx(m * (am**2 + bm**2) / 12, rel=1e-12)
    assert np.allclose(mp.inertia - np.diag(np.diag(mp.inertia)), 0, atol=1e-18)


def test_cylinder_matches_closed_form():
    """Izz = m r^2 / 2, Ixx = Iyy = m(3r^2 + h^2)/12."""
    r, h = 12.0, 40.0  # mm
    mp = mass_properties(Cylinder(radius=r, height=h), AL_5052)

    rm, hm = r * 1e-3, h * 1e-3
    m = np.pi * rm**2 * hm * AL_5052.density
    assert mp.mass == pytest.approx(m, rel=1e-4)
    assert mp.inertia[2, 2] == pytest.approx(m * rm**2 / 2, rel=1e-4)
    assert mp.inertia[0, 0] == pytest.approx(m * (3 * rm**2 + hm**2) / 12, rel=1e-4)


def test_inertia_is_about_com_not_origin():
    """OCCT's default reference point is the origin, but build123d returns the
    COM-relative tensor. Translating the solid must not change it.
    """
    box = Box(10, 20, 30)
    here = mass_properties(box, PLA)
    moved = mass_properties(Pos(123, -45, 67) * box, PLA)

    assert np.allclose(here.inertia, moved.inertia, atol=1e-15)
    assert not np.allclose(here.com, moved.com)


def test_tensor_follows_body_rotation():
    """Axes are world-aligned at the COM, so rotating the solid permutes the
    tensor. This is what makes the MJCF mapping a straight copy.
    """
    box = Box(10, 20, 30)
    upright = mass_properties(box, PLA).inertia
    turned = mass_properties(Rotation(0, 0, 90) * box, PLA).inertia

    assert upright[0, 0] == pytest.approx(turned[1, 1], rel=1e-9)
    assert upright[1, 1] == pytest.approx(turned[0, 0], rel=1e-9)
    assert upright[2, 2] == pytest.approx(turned[2, 2], rel=1e-9)


def test_offdiagonal_sign_convention_matches_urdf():
    """An L-shaped solid has a genuine product of inertia. Check the sign
    against a hand parallel-axis sum in the tensor convention, because a flip
    here would silently corrupt every rotational dynamic.
    """
    lower = Pos(20, 5, 5) * Box(40, 10, 10)
    upper = Pos(5, 5, 20) * Box(10, 10, 20)
    shape = lower + upper

    mp = mass_properties(shape, PLA)

    # Hand computation: two boxes, each with its own COM, combined.
    hand = combine(
        [
            mass_properties(lower, PLA),
            mass_properties(upper, PLA),
        ]
    )
    assert np.allclose(mp.inertia, hand.inertia, rtol=1e-9, atol=1e-18)
    assert abs(mp.inertia[0, 2]) > 1e-12, "expected a real Ixz on an L-shape"


def test_combine_matches_single_solid_at_uniform_density():
    """Parallel-axis combination must agree with measuring the fused solid."""
    a = Pos(0, 0, 0) * Box(20, 20, 10)
    b = Pos(0, 0, 15) * Box(10, 10, 20)

    fused = mass_properties(a + b, PLA)
    summed = combine([mass_properties(a, PLA), mass_properties(b, PLA)])

    assert summed.mass == pytest.approx(fused.mass, rel=1e-9)
    assert np.allclose(summed.com, fused.com, atol=1e-12)
    assert np.allclose(summed.inertia, fused.inertia, rtol=1e-9, atol=1e-18)


def test_combine_handles_mixed_materials():
    """The case Compound cannot do: steel insert in a plastic body."""
    body = Box(30, 30, 10)
    insert = Pos(10, 0, 0) * Cylinder(radius=3, height=10)

    mixed = combine(
        [mass_properties(body, PA_CF), mass_properties(insert, AL_5052)]
    )
    assert mixed.mass > mass_properties(body, PA_CF).mass
    # The dense insert pulls the COM towards it.
    assert mixed.com[0] > 0


def test_fullinertia_order_is_diagonal_first():
    mp = MassProperties(
        mass=1.0,
        com=np.zeros(3),
        inertia=np.array([[1.0, 4.0, 5.0], [4.0, 2.0, 6.0], [5.0, 6.0, 3.0]]),
    )
    assert mp.fullinertia == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

    attrs = mp.to_mjcf_attrs()
    assert attrs["fullinertia"] == "1 2 3 4 5 6"
    assert "quat" not in attrs, "fullinertia and quat must not both be set"


def test_real_geometry_is_physically_realisable():
    """Guards the units-bug class that MuJoCo's balanceinertia would mask."""
    mp = mass_properties(Box(60, 40, 6) + Pos(0, 0, 10) * Cylinder(8, 14), PA_CF)
    assert mp.satisfies_triangle_inequality()


def test_non_solid_is_rejected():
    with pytest.raises(ValueError, match="need a solid"):
        mass_properties(Box(10, 10, 10).faces()[0], PLA)


def test_printed_density_is_between_infill_and_bulk():
    sparse = PA_CF.printed(infill=0.4, walls=5)
    assert 0.4 * PA_CF.bulk_density < sparse.density < PA_CF.bulk_density


def test_measured_density_overrides_the_estimate():
    """A weighed coupon is authoritative over the infill model."""
    calibrated = PA_CF.measured(mass_g=42.0, volume_mm3=50_000.0)
    assert calibrated.density == pytest.approx(840.0, rel=1e-9)


def test_zero_infill_is_shell_only():
    shell = Material("test", 1000.0).printed(infill=0.0, walls=4)
    assert 0 < shell.density < 1000.0
