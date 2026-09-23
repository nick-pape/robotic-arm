"""M4 -- actuator interface geometry, and spec requirement I1.

The spec listed the RS06 bolt circle as [UNVERIFIED]: no published text gives
it, and the manual's dimension drawing is a raster image. These tests pin what
the vendor STEP actually says, so a printed motor pocket is designed against
measured geometry rather than a transcription.

They assert *measured* quantities -- BCD, hole count, diameter, spacing. Thread
size is deliberately not asserted as a single value: a drilled diameter alone
cannot distinguish a tap drill from a close-fit clearance hole, so
`fastener_candidates` returns every plausible reading and the tests check
membership rather than equality.
"""

import pytest

from robotic_arm import actuators
from robotic_arm.actuators import RS00, RS06, for_joint


def circles_by_bcd(actuator):
    return {round(c.bcd, 2): c for c in actuator.bolt_circles}


def test_rs06_mounting_circle_is_measured():
    """The value the spec could not obtain anywhere: RS06's housing pattern."""
    mount = RS06().largest_circle()
    assert mount.bcd == pytest.approx(82.0, abs=0.01)
    assert mount.count == 8
    assert mount.pitch_deg == pytest.approx(45.0, abs=0.1)
    assert mount.hole_diameter == pytest.approx(2.5, abs=0.01)
    assert mount.depth == pytest.approx(8.0, abs=0.01)
    assert mount.bottom == "drill_point", "blind, so tapped rather than clearance"
    assert mount.is_plausibly("M3")


def test_rs06_output_face_pattern():
    """The rotor side, which a printed link bolts its moving half to."""
    output = min(RS06().bolt_circles, key=lambda c: c.hole_diameter * -1)
    assert output.bcd == pytest.approx(24.02, abs=0.01)
    assert output.count == 6
    assert output.pitch_deg == pytest.approx(60.0, abs=0.1)
    assert output.hole_diameter == pytest.approx(3.3, abs=0.01)


def test_rs06_has_three_distinct_circles():
    """Two at O24.02 on opposite faces, one at O82. If the recogniser ever
    merges or splits these, a motor pocket would be generated wrong.
    """
    circles = RS06().bolt_circles
    assert len(circles) == 3
    assert sorted(round(c.bcd, 2) for c in circles) == [24.02, 24.02, 82.0]
    # The two O24.02 circles face opposite ways.
    small = [c for c in circles if round(c.bcd, 2) == 24.02]
    assert {c.axis_z for c in small} == {1.0, -1.0}


def test_rs00_published_top_circle_is_confirmed():
    """Published: 'top 6x M3 on O27'. The STEP agrees on BCD and count, and
    O3.0 is a credible M3 close-fit clearance hole.
    """
    top = circles_by_bcd(RS00())[27.0]
    assert top.count == 6
    assert top.pitch_deg == pytest.approx(60.0, abs=0.1)
    assert top.hole_diameter == pytest.approx(3.0, abs=0.01)
    assert top.is_plausibly("M3")


def test_rs00_published_bottom_circle_cannot_be_m3():
    """Published: 'bottom 4x M3 on O38'. The BCD and count are right; M3 is not.

    A O1.6 hole is an M2 tap drill or an M1.6 clearance hole. It cannot be any
    kind of M3 -- an M3 tap drill is O2.5 and an M3 clearance hole O3.2-3.6.
    Designing an M3 pattern against it yields a part that cannot be bolted on,
    so this erratum is pinned rather than left as a footnote.
    """
    bottom = circles_by_bcd(RS00())[38.0]
    assert bottom.count == 4
    assert bottom.hole_diameter == pytest.approx(1.6, abs=0.01)
    assert not bottom.is_plausibly("M3"), actuators.RS00_PUBLISHED_ERRATUM
    assert bottom.counterbore is not None
    assert bottom.counterbore["diameter"] == pytest.approx(2.5, abs=0.01)


def test_every_circle_is_evenly_spaced():
    """An uneven circle would mean the recogniser merged unrelated holes."""
    for actuator in (RS06(), RS00()):
        for circle in actuator.bolt_circles:
            assert circle.pitch_deg is not None, (
                f"{actuator.name} O{circle.bcd} is not evenly spaced"
            )
            assert circle.pitch_deg * circle.count == pytest.approx(360.0, abs=0.5)


def test_envelope_matches_vendor_claim():
    """RobStride publish O82 x 49 for RS06. The STEP gives O82 and 50.5 tall."""
    x, y, z = RS06().bbox_mm
    assert y == pytest.approx(82.0, abs=0.1)
    assert 49.0 <= z <= 51.0
    assert x > y, "expected a connector or boss projecting past the body"


def test_mounting_circle_fits_inside_the_envelope():
    """A BCD at the very rim would leave no material; sanity-check the pair."""
    rs06 = RS06()
    assert rs06.largest_circle().bcd <= max(rs06.bbox_mm[:2]) + 0.01


def test_joint_assignment_matches_upstream_config():
    """J1-J3 are RS06, J4-J6 RS00, per config/rebotarm_rs.yaml."""
    assert [for_joint(f"joint{i}").name for i in range(1, 7)] == [
        "RS06",
        "RS06",
        "RS06",
        "RS00",
        "RS00",
        "RS00",
    ]


def test_ratings_match_the_urdf_effort_limits():
    """The URDF declares effort=36 on J1-J3 and effort=14 on J4-J6, which are
    the peak torques. Cross-checks two independent sources against each other.
    """
    assert RS06().peak_nm == 36.0
    assert RS00().peak_nm == 14.0


def test_derating_is_conservative_but_not_silly():
    rs06 = RS06()
    assert rs06.derated_nm() < rs06.rated_nm
    assert rs06.derated_nm(0.6) < rs06.derated_nm(0.7)


def test_unknown_actuator_is_rejected():
    with pytest.raises(KeyError, match="unknown actuator"):
        actuators.get("RS99")
