"""M1 -- the stock baseline loads, is the arm we think it is, and simulates.

Every later change is measured against this model, so these tests guard the
assumptions the whole project rests on.
"""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from robotic_arm.reference import BASELINE_MJCF, STOCK_URDF


def urdf_joint_origins() -> dict[str, np.ndarray]:
    """Child-link name -> joint origin xyz, straight from the stock URDF."""
    root = ET.parse(STOCK_URDF).getroot()
    origins = {}
    for joint in root.findall("joint"):
        child = joint.find("child").attrib["link"]
        xyz = joint.find("origin").attrib["xyz"]
        origins[child] = np.array([float(v) for v in xyz.split()])
    return origins


def test_menagerie_model_is_the_rs_arm(baseline):
    """The directory is named `seeed_rebot_devarm`, which reads like the DM
    variant. Assert it is actually the RS arm rather than trusting the name.
    """
    origins = urdf_joint_origins()
    checked = 0
    for name, want in origins.items():
        bid = mujoco.mj_name2id(baseline, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            continue  # gripper links are renamed upstream; frames checked below
        assert np.allclose(baseline.body_pos[bid], want, atol=1e-9), (
            f"body {name!r} at {baseline.body_pos[bid]} but URDF says {want}"
        )
        checked += 1
    assert checked >= 6, f"only {checked} bodies cross-checked against the URDF"


def test_masses_match_urdf_inertials(baseline):
    """Menagerie preserved the URDF inertials; if that stops being true, our
    whole "swap only the inertials" premise needs revisiting.
    """
    root = ET.parse(STOCK_URDF).getroot()
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        bid = mujoco.mj_name2id(baseline, mujoco.mjtObj.mjOBJ_BODY, link.attrib["name"])
        if bid < 0:
            continue
        want = float(inertial.find("mass").attrib["value"])
        assert baseline.body_mass[bid] == pytest.approx(want, rel=1e-9)


def test_baseline_simulates_stably(baseline):
    data = mujoco.MjData(baseline)
    key = mujoco.mj_name2id(baseline, mujoco.mjtObj.mjOBJ_KEY, "raised")
    mujoco.mj_resetDataKeyframe(baseline, data, key)
    for _ in range(1000):
        mujoco.mj_step(baseline, data)
    assert np.all(np.isfinite(data.qpos)), "simulation diverged"
    assert np.abs(data.qpos).max() < 10.0


def test_mjspec_roundtrip_preserves_inertials(baseline):
    """Guards MuJoCo issue #2370, where `MjSpec.to_xml()` silently dropped
    explicitly-set mass and fullinertia. Fixed as of 3.14.0 -- but inertials
    are the entire point of our generator, so a regression must fail loudly.
    """
    spec = mujoco.MjSpec.from_file(str(BASELINE_MJCF))
    xml = spec.to_xml()
    assert "<inertial" in xml and "mass=" in xml

    rebuilt = mujoco.MjModel.from_xml_string(xml, _asset_dict())
    assert np.allclose(rebuilt.body_mass, baseline.body_mass)
    assert np.allclose(rebuilt.body_inertia, baseline.body_inertia)


def _asset_dict() -> dict[str, bytes]:
    """Mesh bytes keyed by filename, so from_xml_string can resolve assets."""
    assets_dir = BASELINE_MJCF.parent / "assets"
    return {p.name: p.read_bytes() for p in assets_dir.iterdir() if p.is_file()}
