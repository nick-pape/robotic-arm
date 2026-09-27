"""Do adjacent printed parts occupy the same space?

The most basic question an assembly can be asked, and nothing in this suite
asked it. Mass, envelope, inertia, bolt access and the MuJoCo clearance sweep
are all indifferent to two CAD solids interpenetrating at the home pose: the
clearance sweep runs on collision *proxies*, and adjacent links are in the
upstream contact exclusion list anyway.

It was worth asking. link4 and link5 overlapped by 2,447 mm3 -- a quarter of
link5's flange buried in link4 -- because `motor_side` put J5's motor below
its joint when the real one sits above it. The renders showed two wrist
housings visibly fighting; no number did.
"""

import mujoco
import numpy as np
import pytest
from build123d import Location, Plane

from robotic_arm.parts import REGISTRY
from robotic_arm.reference import load_baseline

#: Boolean and tessellation noise along a shared seam. Real interference on
#: these parts runs to hundreds or thousands of mm3, so this separates
#: cleanly without needing to be delicate.
NOISE_MM3 = 25.0

CHAIN = ("base_link", "link1", "link2", "link3", "link4", "link5", "link6")
PAIRS = [(a, b) for a, b in zip(CHAIN, CHAIN[1:]) if a in REGISTRY and b in REGISTRY]


@pytest.fixture(scope="module")
def placed():
    model = load_baseline()
    data = mujoco.MjData(model)
    data.qpos[:] = 0
    mujoco.mj_forward(model, data)
    out = {}
    for name in REGISTRY:
        solid = REGISTRY[name][0]()
        bid = model.body(name).id
        rotation = data.xmat[bid].reshape(3, 3)
        plane = Plane(
            origin=tuple(data.xpos[bid] * 1000),
            x_dir=tuple(rotation[:, 0]),
            z_dir=tuple(rotation[:, 2]),
        )
        out[name] = solid.moved(Location(plane))
    return out


def _overlap_mm3(a, b) -> float:
    hit = a.intersect(b)
    if hit is None:
        return 0.0
    pieces = hit if hasattr(hit, "__iter__") else [hit]
    return sum(float(s.volume) for piece in pieces for s in piece.solids())


@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: f"{p[0]}-{p[1]}")
def test_adjacent_parts_do_not_interpenetrate(pair, placed):
    """Two solids in the same place cannot both be printed and assembled."""
    a, b = pair
    overlap = _overlap_mm3(placed[a], placed[b])
    assert overlap < NOISE_MM3, (
        f"{a} and {b} interpenetrate by {overlap:,.0f} mm3 at the home pose; "
        f"the arm could not be assembled"
    )


def test_the_check_can_actually_see_an_overlap():
    """A tripwire that never fires guards nothing.

    Every other version of this idea in this project turned out to be inert --
    a circular fit check, a stale bolt circle, a recorded figure nothing read.
    So confirm the measurement reports a known overlap before trusting it to
    report zero.
    """
    from build123d import Box, Pos

    a = Box(20, 20, 20)
    b = Pos(10, 0, 0) * Box(20, 20, 20)
    assert _overlap_mm3(a, b) == pytest.approx(4000.0, rel=0.01)
    assert _overlap_mm3(a, Pos(40, 0, 0) * Box(20, 20, 20)) == 0.0
