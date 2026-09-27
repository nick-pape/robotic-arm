"""Stand the stock reBot and the printed cobot side by side in MuJoCo.

    uv run --extra viz python scripts/compare_viz.py

Both arms are posed upright by the *same* routine, so what you are looking at
is a difference in geometry and nothing else. Posing them by hand, or taking
each model's own zero, would not compare anything: the two models disagree
about what zero means -- that is half the point of `homepose` -- so their raw
zeros put the arms in completely different postures.

The two are attached into one scene with name prefixes, because MuJoCo shows
one model at a time and two arms in one picture means one model containing
both.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from robotic_arm.homepose import upright_pose
from robotic_arm.mjcf import generate_twin
from robotic_arm.reference import BASELINE_MJCF, require

REPO = Path(__file__).resolve().parents[1]

#: Clear air between the two arms at the home pose, in metres.
#:
#: This is the **gap**, not the spacing between their bases, and the
#: difference matters: both arms stand well off their own base axis at home
#: -- the twin's tool is 292 mm out in x and 94 mm in y -- so offsetting the
#: bases by a fixed amount left them overlapping. The offsets are computed
#: from each arm's measured extent instead, so the gap is what is asked for.
GAP = 0.35

SCENE = """<mujoco model="compare">
  <!-- Match the arm's own solver settings. The scene is the parent spec, so
       on a conflict its values win; left at MuJoCo's defaults it would
       silently downgrade both arms' physics and warn about it twice. -->
  <option integrator="implicitfast" cone="elliptic" impratio="10"/>
  <statistic center="0 0 0.5" extent="1.6"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="140" elevation="-18"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7"
             rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="grid" builtin="checker" width="512" height="512"
             rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4"/>
    <material name="grid" texture="grid" texrepeat="6 6"
              texuniform="true" reflectance="0.15"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="grid"/>
  </worldbody>
</mujoco>
"""


def upright_qpos(model, prefix: str, pose: dict[str, float]) -> dict[int, float]:
    """`qpos` indices and values that stand one attached arm upright.

    `pose` is in rotations. A joint's `qpos` is its rotation plus its `ref`,
    and the two arms have different `ref`s -- that is exactly what is being
    compared -- so the conversion has to happen per joint rather than once.
    """
    out = {}
    for name, rotation in pose.items():
        address = int(model.joint(f"{prefix}{name}").qposadr[0])
        # A compiled model does not expose `ref` directly; for a hinge it is
        # exactly `qpos0`, the qpos at which the joint's rotation is zero.
        out[address] = rotation + float(model.qpos0[address])
    return out


def lateral_extent(model, pose: dict[str, float]) -> tuple[float, float]:
    """(min y, max y) the arm's geometry occupies at `pose`, in metres.

    Measured over the geoms rather than the body origins. A body origin says
    where a link starts, not how far its barrel reaches, and at this arm's
    scale that is the difference between a clear gap and two arms sharing the
    same space.
    """
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    for name, rotation in pose.items():
        address = int(model.joint(name).qposadr[0])
        data.qpos[address] = rotation + float(model.qpos0[address])
    mujoco.mj_forward(model, data)
    y = data.geom_xpos[:, 1]
    bound = model.geom_rbound
    return float(np.min(y - bound)), float(np.max(y + bound))


def build(colour: bool) -> tuple[mujoco.MjModel, dict[int, float]]:
    twin_path = generate_twin(
        out=REPO / "sim" / "compare_twin.xml", per_link_colour=colour
    )

    scene_path = REPO / "sim" / "compare_scene.xml"
    scene_path.write_text(SCENE)
    scene = mujoco.MjSpec.from_file(str(scene_path))

    arms = {
        "stock_": mujoco.MjSpec.from_file(str(require(BASELINE_MJCF))),
        "twin_": mujoco.MjSpec.from_file(str(twin_path)),
    }

    # Each arm's upright pose has to be solved on that arm alone, before it is
    # attached: `upright_pose` looks up bodies by their bare names, and after
    # attachment every name carries a prefix.
    poses = {p: upright_pose(s.compile()) for p, s in arms.items()}

    extents = {
        p: lateral_extent(s.compile(), poses[p]) for p, s in arms.items()
    }
    # Push each arm just far enough that their measured extents clear GAP.
    offsets = {
        "stock_": -GAP / 2 - extents["stock_"][1],
        "twin_": GAP / 2 - extents["twin_"][0],
    }

    for prefix, spec in arms.items():
        # Keyframes cannot survive attachment -- each stores a qpos sized for
        # its own model, and the combined model's qpos is the concatenation.
        for key in list(spec.keys):
            spec.delete(key)
        frame = scene.worldbody.add_frame(pos=[0.0, offsets[prefix], 0.0])
        scene.attach(spec, prefix=prefix, frame=frame)

    model = scene.compile()
    qpos = {}
    for prefix, pose in poses.items():
        qpos.update(upright_qpos(model, prefix, pose))
    return model, qpos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--colour", action="store_true", help="tint each printed link"
    )
    args = parser.parse_args()

    model, qpos = build(args.colour)
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    for address, value in qpos.items():
        data.qpos[address] = value
    mujoco.mj_forward(model, data)

    for prefix, label in (("stock_", "stock"), ("twin_", "twin")):
        tip = data.body(f"{prefix}gripper_end").xpos * 1000
        print(f"  {label:6} tool at {np.round(tip, 1)} mm")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
