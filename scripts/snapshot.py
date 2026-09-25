"""Render static views of the twin to `renders/`, for design feedback.

    uv run --extra viz python scripts/snapshot.py
    uv run --extra viz python scripts/snapshot.py --focus link6 --balancer

Writes PNGs to a gitignored folder. Intended to be re-run after every part
change: the physics is only half the feedback, and a mass budget will not tell
you that two bolt circles intersect.

`--focus` adds close-ups of one body from several angles, which is what you
actually want when reviewing the part you just changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from robotic_arm.balancer import DEFAULT_CANCEL_NM, Balancer
from robotic_arm.mjcf import generate_scene, generate_twin, load
from robotic_arm.motion import WAYPOINTS

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "renders"

#: Poses worth seeing: folded, a natural working pose, and full extension,
#: which is the one the whole torque analysis is about.
POSES = {
    "home": "home",
    "raised": "raised",
    "extended": "full extension",
}

#: (azimuth, elevation) for the whole-arm views.
ANGLES = {"iso": (135.0, -18.0), "front": (90.0, -8.0), "side": (180.0, -8.0)}

#: Close-up angles used with --focus.
DETAIL_ANGLES = {"iso": (135.0, -20.0), "top": (135.0, -65.0), "side": (180.0, -5.0)}


def pose_targets(name: str) -> np.ndarray:
    for label, targets in WAYPOINTS:
        if label == name:
            return np.array(targets, dtype=float)
    raise KeyError(f"no waypoint named {name!r}; have {[w[0] for w in WAYPOINTS]}")


def settle(model: mujoco.MjModel, data: mujoco.MjData, target: np.ndarray) -> None:
    """Drive to a pose and let it come to rest, so the render is not mid-wobble."""
    data.ctrl[: len(target)] = target
    for _ in range(4000):
        mujoco.mj_step(model, data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--balancer", action="store_true")
    parser.add_argument(
        "--colour",
        action="store_true",
        help="tint each printed link differently for review",
    )
    parser.add_argument(
        "--focus",
        default="",
        help="comma-separated body names to add close-up views of",
    )
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=750)
    args = parser.parse_args()

    balancer = Balancer.sized_for(DEFAULT_CANCEL_NM) if args.balancer else None
    model = load(
        generate_scene(
            generate_twin(out=REPO / "sim" / "snapshot_model.xml", balancer=balancer, per_link_colour=args.colour)
        )
    )
    data = mujoco.MjData(model)

    args.out.mkdir(parents=True, exist_ok=True)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    options.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True

    written: list[Path] = []
    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
        for pose_key, waypoint in POSES.items():
            mujoco.mj_resetData(model, data)
            settle(model, data, pose_targets(waypoint))

            for angle_key, (azimuth, elevation) in ANGLES.items():
                camera.distance, camera.azimuth, camera.elevation = 1.5, azimuth, elevation
                camera.lookat[:] = [0.0, 0.0, 0.28]
                renderer.update_scene(data, camera=camera, scene_option=options)
                path = args.out / f"arm_{pose_key}_{angle_key}.png"
                Image.fromarray(renderer.render()).save(path)
                written.append(path)

            for focus in [f.strip() for f in args.focus.split(",") if f.strip()]:
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, focus)
                if bid < 0:
                    raise SystemExit(f"no body named {focus!r}")
                for angle_key, (azimuth, elevation) in DETAIL_ANGLES.items():
                    camera.distance = 0.22
                    camera.azimuth, camera.elevation = azimuth, elevation
                    camera.lookat[:] = data.xpos[bid]
                    renderer.update_scene(data, camera=camera, scene_option=options)
                    path = args.out / f"{focus}_{pose_key}_{angle_key}.png"
                    Image.fromarray(renderer.render()).save(path)
                    written.append(path)

    for path in written:
        print(f"  {path.relative_to(REPO)}")
    print(f"\n{len(written)} renders in {args.out.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
