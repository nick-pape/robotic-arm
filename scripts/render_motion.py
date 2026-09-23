"""Render the arm moving, as an animated GIF.

Drives the model's position actuators through a set of waypoints and renders
offscreen, so this works headless and in CI. The motion is simulated rather
than interpolated in joint space -- the arm tracks targets under gravity, so
what you see is the controller doing real work against the load.

    uv run --extra viz python scripts/render_motion.py
    uv run --extra viz python scripts/render_motion.py --balancer --out arm.gif

With --balancer the J2 spring is included and renders as a red line between
its anchor and the upper arm.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from robotic_arm.balancer import Balancer
from robotic_arm.mjcf import generate, generate_scene, load
from robotic_arm.motion import trajectory

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "docs" / "arm-motion.gif"

def render(
    out: Path,
    use_balancer: bool,
    width: int,
    height: int,
    steps_per_leg: int,
    substeps: int,
    fps: int,
    colors: int,
) -> Path:
    balancer = Balancer.sized_for(8.25) if use_balancer else None
    model_path = generate(out=REPO / "sim" / "render_model.xml", balancer=balancer)
    model = load(generate_scene(model_path))
    data = mujoco.MjData(model)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = 1.5
    camera.elevation = -18.0
    camera.lookat[:] = [0.0, 0.0, 0.28]

    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    # Tendons are drawn by default; make sure the spring is visible.
    options.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True

    frames: list[Image.Image] = []
    targets = trajectory(steps_per_leg)

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for i, target in enumerate(targets):
            data.ctrl[: len(target)] = target
            for _ in range(substeps):
                mujoco.mj_step(model, data)

            # Slow orbit, so the shape of the arm reads in three dimensions.
            camera.azimuth = 135.0 + 40.0 * np.sin(2 * np.pi * i / len(targets))
            renderer.update_scene(data, camera=camera, scene_option=options)
            frames.append(Image.fromarray(renderer.render()))

    out.parent.mkdir(parents=True, exist_ok=True)
    # Quantise every frame against one shared palette. Per-frame palettes cost
    # far more bytes and make the background shimmer between frames.
    palette = frames[len(frames) // 2].quantize(colors=colors, method=Image.MEDIANCUT)
    quantised = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    quantised[0].save(
        out,
        save_all=True,
        append_images=quantised[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--balancer", action="store_true", help="include the J2 gravity spring"
    )
    parser.add_argument("--width", type=int, default=420)
    parser.add_argument("--height", type=int, default=315)
    parser.add_argument("--steps-per-leg", type=int, default=18)
    parser.add_argument("--colors", type=int, default=96, help="GIF palette size")
    parser.add_argument(
        "--substeps", type=int, default=12, help="physics steps per rendered frame"
    )
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    path = render(
        args.out,
        args.balancer,
        args.width,
        args.height,
        args.steps_per_leg,
        args.substeps,
        args.fps,
        args.colors,
    )
    print(f"wrote {path.relative_to(REPO)} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
