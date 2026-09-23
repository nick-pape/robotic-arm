"""Open the digital twin in MuJoCo's interactive viewer, arm moving.

    uv run python scripts/view.py                # stock inertials
    uv run python scripts/view.py --balancer     # with the J2 gravity spring
    uv run python scripts/view.py --static       # hold still, just look at it

The viewer is passive: physics is stepped here, so the arm keeps moving while
you orbit, zoom and inspect. Mouse drags rotate, scroll zooms, double-click
selects a body, and the left panel toggles visual flags such as tendons and
contact forces.

Live J2 torque is printed as it moves, so the pose that loads the shoulder
hardest is visible as it happens rather than only in the analysis.
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from robotic_arm.actuators import RS06
from robotic_arm.balancer import Balancer, static_torques
from robotic_arm.mjcf import generate_scene, generate_twin, load
from robotic_arm.motion import leg_names, trajectory
from robotic_arm.thermal import sustainability

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--balancer", action="store_true", help="include the J2 gravity spring"
    )
    parser.add_argument(
        "--static", action="store_true", help="do not drive the trajectory"
    )
    parser.add_argument("--steps-per-leg", type=int, default=90)
    parser.add_argument(
        "--substeps", type=int, default=4, help="physics steps per viewer frame"
    )
    args = parser.parse_args()

    balancer = Balancer.sized_for(8.25) if args.balancer else None
    model_path = generate_twin(out=REPO / "sim" / "view_model.xml", balancer=balancer)
    model = load(generate_scene(model_path))
    data = mujoco.MjData(model)

    rs06 = RS06()
    targets = trajectory(args.steps_per_leg)
    names = leg_names(args.steps_per_leg)
    print(
        f"viewer: {'balanced' if balancer else 'unbalanced'}, "
        f"{model.nbody} bodies, {model.ngeom} geoms. Close the window to exit."
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
        cycle = itertools.cycle(range(len(targets)))
        last_report = 0.0

        while viewer.is_running():
            frame_start = time.perf_counter()
            i = next(cycle)

            if not args.static:
                data.ctrl[: len(targets[i])] = targets[i]
            for _ in range(args.substeps):
                mujoco.mj_step(model, data)

            viewer.sync()

            now = time.perf_counter()
            if now - last_report > 0.5:
                tau = static_torques(model, data.qpos[: model.nq])[1]
                print(
                    f"\r{names[i]:>16s}  J2 {tau:6.2f} N*m  "
                    f"{sustainability(rs06, tau):<13s}",
                    end="",
                    flush=True,
                )
                last_report = now

            # Keep roughly real time rather than running as fast as possible.
            remaining = model.opt.timestep * args.substeps - (
                time.perf_counter() - frame_start
            )
            if remaining > 0:
                time.sleep(remaining)

    print()


if __name__ == "__main__":
    main()
