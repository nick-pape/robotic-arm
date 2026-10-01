"""Run a policy in the pick-cube env, and save what its cameras saw.

    uv run python -m learning.rollout --policy scripted
    uv run python -m learning.rollout --policy act --episodes 2 --video mp4
    uv run python -m learning.rollout --policy random --video none
    uv run python -m learning.rollout --policy act --viewer
    uv run python -m learning.rollout --policy pickplace --viewer
    uv run python -m learning.rollout --policy act --viewer         --checkpoint outputs/act_pick_cube/checkpoints/last/pretrained_model

With --viewer the run plays live in MuJoCo's viewer instead of being saved:
the viewer shows the env's own model and data, so what you see is exactly
what the policy is driving. It plays at real time when the machine keeps up,
and runs episodes until the window is closed. The grasp site's path is drawn
over the scene -- blue where it has been, orange where the policy plans to
go next -- and each episode keeps running for --linger seconds after a
success, so you see what the policy does with the cube once it has it.

Videos go to renders/rollouts/ (gitignored), one per episode, with the scene
camera on the left and the wrist camera on the right -- the two images the
policy actually received, not a separate render.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from learning.env import PickCubeEnv
from learning.scene import CONTROL_HZ, TWIN_PATH, ensure_twin

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "renders" / "rollouts"


def make_policy(name: str, env: PickCubeEnv, seed: int, checkpoint: Path | None = None):
    if name == "scripted":
        from learning.scripted import ScriptedPick

        return ScriptedPick(env)
    if name == "pickplace":
        from learning.scripted import ScriptedPickPlace

        return ScriptedPickPlace(env, seed=seed)
    if name == "random":
        from learning.policies import RandomPolicy

        return RandomPolicy(env, seed=seed)
    if name == "act":
        from learning.policies import ActPolicy

        if checkpoint is not None:
            return ActPolicy.from_checkpoint(checkpoint)
        return ActPolicy(env, seed=seed)
    raise ValueError(f"unknown policy {name!r}")


def save_video(frames: list[np.ndarray], path: Path, fps: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".mp4":
        import cv2

        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
    else:
        from PIL import Image

        # Every other frame: a 30 fps GIF of a 10 s episode is tens of MB.
        images = [Image.fromarray(f) for f in frames[::2]]
        images[0].save(
            path, save_all=True, append_images=images[1:], duration=2000 // fps, loop=0
        )
    return path


class TrajectoryOverlay:
    """The grasp site's path, drawn into the viewer: where it has been, and
    where the policy currently intends to take it.

    Past is the path the site actually travelled. Future is the policy's
    queued actions put through forward kinematics -- its *intent*, before the
    env's per-step clip, so a future line that leaps somewhere the arm then
    crawls toward is the clip at work. Drawn in the viewer's own scene, so
    the env's cameras never see it and the policy's inputs stay clean.
    """

    PAST_RGBA = np.array([0.2, 0.75, 1.0, 1.0], dtype=np.float32)
    FUTURE_RGBA = np.array([1.0, 0.55, 0.1, 1.0], dtype=np.float32)
    RADIUS = 0.0025
    MAX_PAST = 600

    def __init__(self, env: PickCubeEnv):
        from learning.ik import SiteIK
        from learning.scene import GRASP_SITE

        self.fk = SiteIK(env.model)
        self.site = env.model.site(GRASP_SITE).id
        self.past: list[np.ndarray] = []

    def reset(self):
        self.past = []

    def update(self, scene, env: PickCubeEnv, policy) -> None:
        here = env.data.site_xpos[self.site].copy()
        self.past.append(here)
        planned = getattr(policy, "planned_actions", lambda: [])()
        future = [self.fk.site_pose(np.asarray(a[:6], dtype=float))[0] for a in planned]
        scene.ngeom = 0
        self._polyline(scene, self.past[-self.MAX_PAST :], self.PAST_RGBA)
        self._polyline(scene, [here, *future], self.FUTURE_RGBA)

    def _polyline(self, scene, points, rgba) -> None:
        import mujoco

        for a, b in zip(points, points[1:]):
            if scene.ngeom >= scene.maxgeom:
                return
            if np.linalg.norm(b - a) < 1e-5:
                continue  # a hold: no segment to draw
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), rgba
            )
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, self.RADIUS, a, b)
            scene.ngeom += 1


def run_episode(
    env: PickCubeEnv,
    policy,
    seed: int,
    record: bool,
    viewer=None,
    overlay: TrajectoryOverlay | None = None,
    linger_steps: int = 0,
):
    """One episode. With `linger_steps`, keep going that long after success
    instead of stopping on it -- to see what the policy does with the cube."""
    if viewer is not None:
        with viewer.lock():
            observation, info = env.reset(seed=seed)
        viewer.sync()
    else:
        observation, info = env.reset(seed=seed)
    policy.reset()
    if overlay is not None:
        overlay.reset()
    frames, policy_time, env_time = [], 0.0, 0.0
    succeeded_at = None
    for step in range(env.max_steps):
        step_start = time.perf_counter()
        if record:
            frames.append(np.hstack([observation["pixels"][c.name] for c in env.cameras]))
        start = time.perf_counter()
        action = policy.select_action(observation)
        policy_time += time.perf_counter() - start

        start = time.perf_counter()
        if viewer is not None:
            # The viewer renders from its own thread; hold its lock while the
            # env mutates the data it is reading.
            with viewer.lock():
                observation, _, terminated, truncated, info = env.step(action)
                if overlay is not None:
                    overlay.update(viewer.user_scn, env, policy)
            viewer.sync()
        else:
            observation, _, terminated, truncated, info = env.step(action)
        env_time += time.perf_counter() - start
        if viewer is not None:
            if not viewer.is_running():
                break
            spare = 1.0 / CONTROL_HZ - (time.perf_counter() - step_start)
            if spare > 0:
                time.sleep(spare)
        if terminated and succeeded_at is None:
            succeeded_at = step
        if truncated:
            break
        if succeeded_at is not None and step - succeeded_at >= linger_steps:
            break
    if record:
        frames.append(np.hstack([observation["pixels"][c.name] for c in env.cameras]))
    steps = step + 1
    return {
        "success": succeeded_at is not None,
        "steps": steps if succeeded_at is None else succeeded_at + 1,
        "lift_mm": info["lift"] * 1000,
        "policy_ms": policy_time / steps * 1000,
        "env_ms": env_time / steps * 1000,
        "frames": frames,
    }


def watch(env: PickCubeEnv, policy, name: str, seed: int, linger: float, trajectory: bool):
    """Play episodes in the live viewer until its window is closed."""
    import mujoco.viewer

    overlay = TrajectoryOverlay(env) if trajectory else None
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        episode = 0
        while viewer.is_running():
            result = run_episode(
                env, policy, seed + episode, record=False, viewer=viewer,
                overlay=overlay,
                linger_steps=env.max_steps if np.isinf(linger) else int(linger * CONTROL_HZ),
            )
            print(
                f"{name} episode {episode} (seed {seed + episode}): "
                f"{'success' if result['success'] else 'fail'}  {result['steps']} steps  "
                f"{result['policy_ms'] + result['env_ms']:.0f} ms/step",
                flush=True,
            )
            episode += 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--policy", choices=("scripted", "pickplace", "random", "act"), default="scripted")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--video", choices=("gif", "mp4", "none"), default="gif")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--twin", type=Path, default=TWIN_PATH)
    parser.add_argument("--regenerate", action="store_true", help="rebuild the twin from CAD first")
    parser.add_argument("--viewer", action="store_true", help="watch live in the MuJoCo viewer")
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="act: a trained pretrained_model directory (default: untrained)",
    )
    parser.add_argument(
        "--linger", type=float, default=3.0,
        help="viewer: seconds to keep running after a success before resetting",
    )
    parser.add_argument(
        "--no-trajectory", action="store_true",
        help="viewer: hide the past (blue) / planned (orange) grasp-site path",
    )
    args = parser.parse_args(argv)

    ensure_twin(args.twin, regenerate=args.regenerate)
    if args.policy == "pickplace" and args.viewer:
        # Pick-and-place never finishes on its own; let it cycle until the
        # window closes rather than resetting a few seconds after each lift.
        args.max_steps, args.linger = 10**9, float("inf")
    env = PickCubeEnv(twin_path=args.twin, max_steps=args.max_steps)
    policy = make_policy(args.policy, env, args.seed, args.checkpoint)

    if args.viewer:
        watch(env, policy, args.policy, args.seed, args.linger, not args.no_trajectory)
        env.close()
        return

    successes = 0
    for episode in range(args.episodes):
        seed = args.seed + episode
        result = run_episode(env, policy, seed, record=args.video != "none")
        successes += result["success"]
        line = (
            f"episode {episode} (seed {seed}): "
            f"{'success' if result['success'] else 'fail   '}  "
            f"{result['steps']:3d} steps  lift {result['lift_mm']:6.1f} mm  "
            f"policy {result['policy_ms']:6.1f} ms/step  env {result['env_ms']:5.1f} ms/step"
        )
        if result["frames"]:
            path = args.out / f"{args.policy}_seed{seed}.{args.video}"
            save_video(result["frames"], path, CONTROL_HZ)
            line += f"  -> {path.relative_to(REPO) if path.is_relative_to(REPO) else path}"
        print(line)
    print(f"{args.policy}: {successes}/{args.episodes} successful")
    env.close()


if __name__ == "__main__":
    main()
