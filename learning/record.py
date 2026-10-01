"""Record scripted picks as a LeRobot dataset, to train a policy on.

    uv run python -m learning.record                    # 200 episodes
    uv run python -m learning.record --episodes 5 --repo-id local/smoke

Each episode is one scripted pick from a fresh random cube pose, recorded at
the env's 30 Hz: both camera images and `agent_pos` as the observation, and
the action the env actually applied (see `PickCubeEnv.clip_action`). Episodes
where the pick fails are discarded rather than saved -- a demonstration of
failing is not something to imitate.

Recording keeps going for a short while after the lift succeeds, so the
policy also learns to *hold* the cube, not just to arrive at holding it.

The result is a LeRobotDataset v3 under data/lerobot/<repo-id> (gitignored),
which `lerobot-train` reads directly:

    lerobot-train --dataset.repo_id=local/rebot_twin_pick_arc \
        --dataset.root=data/lerobot/local/rebot_twin_pick_arc \
        --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
        --policy.chunk_size=30 --policy.n_action_steps=30 \
        --batch_size=16 --steps=30000 --save_freq=10000 \
        --output_dir=outputs/act_pick_cube

A one-second chunk (30 steps) rather than ACT's default 100: an episode is
under five seconds, and 100 steps would plan most of one blind.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np

from learning.env import PickCubeEnv
from learning.scene import ARM_JOINTS, CONTROL_HZ, TWIN_PATH, ensure_twin
from learning.scripted import ScriptedPick

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "lerobot"
DEFAULT_REPO_ID = "local/rebot_twin_pick_arc"
TASK = "pick up the red cube"

#: Steps recorded after the lift counts as a success: one second of holding.
HOLD_STEPS = CONTROL_HZ

#: Seeds for recording start here, well clear of the low seeds the tests and
#: `rollout` use, so evaluating on seeds 0..N is evaluating on unseen poses.
FIRST_SEED = 10_000

JOINT_NAMES = [*ARM_JOINTS, "gripper"]


def features(env: PickCubeEnv) -> dict:
    vector = {"dtype": "float32", "shape": (len(JOINT_NAMES),), "names": JOINT_NAMES}
    out = {"observation.state": dict(vector), "action": dict(vector)}
    for camera in env.cameras:
        out[f"observation.images.{camera.name}"] = {
            "dtype": "video",
            "shape": (camera.height, camera.width, 3),
            "names": ["height", "width", "channels"],
        }
    return out


def record_episode(env: PickCubeEnv, policy: ScriptedPick, dataset, seed: int) -> int | None:
    """Add one episode's frames to `dataset`; its length, or None if it failed."""
    observation, _ = env.reset(seed=seed)
    policy.reset()
    succeeded_at = None
    for step in range(env.max_steps):
        action = env.clip_action(policy.select_action(observation)).astype(np.float32)
        frame = {
            "observation.state": observation["agent_pos"],
            "action": action,
            "task": TASK,
        }
        for name, image in observation["pixels"].items():
            frame[f"observation.images.{name}"] = image
        dataset.add_frame(frame)

        observation, _, terminated, truncated, _ = env.step(action)
        if terminated and succeeded_at is None:
            succeeded_at = step
        if succeeded_at is not None and step - succeeded_at >= HOLD_STEPS:
            return step + 1
        if truncated:
            break
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    # The cube can be anywhere on a 240 degree arc, 0.2-0.6 m out: a wide
    # task for ACT, so more than the ~50-100 demos a fixed spot would need.
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--root", type=Path, default=None, help="default: data/lerobot/<repo-id>")
    parser.add_argument("--seed", type=int, default=FIRST_SEED)
    parser.add_argument("--twin", type=Path, default=TWIN_PATH)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing dataset")
    args = parser.parse_args(argv)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = args.root or DATA_DIR / args.repo_id
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"{root} exists; pass --overwrite to replace it")
        shutil.rmtree(root)

    ensure_twin(args.twin)
    env = PickCubeEnv(twin_path=args.twin)
    policy = ScriptedPick(env)
    dataset = LeRobotDataset.create(
        args.repo_id,
        fps=CONTROL_HZ,
        features=features(env),
        root=root,
        robot_type="rebot_b601_twin",
        # Frames go to disk as PNGs before each episode is encoded to video;
        # writing them on threads keeps that off the simulation loop.
        image_writer_threads=4,
    )

    saved, seed, start = 0, args.seed, time.perf_counter()
    while saved < args.episodes:
        length = record_episode(env, policy, dataset, seed)
        if length is None:
            dataset.clear_episode_buffer()
            print(f"seed {seed}: pick failed, discarded", flush=True)
        else:
            dataset.save_episode()
            saved += 1
            elapsed = time.perf_counter() - start
            print(
                f"episode {saved}/{args.episodes} (seed {seed}): {length} frames  "
                f"[{elapsed / 60:.1f} min, ~{elapsed / saved * (args.episodes - saved) / 60:.1f} left]",
                flush=True,
            )
        seed += 1
    dataset.finalize()
    env.close()
    print(f"wrote {saved} episodes, {dataset.num_frames} frames -> {root}")


if __name__ == "__main__":
    main()
