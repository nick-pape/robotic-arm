"""Check a recorded dataset is complete and is what it claims to be.

    uv run python -m learning.check_dataset
    uv run python -m learning.check_dataset --repo-id local/smoke

Trusting the recorder's log is not the same as checking the files. This
reads them back: every episode present and contiguous, every camera video
holding exactly one frame per row, the signals finite and shaped like a
pick, and the cube actually somewhere different each episode. Exits non-zero
on the first failure, so it can gate a training run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from learning.record import DATA_DIR, DEFAULT_REPO_ID, HOLD_STEPS


def _check(ok: bool, message: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'}  {message}")
    return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.root or DATA_DIR / args.repo_id

    import json

    import av
    import pandas as pd

    info = json.loads((root / "meta" / "info.json").read_text())
    frames = pd.concat(
        pd.read_parquet(f) for f in sorted((root / "data").rglob("*.parquet"))
    )
    episodes = frames.groupby("episode_index")
    n_episodes, n_frames = info["total_episodes"], info["total_frames"]
    results = []

    results.append(_check(
        sorted(frames.episode_index.unique()) == list(range(n_episodes)),
        f"{n_episodes} episodes, indexed 0..{n_episodes - 1}",
    ))
    results.append(_check(len(frames) == n_frames, f"{len(frames)} rows == {n_frames} frames"))
    results.append(_check(
        all((e.frame_index.values == np.arange(len(e))).all() for _, e in episodes),
        "frame_index contiguous within every episode",
    ))

    for video_key in [k for k, v in info["features"].items() if v["dtype"] == "video"]:
        count = 0
        for path in sorted((root / "videos" / video_key).rglob("*.mp4")):
            with av.open(str(path)) as container:
                count += sum(1 for _ in container.decode(container.streams.video[0]))
        results.append(_check(count == n_frames, f"{video_key}: {count} decoded frames"))

    state = np.stack(frames["observation.state"].values)
    action = np.stack(frames["action"].values)
    results.append(_check(
        np.isfinite(state).all() and np.isfinite(action).all(), "state and action finite"
    ))

    # Shaped like a pick: open at the start, closed for the held tail.
    shaped = [
        e["action"].values[0][6] == 1.0
        and all(a[6] == 0.0 for a in e["action"].values[-HOLD_STEPS:])
        for _, e in episodes
    ]
    results.append(_check(all(shaped), f"{sum(shaped)}/{n_episodes} open, then close and hold"))

    # A held cube props the jaws open; an empty grip closes them fully.
    held = np.array([e["observation.state"].values[-1][6] for _, e in episodes])
    results.append(_check(
        bool((held > 0.3).all()),
        f"every episode ends holding something (final opening {held.min():.3f}-{held.max():.3f})",
    ))

    # Different cube poses demand different grasps.
    grasps = np.array([e["observation.state"].values[-1][:6] for _, e in episodes])
    spread = np.degrees(grasps.std(axis=0))
    results.append(_check(
        spread[0] > 20.0,
        f"J1 spread {spread[0]:.0f} deg across episodes (the cube moves around the base)",
    ))

    print(f"{sum(results)}/{len(results)} checks passed: {root}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
