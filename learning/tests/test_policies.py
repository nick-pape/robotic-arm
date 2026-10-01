"""Learned policies run in the env end to end, trained or not."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lerobot")

from learning.policies import ActPolicy  # noqa: E402


def test_untrained_act_drives_the_env(env):
    policy = ActPolicy(env, chunk_size=10)
    obs, _ = env.reset(seed=0)
    policy.reset()
    for _ in range(12):  # crosses a chunk boundary
        action = policy.select_action(obs)
        assert action.shape == env.action_space.shape
        assert np.isfinite(action).all()
        obs, *_ = env.step(action)
