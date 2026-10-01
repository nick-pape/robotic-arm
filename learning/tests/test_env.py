"""The env obeys the Gymnasium contract, and the task is actually doable."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from learning.env import MAX_JOINT_STEP
from learning.scene import SPAWN_BEARING, SPAWN_RADIUS
from learning.scripted import ScriptedPick


def test_gymnasium_contract(env):
    check_env(env, skip_render_check=True)


def test_observation_layout(env):
    obs, _ = env.reset(seed=0)
    assert obs["agent_pos"].shape == (7,)
    for camera in env.cameras:
        assert obs["pixels"][camera.name].shape == (camera.height, camera.width, 3)
        assert obs["pixels"][camera.name].dtype == np.uint8
        assert obs["pixels"][camera.name].std() > 5, f"{camera.name} rendered blank"


def test_a_wild_action_moves_each_joint_one_step_at_most(blind_env):
    obs, _ = blind_env.reset(seed=0)
    q = obs["agent_pos"][:6].astype(float)
    blind_env.step(blind_env.action_space.high)
    target = blind_env.data.ctrl[blind_env._arm_act]
    assert np.all(np.abs(target - q) <= MAX_JOINT_STEP + 1e-6)


def test_reset_is_deterministic(blind_env):
    a, _ = blind_env.reset(seed=7)
    cube_a = blind_env.task_state()["cube_pos"]
    b, _ = blind_env.reset(seed=7)
    np.testing.assert_array_equal(a["agent_pos"], b["agent_pos"])
    np.testing.assert_array_equal(cube_a, blind_env.task_state()["cube_pos"])


def _pick(env, options=None, seed=0) -> bool:
    policy = ScriptedPick(env)
    obs, info = env.reset(seed=seed, options=options)
    policy.reset()
    for _ in range(env.max_steps):
        obs, _, terminated, truncated, info = env.step(policy.select_action(obs))
        if terminated or truncated:
            break
    return info["is_success"]


@pytest.mark.parametrize("radius", SPAWN_RADIUS)
@pytest.mark.parametrize("bearing", (*SPAWN_BEARING, 0.0))
def test_spawn_ring_edges_are_pickable(blind_env, radius, bearing):
    """The ring's corners and centreline: where a measured region goes wrong."""
    base = np.arctan2(blind_env._heading[1], blind_env._heading[0])
    xy = radius * np.array([np.cos(base + bearing), np.sin(base + bearing)])
    assert _pick(blind_env, {"cube_xy": xy, "cube_yaw": 0.3})


def test_spawns_fill_the_ring(blind_env):
    """Positions cover the ring, not a patch of it."""
    rng = np.random.default_rng(0)
    xy = np.array([blind_env.sample_spawn(rng)[0] for _ in range(2000)])
    radius = np.linalg.norm(xy, axis=1)
    base = np.arctan2(blind_env._heading[1], blind_env._heading[0])
    bearing = (np.arctan2(xy[:, 1], xy[:, 0]) - base + np.pi) % (2 * np.pi) - np.pi
    assert radius.min() >= SPAWN_RADIUS[0] and radius.max() <= SPAWN_RADIUS[1]
    assert np.percentile(radius, 5) < SPAWN_RADIUS[0] + 0.05
    assert np.percentile(radius, 95) > SPAWN_RADIUS[1] - 0.05
    assert bearing.min() < SPAWN_BEARING[0] + 0.1 and bearing.max() > SPAWN_BEARING[1] - 0.1


def test_scripted_pick_succeeds_across_random_spawns(blind_env):
    results = [_pick(blind_env, seed=seed) for seed in range(20)]
    assert sum(results) >= 19, f"{sum(results)}/20"


def test_grip_does_not_sink_into_the_cube(blind_env):
    """A 40 mm cube should hold the jaws ~40 mm apart, not 30."""
    policy = ScriptedPick(blind_env)
    obs, _ = blind_env.reset(seed=0)
    policy.reset()
    for _ in range(blind_env.max_steps):
        obs, _, terminated, truncated, _ = blind_env.step(policy.select_action(obs))
        if terminated or truncated:
            break
    finger = blind_env.data.qpos[blind_env._finger_qpos].mean()
    gap = 2 * finger - 0.002  # inner faces sit 1 mm past centre when closed
    assert gap == pytest.approx(0.040, abs=0.0015)
