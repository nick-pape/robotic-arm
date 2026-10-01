"""A Gymnasium env for picking a cube with the twin.

Observations follow the layout LeRobot's gym envs use, so its
`preprocess_observation` turns them into policy inputs with no glue:

    agent_pos            float32[7]  six joint angles (rad), gripper opening (0..1)
    pixels/<camera>      uint8[H,W,3] one entry per camera in `scene.CAMERAS`

Actions are absolute targets in the same units: six joint angles in radians
and a gripper command from 0 (closed) to 1 (open). Radians rather than the
real follower's degrees, because the sim is native in them; the conversion
belongs in the one place that talks to hardware, not in every policy.

Every action is clipped twice before it reaches a servo: to the joint ranges,
and to `MAX_JOINT_STEP` from where the joint is now. That second clip is the
sim's version of the safety layer the real arm needs between a policy and its
motors, and it means an untrained policy flails slowly instead of violently.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from learning.scene import (
    ARM_JOINTS,
    GRIPPER_ACTUATOR,
    CAMERAS,
    CONTROL_HZ,
    CUBE,
    CUBE_HALF,
    FINGER_JOINTS,
    GRASP_SITE,
    SCENE_CAMERA,
    SPAWN_BEARING,
    SPAWN_RADIUS,
    SUBSTEPS,
    home_heading,
    load_task_model,
)

#: Largest change a single action may make to any arm joint, rad. At 30 Hz
#: this is 4.5 rad/s, well inside what an RS00 or RS06 can do unloaded.
MAX_JOINT_STEP = 0.15

#: The cube is square, so a quarter turn covers every grasp it can offer.
SPAWN_YAW = (-np.pi / 4, np.pi / 4)

#: Success: the cube is this far above where it rests, and still in the jaws.
LIFT_HEIGHT = 0.05
HELD_DISTANCE = 0.03

#: Time for the arm to settle under gravity after a reset, before the first
#: observation -- otherwise the first frames show the servos taking up sag.
SETTLE_SECONDS = 0.3


class PickCubeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": CONTROL_HZ}

    def __init__(
        self,
        twin_path: Path | None = None,
        max_steps: int = 300,
        render_mode: str | None = "rgb_array",
        cameras=CAMERAS,
    ):
        self.model = load_task_model(twin_path)
        self.data = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.cameras = tuple(cameras)

        m = self.model
        arm = [m.joint(n) for n in ARM_JOINTS]
        self._arm_qpos = np.array([int(j.qposadr[0]) for j in arm])
        self._arm_act = np.array([m.actuator(n).id for n in ARM_JOINTS])
        self._finger_qpos = np.array([int(m.joint(n).qposadr[0]) for n in FINGER_JOINTS])
        self._gripper_act = m.actuator(GRIPPER_ACTUATOR).id
        self._finger_open = float(m.joint(FINGER_JOINTS[0]).range[1])
        self._cube_qpos = int(m.joint(CUBE).qposadr[0])
        self._cube_body = m.body(CUBE).id
        self._grasp_site = m.site(GRASP_SITE).id
        self._home_key = m.key("home").id
        self._heading = home_heading(m)

        low = np.array([j.range[0] for j in arm])
        high = np.array([j.range[1] for j in arm])
        self.action_space = spaces.Box(
            low=np.append(low, 0.0).astype(np.float32),
            high=np.append(high, 1.0).astype(np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(
                    low=self.action_space.low,
                    high=self.action_space.high,
                    dtype=np.float32,
                ),
                "pixels": spaces.Dict(
                    {
                        c.name: spaces.Box(0, 255, (c.height, c.width, 3), np.uint8)
                        for c in self.cameras
                    }
                ),
            }
        )

        self._renderers = {}
        self._steps = 0
        self._cube_rest_z = CUBE_HALF

    # -- state ---------------------------------------------------------------

    def _gripper_opening(self) -> float:
        return float(np.mean(self.data.qpos[self._finger_qpos]) / self._finger_open)

    def _agent_pos(self) -> np.ndarray:
        q = self.data.qpos[self._arm_qpos]
        return np.append(q, self._gripper_opening()).astype(np.float32)

    def _render(self, camera) -> np.ndarray:
        renderer = self._renderers.get(camera.name)
        if renderer is None:
            renderer = mujoco.Renderer(self.model, camera.height, camera.width)
            self._renderers[camera.name] = renderer
        renderer.update_scene(self.data, camera=camera.name)
        # Shadows triple render time on the twin's ~850k-triangle CAD meshes,
        # and a policy does not need them.
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        return renderer.render()

    def _observation(self) -> dict:
        return {
            "agent_pos": self._agent_pos(),
            "pixels": {c.name: self._render(c) for c in self.cameras},
        }

    def task_state(self) -> dict:
        """Privileged state a scripted expert may use and a policy may not."""
        quat = self.data.qpos[self._cube_qpos + 3 : self._cube_qpos + 7].copy()
        return {
            "cube_pos": self.data.xpos[self._cube_body].copy(),
            "cube_yaw": float(2.0 * np.arctan2(quat[3], quat[0])),
            "grasp_pos": self.data.site_xpos[self._grasp_site].copy(),
            "qpos": self.data.qpos[self._arm_qpos].copy(),
        }

    def _info(self) -> dict:
        state = self.task_state()
        lifted = state["cube_pos"][2] - self._cube_rest_z
        held = np.linalg.norm(state["cube_pos"] - state["grasp_pos"]) < HELD_DISTANCE
        return {"is_success": bool(lifted > LIFT_HEIGHT and held), "lift": float(lifted)}

    # -- gym API -------------------------------------------------------------

    def sample_spawn(self, rng: np.random.Generator) -> tuple[np.ndarray, float]:
        """A random cube position (x, y) and yaw inside the spawn ring.

        Radius is drawn so positions are uniform by *area*: drawing it
        uniformly would crowd the cube toward the base, where the ring's
        circumference is shortest.
        """
        r_in, r_out = SPAWN_RADIUS
        radius = np.sqrt(rng.uniform(r_in**2, r_out**2))
        bearing = rng.uniform(*SPAWN_BEARING) + np.arctan2(self._heading[1], self._heading[0])
        xy = radius * np.array([np.cos(bearing), np.sin(bearing)])
        return xy, float(rng.uniform(*SPAWN_YAW))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, self._home_key)

        xy, yaw = self.sample_spawn(self.np_random)
        # An explicit pose, for sweeps and for replaying a specific setup.
        options = options or {}
        if "cube_xy" in options:
            xy = np.asarray(options["cube_xy"], dtype=float)
        if "cube_yaw" in options:
            yaw = float(options["cube_yaw"])
        pos = np.array([xy[0], xy[1], CUBE_HALF])
        d.qpos[self._cube_qpos : self._cube_qpos + 3] = pos
        d.qpos[self._cube_qpos + 3 : self._cube_qpos + 7] = [
            np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)
        ]

        d.qpos[self._finger_qpos] = self._finger_open
        d.ctrl[self._arm_act] = d.qpos[self._arm_qpos]
        d.ctrl[self._gripper_act] = self._finger_open
        mujoco.mj_forward(m, d)
        for _ in range(int(SETTLE_SECONDS / m.opt.timestep)):
            mujoco.mj_step(m, d)

        self._cube_rest_z = float(d.xpos[self._cube_body][2])
        self._steps = 0
        return self._observation(), self._info()

    def clip_action(self, action) -> np.ndarray:
        """The action as it will actually be applied: inside the joint ranges,
        and within `MAX_JOINT_STEP` of where each joint is now. Recording
        this, not the raw request, keeps datasets to commands the arm obeyed."""
        action = np.clip(
            np.asarray(action, dtype=float), self.action_space.low, self.action_space.high
        )
        q = self.data.qpos[self._arm_qpos]
        action[:6] = np.clip(action[:6], q - MAX_JOINT_STEP, q + MAX_JOINT_STEP)
        return action

    def step(self, action):
        action = self.clip_action(action)
        self.data.ctrl[self._arm_act] = action[:6]
        self.data.ctrl[self._gripper_act] = action[6] * self._finger_open

        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        self._steps += 1

        info = self._info()
        terminated = info["is_success"]
        truncated = self._steps >= self.max_steps
        return self._observation(), float(terminated), terminated, truncated, info

    def render(self) -> np.ndarray:
        return self._render(SCENE_CAMERA)

    def close(self):
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()


gym.register(id="RoboticArm/PickCube-v0", entry_point="learning.env:PickCubeEnv")
