"""A scripted pick: the expert every learned policy gets measured against.

It cheats on purpose -- it reads the cube's pose straight out of the
simulator -- because its jobs are to prove the arm *can* do the task in this
scene (reach, grasp contacts, lift under the grip force cap), and later to
generate demonstrations. A learned policy sees only `agent_pos` and pixels.

Open loop once planned: approach above the cube, descend in a straight line,
close, lift. The cube is square, so there are four ways to grasp it; the plan
takes whichever needs the least joint motion to reach.
"""

from __future__ import annotations

import numpy as np

from learning.env import PickCubeEnv
from learning.ik import SiteIK, top_down
from learning.scene import CUBE_HALF

#: Grasp-site height above the floor at the grasp. The site is 18 mm back
#: from the fingertips, so this leaves the tips ~5 mm clear of the floor.
GRASP_HEIGHT = CUBE_HALF + 0.003
APPROACH_HEIGHT = 0.10  # above the grasp, where the descent starts
LIFT_HEIGHT = 0.15

#: Phase durations, in control steps (30 Hz).
MOVE_STEPS = 45
DESCEND_STEPS = 30
SETTLE_STEPS = 6  # let the servos catch up with the descent before closing
CLOSE_STEPS = 15
LIFT_STEPS = 30

#: Average joint speed for the opening move, rad per step. The eased profile
#: peaks at pi/2 times its average, so this keeps the peak under 0.15.
MAX_MOVE_RATE = 0.08

OPEN, CLOSED = 1.0, 0.0


def _ease(n: int) -> np.ndarray:
    """0 to 1 over n steps, smooth at both ends."""
    t = np.linspace(0.0, 1.0, n + 1)[1:]
    return 0.5 - 0.5 * np.cos(np.pi * t)


class ScriptedPick:
    def __init__(self, env: PickCubeEnv):
        self.env = env
        self.ik = SiteIK(env.model)
        self._plan: list[np.ndarray] = []

    def reset(self):
        self._plan = []

    def select_action(self, observation) -> np.ndarray:
        if not self._plan:
            self._plan = self._make_plan()
        action = self._plan.pop(0) if len(self._plan) > 1 else self._plan[0]
        return action

    def planned_actions(self) -> list[np.ndarray]:
        """Actions still to come, for drawing where the arm is headed."""
        return list(self._plan)

    def _facing(self, cube_pos, q_now) -> np.ndarray:
        """`q_now` with J1 turned to face the cube.

        Seeding from home alone, a cube behind the arm draws the solver into
        reaching back over the shoulder -- a valid IK solution whose gripper
        runs into the upper arm. Turning the base first finds the forward one.
        """
        home = self.env._heading
        turn = np.arctan2(cube_pos[1], cube_pos[0]) - np.arctan2(home[1], home[0])
        seed = q_now.copy()
        seed[0] = (turn + np.pi) % (2 * np.pi) - np.pi
        return seed

    def _grasp_solution(self, cube_pos, cube_yaw, q_now):
        """Best collision-free square grasp, by joint motion from `q_now`."""
        target = np.array([cube_pos[0], cube_pos[1], GRASP_HEIGHT])
        cube = {self.env.model.geom("cube").id}
        candidates = []
        for seed in (q_now, self._facing(cube_pos, q_now)):
            for k in range(4):
                rot = top_down(cube_yaw + k * np.pi / 2)
                solution = self.ik.solve(target, rot, seed, iterations=400)
                if solution.converged and not self.ik.self_collides(solution.q, cube):
                    candidates.append((np.abs(solution.q - q_now).sum(), rot, solution))
        if not candidates:
            raise RuntimeError(f"no top-down grasp reaches the cube at {cube_pos}")
        _, rot, solution = min(candidates, key=lambda c: c[0])
        return target, rot, solution.q

    def _line(self, start, end, rot, q_seed, steps, gripper):
        """Joint targets tracing a straight line of the grasp site."""
        actions, q = [], q_seed
        for s in _ease(steps):
            q = self.ik.solve(start + s * (end - start), rot, q, iterations=50).q
            actions.append(np.append(q, gripper))
        return actions, q

    def _make_plan(self) -> list[np.ndarray]:
        state = self.env.task_state()
        q_now = state["qpos"]
        grasp, rot, q_grasp = self._grasp_solution(
            state["cube_pos"], state["cube_yaw"], q_now
        )
        above = grasp + np.array([0.0, 0.0, APPROACH_HEIGHT])
        q_above = self.ik.solve(above, rot, q_grasp, iterations=400).q

        # Long enough that the eased profile's peak stays inside the env's
        # per-step joint clip, or a far cube (J1 turning half a revolution)
        # gets a plan the arm cannot follow.
        steps = max(MOVE_STEPS, int(np.ceil(np.abs(q_above - q_now).max() / MAX_MOVE_RATE)))
        plan = [np.append(q_now + s * (q_above - q_now), OPEN) for s in _ease(steps)]
        descend, q = self._line(above, grasp, rot, q_above, DESCEND_STEPS, OPEN)
        plan += descend
        plan += [np.append(q, OPEN)] * SETTLE_STEPS
        plan += [np.append(q, CLOSED)] * CLOSE_STEPS
        lifted = grasp + np.array([0.0, 0.0, LIFT_HEIGHT])
        lift, _ = self._line(grasp, lifted, rot, q, LIFT_STEPS, CLOSED)
        plan += lift
        return plan


#: Pick-and-place timings, in control steps.
CARRY_STEPS = 60
RELEASE_STEPS = 12
RETREAT_STEPS = 20

#: The cube is set down this far above resting height, and drops the rest.
PLACE_CLEARANCE = 0.002

#: A new spot must be at least this far from the old one, or it is not a move.
MIN_MOVE = 0.10


class ScriptedPickPlace(ScriptedPick):
    """Pick the cube, put it down somewhere new, then go and pick it again.

    Runs indefinitely: whenever a plan runs out it plans the next cycle from
    wherever the cube actually landed, so every pick is against a fresh,
    unscripted position. The env still reports success at the first lift;
    run it with lingering (see `rollout --viewer`) to see the rest.
    """

    def __init__(self, env: PickCubeEnv, seed: int = 0):
        super().__init__(env)
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation) -> np.ndarray:
        if len(self._plan) <= 1:
            self._plan = self._make_plan()
        return self._plan.pop(0)

    def _new_spot(self, old: np.ndarray) -> np.ndarray:
        while True:
            xy, _ = self.env.sample_spawn(self.rng)
            if np.linalg.norm(xy - old[:2]) >= MIN_MOVE:
                return xy

    def _make_plan(self) -> list[np.ndarray]:
        plan = super()._make_plan()
        q = plan[-1][:6]
        state = self.env.task_state()
        rot = self.ik.site_pose(q)[1]

        lifted = self.ik.site_pose(q)[0]
        spot = self._new_spot(state["cube_pos"])
        place = np.array([spot[0], spot[1], GRASP_HEIGHT + PLACE_CLEARANCE])
        above = place + np.array([0.0, 0.0, lifted[2] - place[2]])

        carry, q = self._line(lifted, above, rot, q, CARRY_STEPS, CLOSED)
        lower, q = self._line(above, place, rot, q, DESCEND_STEPS, CLOSED)
        plan += carry + lower
        plan += [np.append(q, OPEN)] * RELEASE_STEPS
        retreat, q = self._line(
            place, place + np.array([0.0, 0.0, APPROACH_HEIGHT]), rot, q, RETREAT_STEPS, OPEN
        )
        plan += retreat
        plan += [np.append(q, OPEN)] * SETTLE_STEPS  # let the cube come to rest
        return plan
