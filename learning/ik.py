"""Damped least-squares IK for the grasp site.

Numerical, not analytic: the wrist is a UR-style tee now, but the frames have
been shifted by hand (`linkframes.JOINT_SHIFT`) and will be again, and a
Jacobian solver keeps working through every such change without being
re-derived. It runs on a scratch `MjData`, so solving never disturbs the
simulation it is planning for.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from learning.scene import ARM_JOINTS, GRASP_SITE


@dataclass(frozen=True)
class Solution:
    q: np.ndarray
    position_error: float  # m
    rotation_error: float  # rad
    converged: bool


class SiteIK:
    """IK over the six arm joints, for the grasp site's position and frame."""

    def __init__(self, model: mujoco.MjModel, site: str = GRASP_SITE):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site = model.site(site).id
        joints = [model.joint(name) for name in ARM_JOINTS]
        self.qpos = np.array([int(j.qposadr[0]) for j in joints])
        self.dofs = np.array([int(j.dofadr[0]) for j in joints])
        self.low = np.array([j.range[0] for j in joints])
        self.high = np.array([j.range[1] for j in joints])

    def self_collides(self, q: np.ndarray, ignore: set[int] = frozenset()) -> bool:
        """Whether the arm at `q` touches itself or the floor.

        Contacts involving any geom in `ignore` (the object being grasped)
        don't count; the scratch data has objects at their default poses.
        """
        self.data.qpos[self.qpos] = q
        mujoco.mj_forward(self.model, self.data)
        for contact in self.data.contact[: self.data.ncon]:
            if contact.geom1 in ignore or contact.geom2 in ignore:
                continue
            return True
        return False

    def site_pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Grasp-site position and rotation matrix for arm angles `q`."""
        self.data.qpos[self.qpos] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        return (
            self.data.site_xpos[self.site].copy(),
            self.data.site_xmat[self.site].reshape(3, 3).copy(),
        )

    def solve(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        q_init: np.ndarray,
        iterations: int = 200,
        damping: float = 0.05,
        max_step: float = 0.2,
        position_tol: float = 1e-3,
        rotation_tol: float = 1e-2,
    ) -> Solution:
        """Pose the grasp site at `target_pos` with frame `target_rot`.

        Joint limits are enforced by clamping each step, which is crude but
        adequate here: the limits are wide (most joints a full turn), and a
        solution that ends pinned against one reports itself as unconverged.
        """
        q = np.array(q_init, dtype=float)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        err_rot = np.zeros(3)
        quat_err = np.zeros(4)
        quat_now = np.zeros(4)
        quat_goal = np.zeros(4)
        mujoco.mju_mat2Quat(quat_goal, np.asarray(target_rot, dtype=float).ravel())

        for _ in range(iterations):
            pos, rot = self.site_pose(q)
            err_pos = target_pos - pos
            mujoco.mju_mat2Quat(quat_now, rot.ravel())
            mujoco.mju_negQuat(quat_err, quat_now)
            mujoco.mju_mulQuat(quat_err, quat_goal, quat_err)
            mujoco.mju_quat2Vel(err_rot, quat_err, 1.0)
            if np.linalg.norm(err_pos) < position_tol and np.linalg.norm(err_rot) < rotation_tol:
                break

            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site)
            jac = np.vstack([jacp[:, self.dofs], jacr[:, self.dofs]])
            err = np.concatenate([err_pos, err_rot])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(6), err)
            dq *= min(1.0, max_step / (np.abs(dq).max() + 1e-12))
            q = np.clip(q + dq, self.low, self.high)

        pos, rot = self.site_pose(q)
        mujoco.mju_mat2Quat(quat_now, rot.ravel())
        mujoco.mju_negQuat(quat_err, quat_now)
        mujoco.mju_mulQuat(quat_err, quat_goal, quat_err)
        mujoco.mju_quat2Vel(err_rot, quat_err, 1.0)
        pos_err = float(np.linalg.norm(target_pos - pos))
        rot_err = float(np.linalg.norm(err_rot))
        return Solution(
            q=q,
            position_error=pos_err,
            rotation_error=rot_err,
            converged=pos_err < position_tol * 2 and rot_err < rotation_tol * 2,
        )


def top_down(jaw_yaw: float) -> np.ndarray:
    """Tool frame for a vertical grasp: jaws pointing down, opening along yaw.

    Columns are the tool's x (out of the jaws), y (the opening axis) and z.
    """
    x = np.array([0.0, 0.0, -1.0])
    y = np.array([np.cos(jaw_yaw), np.sin(jaw_yaw), 0.0])
    return np.column_stack([x, y, np.cross(x, y)])
