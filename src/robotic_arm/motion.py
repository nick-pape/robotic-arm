"""A demonstration trajectory, shared by the live viewer and the renderer.

Waypoints are chosen to show the working envelope and to pass through the pose
that loads J2 hardest, since that pose is what the whole torque analysis is
about. Targets are fed to the model's position actuators rather than written
into qpos, so the arm tracks them under gravity and what you see is the
controller doing real work against the load.
"""

from __future__ import annotations

import numpy as np

#: (name, joint1..joint6 targets in radians)
WAYPOINTS: list[tuple[str, tuple[float, ...]]] = [
    ("home", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    ("raised", (0.0, -0.70, -1.10, 0.0, 0.0, 0.0)),
    ("reach out", (0.0, -1.60, -0.55, 0.0, -0.60, 0.0)),
    ("full extension", (0.0, -2.60, -0.30, 0.0, -0.40, 0.0)),
    ("swing left", (1.20, -2.20, -0.60, 0.0, -0.40, 0.8)),
    ("swing right", (-1.20, -2.20, -0.60, 0.0, -0.40, -0.8)),
    ("tuck", (0.0, -0.40, -2.20, 0.0, 0.60, 0.0)),
    ("home", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
]


def smoothstep(t: float) -> float:
    """Ease in and out, so the arm does not jerk between waypoints."""
    return t * t * (3.0 - 2.0 * t)


def trajectory(steps_per_leg: int) -> list[np.ndarray]:
    """Interpolated control targets across every consecutive waypoint pair."""
    out: list[np.ndarray] = []
    for (_, start), (_, end) in zip(WAYPOINTS, WAYPOINTS[1:]):
        a, b = np.array(start, dtype=float), np.array(end, dtype=float)
        out.extend(a + (b - a) * smoothstep(i / steps_per_leg) for i in range(steps_per_leg))
    out.append(np.array(WAYPOINTS[-1][1], dtype=float))
    return out


def leg_names(steps_per_leg: int) -> list[str]:
    """The waypoint each trajectory sample is heading towards."""
    names = [end for (_, _), (end, _) in zip(WAYPOINTS, WAYPOINTS[1:])]
    out: list[str] = []
    for name in names:
        out.extend([name] * steps_per_leg)
    out.append(WAYPOINTS[-1][0])
    return out
