"""Access to the vendored upstream reference model.

This module is the only place that knows where reference artifacts live. It
loads the stock arm as published; nothing here is derived from our own CAD.

Units are MuJoCo's: metres, kilograms, newton-metres. (CAD-side code works in
millimetres — see `massprops` for the conversion.)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import mujoco

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "reference"
BASELINE_MJCF = REFERENCE_DIR / "mjcf" / "seeed_rebot_devarm.xml"
SCENE_MJCF = REFERENCE_DIR / "mjcf" / "scene.xml"
STOCK_URDF = REFERENCE_DIR / "urdf" / "RS" / "ReBot_Arm_RS.urdf"
RS06_STEP = REFERENCE_DIR / "step" / "RS06-new.step"
RS00_STEP = REFERENCE_DIR / "step" / "RS00.step"

#: The six revolute arm joints, shoulder to wrist. Excludes the two gripper
#: slides, which are prismatic and driven together.
ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")

#: Actuator assignment, from the upstream `config/rebotarm_rs.yaml`.
JOINT_ACTUATOR = {
    "joint1": "RS06",
    "joint2": "RS06",
    "joint3": "RS06",
    "joint4": "RS00",
    "joint5": "RS00",
    "joint6": "RS00",
}


class ReferenceMissingError(FileNotFoundError):
    """Raised when a fetched-on-demand reference artifact is absent."""

    def __init__(self, path: Path) -> None:
        super().__init__(
            f"{path} is missing. Reference meshes and STEP files are not "
            f"committed; fetch them with:\n"
            f"    uv run python scripts/fetch_reference.py"
        )


def require(path: Path) -> Path:
    """Return `path`, with an actionable error if it has not been fetched."""
    if not path.exists():
        raise ReferenceMissingError(path)
    return path


@lru_cache(maxsize=2)
def load_baseline() -> mujoco.MjModel:
    """Load the stock arm exactly as Menagerie publishes it.

    This is the A/B baseline every later change is measured against, so it is
    deliberately read-only and cached: callers get the same compiled model.
    """
    require(BASELINE_MJCF)
    # from_xml_path resolves meshdir relative to the XML, so assets/ is found
    # without changing the working directory.
    return mujoco.MjModel.from_xml_path(str(BASELINE_MJCF))
