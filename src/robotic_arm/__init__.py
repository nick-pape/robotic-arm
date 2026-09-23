"""Digital twin of the reBot Arm B601-RS with a printed structure.

Two sources of truth, never mixed:

* joint frames and kinematics come from the vendored stock model in
  ``reference/`` and are never edited;
* inertial properties come from our own build123d CAD.

See ``spec/rebot-arm-b601-rs-clone.md`` for the engineering brief.
"""

from robotic_arm.reference import ARM_JOINTS, load_baseline

__all__ = ["ARM_JOINTS", "load_baseline"]
