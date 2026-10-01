"""Policy simulation: run manipulation policies against the digital twin.

Kept apart from `robotic_arm`, which designs and validates the arm. This
package only *uses* the twin it generates -- as a robot to put policies on --
and adds nothing the arm's own checks depend on.

    scene     the twin plus a task: floor, a cube, two cameras, finger pads
    env       a Gymnasium env over that scene, in LeRobot's observation layout
    ik        damped least-squares IK on the grasp site
    scripted  a scripted pick, the policy every learned one is measured against
    policies  learned policies through LeRobot (needs the `policy` extra)
    rollout   run any of them and save what the cameras saw

    uv sync --extra sim --extra policy
    uv run python -m learning.rollout --policy scripted
"""
