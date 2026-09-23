"""Printed replacement parts, worked distal inwards.

Each module exposes a ``build_*`` function returning a build123d solid in its
link frame, and a ``MATERIAL`` naming what it is printed in. ``registry``
maps MJCF body names to those, so the MJCF generator can swap inertials
without knowing anything about the CAD.
"""

from __future__ import annotations

from collections.abc import Callable

from robotic_arm.materials import Material
from robotic_arm.parts.forearm import MATERIAL as FOREARM_MATERIAL
from robotic_arm.parts.upper_arm import MATERIAL as UPPER_ARM_MATERIAL
from robotic_arm.parts.upper_arm import build_upper_arm
from robotic_arm.parts.forearm import build_forearm
from robotic_arm.parts.tool_flange import MATERIAL as TOOL_FLANGE_MATERIAL
from robotic_arm.parts.tool_flange import build_tool_flange
from robotic_arm.parts.wrist_pitch import MATERIAL as WRIST_PITCH_MATERIAL
from robotic_arm.parts.wrist_pitch import build_wrist_pitch
from robotic_arm.parts.wrist_roll import MATERIAL as WRIST_ROLL_MATERIAL
from robotic_arm.parts.wrist_roll import build_wrist_roll

#: MJCF body name -> (builder, material). Populated distal inwards as parts
#: are designed; bodies absent here keep their stock inertials.
REGISTRY: dict[str, tuple[Callable[[], object], Material]] = {
    "link2": (build_upper_arm, UPPER_ARM_MATERIAL),
    "link3": (build_forearm, FOREARM_MATERIAL),
    "link4": (build_wrist_pitch, WRIST_PITCH_MATERIAL),
    "link5": (build_wrist_roll, WRIST_ROLL_MATERIAL),
    "link6": (build_tool_flange, TOOL_FLANGE_MATERIAL),
}

__all__ = [
    "REGISTRY",
    "build_forearm",
    "build_upper_arm",
    "build_tool_flange",
    "build_wrist_pitch",
    "build_wrist_roll",
]


def effective_material(shape, material: Material) -> Material:
    """Effective printed density for a structural part, from its geometry."""
    from robotic_arm.design import RULES
    from robotic_arm.massprops import printed_density

    return printed_density(
        shape,
        material,
        infill=RULES.structural_infill,
        walls=RULES.structural_walls,
        nozzle=RULES.nozzle,
    )


def carried_actuator(body: str):
    """The actuator a link carries, and where it sits in that link's frame.

    A joint's motor is mounted on its parent link, so a link carries the
    actuator for the joint *below* it -- link3 carries the J4 motor, and so on.
    Returns (Actuator, centre_mm, axis) or None for a link that carries none.
    """
    import importlib

    from robotic_arm.actuators import JOINT_ACTUATOR, get
    from robotic_arm.linkframes import link_frame

    frame = link_frame(body)
    if frame.child_name is None:
        return None
    # The child link's own joint is the one this link's motor drives.
    child_joint = None
    for joint, _ in JOINT_ACTUATOR.items():
        if joint == f"joint{frame.child_name.removeprefix('link')}":
            child_joint = joint
    if child_joint is None:
        return None

    module = importlib.import_module(REGISTRY[body][0].__module__)
    drums = getattr(module, "_drums", None)
    if drums is None:
        return None
    _, child = drums()
    return get(JOINT_ACTUATOR[child_joint]), child.centre, child.axis


def actuator_mass_properties(body: str):
    """Mass properties of the actuator a link carries, in the link's frame.

    Modelled as a solid cylinder of the actuator's own envelope, scaled to its
    published mass. That is far better than omitting it: an RS06 is 621 g
    against a 166 g printed shell, so leaving it out would understate the link
    by a factor of four and flatter every torque number downstream.
    """
    from dataclasses import replace

    from robotic_arm.massprops import mass_properties
    from robotic_arm.materials import Material
    from robotic_arm.parts.cobot import Drum

    carried = carried_actuator(body)
    if carried is None:
        return None
    actuator, centre, axis = carried

    diameter = min(actuator.bbox_mm[0], actuator.bbox_mm[1])
    body_solid = Drum(centre, axis, diameter, actuator.bbox_mm[2]).solid()

    # Pick a density that reproduces the published mass on this envelope.
    density = actuator.mass_kg / (body_solid.volume * 1e-9)
    return mass_properties(body_solid, Material(actuator.name, density))


def child_interface_diameter(body: str) -> float | None:
    """Outer diameter of whatever the child link presents at this joint.

    A parent's housing has to open far enough for the child's own mounting
    boss to enter it. Without that opening the housing's end cap and the
    child's boss occupy the same volume -- which is most of the 2,289 mm3 of
    printed-material overlap the design review measured between link2 and
    link3. They cannot both be there.
    """
    import importlib

    from robotic_arm.linkframes import link_frame

    child = link_frame(body).child_name
    if child is None or child not in REGISTRY:
        return None

    module = importlib.import_module(REGISTRY[child][0].__module__)
    drums = getattr(module, "_drums", None)
    if drums is not None:
        parent, _ = drums()
        return float(parent.diameter)
    # A child with no boss presents its own outer body, like the tool flange.
    return float(getattr(module, "OUTER_DIAMETER", 0.0)) or None
