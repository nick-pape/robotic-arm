"""Printed replacement parts, worked distal inwards.

Each module exposes a ``build_*`` function returning a build123d solid in its
link frame, and a ``MATERIAL`` naming what it is printed in. ``registry``
maps MJCF body names to those, so the MJCF generator can swap inertials
without knowing anything about the CAD.
"""

from __future__ import annotations

from collections.abc import Callable

from robotic_arm.materials import Material
from robotic_arm.parts.tool_flange import MATERIAL as TOOL_FLANGE_MATERIAL
from robotic_arm.parts.tool_flange import build_tool_flange

#: MJCF body name -> (builder, material). Populated distal inwards as parts
#: are designed; bodies absent here keep their stock inertials.
REGISTRY: dict[str, tuple[Callable[[], object], Material]] = {
    "link6": (build_tool_flange, TOOL_FLANGE_MATERIAL),
}

__all__ = ["REGISTRY", "build_tool_flange"]


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
