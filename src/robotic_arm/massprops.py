"""Mass properties from build123d geometry, in the form MJCF wants.

build123d works in millimetres and OCCT returns a second moment of *volume*
(density 1), so the tensor is mm^5. MuJoCo wants kg and metres. That conversion
is the only real subtlety here; everything else is a straight copy, because
OCCT's tensor is already:

* about the centre of mass (verified translation-invariant),
* expressed in world-aligned axes at the COM, so rotating the solid rotates the
  tensor -- exactly what a body frame needs, and
* in the true inertia-tensor sign convention (off-diagonals are the negative of
  the xz integral), identical to URDF ixy/ixz/iyz and MJCF fullinertia.

So no sign flip and no re-expression is needed, provided the CAD part is
modelled in its link frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from build123d import CenterOf

from robotic_arm.materials import Material

# mm^3 -> m^3, and mm^5 -> m^5 (i.e. mm^3->m^3 times mm^2->m^2).
_MM3_TO_M3 = 1e-9
_MM5_TO_M5 = 1e-15
_MM_TO_M = 1e-3


@dataclass(frozen=True)
class MassProperties:
    """Mass properties in SI, in the form an MJCF `<inertial>` wants.

    `inertia` is about the centre of mass, in body-frame axes.
    """

    mass: float  # kg
    com: np.ndarray  # m, shape (3,), relative to the body origin
    inertia: np.ndarray  # kg*m^2, shape (3, 3), about the COM

    @property
    def fullinertia(self) -> tuple[float, float, float, float, float, float]:
        """MJCF `fullinertia` order: ixx iyy izz ixy ixz iyz. Diagonal first."""
        i = self.inertia
        return (i[0, 0], i[1, 1], i[2, 2], i[0, 1], i[0, 2], i[1, 2])

    @property
    def principal_moments(self) -> np.ndarray:
        """Eigenvalues of the inertia tensor, ascending."""
        return np.linalg.eigvalsh(self.inertia)

    def satisfies_triangle_inequality(self, tol: float = 1e-12) -> bool:
        """Whether the principal moments are physically realisable.

        MuJoCo's `balanceinertia` silently averages the diagonal when this
        fails. We never enable it: a violation means a units bug, not a physics
        problem, and should surface as a test failure.
        """
        a, b, c = self.principal_moments
        return bool(a + b >= c - tol and a + c >= b - tol and b + c >= a - tol)

    def to_mjcf_attrs(self) -> dict[str, str]:
        """Attributes for an MJCF `<inertial>` element.

        Deliberately omits `quat`: when `fullinertia` is given MuJoCo derives
        the inertial frame by eigendecomposition, and supplying both invites a
        silent inconsistency.
        """
        return {
            "pos": " ".join(f"{v:.9g}" for v in self.com),
            "mass": f"{self.mass:.9g}",
            "fullinertia": " ".join(f"{v:.9g}" for v in self.fullinertia),
        }


def mass_properties(shape, material: Material) -> MassProperties:
    """SI mass properties of a build123d shape at a material's density.

    The shape must be a solid. OCCT will happily return volume properties for a
    face or shell, so this guards explicitly rather than returning nonsense.
    """
    volume_mm3 = shape.volume
    if volume_mm3 <= 0:
        raise ValueError(
            f"shape has volume {volume_mm3}; mass properties need a solid, "
            f"not a face, shell or wire"
        )

    rho = material.density
    com_mm = np.array(tuple(shape.center(CenterOf.MASS)), dtype=float)
    # `matrix_of_inertia` is a property, not a method, and is about the COM.
    tensor_mm5 = np.array(shape.matrix_of_inertia, dtype=float)

    return MassProperties(
        mass=volume_mm3 * _MM3_TO_M3 * rho,
        com=com_mm * _MM_TO_M,
        inertia=tensor_mm5 * rho * _MM5_TO_M5,
    )


def combine(parts: Iterable[MassProperties]) -> MassProperties:
    """Combine sub-part mass properties via the parallel-axis theorem.

    build123d's own `Compound` sums these correctly, but only at uniform
    density. A real link is printed plastic plus steel bearings plus an
    aluminium flange, so the combination happens here, where each part carries
    its own material.
    """
    parts = list(parts)
    if not parts:
        raise ValueError("combine() needs at least one part")

    total = sum(p.mass for p in parts)
    if total <= 0:
        raise ValueError(f"combined mass is {total}; check densities")

    com = sum(p.mass * p.com for p in parts) / total

    inertia = np.zeros((3, 3))
    for p in parts:
        d = p.com - com
        inertia += p.inertia + p.mass * (float(d @ d) * np.eye(3) - np.outer(d, d))

    return MassProperties(mass=total, com=com, inertia=inertia)
