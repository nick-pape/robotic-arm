"""Whether the printed sections are actually strong and stiff enough.

Saving 87% of the stock mass is only good news if the sections that remain can
carry the load. Aluminium is being replaced by a material roughly a tenth as
stiff, so this checks the two things that follow from that:

* **Stiffness** -- tip deflection under load, which the spec bounds at 0.5 mm
  (requirement P3). This is usually the binding constraint on a printed arm,
  not strength.
* **Stress** -- bending stress at each link's root against what printed PC-CF
  can take. The critical number is not the in-plane strength quoted on a
  filament spool but the **interlayer** strength, typically half of it or less,
  because a printed part pulls apart between layers long before it breaks
  along them.

Sections here are treated as thin-walled circular tubes, which is what the
shells actually are along their spans. That ignores the drums, the bolt holes
and the tube-to-drum junctions -- all stress raisers -- so these numbers are an
upper bound on the structure's goodness, not a proof of it. The failure modes
this cannot see are listed in `UNMODELLED`.

Units: SI throughout (metres, newtons, pascals), except where a name says mm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITY = 9.81


@dataclass(frozen=True)
class PrintedMaterial:
    """Mechanical properties of a printed material, not of the bulk polymer.

    Printed properties are strongly anisotropic and depend on the printer as
    much as the filament, so these are conservative mid-range figures. Anything
    structural should be confirmed on a coupon from the machine that will make
    the part.
    """

    name: str
    modulus: float  # Pa, in-plane (XY)
    strength_inplane: float  # Pa, along the layers
    strength_interlayer: float  # Pa, across the layers -- the one that governs

    @property
    def anisotropy(self) -> float:
        return self.strength_interlayer / self.strength_inplane


#: Conservative values for well-dried, well-tuned prints. PC-CF is stiffer and
#: much more heat-tolerant than PETG, which is why it is the structural
#: default, but its interlayer strength is still roughly half its in-plane.
PC_CF_PRINTED = PrintedMaterial(
    name="PC-CF",
    modulus=6.0e9,
    strength_inplane=75.0e6,
    strength_interlayer=35.0e6,
)

#: 5052, for comparison with what is being replaced.
AL_5052 = PrintedMaterial(
    name="Aluminium 5052",
    modulus=70.0e9,
    strength_inplane=195.0e6,
    strength_interlayer=195.0e6,
)

#: Failure modes this analysis cannot see. Listed so the clean numbers below
#: are not mistaken for a full structural case.
UNMODELLED = (
    "stress concentration at bolt holes and the cable bore",
    "interlayer separation at the tube-to-drum junctions, where the section "
    "changes abruptly and the print direction changes with it",
    "pull-out of heat-set inserts under bolt preload",
    "local buckling of a 2 mm wall in compression",
    "torsion, which a thin tube carries well but a split or seam does not",
    "creep under sustained load, which matters because the arm holds poses",
    "fatigue over the duty cycles a working arm actually sees",
)


def second_moment(outer_mm: float, wall_mm: float) -> float:
    """Second moment of area of a hollow circular section, in m^4."""
    if wall_mm <= 0 or wall_mm >= outer_mm / 2:
        raise ValueError(f"wall {wall_mm} is not inside a {outer_mm} mm tube")
    inner_mm = outer_mm - 2 * wall_mm
    return np.pi * ((outer_mm * 1e-3) ** 4 - (inner_mm * 1e-3) ** 4) / 64


def section_modulus(outer_mm: float, wall_mm: float) -> float:
    """Section modulus Z = I / c, in m^3."""
    return second_moment(outer_mm, wall_mm) / (outer_mm * 1e-3 / 2)


def bending_stiffness(outer_mm: float, wall_mm: float, material: PrintedMaterial) -> float:
    """EI in N*m^2."""
    return material.modulus * second_moment(outer_mm, wall_mm)


def tip_deflection(
    force: float, length: float, outer_mm: float, wall_mm: float, material: PrintedMaterial
) -> float:
    """Cantilever tip deflection under an end load, in metres.

    delta = F L^3 / (3 EI). A cantilever is the right model here: each link is
    held at its root by a joint and loaded by everything distal of it.
    """
    return force * length**3 / (3 * bending_stiffness(outer_mm, wall_mm, material))


def bending_stress(moment: float, outer_mm: float, wall_mm: float) -> float:
    """Peak bending stress from a moment, in Pa."""
    return moment / section_modulus(outer_mm, wall_mm)


@dataclass(frozen=True)
class SectionCheck:
    """Structural verdict for one link's span."""

    link: str
    outer_mm: float
    wall_mm: float
    length: float
    distal_mass: float
    payload: float
    deflection_mm: float
    stress_mpa: float
    material: PrintedMaterial

    @property
    def stiffness_ratio_to_aluminium(self) -> float:
        """How this section compares with the same section in 5052."""
        return self.material.modulus / AL_5052.modulus

    @property
    def safety_factor_interlayer(self) -> float:
        """Against the weak direction, which is the one that governs."""
        return self.material.strength_interlayer / (self.stress_mpa * 1e6)

    @property
    def passes_p3(self) -> bool:
        """Spec P3 bounds tip deflection at 0.5 mm."""
        return self.deflection_mm <= 0.5


def distal_mass(model, link: str) -> float:
    """Total mass of everything outboard of a link, kg.

    This is the load the link's own section has to carry, so it is what sets
    both its deflection and its root stress.
    """
    import mujoco

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)
    if bid < 0:
        raise KeyError(link)

    total, frontier = 0.0, [bid]
    seen = set()
    while frontier:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)
        for j in range(model.nbody):
            if model.body_parentid[j] == current and j != current:
                total += float(model.body_mass[j])
                frontier.append(j)
    return total


def check_link(
    model,
    link: str,
    outer_mm: float,
    wall_mm: float,
    payload: float = 0.0,
    material: PrintedMaterial = PC_CF_PRINTED,
) -> SectionCheck:
    """Structural check of one link's span, loaded by everything distal of it."""
    from robotic_arm.linkframes import link_frame

    frame = link_frame(link)
    length = frame.span * 1e-3
    carried = distal_mass(model, link) + payload
    force = carried * GRAVITY
    # Worst case is the link horizontal with the load at its far end.
    moment = force * length

    return SectionCheck(
        link=link,
        outer_mm=outer_mm,
        wall_mm=wall_mm,
        length=length,
        distal_mass=carried - payload,
        payload=payload,
        deflection_mm=tip_deflection(force, length, outer_mm, wall_mm, material) * 1e3,
        stress_mpa=bending_stress(moment, outer_mm, wall_mm) * 1e-6,
        material=material,
    )


@dataclass(frozen=True)
class JointCheck:
    """Whether a bolted flange in printed material can carry its moment.

    Usually the real limit on a printed structure, and the reason the spec says
    to bolt motors through metal spacers rather than into plastic. A
    large-diameter tube is an efficient section and carries bending easily; the
    load then arrives at a small bolt circle where a few M3s bear on a 2 mm
    wall, and the material has to take it in its weakest direction.
    """

    name: str
    moment: float
    bolt_count: int
    bcd_mm: float
    hole_mm: float
    wall_mm: float
    material: PrintedMaterial

    @property
    def bolt_force(self) -> float:
        """Peak tensile load on the worst-placed bolt, N.

        For a circular pattern reacting a moment, the bolts share it as a
        couple: the standard estimate is F = 4M / (n * D).
        """
        return 4 * self.moment / (self.bolt_count * self.bcd_mm * 1e-3)

    @property
    def bearing_stress(self) -> float:
        """Stress where the bolt bears on the printed hole, Pa.

        This is the number that usually governs: the projected area is only the
        hole diameter times the wall, which on a 2 mm wall is very small.
        """
        area = self.hole_mm * 1e-3 * self.wall_mm * 1e-3
        return self.bolt_force / area

    @property
    def safety_factor(self) -> float:
        return self.material.strength_inplane / self.bearing_stress

    @property
    def wall_needed_mm(self) -> float:
        """Wall thickness for a safety factor of 2 in bearing."""
        return 2.0 * self.bolt_force / (self.hole_mm * 1e-3 * self.material.strength_inplane) * 1e3
