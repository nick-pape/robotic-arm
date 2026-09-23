"""Design rules for the printed structure.

Two things live here: the print-process rules the spec lays down (section 3,
"Design rules"), and the styling constants that keep the parts looking like one
machine rather than eight separate exercises.

The styling target is a standard collaborative-robot form: clean cylindrical
shells, mostly closed, joints reading as continuous tubes rather than as
brackets bolted together. That is a Universal-Robots-like language, chosen
because it suits printed parts -- a cylinder is stiff in every bending
direction, prints without support on its axis, and hides the cable run.

Units: millimetres, degrees.
"""

from __future__ import annotations

from dataclasses import dataclass

from robotic_arm.materials import PC_CF, PETG, PETG_CF, Material


@dataclass(frozen=True)
class PrintRules:
    """Process constraints for FDM parts, from the spec's design rules."""

    nozzle: float = 0.4
    layer: float = 0.2

    #: >= 5 perimeters on load paths. At a 0.4 nozzle that is ~2.0 mm.
    structural_walls: int = 5
    cosmetic_walls: int = 3
    structural_infill: float = 0.40
    cosmetic_infill: float = 0.25

    #: Heat-set inserts: M3 wants a 4.0 mm hole, with a boss 8-9 mm across so
    #: at least 1.6 mm of wall remains around it.
    m3_insert_hole: float = 4.0
    m3_insert_depth: float = 6.0
    m3_boss_diameter: float = 9.0
    m4_insert_hole: float = 5.6
    m4_boss_diameter: float = 11.0
    m5_insert_hole: float = 6.4
    m5_boss_diameter: float = 12.0
    m6_insert_hole: float = 8.0
    m6_insert_depth: float = 8.0
    m6_boss_diameter: float = 14.0

    #: Clearance holes for screws passing through.
    m3_clearance: float = 3.4
    m4_clearance: float = 4.5

    #: Bearing seats: nominal + 0.05-0.10 for a light press in PA-CF or PC-CF.
    bearing_press: float = 0.07
    #: Pilots and slip fits.
    slip_fit: float = 0.18

    @property
    def structural_wall_thickness(self) -> float:
        return self.structural_walls * self.nozzle

    @property
    def cosmetic_wall_thickness(self) -> float:
        return self.cosmetic_walls * self.nozzle

    def boss_for(self, thread: str) -> tuple[float, float]:
        """(hole diameter, boss outer diameter) for a heat-set insert."""
        if thread == "M3":
            return self.m3_insert_hole, self.m3_boss_diameter
        if thread == "M4":
            return self.m4_insert_hole, self.m4_boss_diameter
        if thread == "M5":
            return self.m5_insert_hole, self.m5_boss_diameter
        if thread == "M6":
            return self.m6_insert_hole, self.m6_boss_diameter
        raise ValueError(f"no insert rule for {thread!r}")


@dataclass(frozen=True)
class Style:
    """Shared styling so the parts read as one machine."""

    #: Break every outer edge. Small, but it is most of why a printed part
    #: looks finished rather than raw.
    edge_break: float = 0.8
    #: Larger radius where a shell meets a flange.
    shoulder_fillet: float = 2.0
    #: Cosmetic groove marking a joint line, as cobots use to hide the seam.
    seam_groove_width: float = 1.6
    seam_groove_depth: float = 0.6
    #: Gap left between a rotating shell and its neighbour.
    joint_gap: float = 1.5


RULES = PrintRules()
STYLE = Style()


#: Material assignment. PC-CF is the default for anything structural: stiff,
#: HDT well above the 60-80 C an actuator housing reaches, and it prints
#: predictably. PETG-CF is the fallback where stiffness matters less, and plain
#: PETG only for cosmetic parts away from the motors -- its 70 C HDT bars it
#: from anywhere near an actuator.
STRUCTURAL: Material = PC_CF
SEMI_STRUCTURAL: Material = PETG_CF
COSMETIC: Material = PETG


def structural(material: Material = STRUCTURAL) -> Material:
    """Effective density for a structural part at the spec's wall/infill."""
    return material.printed(
        infill=RULES.structural_infill,
        walls=RULES.structural_walls,
        nozzle=RULES.nozzle,
    )


def cosmetic(material: Material = COSMETIC) -> Material:
    """Effective density for a cover or non-load-bearing part."""
    return material.printed(
        infill=RULES.cosmetic_infill,
        walls=RULES.cosmetic_walls,
        nozzle=RULES.nozzle,
    )
