"""Material densities for mass-property calculation.

Densities are bulk (solid) values in kg/m^3. A printed part is not solid, so
callers go through `Material.printed()` to get an effective density.

The infill model here is deliberately crude, and that is the honest state of
things: effective density depends on wall count, infill pattern, layer height
and over/under-extrusion. It will dominate the inertia error budget far more
than any tensor subtlety, so `measured()` exists to override it with a number
from a weighed coupon. Calibrate before trusting a mass budget.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Material:
    """A material and how much of the envelope it actually fills."""

    name: str
    bulk_density: float  # kg/m^3, fully dense
    fill: float = 1.0  # fraction of the envelope occupied, 0-1
    hdt_c: float | None = None  # heat deflection temperature, if relevant

    @property
    def density(self) -> float:
        """Effective density in kg/m^3 for mass-property calculation."""
        return self.bulk_density * self.fill

    def printed(self, *, infill: float, walls: int, nozzle: float = 0.4) -> Material:
        """Effective density for an FDM part.

        Approximates the part as a solid shell of `walls` perimeters around an
        `infill`-fraction core. Wall thickness is walls * nozzle; the shell
        fraction is estimated for a section a few centimetres across, which is
        the scale of this arm's links.

        This is an estimate. Weigh a coupon and use `measured()` for anything
        load-bearing.
        """
        if not 0.0 <= infill <= 1.0:
            raise ValueError(f"infill must be 0-1, got {infill}")
        wall_mm = walls * nozzle
        # Shell fraction of a ~30 mm characteristic section, clamped to <=1.
        shell = min(1.0, 2 * wall_mm / 30.0)
        return replace(self, fill=shell + (1.0 - shell) * infill)

    def measured(self, *, mass_g: float, volume_mm3: float) -> Material:
        """Effective density from a weighed part. Always preferred."""
        density = (mass_g * 1e-3) / (volume_mm3 * 1e-9)
        return replace(self, fill=density / self.bulk_density)


# Structural print materials. HDT matters because motor housings reach 60-80 C
# and the spec bars PETG/PLA within 10 mm of an actuator.
PA_CF = Material("PA-CF", 1160.0, hdt_c=150.0)
PC_CF = Material("PC-CF", 1200.0, hdt_c=140.0)
PETG_CF = Material("PETG-CF", 1300.0, hdt_c=82.0)
PETG = Material("PETG", 1270.0, hdt_c=70.0)
ASA = Material("ASA", 1070.0, hdt_c=95.0)
ABS = Material("ABS", 1040.0, hdt_c=98.0)
PLA = Material("PLA", 1240.0, hdt_c=55.0)

# Non-printed materials in the load path.
AL_5052 = Material("Aluminium 5052", 2680.0)
STEEL = Material("Steel", 7850.0)

#: Materials the spec bars from within 10 mm of an actuator.
HEAT_SENSITIVE = (PETG, PLA)

#: Minimum service temperature the spec expects near an actuator.
ACTUATOR_SURFACE_TEMP_C = 80.0
