"""RobStride actuator specifications.

Two kinds of data meet here:

* **Electrical and torque ratings**, transcribed from RobStride's published
  specification table.
* **Bolt geometry**, measured from the vendor STEP files by
  ``scripts/measure_actuators.py`` and committed as
  ``reference/actuator_geometry.json``. It is measured rather than transcribed
  because the RS06 pattern is not published in any text form, and because the
  published RS00 figures turned out to be wrong (see `RS00_PUBLISHED_ERRATUM`).

Keeping the actuator a parameter is a spec requirement: the DM variant uses
different bolt circles, so printed motor pockets must be generated from this
data rather than hardcoded.

Torque is in N*m, lengths in mm, mass in kg.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

GEOMETRY_JSON = (
    Path(__file__).resolve().parents[2] / "reference" / "actuator_geometry.json"
)

#: The manuals.plus transcription of the RS00 manual gives "bottom: 4x M3 on
#: O38". The BCD is right; the thread size is not. Measured from the vendor
#: STEP, that circle carries 4 x O1.6 holes with a O2.5 counterbore -- M1.6
#: hardware, not M3. Recorded here because designing an M3 pattern against it
#: would produce a part that cannot be bolted on.
RS00_PUBLISHED_ERRATUM = (
    "Published 'bottom 4x M3 on O38' is wrong on thread size: the O38 circle "
    "is 4 x O1.6 with a O2.5 counterbore (M1.6). The O27 top circle (6 x M3) "
    "is confirmed correct."
)


@dataclass(frozen=True)
class BoltCircle:
    """One measured bolt circle on an actuator face."""

    bcd: float  # bolt circle diameter, mm
    count: int
    hole_diameter: float  # mm
    depth: float  # mm
    bottom: str  # through | flat | drill_point | unknown
    plane_z: float  # mm, in the STEP's own frame
    axis_z: float  # +1 or -1: which way the holes are drilled
    pitch_deg: float | None  # uniform angular pitch, None if uneven
    counterbore: dict | None

    @property
    def radius(self) -> float:
        return self.bcd / 2.0

    @property
    def fastener_candidates(self) -> tuple[str, ...]:
        """Thread sizes this hole could take, most likely first.

        A drilled diameter alone is ambiguous: O2.5 is both an M3 tap drill and
        close to an M2.5 clearance hole. The bottom type disambiguates
        partially -- a blind hole is usually tapped, a through hole usually
        clearance -- so that is used to order the candidates. Where both remain
        plausible, both are returned.

        This is a reading aid, not a substitute for the vendor drawing. Check
        before cutting metal; `is_plausibly()` is the safe way to test.
        """
        taps = {"M1.6": 1.25, "M2": 1.6, "M2.5": 2.05, "M3": 2.5, "M4": 3.3}
        # ISO 273 close / medium / coarse. Vendor flanges often use close fits,
        # so all three are candidates.
        clearances = {
            "M1.6": (1.7, 1.8, 2.0),
            "M2": (2.2, 2.4, 2.6),
            "M2.5": (2.7, 2.9, 3.1),
            "M3": (3.2, 3.4, 3.6),
            "M4": (4.3, 4.5, 4.8),
        }
        d = self.hole_diameter

        tapped = [t for t, v in taps.items() if abs(d - v) <= 0.25]
        clear = [
            t for t, vs in clearances.items() if any(abs(d - v) <= 0.25 for v in vs)
        ]

        # A blind hole is far more likely tapped; a through hole, clearance.
        first, second = (
            (clear, tapped) if self.bottom == "through" else (tapped, clear)
        )
        ordered = list(dict.fromkeys(first + second))
        return tuple(ordered)

    def is_plausibly(self, thread: str) -> bool:
        """Whether `thread` is a credible reading of this hole."""
        return thread in self.fastener_candidates


@dataclass(frozen=True)
class Actuator:
    """A RobStride quasi-direct-drive actuator."""

    name: str
    rated_nm: float
    peak_nm: float
    mass_kg: float
    gear_ratio: float
    no_load_rpm: float
    voltage_range: tuple[float, float]
    rated_current_apk: float
    peak_current_apk: float
    bolt_circles: tuple[BoltCircle, ...] = ()
    bbox_mm: tuple[float, float, float] | None = None

    def derated_nm(self, factor: float = 0.7) -> float:
        """Continuous torque without the specified aluminium heat sink.

        RobStride's rated figures assume a heat-sink plate. An actuator mounted
        into a printed link has none, so the spec assumes 60-70% of rated is
        available continuously. `factor` defaults to the optimistic end, so a
        result that still fails is unambiguous.
        """
        return self.rated_nm * factor

    def largest_circle(self) -> BoltCircle:
        """The mounting circle -- the widest, which carries the housing load."""
        if not self.bolt_circles:
            raise ValueError(f"{self.name} has no measured bolt geometry")
        return max(self.bolt_circles, key=lambda c: c.bcd)

    def circles_on_face(self, axis_z: float) -> tuple[BoltCircle, ...]:
        """Circles drilled along a given direction (+1 or -1)."""
        return tuple(c for c in self.bolt_circles if c.axis_z == axis_z)


# Vendor ratings, from RobStride's published specification table.
_RATINGS = {
    "RS06": dict(
        rated_nm=11.0,
        peak_nm=36.0,
        mass_kg=0.621,
        gear_ratio=9.0,
        no_load_rpm=480.0,
        voltage_range=(15.0, 60.0),
        rated_current_apk=14.3,
        peak_current_apk=57.0,
    ),
    "RS00": dict(
        rated_nm=5.0,
        peak_nm=14.0,
        mass_kg=0.310,
        gear_ratio=10.0,
        no_load_rpm=315.0,
        voltage_range=(24.0, 60.0),
        rated_current_apk=4.7,
        peak_current_apk=15.5,
    ),
}


@lru_cache(maxsize=1)
def _geometry() -> dict:
    if not GEOMETRY_JSON.exists():
        raise FileNotFoundError(
            f"{GEOMETRY_JSON} missing. Regenerate with:\n"
            f"    uv run --extra cad python scripts/measure_actuators.py"
        )
    return json.loads(GEOMETRY_JSON.read_text())["actuators"]


@lru_cache(maxsize=4)
def get(name: str) -> Actuator:
    """Load an actuator by name, merging ratings with measured geometry."""
    if name not in _RATINGS:
        raise KeyError(f"unknown actuator {name!r}; have {sorted(_RATINGS)}")

    geom = _geometry().get(name, {})
    circles = tuple(
        BoltCircle(
            bcd=c["bcd"],
            count=c["count"],
            hole_diameter=c["hole_diameter"],
            depth=c["depth"],
            bottom=c["bottom"],
            plane_z=c["plane_z"],
            axis_z=c["axis_z"],
            pitch_deg=c["pitch_deg"],
            counterbore=c["counterbore"],
        )
        for c in geom.get("bolt_circles", ())
    )
    bbox = geom.get("bbox_mm")
    return Actuator(
        name=name,
        bolt_circles=circles,
        bbox_mm=(bbox["x"], bbox["y"], bbox["z"]) if bbox else None,
        **_RATINGS[name],
    )


def RS06() -> Actuator:  # noqa: N802 - reads as a constant at call sites
    return get("RS06")


def RS00() -> Actuator:  # noqa: N802
    return get("RS00")


#: Joint-to-actuator assignment, from upstream `config/rebotarm_rs.yaml`.
JOINT_ACTUATOR = {
    "joint1": "RS06",
    "joint2": "RS06",
    "joint3": "RS06",
    "joint4": "RS00",
    "joint5": "RS00",
    "joint6": "RS00",
}


def for_joint(joint: str) -> Actuator:
    """The actuator driving a named joint."""
    return get(JOINT_ACTUATOR[joint])
