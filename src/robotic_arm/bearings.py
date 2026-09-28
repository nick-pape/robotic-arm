"""Thin-section ball bearings, and which one each joint gets.

A pancake actuator's own bearings carry its rotor, not a robot arm. They are
sized for the motor, close together on a short shaft, and they are not meant
to take the moment a 700 mm lever puts on a joint. So each joint gets its own
bearing at the seam, outboard of the motor, where the lever arm is longest and
the load is a moment rather than a radial force.

**These are catalogue parts, not invented ones.** Every entry below is a
standard metric series -- 6800 (thin section) and 6700 (extra-thin) -- that any
bearing supplier stocks, so the design can be built from parts with a part
number rather than from dimensions someone has to make. The whole point of
picking from a catalogue is that the geometry then has to follow the bearing,
not the reverse.

Dimensions are the ISO 15:2017 boundary dimensions for each designation:
bore x outside diameter x width, in mm. `ZZ` (shielded) and `2RS` (sealed)
variants share them, so the suffix does not matter here -- for an arm joint
that is assembled once and rarely opened, `2RS` is the better buy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Bearing:
    """One catalogue bearing, by designation."""

    designation: str
    bore: float
    outer: float
    width: float

    @property
    def section(self) -> float:
        """Radial thickness of the ring pair -- what "thin section" means."""
        return (self.outer - self.bore) / 2

    def __str__(self) -> str:
        return (
            f"{self.designation} ({self.bore:g}x{self.outer:g}x{self.width:g})"
        )


#: The two extra-thin metric series, as far as this arm could ever need.
#: Listed so a change of joint diameter can be answered by re-selecting rather
#: than by inventing a size no supplier carries.
CATALOGUE: tuple[Bearing, ...] = (
    # 6700 series, extra-thin section
    Bearing("6704", 20, 27, 4),
    Bearing("6705", 25, 32, 4),
    Bearing("6706", 30, 37, 4),
    Bearing("6707", 35, 44, 5),
    Bearing("6708", 40, 50, 6),
    Bearing("6709", 45, 55, 6),
    Bearing("6710", 50, 62, 6),
    Bearing("6711", 55, 68, 7),
    Bearing("6712", 60, 75, 7),
    Bearing("6713", 65, 80, 7),
    # 6800 series, thin section
    Bearing("6806", 30, 42, 7),
    Bearing("6807", 35, 47, 7),
    Bearing("6808", 40, 52, 7),
    Bearing("6809", 45, 58, 7),
    Bearing("6810", 50, 65, 7),
    Bearing("6811", 55, 72, 9),
    Bearing("6812", 60, 78, 10),
    Bearing("6813", 65, 85, 10),
    Bearing("6814", 70, 90, 10),
    Bearing("6815", 75, 95, 10),
    Bearing("6816", 80, 100, 10),
)

BY_DESIGNATION = {b.designation: b for b in CATALOGUE}


def largest_fitting(max_outer: float, min_outer: float = 0.0) -> Bearing:
    """The widest-bore catalogue bearing whose OD fits `max_outer`.

    Bore matters more than OD here. The outer ring is bounded by the barrel,
    which is fixed, so the useful freedom is how much of the joint the inner
    ring spans: a wider bore puts the balls further out, which is what carries
    a moment. Among bearings that fit, the largest is therefore the right one,
    not a compromise.
    """
    fits = [
        b for b in CATALOGUE if min_outer <= b.outer <= max_outer
    ]
    if not fits:
        raise ValueError(
            f"no catalogue bearing with an OD between {min_outer:.1f} and "
            f"{max_outer:.1f} mm; the joint diameter has to move to meet one"
        )
    return max(fits, key=lambda b: (b.bore, -b.width))


#: What each joint actually gets, by actuator.
#:
#: Chosen against two hard limits, and there is very little room between them:
#:
#: * the outer ring seats in the barrel bore, so **OD <= bore**, and
#: * the motor is fitted through that same seat, so **OD >= motor OD +
#:   clearance** -- otherwise the seat is a shoulder the motor cannot pass.
#:
#: That second constraint is the one that is easy to miss, and it is why these
#: are large-bore bearings rather than the small ones that would otherwise sit
#: neatly around the rotor hub. Anything narrower than the motor would have to
#: be fitted before it, which contradicts the assembly order.
SELECTION: dict[str, str] = {
    "RS06": "6814",   # 70x90x10 -- seats in the O90 bore, clears the O87 motor
    "RS00": "6710",   # 50x62x6  -- needs the RS00 barrel at O66, see below
}


def for_actuator(actuator) -> Bearing:
    """The bearing this actuator's joint uses."""
    return BY_DESIGNATION[SELECTION[actuator.name]]


if __name__ == "__main__":
    from robotic_arm.actuators import RS00, RS06

    for act in (RS06(), RS00()):
        b = for_actuator(act)
        print(
            f"{act.name}: {b}  section {b.section:.1f} mm  "
            f"(motor O{act.bbox_mm[0]:.1f})"
        )
