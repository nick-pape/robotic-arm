"""Thermal and current limits -- what actually bounds continuous torque.

A torque figure alone does not say whether a pose is sustainable. These motors
are thermally limited, not torque limited: at stall there is no back-EMF, so
holding a pose is pure I^2R heating and the binding constraint is the 145 C
thermistor, gated in firmware at 80/100/140 C.

This matters for reading the M2 result correctly. J2 needing 15.36 N*m to hold
full extension does **not** mean the stock arm is broken -- it is a shipping
product and it works. It means that pose draws ~14 Arms and dissipates tens of
watts, so it is time-limited rather than impossible, which is exactly what
Seeed's own "stay within about 70% of the workspace" warning is about.

The derating question is where stock and clone genuinely differ:

* the **stock** arm bolts J2 to an aluminium flange and spacer that conduct
  into the sheet-metal link, which is the heat path RobStride's rated figures
  assume;
* a **printed** link has no such path, which is why the spec assumes 60-70% of
  rated continuous and why the clone needs the balancer that stock does not.

Kt here is output-referred (post-gearbox): 14.3 Apk rated / sqrt(2) * 1.09
= 11.0 N*m, which reproduces the published rated torque.

Units: N*m, amperes, watts, degrees Celsius.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from robotic_arm.actuators import Actuator

#: Firmware thresholds from the stock controller, in degrees Celsius.
WARN_C = 80.0
SLOW_RETURN_C = 100.0
EMERGENCY_DISABLE_C = 140.0
THERMISTOR_TRIP_C = 145.0

#: Three-phase copper loss is 1.5 * I_rms^2 * R_line for a wye winding driven
#: sinusoidally, with R_line the line-to-line resistance the vendor quotes.
_PHASE_LOSS_FACTOR = 1.5


@dataclass(frozen=True)
class WindingSpec:
    """Electrical constants needed to turn torque into heat."""

    torque_constant: float  # N*m per Arms, output-referred
    line_resistance: float  # ohms, line-to-line


WINDINGS = {
    "RS06": WindingSpec(torque_constant=1.09, line_resistance=0.23),
    "RS00": WindingSpec(torque_constant=0.36, line_resistance=1.5),
}


def winding(actuator: Actuator) -> WindingSpec:
    try:
        return WINDINGS[actuator.name]
    except KeyError:
        raise KeyError(
            f"no winding data for {actuator.name}; add it to WINDINGS"
        ) from None


def current_arms(actuator: Actuator, torque_nm: float) -> float:
    """RMS phase current to hold `torque_nm` at the output."""
    return abs(torque_nm) / winding(actuator).torque_constant


def current_apk(actuator: Actuator, torque_nm: float) -> float:
    """Peak phase current, which is what the vendor limits are quoted in."""
    return current_arms(actuator, torque_nm) * math.sqrt(2)


def copper_loss_w(actuator: Actuator, torque_nm: float) -> float:
    """Resistive dissipation while holding a torque.

    At stall this is essentially the whole thermal input: no mechanical work is
    done holding a pose, so everything the motor draws becomes heat.
    """
    spec = winding(actuator)
    return _PHASE_LOSS_FACTOR * current_arms(actuator, torque_nm) ** 2 * spec.line_resistance


def within_rated_current(actuator: Actuator, torque_nm: float) -> bool:
    """Whether a torque is inside the actuator's continuous current rating."""
    return current_apk(actuator, torque_nm) <= actuator.rated_current_apk


def within_peak_current(actuator: Actuator, torque_nm: float) -> bool:
    return current_apk(actuator, torque_nm) <= actuator.peak_current_apk


def sustainability(actuator: Actuator, torque_nm: float) -> str:
    """A one-word verdict on holding this torque.

    Deliberately coarse. Turning watts into a time-to-trip needs a thermal
    model with the real mounting, which is a bench measurement (requirement
    T1), not something to guess at here.
    """
    if not within_peak_current(actuator, torque_nm):
        return "infeasible"
    if within_rated_current(actuator, torque_nm):
        return "continuous"
    return "time-limited"


def report(actuator: Actuator, torque_nm: float) -> str:
    """Human-readable thermal summary for one holding torque."""
    return (
        f"{actuator.name} holding {torque_nm:.2f} N*m: "
        f"{current_arms(actuator, torque_nm):.1f} Arms "
        f"({current_apk(actuator, torque_nm):.1f} Apk vs "
        f"{actuator.rated_current_apk:.1f} rated / "
        f"{actuator.peak_current_apk:.1f} peak), "
        f"{copper_loss_w(actuator, torque_nm):.0f} W copper loss -> "
        f"{sustainability(actuator, torque_nm)}"
    )
