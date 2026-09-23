"""Shared parameters for the arm.

Every part imports its dimensions from here so that changing a servo or a
fastener size propagates through the whole assembly instead of being buried
in one part's source.

Units: millimetres, degrees.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ServoSpec:
    """Envelope of a hobby servo, measured over the body and mounting flange."""

    body_length: float
    body_width: float
    body_height: float
    flange_length: float
    flange_thickness: float
    flange_hole_dia: float
    flange_hole_spacing: float
    horn_dia: float


@dataclass(frozen=True)
class BaseSpec:
    """The plate the arm stands on."""

    length: float
    width: float
    thickness: float
    corner_radius: float
    mount_hole_dia: float
    mount_hole_inset: float


@dataclass(frozen=True)
class ArmSpec:
    """Link geometry for the two main arm segments."""

    shoulder_to_elbow: float
    elbow_to_wrist: float
    link_width: float
    link_thickness: float
    joint_bore_dia: float


@dataclass(frozen=True)
class FastenerSpec:
    """Clearance holes for the screws used throughout."""

    m3_clearance: float
    m3_tap: float
    m4_clearance: float


# An MG996R-class standard servo.
SERVO = ServoSpec(
    body_length=40.7,
    body_width=19.7,
    body_height=42.9,
    flange_length=54.0,
    flange_thickness=2.8,
    flange_hole_dia=4.2,
    flange_hole_spacing=49.5,
    horn_dia=21.0,
)

BASE = BaseSpec(
    length=120.0,
    width=120.0,
    thickness=6.0,
    corner_radius=8.0,
    mount_hole_dia=4.5,
    mount_hole_inset=12.0,
)

ARM = ArmSpec(
    shoulder_to_elbow=100.0,
    elbow_to_wrist=80.0,
    link_width=24.0,
    link_thickness=6.0,
    joint_bore_dia=8.0,
)

FASTENERS = FastenerSpec(
    m3_clearance=3.4,
    m3_tap=2.5,
    m4_clearance=4.5,
)
