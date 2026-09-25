"""Stock arm versus the printed twin, on the numbers that decide the goal.

    uv run python scripts/compare_performance.py

The design brief is "same motors, UR5e shape, stock-equivalent performance".
The first two are visible in a render; the third is not, and asserting that
reach is unchanged "by construction" is a claim about the code, not a
measurement of the model. This measures it.

Reach and segment lengths *should* come out identical, because the generator
is forbidden from touching joint frames (requirement F2) -- so if they differ
at all, something has gone wrong that no render would show. Payload and
torque genuinely change, because the printed structure is far lighter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from robotic_arm.actuators import RS06
from robotic_arm.balancer import DEFAULT_CANCEL_NM, Balancer
from robotic_arm.mjcf import generate_twin, load
from robotic_arm.reference import load_baseline
from robotic_arm.torque import max_moment_arm, max_payload, worst_case_j2

REPO = Path(__file__).resolve().parents[1]

#: The chain whose lengths define the arm's geometry.
CHAIN = ("base_link", "link1", "link2", "link3", "link4", "link5", "link6")

#: Seeed's published figures.
ADVERTISED_REACH_MM = 754.0
ADVERTISED_MAX_KG = 5.0


def segment_lengths(model) -> dict[str, float]:
    """Distance between consecutive body origins, in mm."""
    data = mujoco.MjData(model)
    data.qpos[:] = 0
    mujoco.mj_forward(model, data)
    out = {}
    for parent, child in zip(CHAIN, CHAIN[1:]):
        a = data.xpos[model.body(parent).id]
        b = data.xpos[model.body(child).id]
        out[f"{parent}->{child}"] = float(np.linalg.norm(b - a)) * 1000.0
    return out


def reach(model, samples: int = 41) -> float:
    """Furthest the tool centre gets from the J1 axis, in mm."""
    data = mujoco.MjData(model)
    best = 0.0
    for q2 in np.linspace(*model.jnt_range[1], samples):
        for q3 in np.linspace(*model.jnt_range[2], samples):
            data.qpos[:] = 0
            data.qpos[1], data.qpos[2] = q2, q3
            mujoco.mj_forward(model, data)
            tcp = data.body("gripper_end").xpos
            best = max(best, float(np.hypot(tcp[0], tcp[1])))
    return best * 1000.0


def total_mass(model) -> float:
    return float(sum(model.body_mass)) 


def report(name: str, model, envelope_fraction: float = 0.70) -> dict:
    envelope = envelope_fraction * max_moment_arm(model)
    return {
        "mass_kg": total_mass(model),
        "reach_mm": reach(model),
        "segments_mm": segment_lengths(model),
        "payload_kg": max_payload(model, RS06().peak_nm, max_arm=envelope, samples=31),
        "worst_j2_nm": worst_case_j2(model, samples=31).torques[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--balancer", action="store_true", default=True)
    args = parser.parse_args()

    stock = load_baseline()
    twin = load(
        generate_twin(
            out=REPO / "sim" / "performance_twin.xml",
            balancer=Balancer.sized_for(DEFAULT_CANCEL_NM) if args.balancer else None,
            visuals=False,
        )
    )

    a, b = report("stock", stock), report("twin", twin)

    print(f"\n{'':22s} {'stock':>12s} {'twin':>12s}  {'delta':>12s}")
    for label, key, unit in (
        ("total mass", "mass_kg", "kg"),
        ("reach", "reach_mm", "mm"),
        ("payload @ peak", "payload_kg", "kg"),
        ("worst J2 torque", "worst_j2_nm", "N*m"),
    ):
        delta = b[key] - a[key]
        pct = f"{100 * delta / a[key]:+6.1f}%" if a[key] else "     --"
        print(f"{label:22s} {a[key]:9.2f} {unit:>2s} {b[key]:9.2f} {unit:>2s}  "
              f"{delta:+8.2f} {pct}")

    print(f"\n{'segment':22s} {'stock':>12s} {'twin':>12s}  {'delta':>12s}")
    worst = 0.0
    for segment, length in a["segments_mm"].items():
        other = b["segments_mm"][segment]
        worst = max(worst, abs(other - length))
        print(f"{segment:22s} {length:9.2f} mm {other:9.2f} mm  {other - length:+8.4f} mm")

    print(f"\nLargest segment difference: {worst:.6f} mm")
    print(
        "Reach and segment lengths must be identical: the generator may not "
        "touch joint frames (F2). Any difference here is a defect."
    )
    print(
        f"\nAdvertised: {ADVERTISED_REACH_MM:.0f} mm reach, "
        f"{ADVERTISED_MAX_KG:.1f} kg max payload (within 70% of the workspace)."
    )


if __name__ == "__main__":
    main()
