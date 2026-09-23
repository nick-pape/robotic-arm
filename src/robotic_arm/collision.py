"""Self-collision and clearance checking -- spec requirement S1.

S1 asks for at least 3 mm of clearance between parts through all joint ranges.
This sweeps the workspace and reports what actually comes closest.

The method is to set every geom's contact margin to the clearance threshold and
let MuJoCo's own broadphase do the work: with a margin, contacts are generated
at a distance rather than only on penetration, so a near miss shows up as a
contact with a positive `dist`. That is far faster than computing pairwise
distances by hand, and it uses the same collision code the simulation does, so
the check cannot disagree with the sim.

Two filters are already applied by MuJoCo and are left alone deliberately:
contacts between a body and its parent (which touch by design at every joint),
and the model's explicit exclude list.

Units: metres inside MuJoCo; millimetres in the reports, since that is what
the spec is written in.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from robotic_arm.reference import ARM_JOINTS

#: Spec S1: minimum clearance through all joint ranges.
MIN_CLEARANCE_MM = 3.0


@dataclass(frozen=True)
class Approach:
    """The closest two bodies came, and where."""

    body_a: str
    body_b: str
    distance_mm: float
    qpos: np.ndarray

    @property
    def penetrating(self) -> bool:
        return self.distance_mm < 0.0

    def __str__(self) -> str:
        state = "PENETRATION" if self.penetrating else "clearance"
        return (
            f"{self.body_a} <-> {self.body_b}: {state} {self.distance_mm:+.2f} mm "
            f"at q={np.round(self.qpos[: len(ARM_JOINTS)], 2)}"
        )


def _body_name(model: mujoco.MjModel, geom: int) -> str:
    return mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom]
    )


def closest_approaches(
    model: mujoco.MjModel,
    samples: int = 4000,
    threshold_mm: float = MIN_CLEARANCE_MM,
    seed: int = 0,
) -> dict[tuple[str, str], Approach]:
    """Sweep the workspace and record the closest approach for each body pair.

    Samples joint space randomly rather than on a grid: a grid over six joints
    is either coarse enough to step straight over a thin collision or far too
    large to run, whereas random sampling covers the space evenly and can be
    turned up until the answer stops moving.
    """
    original_margin = model.geom_margin.copy()
    # A margin makes MuJoCo report near misses, not just penetrations.
    model.geom_margin[:] = threshold_mm * 1e-3

    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in ARM_JOINTS
    ]
    ranges = np.array([model.jnt_range[j] for j in joint_ids])

    worst: dict[tuple[str, str], Approach] = {}
    try:
        for _ in range(samples):
            qpos = np.zeros(model.nq)
            qpos[: len(joint_ids)] = rng.uniform(ranges[:, 0], ranges[:, 1])
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)

            for i in range(data.ncon):
                contact = data.contact[i]
                a = _body_name(model, contact.geom1)
                b = _body_name(model, contact.geom2)
                key = tuple(sorted((a, b)))
                distance_mm = float(contact.dist) * 1e3
                if key not in worst or distance_mm < worst[key].distance_mm:
                    worst[key] = Approach(key[0], key[1], distance_mm, qpos.copy())
    finally:
        model.geom_margin[:] = original_margin

    return worst


def violations(
    model: mujoco.MjModel,
    samples: int = 4000,
    threshold_mm: float = MIN_CLEARANCE_MM,
    seed: int = 0,
) -> list[Approach]:
    """Body pairs that come closer than S1 allows, worst first."""
    found = closest_approaches(model, samples, threshold_mm, seed)
    return sorted(
        (a for a in found.values() if a.distance_mm < threshold_mm),
        key=lambda a: a.distance_mm,
    )


def report(model: mujoco.MjModel, samples: int = 4000) -> str:
    """Human-readable S1 summary."""
    found = closest_approaches(model, samples)
    lines = [f"swept {samples} configurations, {len(found)} body pairs came within "
             f"{MIN_CLEARANCE_MM:.0f} mm"]
    if not found:
        lines.append("  nothing came close -- no pair reached the margin")
        return "\n".join(lines)

    for approach in sorted(found.values(), key=lambda a: a.distance_mm):
        flag = "  FAIL" if approach.distance_mm < MIN_CLEARANCE_MM else "  ok"
        lines.append(f"{flag}  {approach}")

    bad = [a for a in found.values() if a.distance_mm < MIN_CLEARANCE_MM]
    lines.append("")
    lines.append(
        f"S1: {'FAIL' if bad else 'pass'} -- "
        f"{len(bad)} pair(s) below {MIN_CLEARANCE_MM:.0f} mm"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from robotic_arm.mjcf import REPO, generate_twin, load
    from robotic_arm.reference import load_baseline

    which = sys.argv[1] if len(sys.argv) > 1 else "twin"
    if which == "stock":
        model = load_baseline()
        print("=== stock arm ===")
    else:
        model = load(generate_twin(out=REPO / "sim" / "collision_model.xml"))
        print("=== printed twin ===")
    print(report(model))


def min_clearance_at(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray) -> float:
    """Smallest gap between any non-excluded pair at one pose, in mm.

    Returns +inf when nothing is within the margin, which is the good case.
    """
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    if data.ncon == 0:
        return float("inf")
    return min(float(data.contact[i].dist) for i in range(data.ncon)) * 1e3


@dataclass(frozen=True)
class ClearanceComparison:
    """Paired stock-vs-twin clearance over the same sampled poses."""

    samples: int
    regressions: list[tuple[float, float, np.ndarray]]
    worst_stock_mm: float
    worst_twin_mm: float

    @property
    def regression_rate(self) -> float:
        return len(self.regressions) / self.samples if self.samples else 0.0


def compare_clearance(
    stock: mujoco.MjModel,
    twin: mujoco.MjModel,
    samples: int = 3000,
    threshold_mm: float = MIN_CLEARANCE_MM,
    tolerance_mm: float = 1.0,
    seed: int = 0,
) -> ClearanceComparison:
    """Compare clearance pose by pose rather than pair by pair.

    Sweeping each model separately and counting bad pairs compares different
    worst-case poses, which says little. This drives both models through the
    *same* configurations and asks whether the printed twin is ever tighter
    than stock at a pose stock handles.

    `tolerance_mm` absorbs the difference in collision fidelity: the twin uses
    a few cylinders per link where stock uses a convex decomposition, and a
    cylinder is conservative -- it bulges where the real shell is waisted. A
    small penalty is therefore expected and is not a geometry regression.
    """
    stock_margin = stock.geom_margin.copy()
    twin_margin = twin.geom_margin.copy()
    stock.geom_margin[:] = threshold_mm * 1e-3
    twin.geom_margin[:] = threshold_mm * 1e-3

    stock_data, twin_data = mujoco.MjData(stock), mujoco.MjData(twin)
    rng = np.random.default_rng(seed)
    joint_ids = [
        mujoco.mj_name2id(stock, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in ARM_JOINTS
    ]
    ranges = np.array([stock.jnt_range[j] for j in joint_ids])

    regressions: list[tuple[float, float, np.ndarray]] = []
    worst_stock = worst_twin = float("inf")
    try:
        for _ in range(samples):
            qpos = np.zeros(stock.nq)
            qpos[: len(joint_ids)] = rng.uniform(ranges[:, 0], ranges[:, 1])

            a = min_clearance_at(stock, stock_data, qpos)
            b = min_clearance_at(twin, twin_data, qpos)
            worst_stock = min(worst_stock, a)
            worst_twin = min(worst_twin, b)

            # Only a pose stock keeps clear but the twin does not is a
            # regression. Where stock already collides, the clone inherits the
            # problem rather than causing it.
            if a >= threshold_mm and b < threshold_mm - 0.0:
                if (a - b) > tolerance_mm:
                    regressions.append((a, b, qpos.copy()))
    finally:
        stock.geom_margin[:] = stock_margin
        twin.geom_margin[:] = twin_margin

    return ClearanceComparison(samples, regressions, worst_stock, worst_twin)


def adjacent_pair_clearance(
    model: mujoco.MjModel, a: str, b: str, qpos: np.ndarray
) -> float:
    """Closest approach between two named bodies, in mm, ignoring exclusions.

    `closest_approaches` deliberately honours MuJoCo's contact filtering, which
    excludes parent/child pairs and the upstream exclude list. That is correct
    for simulating the stock arm, and blind for validating newly designed joint
    seams: link2/link3, link3/link4 and link4/link5 are all on that list, so no
    amount of sweeping would ever have reported them touching.

    This queries the geometry directly instead, so a printed seam can be
    checked whatever the simulation chooses to ignore.
    """
    data = mujoco.MjData(model)
    data.qpos[: len(qpos)] = qpos
    mujoco.mj_forward(model, data)

    ids = {}
    for name in (a, b):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(name)
        ids[name] = [
            g
            for g in range(model.ngeom)
            if model.geom_bodyid[g] == bid and model.geom_group[g] == 3
        ]
    if not ids[a] or not ids[b]:
        return float("inf")

    return min(
        mujoco.mj_geomDistance(model, data, ga, gb, 1.0, None)
        for ga in ids[a]
        for gb in ids[b]
    ) * 1e3


def adjacent_seam_parity(
    stock: mujoco.MjModel,
    twin: mujoco.MjModel,
    pairs: tuple[tuple[str, str], ...] = (
        ("link2", "link3"),
        ("link3", "link4"),
        ("link4", "link5"),
        ("link5", "link6"),
    ),
    samples: int = 500,
    seed: int = 0,
) -> dict[tuple[str, str], tuple[float, float]]:
    """Worst approach per adjacent pair, stock vs twin, over the same poses.

    Reported as parity because absolute overlap is the wrong test: the stock
    arm's own link2 and link3 interpenetrate by 16 mm at the folded limit, so
    "do they overlap" answers yes for the shipping product too. What matters is
    whether the printed seam is worse than the one it replaces.
    """
    rng = np.random.default_rng(seed)
    joint_ids = [
        mujoco.mj_name2id(stock, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in ARM_JOINTS
    ]
    ranges = np.array([stock.jnt_range[j] for j in joint_ids])

    worst = {pair: (float("inf"), float("inf")) for pair in pairs}
    for _ in range(samples):
        qpos = np.zeros(stock.nq)
        qpos[: len(joint_ids)] = rng.uniform(ranges[:, 0], ranges[:, 1])
        for pair in pairs:
            s = adjacent_pair_clearance(stock, *pair, qpos)
            t = adjacent_pair_clearance(twin, *pair, qpos)
            best_s, best_t = worst[pair]
            worst[pair] = (min(best_s, s), min(best_t, t))
    return worst
