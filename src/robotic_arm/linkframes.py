"""Where a link's joints are, in that link's own frame.

Designing a shell needs three things: which way the parent joint turns, where
the child joint sits, and which way *it* turns. All three come from the stock
model, so parts stay driven by the reference rather than by numbers typed into
a CAD file -- the same rule the MJCF generator follows.

Everything here is in millimetres and the link's own frame, because that is
what build123d wants. The model is in metres, so this converts.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import mujoco
import numpy as np

from robotic_arm.reference import load_baseline

M_TO_MM = 1000.0


def _quat2mat(quat: np.ndarray) -> np.ndarray:
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(quat, dtype=float))
    return out.reshape(3, 3)


@dataclass(frozen=True)
class LinkFrame:
    """A link's joint geometry, in its own frame, in millimetres."""

    name: str
    parent_axis: np.ndarray  # unit vector, the axis this link turns about
    child_name: str | None
    child_origin: np.ndarray  # mm, where the child link's frame sits
    child_axis: np.ndarray  # unit vector, the child's joint axis in our frame
    stock_extent: np.ndarray  # mm, bounding box of the stock visual geometry
    stock_centre: np.ndarray  # mm, centre of that box

    @property
    def span(self) -> float:
        """Distance from this joint to the child joint, mm."""
        return float(np.linalg.norm(self.child_origin))

    @property
    def axes_are_perpendicular(self) -> bool:
        """Whether parent and child axes cross at a right angle.

        True for every wrist joint on this arm, which is what gives a cobot
        wrist its two-perpendicular-drums look.
        """
        return abs(float(self.parent_axis @ self.child_axis)) < 1e-6


@lru_cache(maxsize=16)
def link_frame(name: str) -> LinkFrame:
    """Joint geometry for one link, read out of the stock model."""
    model = load_baseline()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise KeyError(f"no body named {name!r}")

    joints = [j for j in range(model.njnt) if model.jnt_bodyid[j] == bid]
    parent_axis = (
        np.array(model.jnt_axis[joints[0]], dtype=float)
        if joints
        else np.array([0.0, 0.0, 1.0])
    )

    children = [
        j
        for j in range(model.nbody)
        if model.body_parentid[j] == bid and j != bid
    ]
    if children:
        cid = children[0]
        child_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, cid)
        child_origin = np.array(model.body_pos[cid], dtype=float) * M_TO_MM
        # The child's joint axis is given in the child's frame; rotate it into
        # ours so a drum can be placed without thinking in two frames at once.
        rotation = _quat2mat(model.body_quat[cid])
        cjoints = [j for j in range(model.njnt) if model.jnt_bodyid[j] == cid]
        local_axis = (
            np.array(model.jnt_axis[cjoints[0]], dtype=float)
            if cjoints
            else np.array([0.0, 0.0, 1.0])
        )
        child_axis = rotation @ local_axis
    else:
        child_name, child_origin = None, np.zeros(3)
        child_axis = parent_axis

    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != bid or model.geom_group[g] != 2:
            continue
        mid = model.geom_dataid[g]
        start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        verts = (
            model.mesh_vert[start : start + count] @ _quat2mat(model.geom_quat[g]).T
            + model.geom_pos[g]
        )
        lo = np.minimum(lo, verts.min(axis=0))
        hi = np.maximum(hi, verts.max(axis=0))

    return LinkFrame(
        name=name,
        parent_axis=parent_axis / np.linalg.norm(parent_axis),
        child_name=child_name,
        child_origin=child_origin,
        child_axis=child_axis / np.linalg.norm(child_axis),
        stock_extent=(hi - lo) * M_TO_MM,
        stock_centre=(hi + lo) / 2 * M_TO_MM,
    )


def stock_mass(name: str) -> float:
    """Mass of the stock part, kg -- the budget a replacement must beat (P1)."""
    model = load_baseline()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise KeyError(f"no body named {name!r}")
    return float(model.body_mass[bid])


if __name__ == "__main__":
    for link in ("link4", "link5", "link6"):
        frame = link_frame(link)
        print(f"{link}: stock {stock_mass(link) * 1000:.0f} g")
        print(f"   parent axis {frame.parent_axis.round(3)}")
        print(
            f"   child {frame.child_name} at {frame.child_origin.round(1)} mm, "
            f"axis {frame.child_axis.round(3)}"
        )
        print(
            f"   perpendicular={frame.axes_are_perpendicular}  "
            f"span={frame.span:.1f} mm  extent={frame.stock_extent.round(1)}"
        )


def inscribed_drum_radius(
    name: str,
    station: float,
    half_height: float = 25.0,
    sectors: int = 16,
    axis: np.ndarray | None = None,
) -> float:
    """Largest full drum that fits inside the stock silhouette at a station.

    A printed drum is round, so it only stays within the space the stock part
    occupies if stock has material out to that radius in *every* direction.
    Taking the maximum radius would be badly wrong for a long link, where the
    far end dominates every slice; this takes the minimum across angular
    sectors, which is the radius a full drum can actually reach.

    A sector with no stock material at all returns 0: a drum there would sit
    entirely in open space. That is exactly what link4's oversized parent drum
    did, and it surfaced only later as a self-collision the stock arm does not
    have.

    `station` and the result are millimetres along, and from, the axis.
    """
    model = load_baseline()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise KeyError(f"no body named {name!r}")

    axis = link_frame(name).parent_axis if axis is None else np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(seed @ axis)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, seed)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    clouds = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != bid or model.geom_group[g] != 2:
            continue
        mid = model.geom_dataid[g]
        first, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        verts = (
            model.mesh_vert[first : first + count] @ _quat2mat(model.geom_quat[g]).T
            + model.geom_pos[g]
        )
        clouds.append(verts * M_TO_MM)
    if not clouds:
        return 0.0

    cloud = np.vstack(clouds)
    along = cloud @ axis
    band = cloud[np.abs(along - station) <= half_height]
    if len(band) == 0:
        return 0.0

    x, y = band @ u, band @ v
    radius = np.hypot(x, y)
    angle = np.arctan2(y, x)
    index = ((angle + np.pi) / (2 * np.pi) * sectors).astype(int) % sectors

    reach = np.zeros(sectors)
    for sector in range(sectors):
        mask = index == sector
        reach[sector] = radius[mask].max() if mask.any() else 0.0
    return float(reach.min())


def fits_inside_stock(
    name: str, station: float, diameter: float, half_height: float = 25.0
) -> bool:
    """Whether a drum of `diameter` at `station` stays inside stock's silhouette."""
    return diameter / 2 <= inscribed_drum_radius(name, station, half_height)
