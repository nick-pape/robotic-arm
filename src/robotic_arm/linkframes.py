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


@dataclass(frozen=True)
class ActuatorEnvelope:
    """Where the actuator a link carries actually sits, in that link's frame.

    Taken from the stock model's own motor mesh rather than inferred from the
    joint origin. Inferring it was wrong by 30 mm on link3 and left the motor
    hanging outside the printed shell -- visible the moment the motor meshes
    were rendered, invisible in every number.
    """

    mesh: str
    centre: np.ndarray  # mm
    extent: np.ndarray  # mm, axis-aligned in the link frame

    @property
    def is_plausibly_one_motor(self) -> bool:
        """Whether this mesh is a single actuator rather than a whole assembly.

        link2's `motor_2_3` spans the entire 326 mm link, so it is a combined
        mesh covering more than one motor and cannot be used to place a drum.
        """
        return bool(max(self.extent) < 120.0)


def actuator_envelope(name: str) -> ActuatorEnvelope | None:
    """The motor mesh carried by a link, measured in the link's frame."""
    model = load_baseline()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise KeyError(name)

    for g in range(model.ngeom):
        if model.geom_bodyid[g] != bid or model.geom_group[g] != 2:
            continue
        mesh = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[g])
        if not (mesh or "").startswith("motor"):
            continue
        mid = model.geom_dataid[g]
        first, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        verts = (
            model.mesh_vert[first : first + count] @ _quat2mat(model.geom_quat[g]).T
            + model.geom_pos[g]
        ) * M_TO_MM
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        return ActuatorEnvelope(mesh, (lo + hi) / 2, hi - lo)
    return None


def drum_for_actuator(
    name: str, axis: np.ndarray, clearance: float, wall: float
) -> tuple[np.ndarray, float, float] | None:
    """(centre, diameter, length) of a drum that encloses the carried motor.

    Sized from the motor's measured envelope: its span along the drum axis sets
    the length, its span across sets the diameter.
    """
    envelope = actuator_envelope(name)
    if envelope is None or not envelope.is_plausibly_one_motor:
        return None

    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    along = float(abs(envelope.extent @ axis))
    across = float(max(e for e, a in zip(envelope.extent, abs(axis)) if a < 0.5))

    return (
        np.asarray(envelope.centre, dtype=float),
        across + 2 * (wall + clearance),
        along + 2 * (wall + clearance),
    )


def actuator_centre(name: str, near: np.ndarray | None = None) -> np.ndarray | None:
    """Where the carried actuator actually sits, in the link's frame, mm.

    Position only -- not size. The two are separate decisions and conflating
    them was a mistake: sizing a housing to the motor's full mesh envelope
    (~82 mm across, body plus connectors) gives O92 drums that collide, while
    simply *centring* a body-sized housing on the real motor costs nothing and
    fixes a visible misalignment.

    `near` picks one cluster when a mesh holds more than one motor, as link2's
    `motor_2_3` does: it spans both the J2 and J3 ends of the link.
    """
    envelope = actuator_envelope(name)
    if envelope is None:
        return None
    if envelope.is_plausibly_one_motor or near is None:
        return np.asarray(envelope.centre, dtype=float)

    # Several motors in one mesh: take the vertices nearest the given point.
    model = load_baseline()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != bid or model.geom_group[g] != 2:
            continue
        mesh = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[g])
        if mesh != envelope.mesh:
            continue
        mid = model.geom_dataid[g]
        first, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        verts = (
            model.mesh_vert[first : first + count] @ _quat2mat(model.geom_quat[g]).T
            + model.geom_pos[g]
        ) * M_TO_MM
        near = np.asarray(near, dtype=float)
        # Half the span between clusters is a safe cut-off.
        keep = verts[np.linalg.norm(verts - near, axis=1) < 120.0]
        if len(keep) == 0:
            return None
        return (keep.min(axis=0) + keep.max(axis=0)) / 2
    return None


def parent_actuator_face(name: str) -> float | None:
    """Where the parent joint's actuator actually ends, along the link's own
    parent axis, in millimetres.

    This is the plane a link's mounting boss must sit on. It is *not* the joint
    origin: the J3 motor reaches 6.4 mm past link3's origin, so a boss placed
    on the joint plane is buried 7 mm inside the actuator and the connecting
    tube passes straight through it.

    Read from the stock model's own motor geometry, searching this link and its
    parent -- the arm is not consistent about which body carries a given motor
    mesh, so assuming either one gets it wrong half the time.
    """
    model = load_baseline()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise KeyError(name)

    frame = link_frame(name)
    axis = np.asarray(frame.parent_axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    candidates = []
    for source in (bid, int(model.body_parentid[bid])):
        offset = np.zeros(3)
        if source != bid:
            # Express the parent's geometry in this link's frame.
            offset = -np.array(model.body_pos[bid], dtype=float) * M_TO_MM
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != source or model.geom_group[g] != 2:
                continue
            mesh = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[g])
            if not (mesh or "").startswith("motor"):
                continue
            mid = model.geom_dataid[g]
            first, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            verts = (
                model.mesh_vert[first : first + count]
                @ _quat2mat(model.geom_quat[g]).T
                + model.geom_pos[g]
            ) * M_TO_MM + offset
            # Only the cluster near this joint; a mesh may hold two motors.
            near = verts[np.linalg.norm(verts, axis=1) < 140.0]
            if len(near) == 0:
                continue
            candidates.append(float((near @ axis).max()))

    return max(candidates) if candidates else None
