"""Can the thing actually be built? Bolt access and assembly order.

A closed shell can be perfectly correct as a solid and impossible to assemble.
Nothing else in this project checks that: mass, envelope, clearance, inertia
and even exact CAD interference are all indifferent to whether a hex key can
reach a screw. This module asks that question directly.

The test is deliberately physical. A bolt is reachable if a driver of the
right size can travel along the bolt's own axis, from outside the part to the
screw head, without passing through material. That is modelled as a cylinder
swept along the axis: if it intersects the shell, something is in the way.

Units: millimetres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robotic_arm.design import RULES

#: Across-flats plus clearance for a hex key, by thread. A driver needs more
#: room than the screw it turns, which is the whole point.
DRIVER_DIAMETER = {"M3": 6.0, "M4": 8.0, "M5": 9.0, "M6": 11.0}

#: How far back from the face a driver has to come. A stubby key still needs
#: this much straight run to engage and turn.
DRIVER_REACH = 45.0


@dataclass(frozen=True)
class BoltAccess:
    """Whether one fastener can be reached, and by how much it is blocked."""

    index: int
    position: np.ndarray
    blocked_mm3: float
    driver_diameter: float

    @property
    def reachable(self) -> bool:
        # A sliver is tessellation noise; a real obstruction is not.
        return self.blocked_mm3 < 1.0


def ring_positions(centre, axis, bcd: float, count: int, start_angle: float = 0.0):
    """Bolt positions on a circle, in the part's frame."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(seed @ axis)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, seed)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    out = []
    for i in range(count):
        theta = np.deg2rad(start_angle + 360.0 * i / count)
        out.append(
            np.asarray(centre, dtype=float)
            + (np.cos(theta) * u + np.sin(theta) * v) * bcd / 2
        )
    return out


def check_ring(
    part,
    centre,
    axis,
    circle,
    thread: str = "M3",
    approach: int = 1,
    reach: float = DRIVER_REACH,
) -> list[BoltAccess]:
    """Driver access for every bolt on one circle.

    `approach` is +1 or -1: which way along the axis the driver comes from.
    A bolt reachable from one side may be buried from the other, so the answer
    depends on the assembly direction and both are worth checking.
    """
    from robotic_arm.parts.cobot import _oriented_cylinder

    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    diameter = DRIVER_DIAMETER[thread]

    results = []
    for index, position in enumerate(
        ring_positions(centre, axis, circle.bcd, circle.count)
    ):
        # The corridor runs from the bolt outwards along the approach
        # direction. Start just clear of the head so the screw's own hole is
        # not counted as an obstruction.
        corridor = _oriented_cylinder(
            position + axis * approach * (reach / 2 + 1.0),
            axis * approach,
            diameter / 2,
            reach,
        )
        obstruction = part.intersect(corridor)
        volume = 0.0
        if obstruction is not None:
            pieces = obstruction if hasattr(obstruction, "__iter__") else [obstruction]
            for piece in pieces:
                volume += sum(float(s.volume) for s in piece.solids())
        results.append(BoltAccess(index, position, volume, diameter))
    return results


def report(body: str) -> str:
    """Bolt access for one printed part, from both approach directions."""
    import importlib

    from robotic_arm.linkframes import link_frame
    from robotic_arm.parts import REGISTRY

    if body not in REGISTRY:
        raise KeyError(f"{body} is not a printed part")
    module = importlib.import_module(REGISTRY[body][0].__module__)
    part = REGISTRY[body][0]()
    frame = link_frame(body)
    drums = getattr(module, "_drums", None)
    if drums is None:
        return f"{body}: not built from drums; no rings to check"

    parent, child = drums()
    lines = [f"{body}:"]
    for label, drum, towards in (
        ("parent mount", parent, -np.asarray(parent.axis) * 1000.0),
        ("child mount", child, np.asarray(frame.child_origin)),
    ):
        axis = np.asarray(drum.axis, dtype=float)
        axis /= np.linalg.norm(axis)
        face_a = np.asarray(drum.centre, float) - axis * drum.length / 2
        face_b = np.asarray(drum.centre, float) + axis * drum.length / 2
        face = (
            face_a
            if np.linalg.norm(towards - face_a) < np.linalg.norm(towards - face_b)
            else face_b
        )
        if hasattr(module, "_mount_circle"):
            circle = module._mount_circle()
        elif hasattr(module, "_mount_circles"):
            circles = module._mount_circles()
            circle = circles[0 if label.startswith("parent") else 1]
        else:
            # Wrist parts bolt to the same RS00 output circle at both ends.
            from robotic_arm.actuators import RS00

            circle = RS00().output_circle

        for approach in (1, -1):
            bolts = check_ring(part, face, axis, circle, approach=approach)
            blocked = [b for b in bolts if not b.reachable]
            worst = max((b.blocked_mm3 for b in bolts), default=0.0)
            verdict = "clear" if not blocked else f"{len(blocked)}/{len(bolts)} blocked"
            lines.append(
                f"   {label:13} approach {approach:+d}: {verdict}"
                f"  (worst obstruction {worst:.0f} mm3)"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    from robotic_arm.parts import REGISTRY

    for body in sorted(REGISTRY):
        print(report(body))


def mount_rings(body: str):
    """(label, face point, axis, circle) for each mount ring on a printed part."""
    import importlib

    from robotic_arm.linkframes import link_frame
    from robotic_arm.parts import REGISTRY

    module = importlib.import_module(REGISTRY[body][0].__module__)
    drums = getattr(module, "_drums", None)
    if drums is None:
        return []

    frame = link_frame(body)
    parent, child = drums()
    out = []
    for label, drum, towards in (
        ("parent", parent, -np.asarray(parent.axis) * 1000.0),
        ("child", child, np.asarray(frame.child_origin)),
    ):
        axis = np.asarray(drum.axis, dtype=float)
        axis /= np.linalg.norm(axis)
        a = np.asarray(drum.centre, float) - axis * drum.length / 2
        b = np.asarray(drum.centre, float) + axis * drum.length / 2
        face = a if np.linalg.norm(towards - a) < np.linalg.norm(towards - b) else b

        if hasattr(module, "_mount_circle"):
            circle = module._mount_circle()
        elif hasattr(module, "_mount_circles"):
            circle = module._mount_circles()[0 if label == "parent" else 1]
        else:
            from robotic_arm.actuators import RS00

            circle = RS00().output_circle
        out.append((label, face, axis, circle))
    return out


def reachable_directions(body: str) -> dict[str, list[int]]:
    """Which approach directions give full driver access to each mount ring.

    An empty list means that ring cannot be fastened at all -- the screws are
    modelled, the holes are cut, and no driver can reach them.
    """
    from robotic_arm.parts import REGISTRY

    part = REGISTRY[body][0]()
    out: dict[str, list[int]] = {}
    for label, face, axis, circle in mount_rings(body):
        ok = []
        for approach in (1, -1):
            bolts = check_ring(part, face, axis, circle, approach=approach)
            if all(b.reachable for b in bolts):
                ok.append(approach)
        out[label] = ok
    return out


def assembly_order() -> list[str]:
    """Build order for the printed structure, distal last.

    Each link bolts to the actuator carried by its parent, so the parent must
    be on the bench before the child goes on. Working inwards-out also keeps
    every mount face exposed when its bolts are driven: once a child is
    attached it covers the joint.
    """
    from robotic_arm.parts import REGISTRY

    return [b for b in ("link2", "link3", "link4", "link5", "link6") if b in REGISTRY]


def seated_actuator(body: str):
    """The carried joint's actuator, positioned as it would be bolted on.

    Datum is the **hub face**, not the bolt-circle plane. Those differ -- the
    RS06's bolt circle sits 0.5 mm inside its hub face -- and using the wrong
    one buries the hub in the printed part and reports interference that is an
    artefact of the check rather than of the design.
    """
    import numpy as np
    from build123d import Location, Plane, Vector, import_step

    from robotic_arm.actuators import RS06, RS00
    from robotic_arm.linkframes import link_frame
    from robotic_arm.parts.cobot import boss_mount_face
    from robotic_arm.reference import RS00_STEP, RS06_STEP, require
    import importlib

    from robotic_arm.parts import REGISTRY

    module = importlib.import_module(REGISTRY[body][0].__module__)
    if not hasattr(module, "_drums"):
        return None

    # The actuator this link's parent boss bolts onto is the one driving this
    # link's own joint.
    from robotic_arm.actuators import for_joint

    joint = f"joint{body.removeprefix('link')}"
    actuator = for_joint(joint)
    step = RS06_STEP if actuator.name == "RS06" else RS00_STEP
    solid = import_step(str(require(step)))

    # Hub face is the outermost point along the actuator's own axis.
    hub_face_z = solid.bounding_box().min.Z

    frame = link_frame(body)
    face = boss_mount_face(frame, module.PARENT_LENGTH, module.PARENT_PROTRUSION)
    axis = np.asarray(frame.parent_axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    # The actuator sits on the parent's side, so it extends *away* from this
    # link's own body. Taking the direction from the raw axis gets that
    # backwards on half the links, because the parent axes on this arm are
    # not consistently signed -- the same trap that put link2's boss on the
    # wrong side of its joint. Derive it from where the body actually is.
    reach = float(np.asarray(frame.stock_centre, dtype=float) @ axis)
    toward_body = axis * (1.0 if reach >= 0 else -1.0)
    plane = Plane(origin=Vector(*face), z_dir=Vector(*(-toward_body)))
    return plane * Location((0.0, 0.0, -hub_face_z)) * solid


def actuator_interference(body: str) -> float:
    """Volume of printed material occupying the same space as its actuator, mm3.

    Checked against RobStride's own STEP rather than the stock arm's motor
    meshes: the meshes measure about 82 mm across where the vendor body is
    O57, and it is the vendor geometry we would actually bolt to.
    """
    from robotic_arm.parts import REGISTRY

    motor = seated_actuator(body)
    if motor is None:
        return 0.0
    overlap = REGISTRY[body][0]().intersect(motor)
    if overlap is None:
        return 0.0
    pieces = overlap if hasattr(overlap, "__iter__") else [overlap]
    return sum(float(s.volume) for piece in pieces for s in piece.solids())
