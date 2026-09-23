"""Shared geometry for the cobot-style shells.

The form language, taken from a standard collaborative arm: each joint reads as
a cylindrical drum, drums are joined by a slimmer tube, the whole thing is
closed, and every rotating interface shows a circumferential seam. No fasteners
are visible from outside.

It suits printed parts as well as it looks. A cylinder is stiff in every
bending direction rather than just one, it prints on its axis without support,
and it leaves a natural cavity for the actuator and the cable run.

Shells are built as an outer solid minus an inner solid of the same shape,
rather than with an offset operation. That is more predictable in OCCT on
unions of cylinders, and it keeps the wall thickness explicit.

Units: millimetres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from build123d import (
    Align,
    Axis,
    Cylinder,
    Location,
    Part,
    Plane,
    Rotation,
    Vector,
)

from robotic_arm.design import RULES, STYLE


@dataclass(frozen=True)
class Drum:
    """A cylindrical joint housing.

    `axis` is the rotation axis in the link frame; `centre` is where the drum
    sits. `overhang` is how far it extends past the joint plane on the far
    side, which is what makes a joint look like a continuous cylinder rather
    than a disc stuck on the end of a tube.
    """

    centre: np.ndarray
    axis: np.ndarray
    diameter: float
    length: float

    def solid(self, diameter: float | None = None, length: float | None = None) -> Part:
        """A cylinder on this drum's axis, optionally resized for shelling."""
        d = self.diameter if diameter is None else diameter
        ln = self.length if length is None else length
        return _oriented_cylinder(self.centre, self.axis, d / 2, ln)


def _oriented_cylinder(
    centre: np.ndarray, axis: np.ndarray, radius: float, length: float
) -> Part:
    """A cylinder of `length`, centred on `centre`, pointing along `axis`."""
    direction = Vector(*(float(v) for v in axis))
    plane = Plane(origin=Vector(*(float(v) for v in centre)), z_dir=direction)
    return plane * Cylinder(
        radius=radius, height=length, align=(Align.CENTER, Align.CENTER, Align.CENTER)
    )


def tube(start: np.ndarray, end: np.ndarray, diameter: float) -> Part:
    """A cylinder spanning two points, used to join drums."""
    delta = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = float(np.linalg.norm(delta))
    if length < 1e-9:
        raise ValueError("tube endpoints coincide")
    midpoint = (np.asarray(start, dtype=float) + np.asarray(end, dtype=float)) / 2
    return _oriented_cylinder(midpoint, delta / length, diameter / 2, length)


def shelled_body(
    parent: Drum,
    child: Drum,
    tube_diameter: float,
    wall: float | None = None,
    tube_from: float = 0.0,
) -> Part:
    """Two drums joined by a tube, hollowed to a shell of `wall` thickness.

    `tube_from` slides the tube's start along the parent axis, so the
    connecting tube can leave the drum off-centre where the geometry needs it.
    """
    wall = RULES.structural_wall_thickness if wall is None else wall

    start = np.asarray(parent.centre, dtype=float) + np.asarray(
        parent.axis, dtype=float
    ) * tube_from

    outer = (
        parent.solid()
        + child.solid()
        + tube(start, np.asarray(child.centre, dtype=float), tube_diameter)
    )
    # Same form, one wall thinner in every direction. Shortening each drum by
    # two walls keeps the ends closed rather than open tubes.
    inner = (
        parent.solid(parent.diameter - 2 * wall, parent.length - 2 * wall)
        + child.solid(child.diameter - 2 * wall, child.length - 2 * wall)
        + tube(start, np.asarray(child.centre, dtype=float), tube_diameter - 2 * wall)
    )
    return outer - inner


def seam_groove(drum: Drum, offset_along_axis: float) -> Part:
    """A shallow circumferential groove, the visual mark of a rotating joint.

    Cut rather than added: on a printed part a groove is free, whereas a raised
    ring needs support on one side.
    """
    outer = _oriented_cylinder(
        np.asarray(drum.centre, dtype=float)
        + np.asarray(drum.axis, dtype=float) * offset_along_axis,
        drum.axis,
        drum.diameter / 2 + 1.0,
        STYLE.seam_groove_width,
    )
    inner = _oriented_cylinder(
        np.asarray(drum.centre, dtype=float)
        + np.asarray(drum.axis, dtype=float) * offset_along_axis,
        drum.axis,
        drum.diameter / 2 - STYLE.seam_groove_depth,
        STYLE.seam_groove_width + 2.0,
    )
    return outer - inner


def bolt_ring(
    centre: np.ndarray,
    axis: np.ndarray,
    bcd: float,
    count: int,
    hole_diameter: float,
    depth: float,
    start_angle: float = 0.0,
) -> Part:
    """Holes on a bolt circle, as a solid to subtract.

    Built from individual cylinders rather than a polar-located sketch, so the
    ring can sit on any axis rather than only on a principal plane.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    # Any vector not parallel to the axis gives a starting radial direction.
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(seed @ axis)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, seed)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    holes = None
    for i in range(count):
        theta = np.deg2rad(start_angle + 360.0 * i / count)
        radial = np.cos(theta) * u + np.sin(theta) * v
        position = np.asarray(centre, dtype=float) + radial * bcd / 2
        hole = _oriented_cylinder(position, axis, hole_diameter / 2, depth)
        holes = hole if holes is None else holes + hole
    return holes


def break_edges(part: Part, radius: float | None = None) -> Part:
    """Break the sharpest outer edges, tolerating the ones OCCT refuses.

    A union of cylinders produces tangent intersections that cannot always be
    filleted. Rather than fail the build, this fillets what it can and leaves
    the rest -- a missing edge break is cosmetic, a failed build is not.
    """
    from build123d import fillet

    radius = STYLE.edge_break if radius is None else radius
    circles = [
        e
        for e in part.edges().filter_by(lambda e: e.geom_type.name == "CIRCLE")
    ]
    for edge in circles:
        try:
            part = fillet([edge], radius=radius)
        except Exception:  # noqa: BLE001 - cosmetic only, see docstring
            continue
    return part


@dataclass(frozen=True)
class CollisionCylinder:
    """A cylinder standing in for part of a shell during collision checks.

    MuJoCo treats every mesh geom as its convex hull, and these shells are
    L-shaped: one hull round a two-drum wrist would swallow the space between
    the drums and report collisions that do not exist. A handful of primitives
    tracing the real form is both more accurate and far cheaper, which is what
    the Menagerie models do too.

    Built from the same drums as the visual solid, so the two cannot drift.
    """

    centre: np.ndarray
    axis: np.ndarray
    radius: float
    length: float

    @classmethod
    def from_drum(cls, drum: Drum, inflate: float = 0.0) -> CollisionCylinder:
        return cls(
            centre=np.asarray(drum.centre, dtype=float),
            axis=np.asarray(drum.axis, dtype=float),
            radius=drum.diameter / 2 + inflate,
            length=drum.length,
        )

    @classmethod
    def from_span(
        cls, start: np.ndarray, end: np.ndarray, diameter: float
    ) -> CollisionCylinder:
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length < 1e-9:
            raise ValueError("collision cylinder endpoints coincide")
        return cls(
            centre=(start + end) / 2,
            axis=delta / length,
            radius=diameter / 2,
            length=length,
        )


def mount_boss_diameter(
    bcd: float, hole_diameter: float, margin: float = 2.5, floor: float = 0.0
) -> float:
    """Diameter of a drum that only has to carry a bolt circle.

    A link's *parent* drum does not enclose its own actuator: the motor driving
    a joint is mounted on the parent link, not the child. Only the *child* drum
    houses a motor. Sizing a parent drum to an actuator therefore makes it far
    too fat, filling space the stock arm leaves open -- which shows up as
    self-collisions the stock arm does not have.

    `floor` raises it above that structural minimum for proportion. All three
    wrist joints use the same actuator, so all three housings come out the same
    diameter and the wrist cannot taper; letting the bosses between them shrink
    to the bolt circle turns a continuous stack into a row of lumps joined by
    thin necks. A clearance sweep says O54 is free here and O60 is not.
    """
    structural = bcd + 2 * (hole_diameter / 2 + RULES.structural_wall_thickness + margin)
    return max(structural, floor)


def lofted_tube(
    points: list[np.ndarray], diameters: list[float]
) -> Part:
    """A tube through a series of stations, tapering between given diameters.

    A long link is not a constant-diameter tube on a real cobot, and it should
    not be here either: the stock forearm narrows to about O44 at midspan and
    flares at both ends, because that is where the bending moment is lowest.
    Following that profile both looks right and keeps the part inside the space
    the stock arm occupies.
    """
    from build123d import Circle, Plane, Vector, loft

    if len(points) != len(diameters) or len(points) < 2:
        raise ValueError("need matching points and diameters, at least two")

    sections = []
    for index, (point, diameter) in enumerate(zip(points, diameters)):
        # Orient each section across the local run direction, so the loft does
        # not shear where the path changes angle.
        if index == 0:
            direction = np.asarray(points[1], float) - np.asarray(points[0], float)
        elif index == len(points) - 1:
            direction = np.asarray(points[-1], float) - np.asarray(points[-2], float)
        else:
            direction = np.asarray(points[index + 1], float) - np.asarray(
                points[index - 1], float
            )
        direction = direction / np.linalg.norm(direction)
        plane = Plane(
            origin=Vector(*(float(v) for v in point)),
            z_dir=Vector(*(float(v) for v in direction)),
        )
        sections.append(plane * Circle(diameter / 2))
    return loft(sections)


def shell(outer: Part, inner: Part) -> Part:
    """Hollow a solid by subtracting a matching inner solid."""
    return outer - inner


def boss_centre(frame, length: float, protrusion: float) -> np.ndarray:
    """Centre of a parent boss, placed on the side the link's body is on.

    Getting this wrong is easy and not obvious in the numbers. A boss is not
    symmetric about its joint plane: it stands a little proud on the parent's
    side and extends the rest of the way into its own link. Which way that is
    depends on the sign of the parent axis, and the axes on this arm are not
    consistently signed -- link2's points the opposite way to link3's, link4's
    and link5's.

    Writing the offset against the raw axis therefore placed link2's boss 31 mm
    clear of its own joint, leaving a visible gap at the shoulder. Deriving the
    direction from where the stock part's material actually sits removes the
    chance of getting the sign wrong at all.
    """
    axis = np.asarray(frame.parent_axis, dtype=float)
    # Which way along the axis the stock body lies. Fall back to +axis for a
    # part whose centroid sits on the joint plane.
    reach = float(np.asarray(frame.stock_centre, dtype=float) @ axis)
    toward_body = axis * (1.0 if reach >= 0 else -1.0)
    return toward_body * (length / 2 - protrusion)


def boss_mount_face(frame, length: float, protrusion: float) -> np.ndarray:
    """The outboard face of a parent boss, where its bolt circle sits.

    That is the face that lands against the actuator output, so it is where the
    mounting holes have to break through.
    """
    axis = np.asarray(frame.parent_axis, dtype=float)
    reach = float(np.asarray(frame.stock_centre, dtype=float) @ axis)
    toward_body = axis * (1.0 if reach >= 0 else -1.0)
    return boss_centre(frame, length, protrusion) - toward_body * (length / 2)


def housing_over(frame, motor_centre: np.ndarray, motor_length: float) -> Drum:
    """A housing that covers both the carried motor and the child mount face.

    Two constraints pull in different directions and both matter. The drum must
    sit over the motor, which on these links is 26-30 mm off the joint plane --
    centring it on the joint instead left it visibly beside the motor. It must
    also reach the child joint origin, because that is where the next link
    bolts on; centring it purely on the motor pulled link5's face 1.5 mm short
    of link6's mount.

    So the drum spans from behind the motor to the mount face, whichever is
    further, and sits on the motor's own offset from the axis.
    """
    axis = np.asarray(frame.child_axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    motor_centre = np.asarray(motor_centre, dtype=float)
    child = np.asarray(frame.child_origin, dtype=float)

    motor_at = float(motor_centre @ axis)
    child_at = float(child @ axis)
    low = min(motor_at - motor_length / 2, child_at)
    high = max(motor_at + motor_length / 2, child_at)

    # Perpendicular position follows the motor; only the along-axis extent
    # is stretched to reach the mount face.
    perpendicular = motor_centre - axis * motor_at
    return Drum(
        centre=perpendicular + axis * (low + high) / 2,
        axis=axis,
        diameter=0.0,  # caller fills this in; only placement is decided here
        length=high - low,
    )
