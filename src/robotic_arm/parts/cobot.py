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


def solid_mount_end(drum: Drum, mount_face: np.ndarray, thickness: float) -> Part:
    """The cavity to *keep out* of a boss's mount end, so its cap stays thick.

    The output relief is deeper than a wall-thickness cap, so cutting it into a
    shelled boss removes the cap outside the hub diameter and leaves the
    bearing pad floating. Welding a pad on afterwards produced degenerate
    solids -- coincident faces -- so the cap is made thick where the cavity is
    defined instead. The face that carries the joint load wants to be solid
    regardless.
    """
    axis = np.asarray(drum.axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    mount_face = np.asarray(mount_face, dtype=float)
    inward = np.sign(float((np.asarray(drum.centre, float) - mount_face) @ axis)) or 1.0
    return _oriented_cylinder(
        mount_face + axis * inward * thickness / 2,
        axis,
        drum.diameter / 2 + 2.0,
        thickness,
    )


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
    # Stop at the joint plane. Beyond it is the child link's space, and at a
    # rotating joint the two halves need a running clearance, not shared
    # volume. Letting the housing run past cost ~800 mm3 of overlap per joint;
    # the motor itself may protrude, since it is exposed anyway.
    high = child_at - STYLE.joint_gap / 2

    # Perpendicular position follows the motor; only the along-axis extent
    # is stretched to reach the mount face.
    perpendicular = motor_centre - axis * motor_at
    return Drum(
        centre=perpendicular + axis * (low + high) / 2,
        axis=axis,
        diameter=0.0,  # caller fills this in; only placement is decided here
        length=high - low,
    )


def mount_face_ring(drum: Drum, towards: np.ndarray, circle, hole_diameter: float,
                    depth: float) -> Part:
    """Bolt holes cut into whichever face of `drum` points at `towards`.

    Derived from the drum's actual geometry rather than from a constant
    offset. Every child bolt ring in this project was placed as
    `child_origin + axis * CHILD_LENGTH/2` using the nominal constant, while
    the housing had since been repositioned onto its motor -- so the rings sat
    26-27 mm off the face and removed no material at all.
    """
    axis = np.asarray(drum.axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    towards = np.asarray(towards, dtype=float)

    face_a = np.asarray(drum.centre, float) - axis * drum.length / 2
    face_b = np.asarray(drum.centre, float) + axis * drum.length / 2
    face = face_a if np.linalg.norm(towards - face_a) < np.linalg.norm(towards - face_b) else face_b
    # Cut inwards from that face.
    inward = 1.0 if np.dot(np.asarray(drum.centre, float) - face, axis) > 0 else -1.0

    return bolt_ring(
        centre=face + axis * inward * depth / 2,
        axis=axis,
        bcd=circle.bcd,
        count=circle.count,
        hole_diameter=hole_diameter,
        depth=depth,
    )


def cable_channel(start: np.ndarray, end: np.ndarray, diameter: float) -> Part:
    """A side channel for the harness, as a solid to subtract.

    Neither actuator is a hollow-shaft motor -- the RS06 and RS00 STEP files
    show only a O4 central feature -- so there is no route through the middle.
    A central bore in a printed boss implies a passage that does not exist, and
    on link2 and link3 it also ate into the bolt circle, leaving -2.7 mm and
    -1.7 mm of material. Cable therefore runs beside the actuator, which is
    what the stock arm does with its clipped XT30 daisy chain.
    """
    return tube(start, end, diameter)


def mating_face_opening(
    drum: Drum, towards: np.ndarray, diameter: float, depth: float | None = None
) -> Part:
    """An opening cut through a housing's mating face, as a solid to subtract.

    The face a child link bolts to cannot be closed: the actuator output and
    the child's own boss come through it. Shelling a drum leaves both ends
    capped, so this reopens the one that matters.
    """
    axis = np.asarray(drum.axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    centre = np.asarray(drum.centre, dtype=float)
    towards = np.asarray(towards, dtype=float)

    face_a = centre - axis * drum.length / 2
    face_b = centre + axis * drum.length / 2
    outward = axis if np.linalg.norm(towards - face_b) < np.linalg.norm(towards - face_a) else -axis
    face = face_b if np.allclose(outward, axis) else face_a

    depth = (RULES.structural_wall_thickness * 3) if depth is None else depth
    # Start just inside the shell and cut outward through the cap.
    return _oriented_cylinder(face + outward * (depth / 2 - depth), outward, diameter / 2, depth * 2)


def access_ports(
    drum: Drum,
    mount_face: np.ndarray,
    circle,
    driver_diameter: float,
    depth: float,
) -> Part:
    """Driver-sized holes through the far wall, coaxial with a bolt circle.

    A mount face can be perfectly reachable in principle and unusable in
    practice: on link2 and link3 the connecting tube leaves the boss directly
    over four of the six mounting bolts, so a hex key cannot reach them from
    either direction. The screws exist, the holes exist, and the part cannot be
    fastened.

    Ports are the conventional answer -- a clear bore through the opposite wall
    so the driver passes straight through. They cost a little stiffness and
    leave openings that want plugs or a cover, which is a fair trade against
    not being buildable.
    """
    axis = np.asarray(drum.axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    mount_face = np.asarray(mount_face, dtype=float)

    # Bore away from the mount face, into the part. Deriving the direction
    # rather than passing it avoids the sign trap that runs through this arm:
    # link2's parent axis points opposite link3's, so a hardcoded direction
    # drills out of one part and through the other.
    inward = np.sign(float((np.asarray(drum.centre, float) - mount_face) @ axis)) or 1.0
    return bolt_ring(
        centre=mount_face + axis * inward * depth / 2,
        axis=axis,
        bcd=circle.bcd,
        count=circle.count,
        hole_diameter=driver_diameter,
        depth=depth,
    )


def output_interface(
    drum: Drum, mount_face: np.ndarray, actuator, clearance: float | None = None
) -> Part:
    """Relief so a link seats on the actuator's output hub, not on its stator.

    This is the interface that actually carries the joint, and getting it wrong
    is not subtle: an actuator's output is a raised hub -- O52 standing 1.5 mm
    proud on the RS06, O34.6 by 0.4 mm on the RS00 -- carrying the bolt circle.
    The wider face beside it is the **stator**, which does not rotate.

    A flat printed boss pressed against both lands on the stator and the joint
    binds. Earlier revisions did exactly that: the boss was O76 against a O52
    hub, with no spigot, no relief and nothing locating it, so the parts did
    not so much mount as sit alongside each other.

    So the face is cut back everywhere outside the hub, by the hub's own
    protrusion plus a running gap.
    """
    clearance = STYLE.joint_gap if clearance is None else clearance
    if actuator.hub_diameter <= 0:
        return None

    axis = np.asarray(drum.axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    mount_face = np.asarray(mount_face, dtype=float)
    inward = np.sign(float((np.asarray(drum.centre, float) - mount_face) @ axis)) or 1.0

    depth = actuator.hub_protrusion + clearance
    outer = _oriented_cylinder(
        mount_face + axis * inward * depth / 2, axis, drum.diameter / 2 + 2.0, depth
    )
    # Leave the bearing annulus over the hub itself.
    inner = _oriented_cylinder(
        mount_face + axis * inward * depth / 2,
        axis,
        actuator.hub_diameter / 2,
        depth + 2.0,
    )
    return outer - inner


def mount_flange(drum: Drum, mount_face: np.ndarray, thickness: float) -> Part:
    """A solid pad at a boss's mount face, to union in before cutting relief.

    A shelled drum ends in a wall-thickness cap, and the output relief is
    deeper than that -- so cutting the relief straight into a shelled boss
    removes the whole cap outside the hub diameter and leaves the bearing pad
    floating as a separate solid. The face that carries the joint load needs to
    be solid anyway.
    """
    axis = np.asarray(drum.axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    mount_face = np.asarray(mount_face, dtype=float)
    inward = np.sign(float((np.asarray(drum.centre, float) - mount_face) @ axis)) or 1.0
    return _oriented_cylinder(
        mount_face + axis * inward * thickness / 2,
        axis,
        drum.diameter / 2,
        thickness,
    )


#: Datum convention for a joint.
#:
#: The actuator's **rotor hub face sits on the joint plane** -- that is, on the
#: child link's frame origin. Everything else follows: the motor body extends
#: from there back into the parent, its stator flange sits one hub-protrusion
#: behind the hub face, and the two links bolt onto those two faces
#: concentrically.
#:
#: Fixing this datum is what makes the interfaces placeable at all. Without it
#: each part guessed its own mount plane from the joint origin, which is where
#: the actuator *starts*, not where it ends -- so bosses ended up buried inside
#: the motors they were meant to bolt to.


#: ISO 4762 socket-cap head across-corners for M3. Sets how far a cup's bore
#: can open up before it starts eating its own bolt heads.
M3_HEAD_DIAMETER = 5.5


def carried_boss_diameter(actuator, clearance: float | None = None) -> float:
    """The boss diameter the *child* link will present into this cup."""
    return driven_boss(
        (0.0, 0.0, 1.0), np.zeros(3), actuator, 1.0, clearance=clearance,
        toward_body=(0.0, 0.0, 1.0),
    ).diameter


def driven_boss(axis, joint_plane, actuator, length: float, clearance: float | None = None,
                toward_body=None):
    """The driven link's half of a joint: a boss inside the stator ring.

    Bolts to the rotor circle and bears on the hub face. Sized under the hub
    so it can turn inside the other link's cup without rubbing -- the motor's
    own radial gap is only 0.25 mm, which is fine for machined parts and not
    for printed ones.
    """
    clearance = STYLE.joint_gap if clearance is None else clearance
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    joint_plane = np.asarray(joint_plane, dtype=float)

    # Wide enough for a driver to reach its own bolts. Sizing this purely to
    # duck under the hub gave the RS00 links a O35 boss carrying a O27 bolt
    # circle, where a O6 driver needs material out to r=16.5 -- so those bolts
    # were drilled and could never be turned. The hub is not the real limit:
    # the boss turns inside the *cup*, whose bore is ours to choose, and it
    # only has to stay clear of the stator bolt heads.
    from robotic_arm.assembly import DRIVER_DIAMETER

    wall = RULES.structural_wall_thickness
    diameter = max(
        actuator.rotor_circle.bcd + DRIVER_DIAMETER["M3"] + 2 * wall,
        actuator.hub_diameter - 2 * clearance,
    )
    headroom = actuator.stator_circle.bcd - M3_HEAD_DIAMETER - 4 * clearance
    if diameter > headroom:
        raise ValueError(
            f"{actuator.name}: a O{diameter:.1f} boss would foul the stator "
            f"bolt heads at O{actuator.stator_circle.bcd:.1f}"
        )
    # Extends away from the motor, into the link it drives. The direction has
    # to be given rather than taken from the axis: parent axes on this arm are
    # not consistently signed, and using the raw axis put link2's boss on the
    # far side of its own joint.
    if toward_body is None:
        toward_body = -axis
    toward_body = np.asarray(toward_body, dtype=float)
    toward_body = toward_body / np.linalg.norm(toward_body)
    return Drum(
        centre=joint_plane + toward_body * length / 2,
        axis=axis,
        diameter=diameter,
        length=length,
    )


def boss_direction(boss: Drum) -> np.ndarray:
    """Unit vector from the joint plane into the body of the driven link.

    `driven_boss` is built along a direction that is *not* always `+axis` --
    parent axes on this arm are not consistently signed -- and the boss centre
    is the only record of which way it went. Every feature cut into a boss has
    to follow it. Reading the sign off the raw axis instead is the single most
    repeated bug in this file: it has now put a boss, an access port, a seated
    actuator and a bolt ring on the wrong side of their own joint.

    The boss is placed against the joint plane at the origin, so the direction
    is simply the direction of its centre.
    """
    centre = np.asarray(boss.centre, dtype=float)
    length = float(np.linalg.norm(centre))
    if length < 1e-9:
        raise ValueError("boss is centred on its own joint plane; no direction")
    return centre / length


def carrying_cup(axis, joint_plane, actuator, length: float, clearance: float | None = None):
    """The other half: a ring bolted to the stator, around the driven boss.

    Bears on the stator flange, one hub-protrusion behind the hub face, and is
    bored out to clear the rotor and the driven link turning inside it.
    """
    clearance = STYLE.joint_gap if clearance is None else clearance
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    joint_plane = np.asarray(joint_plane, dtype=float)

    # Sits on the far side of the joint plane from the driven link.
    face = joint_plane + axis * actuator.hub_protrusion
    # Two constraints, both previously missed. The cup must be wide enough
    # that a driver port on the stator circle still leaves a full structural
    # wall outside it -- at bcd + 10 the port broke out to within 2 mm of the
    # surface -- and its *cavity* must clear the motor it skirts over. The
    # second one is easy to forget because the cup looks generous from the
    # outside: a O60 cup has a O56 cavity, and an RS00's stator flange is
    # O57, so the motor rim sat inside the cup wall.
    from robotic_arm.assembly import DRIVER_DIAMETER

    wall = RULES.structural_wall_thickness
    diameter = max(
        actuator.stator_circle.bcd + DRIVER_DIAMETER["M3"] + 2 * wall,
        actuator.stator_outer_diameter + 2 * clearance + 2 * wall,
    )
    return Drum(
        centre=face + axis * length / 2,
        axis=axis,
        diameter=diameter,
        length=length,
    )


def cup_bore(cup: Drum, actuator, clearance: float | None = None,
             boss_diameter: float | None = None) -> Part:
    """The opening through a carrying cup, as a solid to subtract.

    Must clear the rotor hub *and* the driven link's boss turning inside it,
    which is usually the larger of the two.
    """
    clearance = STYLE.joint_gap if clearance is None else clearance
    diameter = max(
        actuator.stator_inner_diameter,
        actuator.hub_diameter + 2 * clearance,
        boss_diameter + 2 * clearance if boss_diameter else 0.0,
    )
    return _oriented_cylinder(
        np.asarray(cup.centre, dtype=float), cup.axis, diameter / 2, cup.length + 8.0
    )


def link_interfaces(body: str, boss_length: float, cup_length: float):
    """Both ends of a link, each on whichever ring the stock arm puts it.

    A link meets its own joint at one end and its child's joint at the other.
    Which *ring* each end bolts to is measured, not assumed -- see
    `robotic_arm.mounts`. The rule this code used to hardcode ("own end drives
    off the rotor, child end carries the stator") holds at J3, J4 and J5 and
    is wrong at J2: stock link2 carries the stators of both J2 and J3 and has
    no driven boss at all.

    * a **rotor** end is a driven boss -- small, inside the stator ring,
      turning;
    * a **stator** end is a carrying cup -- larger, bolted to the fixed
      flange, bored to let the other side's hub turn through it.

    Both sit against their joint planes, where the rotor hub faces are by the
    datum convention above.
    """
    from robotic_arm.actuators import for_joint
    from robotic_arm.linkframes import link_frame
    from robotic_arm.mounts import child_joint_role, own_joint_role

    frame = link_frame(body)
    index = int(body.removeprefix("link"))

    own = for_joint(f"joint{index}")
    axis = np.asarray(frame.parent_axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    reach = float(np.asarray(frame.stock_centre, dtype=float) @ axis)
    into_body = axis * (1.0 if reach >= 0 else -1.0)

    if own_joint_role(body) == "rotor":
        boss = driven_boss(axis, np.zeros(3), own, boss_length,
                           toward_body=into_body)
    else:
        # A stator end. The motor sits on this link's side of the plane, so
        # the cup runs the same way a boss would -- into the body -- but it
        # bolts to the outer ring and is bored for the hub turning inside it.
        boss = carrying_cup(into_body, np.zeros(3), own, boss_length)

    cup = None
    carried = None
    if frame.child_name and frame.child_name.startswith("link"):
        child_index = int(frame.child_name.removeprefix("link"))
        if child_index <= 6:
            carried = for_joint(f"joint{child_index}")
            child_axis = -np.asarray(frame.child_axis, dtype=float)
            if child_joint_role(body) == "stator":
                # The cup skirts back over the motor, which lies on this
                # link's side of the child joint.
                cup = carrying_cup(child_axis, frame.child_origin, carried,
                                   cup_length)
            else:
                cup = driven_boss(child_axis, frame.child_origin, carried,
                                  cup_length, toward_body=child_axis)
    return boss, cup, own, carried


def interface_role(body: str, end: str) -> str:
    """"rotor" or "stator" for the named end of a link. `end` is own|child."""
    from robotic_arm.mounts import child_joint_role, own_joint_role

    return own_joint_role(body) if end == "own" else child_joint_role(body)


def build_link(
    body: str,
    boss_length: float,
    cup_length: float,
    tube_stations: tuple[float, ...],
    tube_diameters: tuple[float, ...],
    wall: float | None = None,
    tube_end_offset: float = 0.0,
):
    """A whole link: driven boss, carrying cup, tube between, properly bolted.

    One builder rather than four near-copies. The four parts previously
    repeated this sequence with small divergences, which is how the same fault
    -- both interfaces bolted to inner rings -- ended up in all of them, and
    how fixes landed in some and not others.
    """
    from robotic_arm.assembly import DRIVER_DIAMETER

    wall = RULES.structural_wall_thickness if wall is None else wall
    boss, cup, own, carried = link_interfaces(body, boss_length, cup_length)
    if cup is None:
        return None

    # A tube wider than the interface it leaves is not a taper, it is a hole:
    # the tube's *cavity* then exceeds the boss's outer diameter and eats its
    # wall, severing the boss from the link. Caught here rather than left to
    # surface as a mysterious extra solid.
    if tube_diameters[0] > boss.diameter - wall:
        raise ValueError(
            f"{body}: tube starts at O{tube_diameters[0]:.0f} on a "
            f"O{boss.diameter:.0f} boss; its cavity would cut the boss wall"
        )
    if tube_diameters[-1] > cup.diameter - wall:
        raise ValueError(
            f"{body}: tube ends at O{tube_diameters[-1]:.0f} on a "
            f"O{cup.diameter:.0f} cup; its cavity would cut the cup wall"
        )

    # Where the tube aims. Aiming at the cup's centre is right for an in-line
    # joint and wrong for a perpendicular one: on a wrist the tube then runs
    # diagonally through the joint bore. `tube_end_offset` slides the target
    # deeper into the cup, so the tube meets the housing's side and leaves the
    # bore clear -- which is how a UR-style wrist is actually shaped.
    cup_axis = np.asarray(cup.axis, float) / np.linalg.norm(cup.axis)
    end = np.asarray(cup.centre, float) + cup_axis * tube_end_offset
    start = np.asarray(boss.centre, float)
    points = [start + (end - start) * f for f in tube_stations]

    cup_out = -np.asarray(cup.axis, float) / np.linalg.norm(cup.axis)
    mating_face = np.asarray(cup.centre, float) + cup_out * cup.length / 2
    tube_solid = lofted_tube(points, list(tube_diameters))

    outer = boss.solid() + cup.solid() + tube_solid
    inner = (
        boss.solid(boss.diameter - 2 * wall, boss.length - 2 * wall)
        + cup.solid(cup.diameter - 2 * wall, cup.length - 2 * wall)
        + lofted_tube(points, [d - 2 * wall for d in tube_diameters])
    )
    part = outer - inner

    # Bore the cup so the motor it carries, and the link turning inside it,
    # actually fit.
    part -= cup_bore(cup, carried, boss_diameter=carried_boss_diameter(carried))

    # This link's own end, on whichever ring the stock arm puts it. Placed
    # along the end's own direction, not `-axis`: see `boss_direction`.
    into = boss_direction(boss)
    own_role = interface_role(body, "own")
    own_circle = own.rotor_circle if own_role == "rotor" else own.stator_circle
    if own_role == "stator":
        # Bored for the other side's hub, which turns through it.
        part -= cup_bore(boss, own, boss_diameter=carried_boss_diameter(own))
    part -= bolt_ring(
        centre=into * wall * 1.5,
        axis=into,
        bcd=own_circle.bcd,
        count=own_circle.count,
        hole_diameter=RULES.m3_clearance,
        depth=wall * 4,
    )
    # Stator ring on the cup's mating face.
    cup_face = np.asarray(cup.centre, float) - np.asarray(cup.axis, float) * cup.length / 2
    part -= bolt_ring(
        centre=cup_face + np.asarray(cup.axis, float) * wall * 1.5,
        axis=cup.axis,
        bcd=carried.stator_circle.bcd,
        count=carried.stator_circle.count,
        hole_diameter=RULES.m3_clearance,
        depth=wall * 4,
    )
    # Driver access to the rotor bolts, but only where the boss has room for
    # it. A O6 driver bore on a O27 circle needs wall out to r=16.5, which a
    # O35 boss (wall from r=15.5) does not have -- cutting it anyway severs
    # the boss from the rest of the link. Where there is no room the bolts are
    # reached down the link's own hollow interior instead, and
    # `assembly.reachable_directions` reports which links actually manage it
    # rather than this assuming either way.
    driver = DRIVER_DIAMETER["M3"]
    if own_circle.bcd / 2 + driver / 2 <= boss.diameter / 2 - wall:
        # From the far end down to the seating cap -- not through it. A port
        # spanning the whole boss removes the very face the bolt head bears
        # on, which reads as "reachable" while describing a screw with
        # nothing to pull against.
        part -= bolt_ring(
            centre=into * (wall + boss.length) / 2,
            axis=into,
            bcd=own_circle.bcd,
            count=own_circle.count,
            hole_diameter=driver,
            depth=boss.length - wall,
        )

    # The same port through the carrying cup. Without it the stator bolts were
    # drilled into a blind annulus: modelled, cut, and unfastenable.
    if carried.stator_circle.bcd / 2 + driver / 2 <= cup.diameter / 2 - wall:
        part -= bolt_ring(
            centre=cup_face + cup_axis * (wall + cup.length) / 2,
            axis=cup_axis,
            bcd=carried.stator_circle.bcd,
            count=carried.stator_circle.count,
            hole_diameter=driver,
            depth=cup.length - wall,
        )

    # 5 mm back from the mating face, expressed along the drum's axis, which
    # may point the opposite way to the boss.
    facing = float(into @ (np.asarray(boss.axis, float)
                           / np.linalg.norm(boss.axis)))
    # Where the boss is wider than the hub it bears on, its rim would otherwise
    # sweep the stator face -- only 0.4 mm behind the hub face on an RS00.
    # Relieve it back to a clearance the joint can actually hold.
    if own_role == "rotor" and boss.diameter > own.hub_diameter:
        relief = own.hub_protrusion + STYLE.joint_gap
        part -= (
            _oriented_cylinder(into * relief / 2, into,
                               (boss.diameter + 2.0) / 2, relief)
            - _oriented_cylinder(into * relief / 2, into,
                                 own.hub_diameter / 2, relief + 2.0)
        )

    part -= seam_groove(boss, offset_along_axis=facing * (-boss.length / 2 + 5.0))
    part -= seam_groove(cup, offset_along_axis=-cup.length / 2 + 5.0)

    part = break_edges(part)

    # The joint bore must stay clear right through the mating face. Past that
    # face is the *child* link, not the motor -- the motor sits inside the cup
    # on this side -- so the local invariant is the bore, not the whole cup
    # footprint. Clearance to the moving neighbour is a different question,
    # answered over the full joint range in `robotic_arm.collision`, and
    # probing at cup diameter here just re-asked it badly: it flagged link4's
    # tube passing 31 mm off the joint axis as a collision.
    bore = max(
        carried.stator_inner_diameter,
        carried.hub_diameter + 2 * STYLE.joint_gap,
        carried_boss_diameter(carried) + 2 * STYLE.joint_gap,
    )
    probe = _oriented_cylinder(mating_face + cup_out * 15.0, cup_out, bore / 2, 30.0)
    intrusion = part.intersect(probe)
    intruding = 0.0
    if intrusion is not None:
        pieces = intrusion if hasattr(intrusion, "__iter__") else [intrusion]
        for piece in pieces:
            intruding += sum(float(x.volume) for x in piece.solids())
    if intruding > 1.0:
        raise ValueError(
            f"{body}: {intruding:,.0f} mm^3 of the link blocks its own O"
            f"{bore:.0f} joint bore past the cup's mating face; narrow "
            f"tube_diameters[-1] (O{tube_diameters[-1]:.0f}) or pull the last "
            f"station back"
        )

    if len(part.solids()) != 1:
        volumes = sorted((float(x.volume) for x in part.solids()), reverse=True)
        raise ValueError(
            f"{body}: built {len(part.solids())} solids "
            f"({', '.join(f'{v:,.0f}' for v in volumes)} mm^3); a link is one "
            f"printed piece, so a second solid is loose geometry, not a part"
        )
    return part
