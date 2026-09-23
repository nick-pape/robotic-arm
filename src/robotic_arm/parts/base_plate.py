"""The base plate: a rounded plate with corner mounting holes and a
central bore for the shoulder servo's output shaft.
"""

from build123d import (
    Axis,
    BuildPart,
    BuildSketch,
    GridLocations,
    Hole,
    Locations,
    Part,
    Plane,
    RectangleRounded,
    extrude,
    fillet,
)

from robotic_arm.params import BASE, SERVO


def build_base_plate() -> Part:
    """Return the base plate as a solid."""
    hole_pitch = BASE.length - 2 * BASE.mount_hole_inset

    with BuildPart() as plate:
        with BuildSketch(Plane.XY):
            RectangleRounded(BASE.length, BASE.width, BASE.corner_radius)
        extrude(amount=BASE.thickness)

        top = plate.faces().sort_by(Axis.Z)[-1]

        # Four corner mounting holes, plus a central clearance bore for the
        # shoulder servo horn. Hole() with no depth drills all the way through.
        with Locations(top):
            with GridLocations(hole_pitch, hole_pitch, 2, 2):
                Hole(radius=BASE.mount_hole_dia / 2)
            Hole(radius=SERVO.horn_dia / 2)

        # Break the top edges so the printed part is easier to handle.
        fillet(plate.edges().group_by(Axis.Z)[-1], radius=1.0)

    return plate.part


if __name__ == "__main__":
    part = build_base_plate()
    print(f"volume: {part.volume:.1f} mm^3")
    print(f"bbox:   {part.bounding_box().size}")
