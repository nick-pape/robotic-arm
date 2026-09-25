"""Re-derive actuator bolt geometry from the vendor STEP files.

The spec listed the RS06 bolt circle as [UNVERIFIED] -- no published text gives
it, and the manual's dimension drawing is a raster image. The authoritative
source is RobStride's own STEP, so we measure it rather than transcribe it.

This also caught a real error in the published RS00 figures: see the note in
`reference/actuator_geometry.json`.

Output is committed, so `actuators.py` does not need the STEP files or this
script at runtime.

    uv run --extra cad python scripts/measure_actuators.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from build123d import import_step
from quiddity import recognise_hole_patterns, recognise_holes

REPO = Path(__file__).resolve().parents[1]
STEP_DIR = REPO / "reference" / "step"
OUT = REPO / "reference" / "actuator_geometry.json"

ACTUATORS = {"RS06": "RS06-new.step", "RS00": "RS00.step"}


def circle_angles(holes) -> list[float]:
    """Hole angles in degrees, sorted, for checking even spacing."""
    return sorted(
        round(math.degrees(math.atan2(h["location"][1], h["location"][0])), 3)
        for h in holes
    )


def even_spacing(angles: list[float]) -> float | None:
    """Uniform angular pitch in degrees, or None if the holes are not even."""
    if len(angles) < 2:
        return None
    gaps = [
        round((b - a) % 360, 3)
        for a, b in zip(angles, angles[1:] + [angles[0] + 360])
    ]
    return gaps[0] if max(gaps) - min(gaps) < 0.5 else None


def mounting_face(shape) -> dict:
    """The joint interface: two concentric annular faces on one end.

    These are pancake actuators, and both links bolt to the **same** face. The
    centre is the rotor hub, which turns; the annulus around it is the stator
    flange, which does not. Each carries its own bolt ring, so one link goes on
    the inside and the other on the outside, concentric, with the motor's own
    radial gap between them.

    Missing this is what made the printed links wrong: both of their interfaces
    were bolted to inner rings, so nothing ever held a stator.

    Bands are read directly off the solid, because no published figure gives
    the hub diameter or the stator flange's inner edge.
    """
    import numpy as np

    verts = np.array([[v.X, v.Y, v.Z] for v in shape.vertices()])
    z = verts[:, 2]
    radius = np.hypot(verts[:, 0], verts[:, 1])

    hub_face_z = float(z.min())
    hub = radius[z <= hub_face_z + 0.25]
    hub_diameter = float(hub.max()) * 2

    # The stator face is the next distinct plane outboard of the hub radius.
    wider = verts[radius > hub_diameter / 2 + 0.2]
    if len(wider) == 0:
        return {}
    stator_face_z = float(wider[:, 2].min())
    ring = wider[wider[:, 2] <= stator_face_z + 0.25]
    ring_radius = np.hypot(ring[:, 0], ring[:, 1])

    return {
        "hub_diameter_mm": round(hub_diameter, 2),
        "hub_face_z_mm": round(hub_face_z, 2),
        "hub_protrusion_mm": round(stator_face_z - hub_face_z, 2),
        "stator_inner_diameter_mm": round(float(ring_radius.min()) * 2, 2),
        "stator_outer_diameter_mm": round(float(ring_radius.max()) * 2, 2),
        "stator_face_z_mm": round(stator_face_z, 2),
    }


def measure(path: Path) -> dict:
    shape = import_step(str(path))
    bbox = shape.bounding_box()
    patterns = []

    for pat in recognise_hole_patterns(recognise_holes(shape)):
        # quiddity carries the pattern kind in the class, not in to_dict().
        if type(pat).__name__ != "BoltCircle":
            continue
        data = pat.to_dict()
        holes = data["holes"]
        first = holes[0]
        angles = circle_angles(holes)
        patterns.append(
            {
                "bcd": round(data["diameter"], 4),
                "count": len(holes),
                "hole_diameter": first["diameter"],
                "depth": first["depth"],
                "bottom": first["bottom"],
                "counterbore": first.get("cbore"),
                "plane_z": round(data["center"][2], 4),
                "axis_z": first["axis"][2],
                "angles_deg": angles,
                "pitch_deg": even_spacing(angles),
            }
        )

    patterns.sort(key=lambda p: p["bcd"])
    return {
        "mounting_face": mounting_face(shape),
        "source_file": path.name,
        "volume_mm3": round(shape.volume, 3),
        "bbox_mm": {
            "x": round(bbox.size.X, 3),
            "y": round(bbox.size.Y, 3),
            "z": round(bbox.size.Z, 3),
        },
        "bolt_circles": patterns,
    }


def main() -> None:
    result = {
        "_note": (
            "Measured from RobStride vendor STEP files; see "
            "reference/PROVENANCE.md for pinned commits. Regenerate with "
            "scripts/measure_actuators.py. Coordinates are the STEP's own "
            "frame, millimetres."
        ),
        "actuators": {},
    }
    for name, filename in ACTUATORS.items():
        path = STEP_DIR / filename
        if not path.exists():
            raise SystemExit(
                f"{path} missing; run: uv run python scripts/fetch_reference.py"
            )
        print(f"measuring {name} from {filename}")
        result["actuators"][name] = measure(path)
        for circle in result["actuators"][name]["bolt_circles"]:
            print(
                f"  BCD {circle['bcd']:>7.2f}  x{circle['count']}  "
                f"dia {circle['hole_diameter']:>4} "
                f"z={circle['plane_z']:>6}  pitch={circle['pitch_deg']}"
            )

    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
