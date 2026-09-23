"""Export every part to STEP and STL under exports/.

    uv run python -m robotic_arm.export
"""

from pathlib import Path

from build123d import export_step, export_stl

from robotic_arm.parts import build_base_plate

EXPORT_DIR = Path(__file__).resolve().parents[2] / "exports"

PARTS = {
    "base_plate": build_base_plate,
}


def main() -> None:
    EXPORT_DIR.mkdir(exist_ok=True)
    for name, build in PARTS.items():
        part = build()
        export_step(part, str(EXPORT_DIR / f"{name}.step"))
        export_stl(part, str(EXPORT_DIR / f"{name}.stl"))
        print(f"{name}: {part.volume:.1f} mm^3 -> {name}.step, {name}.stl")


if __name__ == "__main__":
    main()
