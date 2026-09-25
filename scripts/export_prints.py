"""Export the printed parts as meshes ready for a slicer.

    uv run python scripts/export_prints.py
    uv run python scripts/export_prints.py --format step --out somewhere/

Writes one file per printed part into `prints/`, in millimetres and in the
part's own CAD frame -- **not** the pose it occupies on the arm, which is what
the renders show. A slicer wants the part sitting on its own, and orienting it
on the plate is a decision for whoever prints it: these are structural parts
whose layer direction decides where they break, so the script does not silently
pick one.

STL is the default because every slicer reads it. STEP is offered too, since
it keeps the real surfaces rather than a tessellation, and PrusaSlicer,
Orca and Cura all import it.

Each part's material and mass is printed alongside, because that is the number
worth checking before spending a spool: the mass here assumes solid material
at the effective density, and a real print at 40% infill will differ.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from build123d import export_step, export_stl

from robotic_arm.linkframes import stock_mass
from robotic_arm.massprops import mass_properties
from robotic_arm.parts import REGISTRY, effective_material

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "prints"

#: Tessellation fineness for STL. Tighter than the render meshes: this one may
#: actually get printed, and a coarse chord on a O49 bolt boss is a real fit
#: error rather than a cosmetic one.
LINEAR_TOLERANCE = 0.01
ANGULAR_TOLERANCE = 0.1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--format", choices=("stl", "step", "both"), default="both")
    parser.add_argument("--only", default="", help="comma-separated part names")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    wanted = [n.strip() for n in args.only.split(",") if n.strip()] or sorted(REGISTRY)
    rows = []
    for name in wanted:
        if name not in REGISTRY:
            raise SystemExit(f"no printed part named {name!r}")
        builder, material = REGISTRY[name]
        part = builder()
        effective = effective_material(part, material)
        props = mass_properties(part, effective)

        written = []
        if args.format in ("stl", "both"):
            path = args.out / f"{name}.stl"
            export_stl(
                part,
                str(path),
                tolerance=LINEAR_TOLERANCE,
                angular_tolerance=ANGULAR_TOLERANCE,
            )
            written.append(path)
        if args.format in ("step", "both"):
            path = args.out / f"{name}.step"
            export_step(part, str(path))
            written.append(path)

        rows.append((name, material.name, props.mass, stock_mass(name), written))

    width = max(len(r[0]) for r in rows)
    print(f"\n{'part'.ljust(width)}  material    mass      stock     saving")
    total = stock_total = 0.0
    for name, material, mass, stock, written in rows:
        total += mass
        stock_total += stock
        print(
            f"{name.ljust(width)}  {material:10s}  {mass * 1000:6.1f} g  "
            f"{stock * 1000:6.1f} g  {(stock - mass) * 1000:+7.1f} g"
        )
        for path in written:
            print(f"{' ' * width}    {path.relative_to(REPO)}")
    print(
        f"{'total'.ljust(width)}              {total * 1000:6.1f} g  "
        f"{stock_total * 1000:6.1f} g  {(stock_total - total) * 1000:+7.1f} g"
    )
    print(
        "\nOrientation is each part's CAD frame, not its pose on the arm. "
        "Layer direction is a strength decision -- orient on the plate "
        "deliberately."
    )


if __name__ == "__main__":
    main()
