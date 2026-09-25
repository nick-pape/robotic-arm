"""Render each printed part on its own, away from the assembled arm.

    uv run --extra viz python scripts/render_parts.py
    uv run --extra viz python scripts/render_parts.py --only link3

An assembled render answers "does the arm look right"; it cannot answer "is
this part right", because every part is half-hidden by its neighbours and by
the motors. Several faults survived dozens of assembly renders for exactly
that reason -- a flange buried inside the housing it was meant to cap looks
identical, from outside, to no flange at all.

Each part is drawn alone, in its own CAD frame, from four angles.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from build123d import export_stl
from PIL import Image

from robotic_arm.mjcf import REVIEW_RGBA
from robotic_arm.parts import REGISTRY

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "renders" / "parts"

ANGLES = {
    "iso": (135.0, -20.0),
    "front": (90.0, 0.0),
    "end": (0.0, 0.0),
    "top": (90.0, -80.0),
}

SCENE = """<mujoco model="{name}">
  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.4 0.4 0.4" specular="0.15 0.15 0.15"/>
    <global offwidth="1200" offheight="900"/>
  </visual>
  <asset><mesh name="part" file="{file}" scale="0.001 0.001 0.001"/></asset>
  <worldbody>
    <light pos="0.4 -0.4 0.6" dir="-0.5 0.5 -1" directional="true"/>
    <geom type="mesh" mesh="part" rgba="{rgba}"/>
  </worldbody>
</mujoco>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meshes = args.out / "meshes"
    meshes.mkdir(exist_ok=True)

    wanted = [n.strip() for n in args.only.split(",") if n.strip()] or sorted(REGISTRY)
    written = []
    for name in wanted:
        part = REGISTRY[name][0]()
        stl = meshes / f"{name}.stl"
        export_stl(part, str(stl), tolerance=0.03, angular_tolerance=0.15)

        rgba = " ".join(f"{v:.2f}" for v in REVIEW_RGBA.get(name, (0.8, 0.8, 0.82, 1.0)))
        xml = meshes / f"{name}.xml"
        xml.write_text(SCENE.format(name=name, file=stl.name, rgba=rgba))

        model = mujoco.MjModel.from_xml_path(str(xml))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        box = part.bounding_box()
        centre = np.array(tuple(box.center())) / 1000.0
        span = max(box.size.X, box.size.Y, box.size.Z) / 1000.0

        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = centre
        camera.distance = span * 1.5
        options = mujoco.MjvOption()
        mujoco.mjv_defaultOption(options)
        with mujoco.Renderer(model, height=750, width=1000) as renderer:
            for tag, (azimuth, elevation) in ANGLES.items():
                camera.azimuth, camera.elevation = azimuth, elevation
                renderer.update_scene(data, camera=camera, scene_option=options)
                path = args.out / f"{name}_{tag}.png"
                Image.fromarray(renderer.render()).save(path)
                written.append(path)
        print(f"  {name}: {box.size.X:.0f} x {box.size.Y:.0f} x {box.size.Z:.0f} mm")
    print(f"\n{len(written)} part renders in {args.out.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
