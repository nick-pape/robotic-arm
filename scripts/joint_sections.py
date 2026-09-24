"""Render a cut-away through each joint, showing how the parts actually mount.

    uv run --extra viz python scripts/joint_sections.py

The assembled-arm renders cannot answer "where do these two parts meet" --
the interface is inside the shell. That gap hid a real fault for several
revisions: the printed bosses were seating on the actuators' stator faces
rather than their rotating output hubs, and nothing in the suite or the
renders could show it.

Each image pairs one printed part with the **real vendor actuator** it bolts
to, positioned on its hub face, and cuts the pair in half.

Sectioning happens in CAD, because MuJoCo cannot clip. The cut solids are then
exported and rendered through MuJoCo, which is the only renderer this project
has -- the pinned OCP build is the no-VTK variant.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from build123d import Box, Compound, Pos, export_stl
from PIL import Image

from robotic_arm.assembly import seated_actuator
from robotic_arm.linkframes import link_frame
from robotic_arm.parts import REGISTRY

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "renders"

#: Half-space big enough to cut any part here, and a window tight enough that
#: the joint fills the frame rather than being lost in a 320 mm link.
CUT_SIZE = 600.0
WINDOW = 150.0

SCENE = """<mujoco model="joint section">
  <visual>
    <headlight diffuse="0.75 0.75 0.75" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <global offwidth="1600" offheight="1200"/>
  </visual>
  <asset>
    <mesh name="printed" file="{printed}" scale="0.001 0.001 0.001"/>
    <mesh name="actuator" file="{actuator}" scale="0.001 0.001 0.001"/>
  </asset>
  <worldbody>
    <light pos="0.3 -0.3 0.5" dir="-0.5 0.5 -1" directional="true"/>
    <geom type="mesh" mesh="printed" rgba="0.82 0.83 0.85 1"/>
    <geom type="mesh" mesh="actuator" rgba="0.18 0.18 0.2 1"/>
  </worldbody>
</mujoco>
"""


def section(body: str, out_dir: Path) -> Path | None:
    """Cut one joint in half and render it. Returns the image path."""
    actuator = seated_actuator(body)
    if actuator is None:
        return None

    frame = link_frame(body)
    printed = REGISTRY[body][0]()

    # Keep the joint end of the link only; the rest is a long tube that would
    # shrink the interface to nothing.
    axis = np.asarray(frame.parent_axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    reach = float(np.asarray(frame.stock_centre, dtype=float) @ axis)
    toward_body = axis * (1.0 if reach >= 0 else -1.0)
    window = Pos(*(toward_body * WINDOW / 3)) * Box(WINDOW, WINDOW, WINDOW)

    # Cut away everything on one side of the plane through the joint axis.
    half = Pos(0, -CUT_SIZE / 2, 0) * Box(CUT_SIZE, CUT_SIZE, CUT_SIZE)

    meshes = out_dir / "section_meshes"
    meshes.mkdir(parents=True, exist_ok=True)
    paths = {}
    for label, solid in (("printed", printed & window), ("actuator", actuator & window)):
        cut = solid - half
        if cut is None or not cut.solids():
            return None
        path = meshes / f"{body}_{label}.stl"
        export_stl(cut, str(path), tolerance=0.05, angular_tolerance=0.2)
        paths[label] = path.name

    xml = meshes / f"{body}_section.xml"
    xml.write_text(SCENE.format(**paths))

    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = 0.20
    camera.azimuth, camera.elevation = 90.0, 0.0
    camera.lookat[:] = np.zeros(3)

    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    with mujoco.Renderer(model, height=900, width=1200) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=options)
        image = out_dir / f"section_{body}_joint.png"
        Image.fromarray(renderer.render()).save(image)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    written = []
    for body in sorted(REGISTRY):
        path = section(body, args.out)
        if path is None:
            print(f"  {body}: no actuator interface to section")
            continue
        written.append(path)
        print(f"  {path.relative_to(REPO)}")
    print(f"\n{len(written)} joint sections in {args.out.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
