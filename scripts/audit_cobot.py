"""Reproduce mechanical review measurements without changing the design.

Run with the project's Python environment. Writes JSON measurements to
sim/cobot_audit.json; these are diagnostics, not a build acceptance test.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
from build123d import Location, Plane
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.TopAbs import TopAbs_IN
from OCP.gp import gp_Pnt

from robotic_arm.design import RULES
from robotic_arm.linkframes import actuator_centre, link_frame
from robotic_arm.massprops import mass_properties
from robotic_arm.mjcf import cad_inertials
from robotic_arm.parts import REGISTRY, effective_material
from robotic_arm.parts.cobot import Drum, lofted_tube, shelled_body
from robotic_arm.reference import load_baseline


def intersection_volume(a, b):
    intersection = a.intersect(b)
    if intersection is None:
        return 0.0
    if hasattr(intersection, "volume"):
        return float(intersection.volume)
    return sum(float(piece.volume) for piece in intersection)


def uncut_shell(module):
    parent, child = module._drums()
    wall = RULES.structural_wall_thickness
    if hasattr(module, "_tube_path"):
        points, diameters = module._tube_path()
        shape = (
            parent.solid() + child.solid() + lofted_tube(points, diameters)
        ) - (
            parent.solid(parent.diameter - 2 * wall, parent.length - 2 * wall)
            + child.solid(child.diameter - 2 * wall, child.length - 2 * wall)
            + lofted_tube(points, [d - 2 * wall for d in diameters])
        )
    else:
        shape = shelled_body(
            parent, child, module.TUBE_DIAMETER,
            tube_from=6.0 if module.BODY == "link5" else 0.0,
        )
    bore_length = {"link2": 140.0, "link3": 120.0}.get(module.BODY, 200.0)
    return shape - Drum(
        parent.centre, parent.axis, module.BORE_DIAMETER, bore_length
    ).solid()


def geometry_measurements():
    report = {"parts": {}, "adjacent_intersections_mm3": {}}
    shapes = {}
    for name, (builder, material) in REGISTRY.items():
        print(f"Building and measuring {name}", flush=True)
        module = importlib.import_module(builder.__module__)
        captured = []

        if hasattr(module, "bolt_ring"):
            real_bolt_ring = module.bolt_ring

            def capture(**kwargs):
                cutter = real_bolt_ring(**kwargs)
                captured.append((kwargs, cutter))
                return cutter

            with patch.object(module, "bolt_ring", capture):
                shape = builder()
        else:
            shape = builder()
        shapes[name] = shape
        props = mass_properties(shape, effective_material(shape, material))
        record = {"shell_mass_kg": props.mass, "volume_mm3": shape.volume}
        report["parts"][name] = record
        if not captured:
            continue

        blank = uncut_shell(module)
        record["mount_cuts"] = []
        for kwargs, cutter in captured:
            record["mount_cuts"].append({
                "centre_mm": np.asarray(kwargs["centre"]).tolist(),
                "axis": np.asarray(kwargs["axis"]).tolist(),
                "bcd_mm": kwargs["bcd"], "count": kwargs["count"],
                "depth_mm": kwargs["depth"],
                "removed_mm3": intersection_volume(blank, cutter),
            })
            blank = blank - cutter
        record["bore_to_parent_hole_ligament_mm"] = (
            captured[0][0]["bcd"] / 2 - RULES.m3_clearance / 2
            - module.BORE_DIAMETER / 2
        )

        # Check BOTH parent end faces. A cut through the wrong end is not a
        # usable connection at the joint-facing end of the housing.
        parent, child = module._drums()
        axis = np.asarray(parent.axis)
        seed = np.array([1.0, 0.0, 0.0])
        u = np.cross(axis, seed)
        u /= np.linalg.norm(u)
        radius = captured[0][0]["bcd"] / 2
        record["parent_face_first_hole_blocked"] = {
            sign: bool(shape.is_inside(tuple(
                parent.centre + multiplier * axis * (parent.length / 2 - 1)
                + radius * u
            )))
            for sign, multiplier in (("minus", -1), ("plus", 1))
        }

        frame = link_frame(name)
        motor_centre = actuator_centre(name, near=frame.child_origin)
        record["motor_centre_mm"] = motor_centre.tolist()
        record["housing_centre_mm"] = child.centre.tolist()

        # Are vertices of the actual retained motor mesh inside the new shell?
        # A positive count proves overlap; zero does not prove clearance.
        model = load_baseline()
        motor_hits = []
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] != model.body(name).id or model.geom_group[gid] != 2:
                continue
            mesh_id = model.geom_dataid[gid]
            mesh_name = model.mesh(mesh_id).name
            if not mesh_name.startswith("motor"):
                continue
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, model.geom_quat[gid])
            first, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
            verts = (model.mesh_vert[first:first + count] @ rot.reshape(3, 3).T
                     + model.geom_pos[gid]) * 1000
            selected = verts[::max(1, len(verts) // 800)]
            classifier = BRepClass3d_SolidClassifier(shape.solids()[0].wrapped)
            hits = []
            for vertex in selected:
                classifier.Perform(gp_Pnt(*(float(x) for x in vertex)), 1e-6)
                if classifier.State() == TopAbs_IN:
                    hits.append(vertex.tolist())
            motor_hits.append({"mesh": mesh_name, "sampled": len(selected),
                               "inside": len(hits), "example_points_mm": hits[:3]})
        record["motor_vertices_in_shell"] = motor_hits
        print(json.dumps({name: record}), flush=True)

    model = load_baseline()
    data = mujoco.MjData(model)
    poses = {"zero": [0, 0, 0, 0, 0, 0], "working": [0, -1.0, -1.8, 0.4, 0.5, 0]}
    for label, qpos in poses.items():
        data.qpos[:] = 0
        data.qpos[:6] = qpos
        mujoco.mj_forward(model, data)
        placed = {}
        for name, shape in shapes.items():
            bid = model.body(name).id
            rotation = data.xmat[bid].reshape(3, 3)
            plane = Plane(origin=tuple(data.xpos[bid] * 1000),
                          x_dir=tuple(rotation[:, 0]), z_dir=tuple(rotation[:, 2]))
            placed[name] = shape.moved(Location(plane))
        rows = {}
        for index in range(2, 6):
            a, b = f"link{index}", f"link{index + 1}"
            rows[f"{a}/{b}"] = intersection_volume(placed[a], placed[b])
        report["adjacent_intersections_mm3"][label] = {"qpos": qpos, "pairs": rows}
        print(label, rows, flush=True)

    # Compute the actual assembled CAD masses, using exactly the production
    # inertial path (same already-built solids).
    overrides = cad_inertials(shapes)
    report["assembled_cad_mass_kg"] = {name: p.mass for name, p in overrides.items()}
    return report


def simulation_measurements():
    from robotic_arm.actuators import RS00, RS06
    from robotic_arm.balancer import DEFAULT_CANCEL_NM, Balancer, residual_torque
    from robotic_arm.mjcf import generate_twin, load
    from robotic_arm.thermal import current_apk, sustainability
    from robotic_arm.torque import j2_moment_arm, max_moment_arm, max_payload

    stock = load_baseline()
    envelope = 0.7 * max_moment_arm(stock)
    capacity = max_payload(stock, RS06().peak_nm, max_arm=envelope, samples=31)
    data = mujoco.MjData(stock)
    worst, worst_q = 0.0, None
    for q2 in np.linspace(*stock.jnt_range[1], 31):
        for q3 in np.linspace(*stock.jnt_range[2], 31):
            data.qpos[:] = 0
            data.qpos[1:3] = [q2, q3]
            mujoco.mj_forward(stock, data)
            if j2_moment_arm(data, stock) > envelope:
                continue
            translation = np.zeros((3, stock.nv))
            rotation = np.zeros_like(translation)
            mujoco.mj_jacBody(stock, data, translation, rotation, stock.body("gripper_end").id)
            # Unit downward load at the TCP; independent of the existing
            # gripper's COM, where max_payload() currently adds the load.
            torque = abs(float(data.qfrc_bias[1] + capacity * 9.81 * translation[2, 1]))
            if torque > worst:
                worst, worst_q = torque, data.qpos.tolist()

    root = Path(__file__).resolve().parents[1]
    twin = load(generate_twin(out=root / "sim" / "audit_twin.xml",
                             balancer=Balancer.sized_for(DEFAULT_CANCEL_NM), visuals=False))
    residual, qpos = residual_torque(twin, samples=41)
    result = {
        "reported_payload_kg": capacity,
        "envelope_m": envelope,
        "torque_with_that_payload_at_actual_tcp_nm": worst,
        "tcp_load_worst_qpos": worst_q,
        "cad_balanced_residual_nm": residual,
        "cad_balanced_worst_qpos": qpos.tolist(),
        "claimed_continuous_budget_nm": RS06().derated_nm(),
        "rs00_rated_torque_current_apk_in_code": current_apk(RS00(), RS00().rated_nm),
        "rs00_rated_current_apk": RS00().rated_current_apk,
        "rs00_rated_torque_verdict_in_code": sustainability(RS00(), RS00().rated_nm),
    }
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-only", action="store_true")
    args = parser.parse_args()
    report = {} if args.simulation_only else geometry_measurements()
    report["simulation"] = simulation_measurements()
    output = Path(__file__).resolve().parents[1] / "sim" / "cobot_audit.json"
    if args.simulation_only:
        output = output.with_name("cobot_audit_simulation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Measurements: {output}", flush=True)


if __name__ == "__main__":
    main()
