"""The task scene adds a task to the twin without changing the arm."""

from __future__ import annotations

import mujoco
import numpy as np

from learning.scene import (
    CAMERAS,
    CUBE,
    FINGER_BODIES,
    GRIPPER_ACTUATOR,
    GRIP_FORCE,
    load_task_model,
)


def _bodies(model):
    return {model.body(i).name: i for i in range(model.nbody)}


def test_arm_kinematics_untouched(twin_path):
    twin = mujoco.MjModel.from_xml_path(str(twin_path))
    task = load_task_model(twin_path)
    for name, i in _bodies(twin).items():
        j = task.body(name).id
        np.testing.assert_allclose(task.body_pos[j], twin.body_pos[i], atol=1e-12, err_msg=name)
        np.testing.assert_allclose(task.body_quat[j], twin.body_quat[i], atol=1e-12, err_msg=name)
    for i in range(twin.njnt):
        name = twin.joint(i).name
        np.testing.assert_allclose(task.joint(name).range, twin.jnt_range[i], err_msg=name)


def _can_collide(model, a, b):
    return bool(
        model.geom_contype[a] & model.geom_conaffinity[b]
        or model.geom_contype[b] & model.geom_conaffinity[a]
    )


def test_cube_touches_pads_but_not_finger_meshes(twin_path):
    """Grasps happen on the pads; the hull pile behind them stays out of it."""
    model = load_task_model(twin_path)
    cube = model.geom(CUBE).id
    for body in FINGER_BODIES:
        bid = model.body(body).id
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != bid or model.geom_group[g] != 3:
                continue
            is_pad = model.geom(g).name.endswith("_pad")
            assert _can_collide(model, cube, g) == is_pad, model.geom(g).name


def test_cube_still_collides_with_the_links(twin_path):
    """Objects can be knocked over by the arm, not passed through."""
    model = load_task_model(twin_path)
    cube = model.geom(CUBE).id
    link3 = model.body("link3").id
    link_geoms = [
        g for g in range(model.ngeom)
        if model.geom_bodyid[g] == link3 and model.geom_group[g] == 3
    ]
    assert link_geoms and all(_can_collide(model, cube, g) for g in link_geoms)


def test_cameras_match_the_hardware(twin_path):
    model = load_task_model(twin_path)
    for camera in CAMERAS:
        assert model.cam_fovy[model.camera(camera.name).id] == camera.fovy


def test_grip_is_force_limited(twin_path):
    model = load_task_model(twin_path)
    np.testing.assert_allclose(
        model.actuator_forcerange[model.actuator(GRIPPER_ACTUATOR).id],
        [-GRIP_FORCE, GRIP_FORCE],
    )


def test_scene_camera_frames_the_arm_and_the_whole_ring(twin_path):
    """Everything the policy must see is inside the scene camera's image."""
    from learning.scene import (
        SCENE_CAMERA,
        _arm_extent,
        _workspace_extent,
        home_heading,
    )

    model = load_task_model(twin_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam = model.camera(SCENE_CAMERA.name).id
    pos, rot = data.cam_xpos[cam], data.cam_xmat[cam].reshape(3, 3)
    points = np.vstack([_arm_extent(model), _workspace_extent(home_heading(model))])
    local = (points - pos) @ rot
    depth = -local[:, 2]
    tan_v = np.tan(np.radians(SCENE_CAMERA.fovy) / 2)
    tan_h = tan_v * SCENE_CAMERA.width / SCENE_CAMERA.height
    assert np.all(depth > 0)
    assert np.all(np.abs(local[:, 0]) <= tan_h * depth)
    assert np.all(np.abs(local[:, 1]) <= tan_v * depth)
