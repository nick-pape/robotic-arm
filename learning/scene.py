"""The twin, set up for a task: a floor, a cube to pick, and cameras to see it.

Built on top of the generated twin with MjSpec, never by editing its XML: the
twin is regenerated whenever the CAD changes, and anything hand-added to it
would be lost or, worse, silently stale. Nothing here moves a joint frame, so
the arm in the scene is exactly the arm the design pipeline validated.

Three things are added beyond scenery, and each is a decision worth knowing:

* **Finger pads.** The fingers' collision geometry is a convex decomposition
  of stock meshes, and grasping against a pile of hulls is jittery. Each
  finger gets a box on its gripping face, and the object contacts *only*
  those boxes -- see `OBJECT_BIT`.
* **A grip force cap.** The stock gripper servo is good for ~1.9 kN, which is
  the RS00's peak torque through a 16T pinion. Nothing real grips a cube that
  hard; the real gripper runs force-limited, so the sim does too.
* **Cameras placed like the real ones.** A D405-class wrist camera behind the
  jaws and a D435-class scene camera across the table, with matching vertical
  fields of view, so images look like what the hardware will produce.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from robotic_arm.mjcf import GRIPPER_ACTUATOR, SIM_DIR

#: The twin this package builds on. Deliberately its own file: `sim/model.xml`
#: is written by both `mjcf.generate()` (stock + inertials) and
#: `mjcf.generate_twin()`, so what it holds depends on which ran last.
TWIN_PATH = SIM_DIR / "twin.xml"

#: Policy rate. LeRobot records the real B601 at 30 Hz, so the sim matches it.
CONTROL_HZ = 30

#: Physics step, chosen so a whole number of steps makes one control period.
SUBSTEPS = 20
TIMESTEP = 1.0 / (CONTROL_HZ * SUBSTEPS)

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
FINGER_JOINTS = ("joint_left", "joint_right")
FINGER_BODIES = ("gripper_left", "gripper_right")

#: The tool frame. In it, +x points out of the jaws (fingertips at x = 0),
#: y is the axis the fingers open along, and z is the fingers' thickness.
TOOL_BODY = "gripper_end"

#: Where an object sits when it is held: between the pads, 18 mm back from
#: the fingertips. Scripted IK and the success check both aim at this point.
GRASP_SITE = "grasp"
GRASP_POS = np.array([-0.018, 0.0, 0.0])

#: Gripping-face pads, in the tool frame with the fingers closed. Measured
#: off the finger meshes: each fingertip's inner face sits at |y| = 0.001 when
#: closed and opens 1:1 with its slide joint, over the last 40 mm of finger.
PAD_HALF = np.array([0.017, 0.004, 0.012])
PAD_CENTRE = np.array([-0.019, 0.003, 0.0])  # y sign flips per finger

#: Grip force, N -- the squeeze on whatever is between the jaws. The servo's
#: own gain (5000 N/m) saturates this within 8 mm of closing on an object, so
#: it behaves as a force-limited grip.
GRIP_FORCE = 40.0

CUBE = "cube"
CUBE_HALF = 0.02
CUBE_MASS = 0.05
CUBE_RGBA = (0.85, 0.2, 0.15, 1.0)

#: Contact softness for the cube and the pads. MuJoCo's default lets a 40 N
#: grip sink each pad ~5 mm into a 50 g cube, so the jaws read a 30 mm gap
#: on a 40 mm object -- an artifact a policy can learn to rely on. This takes
#: it to ~0.2 mm. The time constant must stay above two physics steps.
CONTACT_SOLREF = [0.004, 1.0]
CONTACT_SOLIMP = [0.95, 0.99, 0.001, 0.5, 2.0]

#: Contact bits. Bit 1 is everything the twin already has (arm and floor).
#: Objects use bit 2, and touch the floor, the pads and the arm's links --
#: but not the finger meshes, which only the pads stand in for.
ARM_BIT = 1
OBJECT_BIT = 2


@dataclass(frozen=True)
class Camera:
    name: str
    fovy: float  # degrees, vertical -- the only field of view MuJoCo takes
    width: int
    height: int


#: The scene camera frames the whole spawn ring, so at 320x240 a 4 cm cube
#: is ~5 px across -- too few to place a grasp from. 640x480 doubles that,
#: and costs little: render time here is set by the arm's ~850k triangles,
#: not by pixels. The wrist camera sees the cube up close and stays small.
SCENE_CAMERA = Camera("scene", fovy=42.0, width=640, height=480)  # D435 colour
WRIST_CAMERA = Camera("wrist", fovy=58.0, width=320, height=240)  # D405
CAMERAS = (SCENE_CAMERA, WRIST_CAMERA)

#: Where the scene camera looks from: bearing off the arm's home heading,
#: and angle above the floor. Its distance is solved to fit everything in.
SCENE_AZIMUTH = np.radians(25.0)
SCENE_ELEVATION = np.radians(50.0)
SCENE_MARGIN = 0.04

#: Cube spawn region: an arc of ring around the base, measured rather than
#: chosen. A sweep of the scripted pick (10 degree, 5 cm grid) succeeds at
#: every radius from 0.20 to 0.60 m for bearings -170 to +70 degrees off the
#: arm's home heading. Past +80 it starts failing, and from about +120 on
#: there is no collision-free top-down grasp at all -- the dead zone is to
#: one side, not straight behind. `learning/tests/test_env.py` re-checks
#: the region's edges, so a design change that shrinks it is caught.
SPAWN_RADIUS = (0.20, 0.60)
SPAWN_BEARING = (np.radians(-170), np.radians(70))


def ensure_twin(path: Path = TWIN_PATH, regenerate: bool = False) -> Path:
    """The generated twin, building it first if it is missing.

    Generation runs the whole CAD pipeline (about a minute and a half), so it
    happens once and is reused; pass `regenerate` after changing the design.
    """
    if regenerate or not path.exists():
        from robotic_arm.mjcf import generate_twin

        # The UR palette, as in `scripts/view.py --colour`: it is how the arm
        # is meant to look, and a vision policy should learn the real arm's
        # colours, not a uniform placeholder.
        generate_twin(out=path, per_link_colour=True)
    return path


def _look_at(pos, target, up) -> np.ndarray:
    """Quaternion for a camera at `pos` looking at `target`, `up` roughly up.

    MuJoCo cameras look down their own -z with +y up in the image.
    """
    pos, target, up = (np.asarray(v, dtype=float) for v in (pos, target, up))
    back = pos - target
    z = back / np.linalg.norm(back)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, y, z]).ravel())
    return quat


def home_heading(model: mujoco.MjModel) -> np.ndarray:
    """Unit vector, in the floor plane, that the tool points along at home.

    The workspace and the scene camera are laid out along it, so they follow
    the arm if `homepose.BASE_YAW` ever changes.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    tool = model.body(TOOL_BODY).id
    base = model.body("link1").id
    flat = (data.xpos[tool] - data.xpos[base])[:2]
    return np.array([*flat / np.linalg.norm(flat), 0.0])


def _add_pads(spec: mujoco.MjSpec, probe: mujoco.MjModel) -> None:
    """Box pads on the fingers' gripping faces, placed via the tool frame."""
    data = mujoco.MjData(probe)
    data.qpos[:] = 0.0  # home, fingers closed: where PAD_CENTRE was measured
    mujoco.mj_forward(probe, data)
    tool = probe.body(TOOL_BODY).id
    r_tool = data.xmat[tool].reshape(3, 3)
    p_tool = data.xpos[tool]

    for body_name, side in zip(FINGER_BODIES, (-1.0, 1.0)):
        finger = probe.body(body_name).id
        r_finger = data.xmat[finger].reshape(3, 3)
        centre = PAD_CENTRE * np.array([1.0, side, 1.0])
        world = p_tool + r_tool @ centre
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (r_finger.T @ r_tool).ravel())
        spec.body(body_name).add_geom(
            name=f"{body_name}_pad",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=PAD_HALF,
            pos=r_finger.T @ (world - data.xpos[finger]),
            quat=quat,
            friction=[1.5, 0.005, 0.0001],
            condim=4,
            solref=CONTACT_SOLREF,
            solimp=CONTACT_SOLIMP,
            contype=OBJECT_BIT,
            conaffinity=OBJECT_BIT,
            group=3,
            rgba=[0.1, 0.1, 0.1, 1.0],
        )


def _let_objects_touch_the_links(spec: mujoco.MjSpec) -> None:
    """Arm collision geoms accept object contacts, except on the fingers.

    Selected by body, not by comparing geom objects: MjSpec hands out a new
    Python wrapper on every access, so identity tests never match.
    """
    for body in spec.bodies:
        if body.name in FINGER_BODIES:
            continue
        for geom in body.geoms:
            if geom.contype & ARM_BIT:
                geom.conaffinity |= OBJECT_BIT


def _compensate_gravity(spec: mujoco.MjSpec) -> None:
    """Hold the arm's own weight with feedforward torque, as the real arm will.

    Without it the position servos sag: J2 droops ~1 degree at kp = 900, which
    puts the jaws 10+ mm off a grasp the IK solved exactly. The real arm adds
    the same term through the MIT command's torque feedforward. Routing it
    through the actuators (`actgravcomp`) keeps it inside each motor's
    force limit, so a pose the motors cannot hold still sags. Only the arm's
    own weight is compensated -- anything it carries is the servos' problem.
    """
    for body in spec.bodies:
        if body.name not in ("world", "base_link", CUBE):
            body.gravcomp = 1.0
    for name in ARM_JOINTS:
        spec.joint(name).actgravcomp = True


def _add_scenery(spec: mujoco.MjSpec) -> None:
    # Sky, floor and lighting match `mjcf.generate_scene`, so this looks like
    # every other view of the twin.
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960

    # A sky, so a camera pointed off the table sees something other than the
    # clear colour -- pure black is not a background any real camera returns.
    spec.add_texture(
        name="sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.3, 0.5, 0.7],
        rgb2=[0.0, 0.0, 0.0],
        width=256,
        height=1536,
    )
    spec.add_texture(
        name="floor_grid",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=[0.2, 0.3, 0.4],
        rgb2=[0.1, 0.2, 0.3],
        markrgb=[0.8, 0.8, 0.8],
        width=300,
        height=300,
    )
    # Not reflective, unlike the standard scene: the wrist camera sees the
    # gripper mirrored in a shiny floor, which no real table will show it.
    floor_mat = spec.add_material(name="floor", texrepeat=[5, 5], texuniform=True)
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_grid"

    world = spec.worldbody
    world.add_light(pos=[0.3, 0.0, 2.0], dir=[0.0, 0.0, -1.0], castshadow=True)
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.05],
        material="floor",
        contype=ARM_BIT,
        conaffinity=ARM_BIT | OBJECT_BIT,
    )


def _add_cube(spec: mujoco.MjSpec, rest: np.ndarray) -> None:
    body = spec.worldbody.add_body(name=CUBE, pos=rest)
    body.add_freejoint(name=CUBE)
    body.add_geom(
        name=CUBE,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[CUBE_HALF] * 3,
        mass=CUBE_MASS,
        rgba=CUBE_RGBA,
        friction=[1.0, 0.005, 0.0001],
        condim=4,
        solref=CONTACT_SOLREF,
        solimp=CONTACT_SOLIMP,
        contype=OBJECT_BIT,
        conaffinity=OBJECT_BIT,
    )


def _arm_extent(probe: mujoco.MjModel) -> np.ndarray:
    """Points bounding every visible part of the arm at home, world frame."""
    data = mujoco.MjData(probe)
    mujoco.mj_forward(probe, data)  # qpos = 0 is home
    points = []
    for g in range(probe.ngeom):
        if probe.geom_group[g] != 2 or probe.geom_bodyid[g] == 0:
            continue
        centre, radius = data.geom_xpos[g], probe.geom_rbound[g]
        points.extend(centre + radius * axis for axis in np.vstack([np.eye(3), -np.eye(3)]))
    return np.array(points)


def _workspace_extent(heading: np.ndarray) -> np.ndarray:
    """Points bounding everywhere a cube can spawn, world frame."""
    base = np.arctan2(heading[1], heading[0])
    reach = SPAWN_RADIUS[1] + CUBE_HALF * np.sqrt(2)
    points = []
    for bearing in np.linspace(*SPAWN_BEARING, 48):
        for z in (0.0, 2 * CUBE_HALF):
            points.append([reach * np.cos(base + bearing), reach * np.sin(base + bearing), z])
    return np.array(points)


def _frame(points: np.ndarray, camera: Camera, direction: np.ndarray, margin: float):
    """Camera position and orientation that fit `points` in frame.

    Looks along -`direction` at the centre of the points' bounding box, and
    backs off until every point sits inside the image with `margin` to
    spare (as a fraction of the half-width). Solved, not hand-placed, so the
    framing follows the arm and the workspace when either changes.
    """
    up = np.array([0.0, 0.0, 1.0])
    target = (points.min(axis=0) + points.max(axis=0)) / 2
    tan_v = np.tan(np.radians(camera.fovy) / 2) * (1 - margin)
    tan_h = tan_v * camera.width / camera.height

    def fits(distance):
        pos = target + distance * direction
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, _look_at(pos, target, up))
        local = (points - pos) @ rot.reshape(3, 3)  # camera frame: x right, y up, -z ahead
        depth = -local[:, 2]
        return bool(
            np.all(depth > 0)
            and np.all(np.abs(local[:, 0]) <= tan_h * depth)
            and np.all(np.abs(local[:, 1]) <= tan_v * depth)
        )

    near, far = 0.1, 20.0
    for _ in range(50):
        mid = (near + far) / 2
        near, far = (near, mid) if fits(mid) else (mid, far)
    pos = target + far * direction
    return pos, _look_at(pos, target, up)


def _add_cameras(spec: mujoco.MjSpec, probe: mujoco.MjModel, heading: np.ndarray) -> None:
    # In front of the arm and a little to one side, looking down: frames the
    # arm standing at home and the whole spawn ring, solved by `_frame`.
    azimuth = np.arctan2(heading[1], heading[0]) + SCENE_AZIMUTH
    elevation = SCENE_ELEVATION
    direction = np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ])
    points = np.vstack([_arm_extent(probe), _workspace_extent(heading)])
    pos, quat = _frame(points, SCENE_CAMERA, direction, SCENE_MARGIN)
    spec.worldbody.add_camera(
        name=SCENE_CAMERA.name, pos=pos, quat=quat, fovy=SCENE_CAMERA.fovy
    )
    # Behind the jaws on the tool's +z side, aimed just past the fingertips:
    # the view a wrist-mounted D405 has of a grasp.
    wrist_pos = np.array([-0.10, 0.0, 0.065])
    spec.body(TOOL_BODY).add_camera(
        name=WRIST_CAMERA.name,
        pos=wrist_pos,
        quat=_look_at(wrist_pos, [0.02, 0.0, 0.0], [0.0, 0.0, 1.0]),
        fovy=WRIST_CAMERA.fovy,
    )


def build_task_spec(twin_path: Path | None = None) -> mujoco.MjSpec:
    """The pick-cube scene, as a spec a caller may extend before compiling."""
    twin_path = ensure_twin() if twin_path is None else twin_path
    spec = mujoco.MjSpec.from_file(str(twin_path))
    probe = spec.compile()
    heading = home_heading(probe)

    spec.option.timestep = TIMESTEP
    _add_scenery(spec)
    _add_pads(spec, probe)
    _let_objects_touch_the_links(spec)
    _compensate_gravity(spec)

    rest = 0.38 * heading + np.array([0.0, 0.0, CUBE_HALF])
    _add_cube(spec, rest)
    _add_cameras(spec, probe, heading)
    # Group 4: kept out of every camera, which renders groups 0-2.
    spec.body(TOOL_BODY).add_site(name=GRASP_SITE, pos=GRASP_POS, size=[0.004] * 3, group=4)

    spec.actuator(GRIPPER_ACTUATOR).forcerange = [-GRIP_FORCE, GRIP_FORCE]

    # The twin's keyframes cover the arm alone; the cube's free joint adds
    # seven qpos entries, which every key must now carry.
    cube_qpos = [*rest, 1.0, 0.0, 0.0, 0.0]
    for key in spec.keys:
        key.qpos = [*key.qpos, *cube_qpos]
    return spec


def load_task_model(twin_path: Path | None = None) -> mujoco.MjModel:
    return build_task_spec(twin_path).compile()
