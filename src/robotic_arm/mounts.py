"""Which side of each joint holds the rotor, measured from the stock arm.

Every revision of this project assumed the same thing: that a link is driven
by the joint below it, so its own end bolts to that joint's **rotor** and its
child end bolts to the child joint's **stator**. That holds at J1 and J3-J5.
It is wrong at J2, and the error is not cosmetic -- it puts a turning boss
where the stock arm has a fixed ring.

The stock geometry settles it without any assumption. A link bolted to a
rotor shows a small central **hub**: material right down to the axis, carrying
the O24.02 (RS06) or O27 (RS00) circle at r = 12-13.5. A link bolted to a
stator shows a ring with a **clearance bore**, nothing inside r ~ 18-19.5,
because the other link's hub has to turn through it. The two signatures do not
overlap, so a single radius reading separates them.

Measured at the zero pose from the vendored MJCF, the answer is:

    J1  base_link stator / link1 rotor      child driven
    J2  link2     stator / link1 rotor      *parent* driven
    J3  link2     stator / link3 rotor      child driven
    J4  link3     stator / link4 rotor      child driven
    J5  link4     stator / link5 rotor      child driven

J2 is the odd one, and it is deliberate rather than arbitrary: **link2 carries
the stators of both J2 and J3**. That is why the vendored model ships a single
`motor_2_3` mesh belonging to link2 -- one part straddling two housings, with
link1 and link3 each reaching in with a rotor hub. Stock link2 therefore has
no driven boss at all; both its ends are stator rings.

Units: millimetres.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

#: Below this radius at the joint plane, a part is a rotor hub; above it, the
#: part is bored to clear one. Measured hubs run to r <= 9.8 and measured
#: bores floor at r >= 18.0, so the gap is wide and the threshold is not
#: doing fine judgement.
HUB_RADIUS_LIMIT = 14.0

#: Meshes that are actuators or cosmetic covers rather than structure.
_NON_STRUCTURAL = ("motor", "pla")


def _mesh_world(model, data, gid):
    import mujoco

    mesh_id = model.geom_dataid[gid]
    first, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    vertices = model.mesh_vert[first : first + count].astype(float)
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[gid])
    vertices = vertices @ rotation.reshape(3, 3).T + model.geom_pos[gid]
    body = model.geom_bodyid[gid]
    return (vertices @ data.xmat[body].reshape(3, 3).T + data.xpos[body]) * 1000.0


@lru_cache(maxsize=1)
def joint_roles() -> dict[str, dict[str, str]]:
    """{joint: {body: "rotor" | "stator"}} for every joint with structure.

    Derived, not declared. If the vendored model is ever updated and a motor
    is flipped, this changes with it and the test that pins the table fails.
    """
    import mujoco

    from robotic_arm.reference import load_baseline

    model = load_baseline()
    data = mujoco.MjData(model)
    data.qpos[:] = 0
    mujoco.mj_forward(model, data)

    roles: dict[str, dict[str, str]] = {}
    for index in range(model.njnt):
        name = model.joint(index).name
        anchor = data.xanchor[index] * 1000.0
        axis = data.xaxis[index] / np.linalg.norm(data.xaxis[index])
        # Only the two bodies the joint actually connects. Without this, any
        # link that merely passes within 12 mm of the joint plane is scored
        # too, and link3 was reported as a party to both J2 and J4.
        child_body = int(model.jnt_bodyid[index])
        adjacent = {child_body, int(model.body_parentid[child_body])}

        found: dict[str, str] = {}
        for gid in range(model.ngeom):
            if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            if model.geom_group[gid] != 2:
                continue
            if model.geom_bodyid[gid] not in adjacent:
                continue
            mesh_name = model.mesh(model.geom_dataid[gid]).name
            if mesh_name.startswith(_NON_STRUCTURAL):
                continue
            vertices = _mesh_world(model, data, gid)
            delta = vertices - anchor
            along = delta @ axis
            radial = np.linalg.norm(delta - np.outer(along, axis), axis=1)
            # Only the material straddling the joint plane speaks to how this
            # part attaches; the rest of the link is far down the arm.
            near = (np.abs(along) < 12.0) & (radial < 50.0)
            if near.sum() < 30:
                continue
            body = model.body(model.geom_bodyid[gid]).name
            role = "rotor" if radial[near].min() < HUB_RADIUS_LIMIT else "stator"
            # A body may contribute several meshes; a hub anywhere wins.
            if found.get(body) != "rotor":
                found[body] = role
        if found:
            roles[name] = found
    return roles


def own_joint_role(body: str) -> str:
    """Whether `body` bolts to the rotor or the stator of its *own* joint."""
    index = int(body.removeprefix("link"))
    roles = joint_roles().get(f"joint{index}", {})
    role = roles.get(body)
    if role is None:
        raise KeyError(f"no measured role for {body} at joint{index}: {roles}")
    return role


def child_joint_role(body: str) -> str:
    """Whether `body` bolts to the rotor or the stator of its *child's* joint.

    The complement of the child's own role -- a joint has exactly two sides.
    """
    from robotic_arm.linkframes import link_frame

    child = link_frame(body).child_name
    if not child or not child.startswith("link"):
        raise KeyError(f"{body} has no printed child link")
    return "stator" if own_joint_role(child) == "rotor" else "rotor"


def summary() -> str:
    lines = ["joint  rotor side        stator side"]
    for joint, roles in sorted(joint_roles().items()):
        rotor = [b for b, r in roles.items() if r == "rotor"]
        stator = [b for b, r in roles.items() if r == "stator"]
        lines.append(f"{joint:6s} {', '.join(rotor) or '-':17s} {', '.join(stator) or '-'}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
