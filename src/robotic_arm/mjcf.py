"""Generate the digital twin's MJCF from the stock model plus CAD inertials.

This is where the project's governing rule is enforced mechanically rather than
by discipline. The generator starts from the vendored stock model and changes
*only* inertial properties. Joint frames, geometry, actuators, contacts and
keyframes all come through untouched, because they are never written here --
there is no code path that could modify them.

That is spec requirement F2 ("only <inertial> blocks differ from stock"), and
`frame_differences()` proves it after the fact rather than assuming it.

One honest caveat: generation round-trips through XML text at 9 significant
figures, so an override-free build is a no-op in intent but not bit-exact.
Frames survive exactly (they are short decimals), but inertia values shift by
~1e-6 relative. That is five orders of magnitude below the uncertainty in
printed-part effective density, so it does not matter -- but `ROUNDTRIP_TOL`
names it rather than leaving it as a mystery in a failing test.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import mujoco
import numpy as np

from robotic_arm.balancer import Balancer, add_to_spec
from robotic_arm.massprops import MassProperties
from robotic_arm.reference import BASELINE_MJCF, REFERENCE_DIR, require

REPO = Path(__file__).resolve().parents[2]
SIM_DIR = REPO / "sim"
SIM_MODEL = SIM_DIR / "model.xml"

#: Mesh assets stay in reference/; the generated model points back at them
#: rather than duplicating ~15 MB per build.
ASSET_DIR = REFERENCE_DIR / "mjcf" / "assets"

#: Relative tolerance for "unchanged" inertials across an XML round-trip.
#: MJCF is written at 9 significant figures; see the module docstring.
ROUNDTRIP_TOL = 1e-5


class UnknownBodyError(KeyError):
    """Raised when an override names a body the stock model does not have."""


def _body_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)
    ]


def apply_inertials(
    spec: mujoco.MjSpec, overrides: Mapping[str, MassProperties]
) -> mujoco.MjSpec:
    """Replace named bodies' inertial properties in place.

    Sets `fullinertia` and deliberately leaves `iquat` alone: MuJoCo derives
    the inertial frame from `fullinertia` by eigendecomposition, and setting
    both invites a silent inconsistency between them.
    """
    available = {b.name for b in spec.bodies}
    for name, mp in overrides.items():
        if name not in available:
            raise UnknownBodyError(
                f"{name!r} is not a body in the stock model. "
                f"Bodies are: {sorted(n for n in available if n)}"
            )
        if not mp.satisfies_triangle_inequality():
            raise ValueError(
                f"{name!r} inertia violates the triangle inequality "
                f"(principal moments {mp.principal_moments}). That is a units "
                f"bug, not a physics result -- check the density and the mm/m "
                f"conversion. Do not paper over it with balanceinertia."
            )
        body = spec.body(name)
        body.mass = float(mp.mass)
        body.ipos = np.asarray(mp.com, dtype=float)
        body.fullinertia = np.asarray(mp.fullinertia, dtype=float)
        body.explicitinertial = True
    return spec


def generate(
    overrides: Mapping[str, MassProperties] | None = None,
    out: Path = SIM_MODEL,
    balancer: Balancer | None = None,
) -> Path:
    """Write the twin's MJCF, with CAD inertials swapped in for `overrides`.

    With no overrides and no balancer this reproduces the stock model, which is
    the useful degenerate case: it proves the generator is a no-op before any
    CAD exists.

    `balancer` adds the J2 gravity spring. That is an addition beyond stock
    rather than an inertial swap, so it is explicit at the call site.
    """
    require(BASELINE_MJCF)
    spec = mujoco.MjSpec.from_file(str(BASELINE_MJCF))
    apply_inertials(spec, overrides or {})
    if balancer is not None:
        add_to_spec(spec, balancer)

    # Point at the shared asset directory rather than duplicating ~15 MB per
    # build. This is an absolute path: meshdir resolves against the *source*
    # model's directory during compile(), not the output's, so a relative path
    # would break as soon as the two differ. sim/ is generated output,
    # regenerated per machine, so a machine-specific path there is fine.
    out.parent.mkdir(parents=True, exist_ok=True)
    spec.meshdir = str(require(ASSET_DIR)).replace("\\", "/")

    spec.compile()  # raises on anything structurally invalid
    out.write_text(spec.to_xml())
    return out


def load(path: Path = SIM_MODEL) -> mujoco.MjModel:
    """Compile a generated model."""
    return mujoco.MjModel.from_xml_path(str(require(path)))


def frame_differences(
    generated: mujoco.MjModel, reference: mujoco.MjModel, tol: float = 1e-12
) -> list[str]:
    """Everything that differs between two models *other than* inertials.

    Returns human-readable descriptions, empty when requirement F2 holds. This
    is the check worth running after every change: if anything here is
    non-empty, a frame moved and the kinematics no longer match stock.
    """
    problems: list[str] = []

    def compare(label: str, a: np.ndarray, b: np.ndarray) -> None:
        if a.shape != b.shape:
            problems.append(f"{label}: shape {a.shape} vs {b.shape}")
        elif not np.allclose(a, b, atol=tol, rtol=0):
            worst = float(np.abs(np.asarray(a) - np.asarray(b)).max())
            problems.append(f"{label}: differs by up to {worst:.3e}")

    if generated.nbody != reference.nbody:
        problems.append(f"body count {generated.nbody} vs {reference.nbody}")
        return problems
    if _body_names(generated) != _body_names(reference):
        problems.append("body names or ordering changed")

    compare("body_pos", generated.body_pos, reference.body_pos)
    compare("body_quat", generated.body_quat, reference.body_quat)
    compare("jnt_pos", generated.jnt_pos, reference.jnt_pos)
    compare("jnt_axis", generated.jnt_axis, reference.jnt_axis)
    compare("jnt_range", generated.jnt_range, reference.jnt_range)
    compare("jnt_type", generated.jnt_type.astype(float), reference.jnt_type.astype(float))
    compare("geom_pos", generated.geom_pos, reference.geom_pos)
    compare("geom_quat", generated.geom_quat, reference.geom_quat)
    compare("geom_size", generated.geom_size, reference.geom_size)
    compare("actuator_ctrlrange", generated.actuator_ctrlrange, reference.actuator_ctrlrange)
    compare("actuator_forcerange", generated.actuator_forcerange, reference.actuator_forcerange)

    return problems


def inertial_differences(
    generated: mujoco.MjModel, reference: mujoco.MjModel, tol: float = 1e-12
) -> dict[str, dict[str, float]]:
    """Per-body inertial deltas, for reporting what a CAD swap actually changed."""
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(_body_names(generated)):
        dm = float(generated.body_mass[i] - reference.body_mass[i])
        dc = float(np.abs(generated.body_ipos[i] - reference.body_ipos[i]).max())
        di = float(np.abs(generated.body_inertia[i] - reference.body_inertia[i]).max())
        if abs(dm) > tol or dc > tol or di > tol:
            out[name] = {"mass": dm, "com": dc, "inertia": di}
    return out


if __name__ == "__main__":
    path = generate()
    model = load(path)
    print(f"wrote {path.relative_to(REPO)}  ({path.stat().st_size:,} B)")
    print(f"  {model.nbody} bodies, {model.njnt} joints, {model.ngeom} geoms")


#: A viewable scene needs lighting, a floor and a skybox, which the bare model
#: deliberately does not carry. Mirrors Menagerie's own scene.xml so the twin
#: looks like the stock model does in the viewer.
_SCENE_TEMPLATE = """<mujoco model="robotic-arm twin scene">
  <include file="{model_file}"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="140" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
      width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
      width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
      texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""


def generate_scene(model_path: Path, out: Path | None = None) -> Path:
    """Write a viewable scene wrapping an already-generated model.

    Kept separate from `generate` so the model itself stays a faithful
    stock-plus-inertials artifact: the floor and lights are presentation, and
    adding them to the model would make the F2 diff meaningless.
    """
    out = out or model_path.with_name(model_path.stem + "_scene.xml")
    out.write_text(_SCENE_TEMPLATE.format(model_file=model_path.name))
    return out
