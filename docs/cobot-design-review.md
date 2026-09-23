# Cobot design review — 2026-09-23

Reviewed revision: `3d5cd6f` and the preceding printed-part changes.

**The current design is not ready to fabricate.** There are reproducible
assembly interferences, unusable motor mounting features, and validation
checks that do not exercise the new design. These conclusions come from the
generated build123d solids, the actual MuJoCo model, and the vendor drawings.

The existing suite reports **128 passed, 4 skipped**. All **117 downloaded
reference artifacts** match their recorded checksums. Those results establish
reproducibility; they do not establish a buildable assembly.

This review adds a diagnostic script and this report. Production CAD,
reference files, specifications, and existing tests have not been changed.

## Reproducing the measurements

From the repository root, with the existing environment:

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts/fetch_reference.py --check
.venv\Scripts\python.exe scripts/audit_cobot.py
```

The audit writes [local measurements](../sim/cobot_audit.json). It builds all
five printed solids, measures Boolean intersections in assembled poses,
checks mounting cutters, samples retained motor meshes against the shells,
and evaluates the actual printed twin. It does not modify design inputs.
`--simulation-only` runs just the numerical checks and writes a separate JSON.
The diagnostic is specific to this design revision; it is not an acceptance
test or a full workspace/strength certification.

## 1. P1 — Neighboring printed shells occupy the same space

**Evidence:** Boolean intersections of the final, filleted build123d solids
after applying MuJoCo's body transforms:

| Printed pair | Zero pose overlap | Working pose overlap |
|---|---:|---:|
| link2 / link3 | 2,289.18 mm³ | 1,308.27 mm³ |
| link3 / link4 | 1,172.37 mm³ | 1,175.18 mm³ |
| link4 / link5 | 798.56 mm³ | 797.68 mm³ |
| link5 / link6 | 0 mm³ | 0 mm³ |

The working pose is `q = [0, -1.0, -1.8, 0.4, 0.5, 0]` radians, inside the
six joint limits. These are intersections of printed material, not overlaps
of conservative collision cylinders or motor meshes. Separate printed parts
cannot occupy these shared volumes.

**Why tests miss it:** [collision.py](../src/robotic_arm/collision.py), lines
13–15, intentionally preserves parent/child filtering and the upstream
exclusion list. [The reference MJCF](../reference/mjcf/seeed_rebot_devarm.xml),
lines 332–335, explicitly excludes these pairs. That exclusion made sense for
the upstream simulation assets; it cannot validate newly designed joint seams.

**Correction:** Establish actual mating planes and running clearances at every
joint. Check adjacent shell pairs with exact geometry, allowing only explicitly
defined mechanical contact surfaces. Keep simulation contact exclusions
separate from assembly-fit validation.

## 2. P1 — RS06 mounts use the wrong pattern and fastener size

[upper_arm.py](../src/robotic_arm/parts/upper_arm.py), line 74, and
[forearm.py](../src/robotic_arm/parts/forearm.py), lines 89–91, index measured
circles by BCD alone. The reference JSON has **two circles at 24.02 mm**:

- Six holes, measured diameter 3.3 mm, plane Z = −1.0 mm.
- Three holes, measured diameter 2.5 mm, plane Z = +5.8 mm.

The dictionary silently overwrites the six-hole entry with the three-hole
entry. The builders therefore generate **three Ø3.4 clearance cuts**.

The [RS06 vendor manual, drawing on printed page 8](https://github.com/RobStride/Product_Information/blob/main/Product%20Literature/RS06/RS06User%20Manual260713.pdf)
specifies the output pattern as **6 × M4 on Ø24**, with 6 mm thread depth.
Its housing mount is **8 × M3 on Ø82**. Thus merely selecting the six-hole
record would still leave the wrong clearance-hole size. The same output circle
is also reused for the carried actuator without defining a separate housing
attachment.

**Correction:** Define named interfaces for the output, housing, and locating
features, including thread, face, angular phase, and datum. Select them by
identity, not just diameter. Preserve a complete housing-to-link and
output-to-neighbor load path.

## 3. P1 — Cable bores remove the material needed by the RS06 mounting holes

The parent bores are Ø26 on link2 and Ø24 on link3, while the selected bolt
circle is only Ø24.02. With the current Ø3.4 cuts, the minimum radial material
between a bolt hole and the bore is:

| Part | `BCD/2 − clearance/2 − bore/2` |
|---|---:|
| link2 | **−2.69 mm** |
| link3 | **−1.69 mm** |
| link4 | +0.80 mm |
| link5 | +1.80 mm |

Negative values mean the holes merge into the central opening. In link2 even
the nominal bolt center is inside the cable bore. These are not closed bolt
holes with adequate surrounding material. Correcting RS06 clearance to M4
makes the interference larger.

**Correction:** Redesign the bore and mounting flange together. Include screw
heads, washers, support thickness, locating features, and the actual available
cable route through/around the motor. A bore in a printed part alone does not
prove a continuous passage through the actuator.

## 4. P1 — Some mounting cuts miss the part or drill the wrong face

Measured against the corresponding shell before mounting holes and fillets:

- **link2 child mount:** cutter center Z = +29 mm, depth 44 mm, so the cut
  occupies Z = +7…+51 mm. It removes **0 mm³** from the shell.
- **link3 child mount:** cutter center Z = +31.5 mm, depth 40 mm, occupying
  Z = +11.5…+51.5 mm. It removes **0 mm³**.
- **link4 parent mount:** cutter center Z = −24 mm, depth 40 mm, occupying
  Z = −44…−4 mm. The joint-facing boss end is at **Z = +10 mm**. A point on
  the intended first hole center at Z = +9 mm is still inside solid material.

Sources: [upper_arm.py](../src/robotic_arm/parts/upper_arm.py):165,
[forearm.py](../src/robotic_arm/parts/forearm.py):202,
[wrist_pitch.py](../src/robotic_arm/parts/wrist_pitch.py):144.
The housing placement was updated, but these cuts still use older offsets.

**Correction:** Derive cuts from the actual mating faces. Verify that each cut
removes material and opens through the intended face with sufficient support
for its screw. A valid solid and a correct bounding box do not check this.

## 5. P1 — The passing thermal test uses stock inertials; the printed twin fails

[test_p2_regression.py](../tests/test_p2_regression.py), lines 44–56, still
constructs its `clone` with `generate(balancer=...)`, without CAD overrides.
Its comment says no printed parts are committed. Consequently P2a, P2b, and
P2c never evaluate the current printed parts.

Using `generate_twin(balancer=Balancer.sized_for(8.25), visuals=False)` and the
same 41 × 41 residual-torque sweep gives:

- Maximum remaining J2 torque: **8.609 N·m**.
- Project's assumed continuous budget: **7.700 N·m**.
- Example worst pose: `q = [0, -0.157, 0, 0, 0, 0]` radians.

This exceeds the project's own criterion by about **11.8%**, while its
regression test passes. The stock spring sizing has not been revalidated for
the much lighter printed model.

**Correction:** Test the production twin. First correct its assembled mass
inventory, then resize the spring and validate the intended operating poses.
The 70% derating remains an assumption, not a measured continuous capability.
The current RS06 manual specifies a **130 × 160 mm heat sink** for its rated
torque; an arbitrary aluminum bracket is not demonstrated equivalent.

## 6. P1 — The actuator inventory still omits the shoulder motor

[parts/__init__.py](../src/robotic_arm/parts/__init__.py), lines 59–87, assumes
each link carries only the motor for its child joint. On link2 it therefore
adds one RS06, positioned at the elbow housing.

But the retained stock visual is **`motor_2_3`**, containing motors at both
ends of the upper arm. `linkframes.actuator_centre()` explicitly recognizes
that it contains two clusters. The shoulder motor is not added elsewhere by
the new inertial code. The stock link1 mass is only 465.5 g, less than one
published 621 g RS06, so treating link1 as an intact extra RS06 assembly does
not reconcile the accounting either.

The current link2 inertial is **164.48 g shell + 621 g = 785.48 g**. It does
not account for both retained motors, nor a complete fastener/bearing/metal
interface inventory. The assumption that every motor housing belongs to the
parent link is also inappropriate for the retained shoulder arrangement.

Separately, [test_parts_all.py](../tests/test_parts_all.py):40 and :126 compare
**bare printed shells** with **whole stock body inertials**. For example,
link5 passes with a 51.16 g shell even though the generated body weighs
**361.16 g**, versus **201.20 g** in the baseline. That upstream body mass is
itself less than the published 310 g RS00; the baseline needs reconciliation
before treating parity as physical validation.

**Correction:** Make an explicit assembly BOM with every retained component,
its body assignment, mass, and transform. Compare the same component scope
on both sides. Resolve the upstream mass inconsistencies instead of inferring
manufacturable mass savings from them.

## 7. P1 — Payload is added at the gripper COM, not at the TCP

[torque.py](../src/robotic_arm/torque.py):175 increases `gripper_end.body_mass`
without changing its COM. That existing COM is **112.75 mm behind the tool
origin**. `balancer.max_continuous_payload()` makes the same assumption.

With the stock model and existing 70%-envelope calculation, `max_payload()`
reports **5.0 kg** in this environment. Applying that load at the actual tool
origin using its translational Jacobian gives **38.237 N·m** at J2, exceeding
the **36 N·m** limit, at a pose admitted by that same envelope:
`[0, -2.6166667, -1.57, 0, 0, 0]` radians.

This reproduces the analysis bug within the model's own sweep; it is not a
claim about a measured hardware payload rating. The approximate agreement
with the advertised 5 kg is not validation of the inertials. The conversion of
the manufacturer's workspace guidance to this scalar shoulder moment-arm
filter is also a project assumption.

**Correction:** Add payload at an explicit tool/load COM, using an attached
body or correctly combined inertials. Check every actuator and the intended
collision-free operating envelope. Use a joint Jacobian for the moment arm;
`j2_moment_arm()` currently uses a world-X difference that is not invariant
under base rotation.

## 8. P2 — Motor fit and collision geometry disagree

The final shells contain sampled vertices of their retained motor meshes:

| Body | Motor mesh | Strictly inside printed material |
|---|---|---:|
| link2 | motor_2_3 | 28 / 838 sampled vertices |
| link3 | motor_4 | 98 / 811 |
| link4 | motor_5 | 60 / 817 |
| link5 | motor_6 | 93 / 852 |

The audit uses OpenCascade solid classification with a 1e-6 mm tolerance and
counts strictly interior points, excluding boundary points. This proves
interference with the retained assets. It does **not** establish that those
upstream motor meshes accurately represent every current actuator revision.
The substantial differences between those assets and the nominal vendor
envelopes need resolution.

[mjcf.py](../src/robotic_arm/mjcf.py):372 replaces all stock body collision
hulls with shell cylinders, while motor visuals remain non-colliding. Motor
and connector projections outside those cylinders therefore also disappear
from inter-link collision checking. The wrist enclosure tests only compare
diameters, not occupied volumes or connector access.

**Correction:** Reconcile vendor geometry with the retained motor assets,
place each by a mechanical datum, and include the complete assembly in fit
and collision checks. Define an insertion/removal path or split housing, cable
access, and screw access; the current closed-shell construction has no
documented assembly sequence.

## 9. P2 — RS00 electrical data and the claimed mounting erratum are wrong or unsupported

[thermal.py](../src/robotic_arm/thermal.py):56 uses **0.36 N·m/Arms** for RS00.
The [current vendor manual](https://github.com/RobStride/Product_Information/blob/main/Product%20Literature/RS00/RS00User%20Manual260713.pdf),
printed page 8, specifies **1.48 N·m/Arms**. At its rated 5 N·m the code
calculates **19.64 Apk** and labels the motor **infeasible**, despite the
published rated and peak currents being 4.7 and 15.5 Apk. Holding resistance
fixed, its copper-loss estimate is approximately **16.9 times** the estimate
using the documented torque constant. Tests verify rated-torque consistency
for RS06 but not RS00.

The same manual's drawing on printed page 7 explicitly specifies **4 × M3 on
Ø38** for the back mount. [actuators.py](../src/robotic_arm/actuators.py):36
declares the published M3 pattern wrong based on smaller cylindrical features
recognized in a STEP file. That is not enough evidence to overrule the
installation drawing, and the test suite enshrines this inference. Reconcile
the STEP revision, feature purpose, and drawing before assigning threads.

The RS00 rated torque also specifies a **90 × 85 mm heat sink**. A current-only
`continuous` label does not establish acceptable temperatures in the printed
enclosure.

## 10. P2 — The structural calculation does not establish requirement P3

[structure.py](../src/robotic_arm/structure.py):176 models each link as a
constant-section cantilever with all downstream mass at its child joint. It
omits the carried motor on that same body, the actual downstream moment arm,
and transmission of link rotation into TCP displacement. `passes_p3` compares
that individual link's displacement to 0.5 mm, whereas P3 is a **whole-arm
TCP displacement under 2.5 kg at 70% reach**. The default payload is zero.

Thus even a passing section result cannot establish P3. The module is not
called by the existing tests. Its own documented limitations also exclude
joint compliance, stress concentrations, creep, and other relevant effects.
The joint helper further estimates axial bolt tension but divides it by a
hole-bearing area; those are different load/failure modes.

**Correction:** Use the full load path, carried hardware, distal lever arms,
and accumulated translations/rotations. Evaluate flange bending, bolt tension,
pull-through, and bearing separately. Validate material properties and final
deflection on representative prints/hardware.

## Additional integration and documentation issues

- **Joint signs:** All six arm axes in the vendored MJCF are opposite the
  vendored control URDF, not just J2/J3 as the provenance note highlights.
  Independently evaluated FK with all six angles negated agrees to about
  **0.0023 mm** at one nontrivial test pose. Negating only J2/J3 gives
  **179.08 mm** TCP error at that pose. No hardware/simulation sign adapter or
  randomized URDF-to-MJCF FK guard is present. Keep the upstream models
  unchanged, but define and test the interface mapping before software reuse.
- **Architecture claims:** README still says there are no part designs and
  that only inertials change. Five parts are registered, and `generate_twin()`
  changes visual and collision geometry. `frame_differences()` does not check
  everything its docstring claims, including contact exclusions and complete
  actuator configuration. Separate kinematic parity from permitted geometry
  changes and test the actual production path.
- **Requirements:** Passing stock-parity checks cannot establish assembly fit,
  stiffness, thermal capability, or hardware safety. Those independent
  requirements should not be inferred from a shipping product's existence.

## Recommended correction sequence

1. Reconcile the stock assembly, vendor interfaces, motor ownership, and masses.
2. Redesign joint interfaces and assembly access, then remove exact CAD
   interferences and prove that mounting cuts work.
3. Validate complete assembly collisions, including adjacent printed parts
   and retained motors/connectors.
4. Point regression tests at the actual twin; correct payload placement and
   RS00 electrical data; recalculate the spring and load budgets.
5. Complete structural/thermal validation and update the requirements/status
   documentation to reflect what has actually been demonstrated.

The immediate blockers are interface and assembly geometry. Cosmetic taper
or rendering adjustments cannot resolve them.
