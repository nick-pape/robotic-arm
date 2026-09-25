# robotic-arm

A MuJoCo digital twin of the [reBot Arm B601-RS](https://wiki.seeedstudio.com/rebot_b601_rs_getting_started/)
with a custom FDM-printed structure, and the [build123d](https://build123d.readthedocs.io/)
CAD that generates it.

The engineering brief is `spec/rebot-arm-b601-rs-clone.md`.

## Setup

Requires [uv](https://github.com/astral-sh/uv) and Python 3.11-3.14.

```bash
uv sync --extra dev
uv run python scripts/fetch_reference.py   # ~16 MB of upstream meshes and STEP
```

The fetch step is separate because reference meshes are pinned but not
committed — see `reference/PROVENANCE.md`.

## Usage

```bash
uv run pytest                          # verification suite
uv run python -m robotic_arm.assembly  # bolt access per part
uv run python -m robotic_arm.torque    # actuator torque budget
uv run python -m robotic_arm.mjcf      # generate sim/model.xml
python -m mujoco.viewer --mjcf=sim/model.xml

# Design review images, into a gitignored renders/
uv run --extra viz python scripts/snapshot.py --balancer     --focus link2,link3,link4,link5,link6
uv run --extra viz python scripts/joint_sections.py
```

The joint sections are the ones worth looking at first. An assembled-arm
render cannot show where two parts meet, because the interface is inside the
shell -- and that gap hid a real fault for several revisions, with the printed
bosses seating on the actuators' stator faces instead of their rotating output
hubs. Each section pairs a printed part with the real vendor actuator it bolts
to and cuts the pair in half.

## Two sources of truth

The spec's requirement F2 is "only `<inertial>` blocks differ from stock". That
is the architecture, not a convention:

- **joint frames and kinematics** come from the stock Seeed URDF, vendored at a
  pinned commit in `reference/` and never edited;
- **inertial properties** come from our build123d CAD.

`mjcf.generate()` writes only inertials, and `frame_differences()` checks that
afterwards. Note that `generate_twin()` deliberately goes further: it also
swaps visual meshes and collision geometry, so it is *not* a frames-only
transform and must not be compared against stock as though it were.

## Status

**All seven structural parts are now designed** (base_link, link1-link6) on a
single UR5e-style archetype. Goal: look and assemble like a UR5e, using the
same RobStride actuators, with stock-equivalent performance.

Every link is the same shape::

    (=)  ================  [  M  ]
     rotor flange          stator housing

The flange caps the previous joint's housing and bolts to its **rotor** ring;
the housing encloses this link's own child motor and bolts to its **stator**
ring. Both rings are on one face of a pancake actuator, so a joint is two
links bolted to the same face at different radii, and the seam between flange
and housing is all that shows from outside. Housing diameters come from the
motors, not from taste: an RS06 is O87 across its flange, so no housing that
actually encloses one can be slimmer than O94.

### Measured against the stock arm

| | stock | twin | |
|---|---|---|---|
| total mass | 6.01 kg | 3.81 kg | **-36.6%** |
| reach | 778.72 mm | 778.72 mm | identical |
| payload, 70% envelope | 4.72 kg | 5.22 kg | **+10.6%** |
| worst J2 torque | 15.36 N*m | 12.99 N*m | -15.4% |
| every segment length | | | identical to 0.000000 mm |

Reproduce with `uv run python scripts/compare_performance.py`. Reach and
segment lengths *must* be identical, because the generator may not touch
joint frames (F2); any difference there is a defect, not a design choice.
Payload improves because the arm carries 2.2 kg less of itself. Printed
structure is 836 g against 5208 g of stock structure.

The cost is clearance. Stock keeps its links slim by leaving the motors
bare between two fork plates; enclosing an O87 RS06 needs an O94 housing,
3.5 mm fatter in every radial direction than the motor already sitting
there. 68 of 2000 sampled poses are clear on stock and collide on the
twin, worst -5.2 mm. Slimming the barrels took that from 114 poses and
-6.1 mm; the rest is inherent, and the ways out are a thinner wall, a
smaller actuator, or reduced joint limits. Tracked as a strict xfail in
`tests/test_s1_clearance.py`.

| Milestone | State |
|---|---|
| M0 vendor ground truth | done — URDF, Menagerie MJCF, RobStride STEP, checksummed |
| M1 stock baseline in MuJoCo | done — loads, simulates, verified to be the RS arm |
| M2 torque budget from real inertials | done — see below |
| M3 mass-properties pipeline | done — validated against closed-form solutions |
| M4 RS06 bolt pattern from STEP | done — measured, and corrected a published error |
| M5 CAD→MJCF generator | done — path proven end to end, F2 enforced |
| M6 J2 balancer | done — sized by optimisation |
| M7 thermal and current limits | done |
| base_link, link1-link6 printed | designed on the UR5e archetype; performance validated |
| link1, base_link | done |

## What the real inertials say

The spec's load model was built on an estimated mass distribution and says so:
*"Replace them with the URDF inertials."* Doing that moves the answer —
`uv run python -m robotic_arm.torque`:

```
J2 self-weight only: 15.36 N*m = 140% of rated, 43% of peak
spec estimated 10.7 N*m -> real is +44%
```

The gap is distal mass: the spec assumed 3.3 kg beyond J2, the real figure is
**4.37 kg**.

### This does not mean the stock arm is broken

It is a shipping product and it works. `thermal.py` gives the physically
meaningful reading: holding full extension draws 19.9 Apk against a 14.3 Apk
continuous rating but only 57 Apk peak, dissipating ~69 W. That pose is
**time-limited by heat, not impossible** — which is exactly what Seeed's own
"stay within about 70% of the workspace" warning and the firmware's
80/100/140 °C gating are about.

Where stock and clone genuinely differ is the heat path. Stock bolts J2 to an
aluminium flange that conducts into the sheet-metal link, which is what
RobStride's rated figures assume. A printed link has none, so the spec assumes
60–70% of rated continuously (~7.7 N·m) — and against *that* budget the
unbalanced clone cannot hold itself at reach at all.

## What the balancer achieves

Sizing the J2 spring is an optimisation, not "cancel the peak": over-springing
makes the worst case worse, because at folded poses the gravity moment is small
and an oversized spring drives the joint the other way.

The optimum cancels **8.25 N·m** (k = 1375 N/m, 220 N peak force), taking
worst-case J2 from 15.36 → **7.13 N·m**, inside the clone's continuous budget.
That is the figure the spec recommended — which its own low self-weight
estimate did not actually support.

## Requirements are regression guards

The point of the verification suite is not "is this arm good enough" in the
abstract — it is whether swapping metal for plastic costs anything. The clone
reuses stock actuators, kinematics and software, so most requirements are
stated as **parity against the stock model** rather than as absolute targets.
That also makes them self-calibrating: re-measure the baseline and the targets
move with it instead of going stale.

P2 was rewritten on this basis; `spec/` carries the full reasoning. As
originally written it asked for payload held continuously at the worst pose in
the whole workspace within a derated budget — and **the stock arm fails that at
zero payload**, because its own self-weight exceeds the budget at extension. A
requirement a shipping product fails is measuring the wrong thing. It is now:

| | Requirement | Stock reference |
|---|---|---|
| P2a | Clone J2 torque ≤ stock + 5% | 15.36 N·m |
| P2b | Clone payload capacity ≥ 90% of stock, within the 70% envelope | 4.97 kg |
| P2c | Balanced self-weight ≤ derated continuous budget (absolute) | 7.13 / 7.70 N·m |

P2c is the one deliberate absolute, because it covers the single place the
clone is worse off by construction: stock bolts J2 to an aluminium flange that
conducts heat into the sheet-metal link, and a printed link has none. Stock
does not need a balancer to hold itself up; the clone does.

Until printed parts exist, P2a and P2b pass trivially — they are tripwires
armed ahead of the change they guard, not measurements of work already done.

## Layout

```
reference/          vendored upstream, read-only — URDF, MJCF, STEP
src/robotic_arm/
  reference.py      load the stock baseline model
  actuators.py      RS06/RS00 ratings + measured bolt geometry
  materials.py      densities, with an infill model
  massprops.py      build123d solid -> mass, COM, inertia tensor
  mjcf.py           generate the twin: stock frames + CAD inertials
  balancer.py       J2 zero-free-length spring, sizing and residuals
  torque.py         gravity-torque analysis vs actuator limits
  thermal.py        current and heat — what actually bounds continuous torque
  motion.py         demonstration trajectory, shared by viewer and renderer
scripts/
  fetch_reference.py    pinned, checksummed fetch of large artifacts
  measure_actuators.py  re-derive bolt geometry from vendor STEP
  view.py               open the twin in the interactive viewer
  render_motion.py      same, offscreen to a GIF (output gitignored)
spec/               engineering brief
tests/              one case per spec requirement ID
```

## Known defects

An independent review (`docs/cobot-design-review.md`, 2026-09-23) found faults
that the passing suite did not. Verified and fixed so far:

- RS06 bolt circles were selected by diameter alone, and it has **two** circles
  at O24.02. The dict silently kept the three-hole one, so link2 and link3
  generated three mounting holes where the actuator has six.
- The RS00 torque constant was 0.36 N*m/Arms, unsourced and wrong by a factor
  of four; the motor read as infeasible at its own rated torque. The
  consistency test covered only the RS06, which is how it survived.
- `test_p2_regression.py` built its "clone" with `generate()` rather than
  `generate_twin()`, so every P2 check ran against the stock model. Pointed at
  the real twin, P2c failed at 8.61 N*m against a 7.70 budget.
- The balancer was still sized for stock's 15.36 N*m self-weight. Re-optimised
  for the lighter twin: 6.5 N*m cancellation, 6.90 N*m residual.

Also fixed:

- **Cable bores swallowed the bolt circles** (-2.69 mm of material on link2).
  Both actuators turn out not to be hollow-shaft motors -- their STEP files
  show only a O4 central feature -- so a through-bore implied a cable route
  that does not exist. Removed; the harness now runs beside the actuator in a
  side channel, as the stock arm's clipped XT30 daisy chain does.
- **Mounting cuts removed 0 mm3.** They were placed from constants while the
  housings had since been moved onto their motors. Now derived from the
  drums' actual faces; they remove ~112 mm3 each.
- **Parent axes on this arm are not consistently signed** ((0,0,-1) on some
  links, (0,0,+1) on others), and deriving a direction from the raw axis has
  now put a boss, an access port, a seated actuator and two bolt rings on the
  wrong side of their own joint -- five separate times, each found by a render
  or a test rather than by reading the code. Direction is now taken from
  `cobot.boss_direction`, which reads it off the geometry that was actually
  built.
- **A tube wider than its cup bulged out through the mating face.** link2's
  O58 tube was centred in a 36 mm cup, so it protruded 6 mm past the face; the
  bore then clipped the protrusion and left a floating chip of a second solid.
  Cups are now at least as long as the tube is wide (which also makes them
  cover the motors properly), `build_link` refuses to return more than one
  solid, and a probe rejects anything blocking the joint bore.
- **Housing mating faces were capped**, so the child link's boss punched
  through them. Opened to clear the child's boss plus a running gap.

Still open, and blocking fabrication:

- **Printed seams are worse than the stock ones they replace** -- link2/link3
  by 36 mm, link3/link4 by 32 mm, link4/link5 by 14 mm. Note that *both* arms
  interpenetrate at folded joint limits (stock's link2/link3 by 16 mm), so
  this is a parity regression rather than overlap as such. The printed drums
  are round and fatter at the seams than stock's flat beams. Fixing it needs
  slimmer seam geometry or reduced joint limits. Tracked as a strict xfail.
- ~~Payload applied at the gripper COM~~ **fixed.** It now hangs at the
  grasp point between the fingers. Stock workpiece capacity drops 5.00 ->
  4.00 kg as a result, and the earlier "reproduces the advertised 5 kg
  exactly" claim was an artefact of the 113 mm error. With the 0.8 kg gripper
  counted, total at the tool is 4.8 kg against the advertised 5 -- consistent,
  but the quoting convention is an assumption, so it is not validation.
- **The flange and housing each derived their own direction**, so at J3 and J4 the flange extended the *same* way as the parent's housing and sat 14 mm inside it -- which is why the elbow rendered as an open cylinder with the motor visible. J5 looked right only because link4's housing was independently wrong in the same direction, two errors cancelling. Both ends now derive from one measured fact, which side the motor slug sits on, so they are opposite by construction.
- **Stock's fork told us nothing about the motor interface.** Its rings bolt at O54 to the plate-to-plate standoff collar, not to the actuator; the RS06 has no O54 circle. The vendor STEP is the only authority for the bolt patterns.
- ~~No bolt access~~ **fixed, then found to have been measuring nothing.**
  `mount_rings` had gone stale against the redesign: with no part still
  exposing `_mount_circle`, it fell back to a single RS00 O27 circle for both
  rings on every part -- a bolt circle that exists on neither interface. It
  now reads the actual rotor and stator circles, and measures from the **bolt
  head** rather than the mating face, since the screws are counterbored and a
  corridor starting at the mating face begins inside material by construction.
  Against the real rings, three further defects surfaced and were fixed: the
  cup had no driver port at all, both ports cut away the very cap the bolt
  head bears on, and the RS00 links' O35 boss was too narrow to admit a driver
  to its own O27 bolt circle. All four links now have both rings reachable.
- **Every printed part still intersects at least one motor** where the motors
  actually sit: link2 29, link3 45, link4 59, link5 41, link6 7 sampled
  vertices, down from 260 total before the redesign. Tracked as a strict
  xfail, with a per-part ratchet test so the numbers cannot drift -- the
  previous record claimed exactly that and was never read, so the redesign
  moved all five while the stale figures sat unchallenged.
- **`max_payload` was silently wrong**, and it underpins requirement P2. Its
  bisection read the tool mass *it had just written* as the base for the next
  probe, so every trial carried the sum of all previous trials. The search
  then stopped at whatever dyadic value passed first and returned exactly
  4.0000 kg for any arm that could hold 4 kg -- stock and twin alike, which is
  why a clone 2.2 kg lighter appeared to gain no payload at all. Fixed by
  restoring the tool state before each probe; stock reads 4.73 kg and the twin
  5.23 kg.
- Cable route is a side channel beside each actuator, but no harness
  routing, strain relief or connector access has been designed.
- `structure.py` does not establish P3: it is a per-link cantilever, while P3
  is whole-arm TCP deflection.

## Known limitations

- **Effective density is the weakest number in the model.** A 40%-infill PA-CF
  part is not ρ=1160, and `Material.printed()` is a crude shell-plus-core
  estimate. It will dominate the inertia error budget. `Material.measured()`
  overrides it from a weighed coupon; deriving it from slicer GCode is the
  better long-term answer and is left as a seam rather than built.
- Bench-only requirements (S2, S3, T1, V1, P3) have no hardware to run against
  yet.

## Modelling with an AI assistant

`.mcp.json` registers [build123d-mcp](https://github.com/pzfreo/build123d-mcp),
which gives an assistant a live CAD session: execute build123d code, render and
measure geometry, check printability, and export STEP/STL. Claude Code picks it
up from `.mcp.json` on start.

The build123d pin (`>=0.10,<0.12`) matches the MCP server's build123d 0.11.1, so
CAD developed in an MCP session behaves identically in the repo.
