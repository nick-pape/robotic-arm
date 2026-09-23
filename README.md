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
uv run python -m robotic_arm.torque    # actuator torque budget
uv run python -m robotic_arm.mjcf      # generate sim/model.xml
python -m mujoco.viewer --mjcf=sim/model.xml
```

## Two sources of truth

The spec's requirement F2 is "only `<inertial>` blocks differ from stock". That
is the architecture, not a convention:

- **joint frames and kinematics** come from the stock Seeed URDF, vendored at a
  pinned commit in `reference/` and never edited;
- **inertial properties** come from our build123d CAD.

`mjcf.generate()` cannot violate this, because it has no code path that writes
anything but inertials — and `frame_differences()` proves it afterwards rather
than assuming it.

## Status

Infrastructure is complete; no part designs are committed yet, pending design
input.

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
| Part design | **next — needs design input** |

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
