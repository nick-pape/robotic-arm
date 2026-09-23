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
uv run pytest                        # verification suite
uv run python -m robotic_arm.torque  # actuator torque budget
```

## Two sources of truth

The spec's requirement F2 is "only `<inertial>` blocks differ from stock". That
is the architecture, not a convention:

- **joint frames and kinematics** come from the stock Seeed URDF, vendored at a
  pinned commit in `reference/` and never edited;
- **inertial properties** come from our build123d CAD.

Tests assert the frames never drift. `reference/` is upstream material only.

## Status

| Milestone | State |
|---|---|
| M0 vendor ground truth | done — URDF, Menagerie MJCF, RobStride STEP, checksummed |
| M1 stock baseline in MuJoCo | done — loads, simulates, verified to be the RS arm |
| M2 torque budget from real inertials | done — see below |
| M3 mass-properties pipeline | next |
| M4 RS06 bolt pattern from STEP | |
| M5 printed parts, distal-first | |
| M6 J2 balancer | |
| M7 thermal, CAN, e-stop | |

## What the real inertials say

The spec's load model was built on an estimated mass distribution and says so:
*"Replace them with the URDF inertials."* Doing that moves the answer
materially — `uv run python -m robotic_arm.torque`:

```
J2 self-weight only: 15.36 N*m = 140% of rated, 199% of derated, 43% of peak
spec estimated 10.7 N*m -> real is +44%
```

The gap is distal mass: the spec assumed 3.3 kg beyond J2, the real figure is
**4.37 kg**. So **the unbalanced arm cannot hold its own weight at full reach
continuously**, before any payload at all. That makes the J2 gravity balancer
load-bearing for the design rather than an optimisation, and it means the
balancer must cancel more than the spec's 8 N·m target.

Requirement P2 is currently a deliberate `xfail` in the suite; it flips to
passing when the balancer lands in M6.

## Layout

```
reference/          vendored upstream, read-only — URDF, MJCF, STEP
src/robotic_arm/
  reference.py      load the stock baseline model
  torque.py         gravity-torque analysis vs actuator limits
scripts/
  fetch_reference.py  pinned, checksummed fetch of large artifacts
spec/               engineering brief
tests/              one case per spec requirement ID
```

## Modelling with an AI assistant

`.mcp.json` registers [build123d-mcp](https://github.com/pzfreo/build123d-mcp),
which gives an assistant a live CAD session: execute build123d code, render and
measure geometry, check printability, and export STEP/STL. Claude Code picks it
up from `.mcp.json` on start.

The build123d pin (`>=0.10,<0.12`) matches the MCP server's build123d 0.11.1, so
CAD developed in an MCP session behaves identically in the repo.
