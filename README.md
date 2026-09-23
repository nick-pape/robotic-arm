# robotic-arm

Parametric CAD models for a servo-driven robotic arm, written in
[build123d](https://build123d.readthedocs.io/) and set up to be modelled
interactively through [build123d-mcp](https://github.com/pzfreo/build123d-mcp).

All dimensions are in millimetres.

## Setup

Requires [uv](https://github.com/astral-sh/uv) and Python 3.11–3.14.

```bash
uv sync --extra dev
```

## Usage

Build and export every part to `exports/`:

```bash
uv run python -m robotic_arm.export
```

Check dimensions:

```bash
uv run pytest
```

Print a single part's stats:

```bash
uv run python -m robotic_arm.parts.base_plate
```

## Modelling with an AI assistant

`.mcp.json` registers the `build123d-mcp` server, which gives an assistant CAD
tools: execute build123d code in a live session, render PNG/SVG previews,
measure volumes and bounding boxes, check printability, and export STEP/STL/DXF.
It lets the assistant check geometry as it builds instead of writing a script
blind.

Claude Code picks the server up from `.mcp.json` on start — approve it when
prompted. Verify the server can launch:

```bash
uv tool run --python 3.12 build123d-mcp@latest --version
```

The server runs in its own isolated environment, so it does not need
`uv sync` to have been run first.

A useful prompt shape:

> Use build123d-mcp. Build the forearm link incrementally, render after the main
> features, measure the final dimensions, run `validate()`, and export STEP if it
> passes.

You can also ask the assistant to call
`install_skill(target="claude", skill="modeling")` to drop the server's own
modelling workflow guidance into the repo.

## Layout

```
src/robotic_arm/
  params.py          shared dimensions — servo, base, links, fasteners
  export.py          builds every part and writes exports/
  parts/
    base_plate.py    rounded base plate, corner holes, servo horn bore
tests/               dimensional checks
exports/             generated STEP/STL (gitignored)
```

Parts import their dimensions from `params.py` rather than hardcoding them, so
changing a servo or fastener size propagates through the assembly. Add a new
part as a `build_*` function in `parts/`, export it from `parts/__init__.py`,
and register it in `export.py`.

## Version note

The build123d dependency is pinned to `>=0.10,<0.12` to match the range
`build123d-mcp` supports, so code developed in an MCP session behaves the same
way when it lands in this repo. Newer build123d releases exist; if you stop
using the MCP server, this pin can be relaxed.
