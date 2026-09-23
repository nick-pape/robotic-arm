# Reference artifacts

Everything here is **upstream material, vendored unchanged**. Nothing in this
directory is edited by this project. It is the source of truth for joint frames
and for the stock arm's mass properties; our CAD supplies only new inertials.

Regenerate the non-committed files with:

```bash
uv run python scripts/fetch_reference.py           # fetch + verify
uv run python scripts/fetch_reference.py --check   # verify only, no network
```

Every fetched byte is checksummed against `manifest.json` (117 entries), so
"not committed" never means "not pinned".

## Sources

| Path | Upstream | Commit | Licence |
|---|---|---|---|
| `urdf/RS/ReBot_Arm_RS.urdf` | [Seeed-Projects/reBotArm_control_py](https://github.com/Seeed-Projects/reBotArm_control_py) `urdf/RS/urdf/` | `6415d43130d1e143c70dc106096a857ac5556f81` | see upstream |
| `mjcf/seeed_rebot_devarm.xml`, `mjcf/scene.xml`, `mjcf/assets/*` | [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) `seeed_rebot_devarm/` | `da76818e269b82289eba39808e2fb91d679d6994` | MIT (`mjcf/LICENSE`) |
| `step/RS06-new.step`, `step/RS00.step` | [RobStride/Product_Information](https://github.com/RobStride/Product_Information) `Product Literature/` | `0f4ad74fdb67023e75bbcbeebecd6f1a003ce000` | vendor material |

Retrieved 2026-09-23.

## Committed vs fetched

Committed: the URDF, the two MJCF files, the Menagerie licence, and
`manifest.json`. These are small text files worth diffing.

Fetched on demand: 115 Menagerie mesh assets and 2 RobStride STEP files,
~16 MB of binary that already has a canonical upstream home. They are covered
by `.gitignore`'s `*.stl` / `*.step` rules.

## Why the Menagerie model is the RS arm

The directory is named `seeed_rebot_devarm`, which reads like the DM variant.
It is not. Its body positions match the RS URDF joint origins digit-for-digit
(`link3 pos="-0.236 0 0"`, `link5 pos="0.087 -0.048 -0.03075"`,
`gripper_end pos="0 0 0.16621"`), all ten masses match the URDF inertials
exactly, and its README describes RS-06 / RS-00 actuators. `tests/` asserts
this rather than trusting the name.

## Known upstream quirks

- **Joint sign convention.** MJCF `joint2` / `joint3` carry
  `ctrlrange="-3.14 0"` while the URDF declares `lower="0" upper="3.14"`. The
  URDF axes are `0 0 1` and `0 0 -1`, so the converter absorbed the sign into
  the axis. Any URDF-to-MJCF comparison must map this explicitly.
- **Gripper asymmetry.** `gripper_joint1` upper limit is `0.05`,
  `gripper_joint2` is `0.0715`. That is what upstream says; it is not a
  transcription error.
