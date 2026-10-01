# Training plan — first learned policy in sim

*Written 2026-10-01, at the hand-off from a Windows laptop to a Codespace on
the homelab (RTX PRO 4500). The laptop proved the pipeline end to end; the
homelab records the real dataset and trains on it.*

## Goal

An ACT policy that picks the red cube from anywhere the arm can reach, using
only what a real policy would get: joint angles and two camera images. It is
the first rung of the ladder in the earlier research (imitation in sim →
VLA fine-tune → real teleop data), and it exercises every stage the real arm
will use: record → LeRobot dataset → `lerobot-train` → closed-loop rollout.

Success criterion for this run: **≥ 80 % success over 50 unseen cube poses**
(`learning.rollout --episodes 50`). The scripted expert it learns from scores
≥ 95 %.

## What exists (`learning/`)

| Module | What it does |
|---|---|
| `scene` | The generated twin plus a task: floor, a 4 cm 50 g cube, finger pads, gravity feedforward, a D435-style scene camera (640×480, framing solved to fit the arm and the whole spawn region) and a D405-style wrist camera (320×240) |
| `env` | Gymnasium env, 30 Hz, LeRobot observation layout (`agent_pos`, `pixels/{scene,wrist}`); actions are 6 absolute joint angles (rad) + gripper 0..1, clipped to ±0.15 rad/step |
| `ik`, `scripted` | Damped least-squares IK and a scripted pick (privileged: reads the cube pose). `ScriptedPickPlace` loops pick → carry → place forever |
| `record` | Scripted picks → LeRobotDataset v3; failed picks discarded; 1 s of holding recorded after each lift |
| `check_dataset` | Reads a recorded dataset back and verifies it (episodes, video frame counts, signals, workspace coverage). Gate training on it |
| `policies` | Untrained or trained ACT through LeRobot, plus a random baseline |
| `rollout` | Runs any policy: success rate, MP4/GIF, or `--viewer` live with the planned/past trajectory drawn |

The spawn region was **measured, not chosen**: the scripted pick succeeds at
every radius 0.20–0.60 m for bearings −170°…+70° off the arm's home heading.
Past +80° there is no collision-free top-down grasp. Tests re-check the
region's edges, so a design change that shrinks it fails loudly.

## Decisions already made, and why

- **ACT first, not a VLA.** One task, one scene, no language: ACT's home
  ground. It debugs the pipeline at the lowest cost before π0.5/SmolVLA.
- **Chunk 30 (1 s), not ACT's default 100.** Episodes are ~4.5 s; a
  100-step chunk would execute most of one blind.
- **200 demos, not 50.** The cube spans a 240° arc, so the policy has to
  learn to turn the base, not one fixed motion. 50 is the textbook number for
  a *fixed* spot.
- **Absolute joint targets in radians.** The sim is native in them;
  conversion to the real follower's degrees belongs at the hardware boundary.
- **Actions recorded after the env's clip** (`PickCubeEnv.clip_action`): the
  dataset only contains commands the arm actually obeyed.
- **Evaluation seeds 0..N, recording seeds from 10 000:** every evaluated
  pose is unseen in training.

## Running it on the homelab

### 1. Environment

The Codespace container needs the GPU passed through (`--gpus all`, or the
devcontainer's GPU host requirement) and, for headless MuJoCo rendering:

```bash
sudo apt-get install -y libegl1 libgl1 libglvnd0 ffmpeg   # EGL for MuJoCo, FFmpeg for torchcodec
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
uv sync --extra dev --extra sim --extra policy
uv run python scripts/fetch_reference.py                  # vendor meshes; the twin is built from them
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

`uv.lock` resolves Linux torch 2.11 as a CUDA 13 build, which needs an NVIDIA
driver ≥ 580. If `cuda.is_available()` is False, check `nvidia-smi` first.

### 2. Validate before spending hours

```bash
uv run pytest learning/tests -q          # ~2 min incl. one twin build; all must pass
uv run python -m learning.rollout --policy scripted --episodes 3 --video mp4
```

The rollout videos land in `renders/rollouts/`; look at one. If the scene or
wrist image is black, EGL is not working.

Run `learning/tests`, not the whole suite. The design suite (`tests/`) has 30
known failures at the time of writing -- link interpenetration, motor
interference ceilings, J2 torque parity -- from assemblability work still in
progress on the CAD, verified to be independent of everything in `learning/`.
They are not blockers for training, and not this task's to fix.

### 3. Record

```bash
uv run python -m learning.record                     # 200 episodes -> data/lerobot/local/rebot_twin_pick_arc
uv run python -m learning.check_dataset              # must print 9/9
```

On the laptop this ran ~1 min/episode (CPU rendering and AV1 encoding).
Expect much faster here; if it isn't, rendering is not on the GPU.

### 4. Train

```bash
uv run lerobot-train \
  --dataset.repo_id=local/rebot_twin_pick_arc \
  --dataset.root=data/lerobot/local/rebot_twin_pick_arc \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --policy.chunk_size=30 --policy.n_action_steps=30 \
  --batch_size=16 --steps=30000 --save_freq=10000 --num_workers=8 \
  --output_dir=outputs/act_pick_arc
```

A 4-step CPU run of exactly this command (on the earlier dataset) trained and
checkpointed correctly; the full run has not been done. Watch the logged
`l1_loss` fall. The ResNet-18 backbone downloads ImageNet weights on first
run.

### 5. Evaluate

```bash
CKPT=outputs/act_pick_arc/checkpoints/last/pretrained_model
uv run python -m learning.rollout --policy act --checkpoint $CKPT --episodes 50 --video none
uv run python -m learning.rollout --policy act --checkpoint $CKPT --episodes 5 --video mp4
```

Also evaluate the 10k and 20k checkpoints — ACT can overfit, and the best
checkpoint is not always the last. Record the success rate of each here.

## If it falls short

In rough order of likely payoff:

1. **Watch the failures** (`--video mp4`). Which phase fails — finding the
   cube, aligning the jaws, or the grip — says which fix applies.
2. **More data or longer training** — 400 demos, or 60k steps.
3. **Temporal ensembling** (`--policy.n_action_steps=1
   --policy.temporal_ensemble_coeff=0.01`): re-plans every step, smoother
   and more reactive, slower at inference.
4. **The camera.** The cube is ~10 px in the scene view. Larger renders or a
   second scene camera are cheap here.
5. **The home pose.** The wrist camera sees only sky for the first ~45 frames
   while the arm leaves home; a start pose that looks at the table gives
   the policy its closest view from the first frame.

## Next steps after a working ACT

1. **Pick-and-place.** `ScriptedPickPlace` already exists; record it as a
   second task and train one policy on both (introduces language: SmolVLA or
   Multitask DiT).
2. **Domain randomization** — lighting, textures, cube size and mass,
   camera pose ±2 cm, actuator gains and latency. Without it, nothing
   learned here survives contact with the real arm.
3. **An actuator model and system ID** — the sim's servos are ideal PD
   with gravity feedforward. The real RobStride MIT loop has latency,
   friction and torque limits; the research notes have the recipe.
4. **VLA fine-tune** — SmolVLA locally, then π0.5 (full fine-tune on a
   rented 80 GB GPU; LoRA if not). Same dataset format, same rollout.
5. **Real data.** The same `record` → `check_dataset` → `lerobot-train`
   path takes teleop demos from a leader arm instead of the scripted
   expert. That is the point of having built it this way.

## Known caveats

- The scripted expert is privileged (it reads the cube pose) and open-loop
  once planned. It is a source of clean demos, not a recovery-capable
  teacher; policies trained on it will not have seen recoveries.
- On Windows: LeRobot can't create the `checkpoints/last` symlink without
  Developer Mode (use the numbered folder), and torchcodec prints harmless
  tracebacks before falling back to PyAV. Neither applies on Linux.
- Rendering is triangle-bound (~850k from the CAD meshes), not pixel-bound.
  Decimated visual meshes would speed recording several-fold if it matters.
