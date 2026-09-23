# reBot Arm B601-RS Clone with Custom FDM Structure: What I Could Verify, Budgets, and Requirements

**Bottom line:** A printed-structure clone of the B601-RS can reuse the whole stock software stack if two things hold: (1) you keep the joint frames identical to Seeed's RS URDF, and (2) you only change the inertial blocks in the URDF/MJCF. The limiting joint is J2. Its RS06 has 11 N·m rated / 36 N·m peak, and at full reach the arm's own weight plus a 2.5 kg payload needs roughly 1.5-2.5x that rated torque. So Seeed's "2.5 kg rated" figure only holds inside its recommended 70% workspace. A J2 gravity balancer and metal (not printed) parts at the J1/J2 load path are what buy back continuous margin.

This run ran out of turns before three planned steps: the RS-specific BOM page (the URL was blocked because no search returned it), the URDF joint offsets, and the RobStride 2026-07-13 spec PDF with its RS06/RS00 heat-sink conditions. I did not spawn the research subagent or run the enrichment pass. Values not confirmed from a source are marked **[UNVERIFIED]**.

## TL;DR
- **Kinematics and software:** keep the joint frames from Seeed's `reBotArm_control_py/urdf/RS` exactly, and change only mass, center of mass (COM) and inertia. Gate this with a pytest forward-kinematics (FK) check under 0.1 mm across random configurations, plus a mass budget where each printed link weighs no more than the part it replaces.
- **Load and margin:** J2 (RS06, 11 N·m rated / 36 N·m peak, 14.3 Apk rated / 57 Apk peak phase current) is the bottleneck. Seeed's own warning points the same way: stay within about 70% of the workspace or J2 stall protection can trip and the arm can drop. Size a zero-free-length spring balancer to cancel about 70-80% of the arm's self-weight moment. Plan on about 1.0-1.5 kg of continuous payload at full reach, not 2.5 kg.
- **Build strategy:** print the arm shells, links and covers in PA-CF or PC-CF. Keep machined aluminium for the J1 bearing mount, the motor front/rear spacers and flanges at J2-J4, the wrist bracket and the gripper slider bracket; Seeed's own BOM marks these as needing metal or as "not recommended" in print for long-term use. Build the stock arm first as an A/B baseline, then swap parts distal-first.

## Key Findings (sourced)
| Item | Value | Source / status |
|---|---|---|
| Actuators | J1-J3 RS06, J4-J6 RS00, gripper = CAN ID 7 (RS00); host ID 0xFD | rebot_control README |
| RS06 | 11 N·m @100 rpm rated, 36 N·m peak, 621 g, 9:1, 480 rpm no-load, 15-60 V, 14.3 Apk rated / 57 Apk peak, Kt 1.09 N·m/Arms, 0.23 Ω line resistance, Ø82 x 49 mm | RobStride README (2026-07-13), RS06 manual, Seeed wiki |
| RS00 | 5 N·m rated, 14 N·m peak, 310 g, 10:1, 315 rpm, 24-60 V, 4.7 Apk rated / 15.5 Apk peak, 1.5 Ω line resistance, Ø57 x 51 mm; connector XT30PB(2+2) (AMASS) | RobStride README, RS00 manual, Seeed wiki |
| RS00 bolt pattern | Top 6x M3 on Ø27 ±0.1 mm; bottom 4x M3 on Ø38 ±0.1 mm | manuals.plus copy of the manual **[verify against the STEP file]** |
| RS06 bolt pattern | Not captured; get it from `RS06-new.step` and the installation drawing | **[UNVERIFIED]** |
| Rated-torque condition | "Rated torque values require the aluminum heat sinks specified"; RS04 needs a 345x345 mm plate for 40 N·m (35 N·m on 220x200 mm) | RobStride README; the RS06/RS00 plate sizes are **[UNVERIFIED]** because the PDF was not read |
| Protections | Fault bits: stall-overload (bit14), overtemperature at 145 °C thermistor, undervoltage 12 V, overvoltage 60 V; CAN_TIMEOUT (20000 = 1 s) puts the motor in reset | RS06 manual |
| MIT law | τ = Kd(v_set − v) + Kp(p_set − p) + τ_ff; P ±12.57 rad, V ±50 rad/s, Kp 0-5000, Kd 0-100, T ±36 N·m (RS06) | RS06 manual |
| Unpowered behaviour | Motors apply anti-backdrive damping by default when rotated quickly while unpowered; this is damping, not a brake | RS06 manual |
| Arm specs | Reach 754.7 mm with gripper / 587.5 mm without; 6.7 kg; 2.5 kg rated / 5 kg max (both within 70% workspace); repeatability 0.1 mm (README says < 0.2 mm); PSU 48 V 15 A (wiki) vs 12.5 A LRS-350-48 (README); joint ranges as given in the task brief | Seeed wiki, blog, README |
| Stock gravity compensation | τ = g(q) from Pinocchio with kp = 2, kd = 1; Seeed suggests scaling tau_g[2], [3] by about 1.2 for friction or assembly error; motors are disabled on exit, so the arm drops | Seeed Pinocchio wiki |
| Stock rebot_control loop | MIT at 200 Hz, telemetry at 2 Hz; temperature thresholds 80 °C warn / 100 °C slow return to zero / 140 °C emergency disable | rebot_control README |
| Printed vs metal (DM BOM, same mechanics as RS) | Printed: ABS for load-bearing parts at 30% infill, PLA at 15% for covers. CNC 5052: J1 bearing mount (printable in high-infill ABS), J2-J5 front spacers (printable in ABS), J2-J4 rear spacers and flanges, wrist J5 bracket, gripper connectors A/B, slider bracket ("printable but not recommended for long-term use"), Link1/2/3/5 in CNC plus sheet metal | DM readme v1.1 |
| Production-vs-BOM delta | "some 3D printed parts will be replaced with metal for durability"; v1.1 added a base reinforcement part because base rigidity was insufficient | DM readme |
| Third-party URDF check | Viam port measured FK drift of 1.045 mm axis offset for the RS URDF vs 0.084 mm for DM; they keep gravity scale at 0 on RS until checked on the bench | viam-devrel PR #3 |

The takeaway from the printed-vs-metal rows: the parts Seeed moved to metal are the base and J1, the J2-J4 motor spacers and flanges, and the gripper slider. Those are the places where printed plastic lacks margin. Keep them metal in the clone.

## 1. BOM

### A. Reuse from the stock B601-RS
| Part | Qty | Approx. price | Note |
|---|---|---|---|
| RobStride RS06 | 3 | ¥849 list (about $120); retail about $150-200 | J1-J3 |
| RobStride RS00 | 4 | ¥598 (about $85); retail about $110-140 | J4-J6 and gripper |
| XT30(2+2) daisy-chain harness | 7 segments (DM pattern: 350 mm x 3, 200 mm x 4) | $3-4 each | Mating connectors: XT30PB(2+2)-M board side / XT30(2+2)-F cable side |
| Bearings (RS assumed same as DM) | 6707ZZ x1 (J1 radial), 6803ZZ x3, AXK5578 thrust x1 (J1) | $12-13 each | **[verify against RS BOM]** |
| Gripper | MGN9 170 mm rail x1 + 2 blocks; module-1 16T gear (6 mm bore, boss type); 2 racks | $23 + $20 + $44 | |
| Machined 5052 parts kept (see list above) | Set | About $250 (DM reference) | Metal, not printed |
| Fasteners | M3 socket head 6/12/25 mm; M3 countersunk 7/9/12/16 mm; M4x75 mm; M4 dowels 8/12 mm; M3 low-profile head; KA3x12 self-tappers | About $30 | Use medium-strength thread-locker; tighten to 3-6 kgf·cm (per Seeed) |

### B. Custom printed replacements
Masses and print times are estimates for a 0.4 mm nozzle and 0.2 mm layers **[estimate]**.

| Part | Material | Walls / infill | Mass | Time |
|---|---|---|---|---|
| Base plate + base link | PA-CF (or PC-CF) | 6 walls, 50% gyroid | 250-350 g | 10-14 h |
| Upper arm structure (replaces Link2 fillers and cover; Link2 sheet metal kept at first) | PA-CF | 5 walls, 40% | 150-250 g | 8-10 h |
| Lower arm (Link3 L/R fillers and cover) | PA-CF | 5 walls, 40% | 120-200 g | 6-8 h |
| Arm handle, limit blocks, motor-5 cover, cable restraints | PETG / ASA | 3 walls, 20-30% | 80 g total | 5 h |
| Fingers x2, rail bracket | PA-CF | 100% at the jaws | 40 g | 3 h |
| Harness clips | PETG | | 10 g | 1 h |

### C. New additions
| Part | Spec | Approx. price |
|---|---|---|
| CAN adapter | CANable 2.0 with candleLight (gs_usb) firmware, which shows up as `can0` in SocketCAN; stock firmware is slcan. Note: STM32G431 "CANable-MKS 2.0" boards are not supported by candleLight_fw | $25-40 |
| CAN termination | 120 Ω at both bus ends (the adapter DIP switch counts as one) | $1 |
| PSU | Mean Well LRS-350-48 (7.3 A) is marginal. Recommended: LRS-600-48 or a 48 V 12.5-15 A unit | $35-90 |
| Regen protection | RobStride discharge module (in their repo), or an adjustable shunt/brake chopper clamping at about 54-56 V into a 10-20 Ω / 50 W resistor, plus a 1000-2200 µF 63 V bulk capacitor at the bus | $15-40 |
| E-stop | 22 mm mushroom, NC, driving a 48 V DC contactor or SSR rated at least 20 A DC. Place it after the bulk capacitor so regen still has somewhere to go, or cut AC and keep the shunt live | $30-60 |
| Fuse | 15-20 A blade fuse on the 48 V line | $5 |
| Table clamps | 2x 6-inch G-clamps (≥ 3-inch per Seeed) | $40 |
| J2 balancer | 2x extension springs about 1.5-2 N/mm with Dyneema cable over a Ø20 mm idler (zero-free-length equivalent), M5 anchor pins, 625ZZ pulleys | $30 |
| Heat-set inserts | M3 (Ø4.0 mm hole), about 150 pcs; M4 (Ø5.6 mm hole) | $20 |

### DM variant delta
Swap to 3x DM4340P V4 (about $175 each) on J1-J3 and 4x DM4310 V4 (about $120 each) on J4-J6 plus gripper. Use a 24 V PSU (LRS-350-24, 14.6 A) and the DM CAN-USB board ($15). Rated payload drops to 1.5 kg. Bearings (6707ZZ, 6803ZZ x3, AXK5578), the MGN9 rail and the 16T gear are the same as listed. The DM motors have different bolt circles, so the printed motor pockets must be parametric per actuator.

## 2. Engineering specification

### Load model
The self-weight numbers below are derived from the 6.7 kg total and actuator masses **[estimate]**. Replace them with the URDF inertials.

- J2 to J3 distance ≈ 0.26 m; J3 to wrist ≈ 0.25 m; wrist to TCP ≈ 0.24 m.
- Distal mass beyond J2 ≈ 3.3 kg: RS06 at J3 (0.62 kg), 4x RS00 (1.24 kg), structure about 1.4 kg.

| Case (arm horizontal, full reach) | J2 τ required | vs rated 11 N·m | vs peak 36 N·m |
|---|---|---|---|
| Self-weight only (COM ≈ 0.33 m) | ≈ 10.7 N·m | 1.03x (0.97 margin) | 3.4x margin |
| + 1.0 kg at 0.72 m | ≈ 17.8 N·m | 0.62x | 2.0x |
| + 2.5 kg | ≈ 28.4 N·m | 0.39x | 1.27x |
| + 5 kg | ≈ 46 N·m | 0.24x | 0.78x (infeasible) |
| 70% reach (0.53 m), 2.5 kg, self-weight moment about 7.5 N·m | ≈ 20.5 N·m | 0.54x | 1.76x |
| Full reach, 2.5 kg, with balancer cancelling 8 N·m | ≈ 20 N·m | 0.55x | 1.8x |

Required torque at the other joints, full reach, 2.5 kg:
- J3: ≈ 2.5·9.81·0.49 + ≈ 2.2 N·m ≈ 14 N·m, above rated by about 1.3x.
- J5 (RS00): 2.5 kg at about 0.12 m (wrist to grasp) is ≈ 3 N·m plus gripper ≈ 1 N·m, so 4 N·m vs 5 N·m rated (1.25x margin). Longer fingers or offset tools exceed this.
- J1: horizontal, so gravity-free. Dynamic only: inertia ≈ 1.5-2 kg·m² with 2.5 kg at 0.72 m; at 2 rad/s² that needs 4 N·m, which is fine.

**Conclusion:** continuous payload at full reach without a balancer is about 0 kg (J2 is thermally limited just by the arm's own weight). With a balancer it is about 1.0 kg continuous, or 2.5 kg at 20-40% duty. Seeed's 2.5 kg figure is effectively a "70% workspace, intermittent" rating. These figures assume rated torque is available, which depends on heat sinking.

### Thermal derating
RobStride rated torque assumes an aluminium heat-sink plate, and an RS06 mounted into a printed link has none. Assume 60-70% of rated torque continuously, about 7 N·m **[engineering estimate; get the plate size from the PDF]**. Resistive loss scales with current squared: at 11 N·m, about 10 Arms phase current, so I²R ≈ 1.5·(10²)·0.23 ≈ 35 W. That is why J2 must be mounted to the retained aluminium flange and spacer, which conduct heat into the Link2 sheet metal. Mitigations:
- Keep metal contact at motor faces, with thermal pads.
- Add ventilation slots in the shells.
- Gate on the 80/100/140 °C thresholds.

### J2 gravity balancer
With a zero-free-length spring anchored at height a above J2 and attached at distance b along the upper arm, the spring torque is k·a·b·cosθ. That exactly cancels a self-weight moment of m·g·r·cosθ when **k·a·b = m·g·r**. It is exact only for the upper arm alone; the forearm angle (J3) changes the moment, so the spring can cancel only the constant part.

Worked sizing:
- Target the J3-folded average moment of about 8 N·m.
- With a = 0.06 m and b = 0.10 m, k·a·b = 8 gives k ≈ 1.33 N/mm.
- Realise zero free length with a cable over an idler at the anchor point, with the spring body parked in the base.
- Peak spring force ≈ k·(a + b) ≈ 213 N. That load goes into the base and J1 bearing stack, so the anchor must be metal.

Feed the spring term into gravity compensation: τ_ff = g(q) − τ_spring(q2). Model the spring as an MJCF `tendon` plus `spring`, or as a custom passive force, so MuJoCo and Pinocchio agree.

### Power budget
- Continuous: J2 plus J3 at rated ≈ 2x35 W copper loss + mechanical ≈ 100-150 W total.
- Peak: two RS06 at 57 Apk ≈ 2x(36 N·m x 2 rad/s ≈ 72 W mechanical + about 1.1 kW I²R at peak current). Bus current is lower than phase current, but expect short 15-25 A transients. The 48 V 15 A supply is adequate with a bulk capacitor; the LRS-350-48 is marginal.
- Regen: stopping the loaded arm falling from 1 rad/s ≈ ½Iω² ≈ 1-2 J plus potential energy; a gravity drop of 3.3 kg by 0.3 m ≈ 10 J.
  - 10 J into 2200 µF raises the bus from 48 V to √(48² + 2·10/0.0022) ≈ 115 V, which exceeds the 60 V overvoltage fault.
  - So a shunt clamp is mandatory. Switch-mode supplies cannot absorb reverse current, and the RS06 manual warns that damping mode generates power.

### Control specification
- **Bus:** CAN 2.0 at 1 Mbps with extended frames. One MIT command plus reply ≈ 2x ~130 bits ≈ 260 µs per motor. Seven motors ≈ 1.8 ms, which works out to about 500 Hz at about 90% bus load.
- **Rates:** use 250-400 Hz for about 50-70% load. Seeed runs 200 Hz. Beyond that, use a second bus (J1-J3 and J4-J7 on separate adapters).
- **Mode:** MIT with τ_ff = g(q) − τ_spring + friction feedforward.
- **Protection:** set CAN_TIMEOUT to about 50-100 ms (1000-2000 counts) so a host crash disables the motors. Poll temperature at ≥ 2 Hz and watch fault bit14 (stall).
- **Power loss:** there are no brakes. Rely on the balancer (which halves the drop torque), the unpowered anti-backdrive damping, and a mechanical J2 rest pad. Add a "park" pose on shutdown, as in the contributed ROS2 safe-park.

## 3. Mechanical specification
- **Kinematics:** 6R chain plus gripper. The joint origins must be copied from the `urdf/RS` package at a pinned commit; I did not retrieve the numeric DH parameters. Generate the MJCF from the same parameter file using `mujoco.MjSpec` and assert equality against the stock URDF with Pinocchio FK. Note the Viam finding that the RS URDF has a 1.045 mm axis offset relative to its meshes, so treat the URDF as the reference and the meshes as secondary.
- **Load paths:** J1 overturning moment at full reach with 2.5 kg ≈ (3.3·0.33 + 2.5·0.72)·9.81 ≈ 28 N·m, plus balancer preload. Carry it through the AXK5578 thrust bearing and the 6707ZZ radial bearing into the metal bearing mount, not through the RS06 output bearing (its moment rating is unpublished **[UNVERIFIED]**).
- **Materials:**
  - PETG: HDT 65-75 °C, keep below about 60 °C long-term, and it creeps. Use it only for covers.
  - PETG-CF: HDT 74-90 °C.
  - PA-CF: modulus about 5-8 GPa, HDT above 150 °C annealed. Dry it before printing. This is the preferred structural material.
  - PC-CF: similar performance.
  - Motor housings can exceed 60-80 °C, so avoid PETG and PLA within 10 mm of an actuator.
- **Design rules:**
  - Walls ≥ 5 perimeters (2.0-2.4 mm) on load paths.
  - M3 insert hole Ø4.0 mm, depth = insert length + 1 mm, boss outer diameter ≈ 8-9 mm (≥ 1.6 mm wall).
  - Bearing seats: model at nominal +0.05-0.10 mm for a light press in PA-CF, test with a coupon first, or use a metal sleeve.
  - Pilots and slip fits: +0.15-0.2 mm.
  - Orient layers so bending stress runs along the layers, never across Z.
  - Bolt motors through metal spacers, not plastic threads.
- **Deflection:** a printed PA-CF tube section at E ≈ 6 GPa vs 5052 aluminium at 70 GPa is about 12x less stiff for the same section. Give the section 2-3x the stock depth to keep tip deflection under 0.5 mm at 2.5 kg. A Link2 cantilever of 0.26 m needs EI ≥ F·L³/(3δ) ≈ 25·0.0176/(3·0.0005) ≈ 290 N·m². Keeping the sheet-metal Link2 and Link3 in the first builds is recommended.
- **Cable routing:** keep the XT30 daisy chain in the DM harness pattern, with clips at motor 1 (Seeed notes long-term connector abrasion) and strain relief on J5-J7.

## 4. Requirements and verification
| ID | Requirement | Verification |
|---|---|---|
| F1 | FK of the clone model equals the stock URDF to ≤ 0.1 mm / 0.05° over 10,000 random configurations | pytest with Pinocchio |
| F2 | Only `<inertial>` blocks differ from stock in the URDF/MJCF | diff test |
| P1 | Each custom link mass ≤ the stock part it replaces; total ≤ 6.7 kg | CAD mass + scale |
| P2 | J2 static torque at full reach with target payload, minus spring torque, ≤ 0.7x derated rated torque (about 7 N·m) | analytic pytest + bench current reading |
| P3 | Tip deflection ≤ 0.5 mm under 2.5 kg at 70% reach | dial indicator |
| S1 | Minimum 3 mm clearance through all joint ranges | MuJoCo collision sweep |
| S2 | Bus stays ≤ 58 V during an emergency-stop drop test with max payload | oscilloscope |
| S3 | CAN_TIMEOUT ≤ 100 ms; e-stop cuts drive power within 50 ms | test |
| T1 | J2 holding 1 kg at full reach for 20 min stays below 80 °C; within 5 °C of the stock baseline after each part swap | thermal log |
| I1 | RS06/RS00 bolt circles and pilots match the STEP files ±0.1 mm | CMM or gauge print |
| V1 | Repeatability ≤ 0.2 mm over 50 cycles | dial indicator |

**Swap order:**
1. Assemble the stock arm and record the baseline: FK/camera, T1, current per joint, gravity-compensation scale factors.
2. Swap the gripper fingers and rail bracket.
3. Swap the J5-J6 covers and cable restraints.
4. Swap the lower-arm fillers and cover.
5. Swap the upper-arm fillers and cover.
6. Install the balancer, and re-tune tau_g before the next swap.
7. Swap the base link and plate.
8. Only then consider printed Link2 or Link3.

Repeat V1, T1 and the P2 current check after each step, and roll back if any metric degrades by more than 10%.

**Scaling to a 1 m / 2 kg arm:** J2 load scales roughly with L². At 1 m, the self-weight plus a 2 kg payload moment ≈ 35-45 N·m continuous. That is RS04 class (40 N·m rated with a 345 mm heat-sink plate) or an RS06 with a large balancer. Keep the actuator a parameter in the code-CAD design.

## Caveats
- The RS BOM page, the URDF numbers and the RobStride spec PDF were not retrieved. Bearings, RS06 bolt geometry, heat-sink sizes and link masses must be confirmed from the repo and STEP files before cutting any metal.
- Torque cases use estimated mass distribution.
- Seeed gives conflicting figures: PSU 12.5 A vs 15 A, and repeatability 0.1 mm vs < 0.2 mm.