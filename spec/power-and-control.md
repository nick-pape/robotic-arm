# Power and Control Electronics: Architecture, Budgets, and Requirements

**Bottom line:** The arm needs one 48 V bus and one CAN bus. Both travel together through a seven-segment XT30(2+2) daisy chain, from the base to the gripper. Everything else goes off the arm, in a control box, on two custom boards. The **power board** takes 48 V in. It holds the fusing, inrush limiting and switching, the regen clamp, the bulk capacitance and the arm output. The **logic board** sits under a Jetson Orin Nano Super dev kit. It provides the Jetson's 19 V supply, the CAN transceiver, bus telemetry, and the E-stop and enable interface. Two failures matter more than anything else. A supply that is too small, or a protection threshold that is too tight, drops the bus. That resets every motor, and **the arm falls, because there are no brakes**. The other is regen above 60 V, which faults every motor. Most of the requirements below exist to prevent one of those two.

This builds on `spec/rebot-arm-b601-rs-clone.md`, which covers the actuators, the arm's own power budget and the control rates. Values not confirmed from a source are marked **[UNVERIFIED]**. Jetson facts come from NVIDIA's *Jetson Orin Nano Developer Kit Carrier Board Specification*, SP-11324-001 v1.3 (hereafter "SP-11324").

## TL;DR
- **Bus:** 48 V from a 12.5–15 A supply. The RS00's 24 V minimum rules out 24 V, and the 60 V overvoltage fault caps the top end. **Set a current limit on each motor so the supply size follows from those limits.** Never size the supply to the worst case on paper (~65 A).
- **Topology:** a single daisy chain carrying power and CAN together, J1 → J6 → gripper. Put 120 Ω at the logic board and at the gripper. Only the first segment carries the whole arm's current.
- **Brain:** the Orin Nano Super dev kit, powered at 19 V through its barrel jack from **before** the motor contactor. Motor CAN uses the Orin's built-in controller (mttcan) on the carrier's **J17** header, with the transceiver on the logic board. A USB CAN adapter is the fallback, and it plugs into the same board connector.
- **Safety:** the E-stop cuts the motor bus in hardware, and releasing it does not restart anything. The Jetson can only *request* enable. A brake chopper at ~54 V is mandatory. All 48 V parts are rated ≥ 80 V.
- **Not a HAT:** a HAT on the dev kit's 40-pin header would sit directly over the heatsink fan. Instead, the logic board is a base plate the dev kit stands on, connected by short cables to J12 (40-pin), J14 (button header), J16 (barrel jack) and J17 (CAN).

## Key Findings (sourced)
| Item | Value | Source / status |
|---|---|---|
| Actuator supply | RS06 15–60 V; RS00 24–60 V; undervoltage fault 12 V, overvoltage fault 60 V | RS06/RS00 manuals, via the arm spec |
| Actuator connector | XT30PB(2+2): two power pins plus CAN_H/CAN_L | RS00 manual, via the arm spec. Whether each motor has **two** ports (pass-through) is **[UNVERIFIED]**. The stock harness's 7 segments for 7 motors implies it does |
| Winding constants | RS06 Kt 1.09 N·m/Arms, 0.23 Ω line; RS00 Kt 1.48 N·m/Arms, 1.5 Ω line | `src/robotic_arm/thermal.py` |
| Dev kit DC input | J16 barrel jack, 9–20 V, 19 V supply provided; 5.5 mm OD / 2.5 mm pin, centre positive | SP-11324 §3.8. The 3.5 A max is from a secondary source **[UNVERIFIED]** |
| Alternate power input | J18 backpower header, 9–20 V, **3 A max**, 1×2 2.54 mm | SP-11324 Table 3-10 |
| CAN on the carrier | J17, **an unpopulated footprint**: 1×4, 2.54 mm. 3.3 V logic. **No transceiver on the carrier** | SP-11324 Table 3-5 |
| J17 pin order | SP-11324: 1 CAN_TX, 2 CAN_RX, 3 GND, 4 3.3 V. NVIDIA's Jetson Linux CAN guide says pin 1 is CAN_RX and pin 2 is CAN_TX | **Conflict.** Verify on the board before committing the logic board's J17 cable (requirement E14) |
| mttcan pin setup | Orin Nano: can0_din 0x0c303018 ← 0xc458, can0_dout 0x0c303010 ← 0xc400 | Jetson Linux Developer Guide, CAN chapter. Forum reports of mttcan trouble on JetPack 6 builds mean the fallback must stay viable |
| Button header J14 | Pins 5–6 shorted = auto power-on disabled. 7–8 shorted = reset (SYS_RESET\*, **1.8 V** input). 9–10 = force recovery (1.8 V). 11–12 = power-on (5 V). 3/4 = debug UART2 | SP-11324 Table 3-4 |
| 40-pin GPIO | Most GPIOs go through **TXB0108** auto-direction level shifters. The output drivers are "very weak" (~4 kΩ in series), 3.3 V levels, ±20 µA rated drive | SP-11324 Table 3-3, note 3 |
| 40-pin I²C | I2C1 on pins 3/5 and I2C0 on pins 27/28. Wired straight to the SoC, open-drain, ±2 mA, with on-board pull-ups (2.2 kΩ on I2C1, 1.5 kΩ on I2C0) | SP-11324 Table 3-3, note 2 |
| 40-pin power | Pins 2/4 5 V and pins 1/17 3.3 V, 1 A per pin | SP-11324 Table 3-3. Never use these to back-power the dev kit; it takes 9–20 V only |

## 1. Architecture

```
                           CONTROL BOX                                         │  ARM
                                                                               │
 AC ─ PSU 48 V 12.5–15 A ─┬──────────────── POWER BOARD ────────────────────┐ │
                          │  input TVS ─ main fuse 20 A                     │ │
                          │     ├─ aux fuse 3 A ─────────── 48V_AUX ──────┐ │ │
                          │     └─ hot-swap (back-to-back FETs, inrush     │ │ │
                          │          + current limit, EN ← E-stop loop)    │ │ │
                          │            │                                   │ │ │
                          │        ARM BUS: bulk 2×1000 µF/80 V            │ │ │
                          │                 brake chopper @ 54 V ─► R ext  │ │ │
                          │                 INA228 shunt                   │ │ │
                          │                 arm fuse 15 A                  │ │ │
                          │            │                                   │ │ │
                          │   XT30PB(2+2) out ◄── CAN_H/L ◄──┐             │ │ │
                          └────────────┼─────────────────────┼─────────────┘ │ │
                                       │             inter-board cable       │ │
                          ┌─ LOGIC BOARD (Jetson stands on it) ─────────────┐ │
                          │ 48V_AUX ─ 36–75 V→19 V buck 60 W ─► J16 barrel  │ │
                          │ CAN transceiver + 120 Ω ◄─► J17 (CAN_TX/RX)     │ │
                          │ CAN fallback header (USB adapter)               │ │
                          │ I²C ◄─► J12 pins 3/5 ; buffered GPIO ◄─► J12    │ │
                          │ E-stop loop, enable latch, J14 reset/power      │ │
                          └─────────────────────────────────────────────────┘ │
                                       │                                       │
           E-stop (NC, 22 mm) ─────────┘                                       │
                                       │                                       │
           box-to-base cable ──────────┴────► J1 ─ J2 ─ J3 ─ J4 ─ J5 ─ J6 ─ gripper [120 Ω]
                                              RS06 RS06 RS06 RS00 RS00 RS00 RS00
```

### Why two boards, and why neither is a HAT
- **What goes on the power board:** two 1000 µF / 80 V caps, an XT60 input, an XT30PB(2+2) output, a PCB fuse holder and the hot-swap FETs. None of these fit in a 65×56 mm HAT. The brake resistor has to go off-board on a heatsink anyway.
- **Keep 48 V away from the brain:** ~30 A switching edges and regen spikes don't belong next to the dev kit's GPIO.
- **Different revision rates:** the power board's trip thresholds come from bench measurements (§7), so it will be respun. The logic board should settle sooner.
- **Why not a HAT:** on the dev kit, a board plugged into the 40-pin header sits right over the heatsink fan, which blocks airflow at 25 W. Instead, the logic board is a base plate the dev kit stands on using its own mounting holes. It connects to J12, J14, J16 and J17 with short cables.

## 2. Power Budget

These figures are bus current at 48 V, computed from the winding constants and the torque budget (`uv run python -m robotic_arm.torque`). Holding a pose does no mechanical work, so the bus supplies only the I²R loss, even when the windings carry 14 A RMS. Driver overhead is ignored and treated as margin.

| Case | Breakdown | Bus |
|---|---|---|
| Idle, upright home | Drivers only | < 0.5 A **[UNVERIFIED: per-motor quiescent draw]** |
| Worst static pose, no payload, no balancer | J2 15.36 N·m → 14.1 Arms → 68.5 W; J3 7.05 N·m → 14.4 W; J4 1.92 N·m → 3.8 W | **~1.8 A** (87 W) |
| Same pose, 1 kg payload | J2 22.9 N·m → 21.0 Arms → 153 W (over rated torque, so time-limited) | ~3.5–4 A |
| One RS06 at peak current, 3 rad/s | 57 Apk = 40.3 Arms → 560 W loss + 108 W mechanical | ~14 A |
| One RS00 at peak current, 3 rad/s | 15.5 Apk = 11.0 Arms → 270 W loss + 42 W mechanical | ~6.5 A |
| J2 and J3 at peak together | | **~28 A** for tens of ms |
| All seven at peak | Only possible on paper | ~65 A |
| Jetson Orin Nano Super, MAXN Super + USB peripherals | Through the 19 V buck at ~90% efficiency | ~0.7–1.0 A |

**Correction to the arm spec:** its power budget gives "about 1.1 kW I²R at peak current" per RS06. That figure treats 57 A peak as if it were RMS. The correct value is 40.3 A RMS, so 1.5 × 40.3² × 0.23 ≈ **560 W**. The arm spec's conclusion still holds: 15 A is enough with bulk capacitance.

**Sizing rule:** set each motor's current limit (`limit_cur`) so that the combinations that can actually happen together fit inside the supply. Pick the supply, the hot-swap controller's current limit and the fuses from those limits, not from the theoretical sum. As a starting point: J2/J3 at 70% of peak and the others at rated, then revise from the bench measurements in §7.

**Why headroom matters:** most LRS-class supplies react to sustained overload by shutting off and retrying. The bus then falls below the 12 V undervoltage threshold, every motor resets, and the arm falls. So an undersized supply is a safety fault, not just a performance limit.

## 3. Power Board

### 3.1 Regen and the clamp
The motors push energy back into the bus when they decelerate, and also when the RS06's unpowered damping is active. A switch-mode supply can't absorb that. The arm spec estimates a ~10 J worst case, from dropping 3.3 kg by 0.3 m. Between 48 V and 56 V, 2200 µF absorbs only ½·C·(56² − 48²) ≈ **0.9 J**. The rest goes to the chopper.

- **Chopper:** a comparator with hysteresis drives a low-side MOSFET into an external 10–15 Ω, ≥ 50 W resistor. It turns on at 54 V and off at 52 V. At 54 V into 10 Ω that is ~290 W instantaneous, so the resistor's rating is set by duty cycle and must be checked against the E-stop drop test (S2). RobStride's discharge module is an acceptable bench stand-in.
- **Location:** on the arm side of the hot-swap switch. The motors are what generate the energy, and the clamp has to keep working after the switch opens.
- **Supply trim:** the supply's output trim must stay ≤ 50 V. Any higher and the chopper fires on the supply's own output.
- **Secondary TVS:** fit one at the board input for surges. It is not the regen clamp: its clamping voltage is far above 60 V.

### 3.2 Switching, inrush and E-stop
- **Instead of a contactor:** a hot-swap controller (an 80 V part, e.g. TI LM5069) with back-to-back MOSFETs. It replaces the contactor and the precharge relay. It limits inrush into the bulk caps and the motors' input capacitance, and its timed current limit doubles as electronic overcurrent protection.
- **Trip threshold:** the timed current limit must ride through the measured acceleration peaks (~28 A for tens of ms). If it trips during normal motion, it causes the arm drop it exists to prevent. Set the threshold from §7 measurements, never from a guess.
- **E-stop loop:** a 22 mm NC mushroom switch, wired as a hardware loop. Opening it pulls the hot-swap controller's enable low directly, with no firmware in the path. This is a Category 0 stop. The balancer and unpowered damping slow the fall but don't stop it. A software-commanded stop followed by a delayed cut (Category 1) is a possible future upgrade, not a replacement.
- **No restart on release:** releasing the E-stop must not re-energise the bus. Re-enabling needs a deliberate action: the Jetson's ENABLE_REQ edge, or a front-panel reset button.
- **No live plugging:** the XT30 must never be connected or disconnected with the bus live. That is a procedure, and the box should state it on a label.

### 3.3 Protection and parts ratings
| Branch | Protection | Rating |
|---|---|---|
| Board input | Main fuse 20 A slow-blow, PCB holder; input TVS | Above the supply's 12.5–15 A, below the 20 A connector/wiring rating |
| Arm output | Arm fuse 15 A; hot-swap current limit | XT30 is ~15 A continuous |
| 48V_AUX to the logic board | Fuse 3 A | Upstream of the hot-swap, so the Jetson stays powered through an E-stop |

**Rate everything on the 48 V side for ≥ 80 V.** That covers capacitors, MOSFETs, controllers and eFuses. 60 V-class eFuses are **not** acceptable: the chopper holds the bus at ~54–56 V, which leaves no margin. Use eFuses only on the 19 V side.

### 3.4 Telemetry
Fit an INA228 on the arm bus, after the hot-swap. It measures bus voltage, current, power and charge over I²C. It provides the bench data for §7, the S2 bus-voltage trace, and live current logging to compare against the §2 budget. The sense resistor goes in the high-current path on the power board, and the I²C runs to the Jetson over the inter-board cable.

## 4. Logic Board

### 4.1 Jetson power
- **Converter:** 36–75 V input to 19 V output, ≥ 60 W (e.g. a 100 V synchronous buck controller such as LM5146), with a 19 V eFuse on the output. It feeds J16 through a 5.5/2.5 mm barrel lead, centre positive. Don't use J18, which is an unpopulated footprint limited to 3 A.
- **Supply source:** fed from 48V_AUX, upstream of the hot-swap switch. An E-stop never reboots the Jetson.
- **Board logic rails:** the board's own 3.3 V comes from 48V_AUX, never from the Jetson's 40-pin power pins.

### 4.2 CAN
- **Transceiver:** a 3.3 V-logic CAN transceiver (TCAN1042V-class, with VIO tied to 3.3 V) on the logic board, connected to J17 by a short cable. The J17 header has to be soldered onto the carrier.
- **Bus protection:** TVS on CAN_H/CAN_L, plus a 120 Ω terminator on a jumper. The logic board is one end of the bus and the gripper is the other.
- **Fallback:** a 4-pin CAN header (CAN_H, CAN_L, GND, shield) wired in parallel with the transceiver's bus side. If mttcan bring-up fails on a given JetPack, cut the J17 cable, plug a candleLight/gs_usb adapter in here and turn its own terminator off. No board change is needed.
- **Isolation:** not used. The 19 V converter is non-isolated, so an isolated transceiver would isolate nothing. Instead, the system has one ground with a single star point on the power board's output.

### 4.3 40-pin interface (J12): proposed assignment
These are all 3.3 V. Pin choices favour pins whose power-on default is pull-down, so an unconfigured Jetson reads as "not requesting enable".

| J12 pin | Signal | Dir (from Jetson) | Notes |
|---|---|---|---|
| 3 / 5 | I2C1 SDA / SCL | bidir | Wired straight to the SoC, with 2.2 kΩ pull-ups already on the carrier. Add **no** extra pull-ups. INA228 plus an ID EEPROM |
| 29 | ENABLE_REQ | out | GPIO01, pull-down at power-on. Goes into a high-impedance CMOS input. A rising edge re-arms the bus only if the E-stop loop is closed |
| 31 | ESTOP_OK | in | GPIO11, pull-down at power-on (reads as "not OK" if unconnected) |
| 33 | BUS_OK | in | GPIO13. Hot-swap power-good and bus within 40–56 V |
| 15 | FAULT_N | in | GPIO12. Wired-OR of INA228 ALERT, eFuse fault and chopper over-temperature |
| 6, 9, 14, 20 | GND | | |

**Level-shifter caveat:** pins 15, 29, 31 and 33 go through the carrier's TXB0108 shifters. Every signal into the Jetson must come from a 3.3 V push-pull buffer (e.g. 74LVC1G125). Pull resistors on these lines must be ≥ 50 kΩ, and nothing may load ENABLE_REQ.

### 4.4 Button header (J14)
- **Pins 7–8, SYS_RESET\*:** an open-drain MOSFET or optocoupler shorts these to reset a hung Jetson. The signal is **1.8 V**, so never drive it from 3.3 V logic. The trigger is a front-panel button, not an automatic watchdog, until a watchdog policy exists.
- **Pins 5–6, auto power-on disable:** leave these open. The Jetson boots whenever 19 V is present.
- **Pins 3–4, debug UART2:** bring these out to a header for console access.

### 4.5 Enable logic
The E-stop loop, the enable latch and the BUS_OK window comparator are discrete logic: a flip-flop and comparators, not a microcontroller. That keeps firmware out of the safety path. Behaviour:
1. The E-stop loop opens → the hot-swap is disabled at once and the latch clears.
2. The loop closes → the bus stays off.
3. The loop is closed and ENABLE_REQ rises (or the reset button is pressed) → the latch sets, the hot-swap soft-starts, and BUS_OK goes high once the bus is in its window.

### 4.6 Inter-board cable
A single locking connector (e.g. Micro-Fit 3.0, 2×6):

| Signal | Direction |
|---|---|
| 48V_AUX, GND (×2 each) | power → logic |
| CAN_H, CAN_L (twisted pair) | bidirectional |
| I2C SDA, SCL | bidirectional (logic board is master) |
| HS_EN (from the latch) | logic → power |
| PG, FAULT_N | power → logic |
| ESTOP loop A / B | through both boards |

## 5. Arm Harness

- **Topology:** the stock DM pattern. Seven XT30(2+2) segments, ~350 mm × 3 and ~200 mm × 4, with the CAN pair twisted.
- **Wire:** 16–18 AWG high-strand silicone for power, 24–26 AWG for CAN. The box-to-base cable is 16 AWG; at 15 A over 2 m round trip that drops ~0.4 V.
- **Current by position:** the base-to-J1 segment carries the whole arm. Past J3, only the RS00s draw current.
- **Termination:** 120 Ω at the gripper end. **[UNVERIFIED]** whether RobStride motors have a switchable built-in terminator. If they do, enable it on the gripper only.
- **Joint travel:** cable wrap limits joint travel. There are no slip rings and no hollow shafts, so each joint needs a service loop sized for ±180°. That matches `jointlimits.TOTAL_CAP`.
- **Strain relief:** clamp the harness on both sides of every joint. Seeed reports connector abrasion at motor 1.

**Gaps in the CAD this depends on**
1. `cable_channel()` (`src/robotic_arm/parts/cobot.py:468`) is defined and never called. No link has a cable path yet.
2. The cable bore in `src/robotic_arm/parts/tool_flange.py:73` assumes a hollow J6 shaft. The RS00 isn't hollow (`cobot.py:471`). The gripper cable has to wrap around the outside of the J6 barrel.
3. Housings are sized to the motor body without its connectors (`src/robotic_arm/linkframes.py:351`), and the service cap (`parts/servicecap.py`) closes over the back of each motor. Each motor's connector position and port count still need to be read from `reference/step/RS06-new.step` and `RS00.step`. If the ports are on the back face, each cap needs a grommeted exit with room for the connector and a bend in the wire.

## 6. Control Interface
- CAN 2.0, 1 Mbps, extended frames, MIT mode. Seven nodes give ~500 Hz at ~90% bus load, so run 250–400 Hz (arm spec). A second bus isn't possible, because the XT30(2+2) carries one CAN pair.
- Motor CAN_TIMEOUT 50–100 ms, so a crashed Jetson disables the motors.
- Poll temperature at ≥ 2 Hz and watch fault bit 14 (stall). Log the INA228 alongside.

## 7. Bring-up Plan

1. **Phase 0, modules on the bench.** Build the architecture from off-the-shelf parts: an LRS-600-48, a contactor with a precharge resistor, RobStride's discharge module, inline fuses, a USB CAN adapter, and an INA228 breakout. Run the stock control stack and log bus current and voltage through representative motions, the worst static pose and an E-stop drop with maximum payload. **Outputs:** measured peak currents and durations, the regen peak voltage, and per-motor quiescent draw. These replace the estimates in §2 and set the hot-swap current-limit timer, the fuse ratings and the brake resistor's power.
2. **Phase 1, logic board rev A** on the Phase 0 power hardware. Settle the J17 pin order (E14), mttcan bring-up, I²C, the GPIOs and the enable latch.
3. **Phase 2, power board rev A**, on a bench with an electronic load and a capacitor bank first, then on the arm.
4. **Phase 3, integration.** Re-run S2, S3 and the P2 current check from the arm spec.

## 8. Requirements and Verification

| ID | Requirement | Verification |
|---|---|---|
| E1 | Bus nominal 48 V; supply trim ≤ 50 V | Meter |
| E2 | Supply, hot-swap current limit and fuses are sized from the configured per-motor current limits, and documented together | Review |
| E3 | No bus dropout (motor undervoltage fault or supply retry) through the Phase 0 motion set at configured limits | INA228 log + motor fault bits |
| E4 | Bus ≤ 58 V during an E-stop drop with maximum payload (= arm spec S2) | Oscilloscope |
| E5 | E-stop removes arm-bus power within 50 ms with no firmware in the path (= arm spec S3) | Oscilloscope on hot-swap gate and bus |
| E6 | Releasing the E-stop does not re-energise the bus | Test |
| E7 | Inrush on enable stays within the hot-swap MOSFETs' safe operating area, and the bus reaches its window in ≤ 200 ms | Oscilloscope + SOA calculation |
| E8 | Every component on the 48 V side is rated ≥ 80 V | BOM review |
| E9 | The Jetson stays powered through E-stop and arm-bus faults | Test |
| E10 | 19 V rail within ±5% from 0 to 60 W | Electronic load |
| E11 | No pull-ups < 50 kΩ and no non-buffered drivers on TXB-shifted J12 pins; no extra pull-ups on I2C1 | Schematic review |
| E12 | Nothing drives J14 SYS_RESET\* above 1.8 V | Schematic review |
| E13 | CAN bus has exactly two 120 Ω terminations (≈ 60 Ω measured H–L with power off) | Meter |
| E14 | The J17 TX/RX pin order is confirmed on the actual carrier before the logic board's J17 cable is made | Scope on CAN_TX during `cansend` |
| E15 | The fallback USB CAN adapter works through the logic board's fallback header with no board change | Test |
| E16 | Motor CAN_TIMEOUT ≤ 100 ms | Config readback + kill-host test |
| E17 | Harness clears all joint ranges with service loops, with no connector under tension at the limits | Range-of-motion test |

## 9. Open Questions
- Each motor's connector count and location (from the STEP files, then on a real motor).
- Whether the RobStride motors have a built-in termination resistor.
- Per-motor quiescent draw and input capacitance. These set the idle budget and the inrush energy.
- The barrel jack's current rating (3.5 A comes from a secondary source).
- The J17 pin order (E14).
- Whether mttcan works on the JetPack version deployed.

## Sources
- NVIDIA, *Jetson Orin Nano Developer Kit Carrier Board Specification*, SP-11324-001 v1.3: <https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/orin_nano/docs/jetson_orin_nano_devkit_carrier_board_specification_sp.pdf>
- NVIDIA, *Jetson Linux Developer Guide*, Controller Area Network (CAN), r36.4: <https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/HR/ControllerAreaNetworkCan.html>
- NVIDIA Developer Forums, "Mttcan does not work on Orin Jetpack 6.0 dp": <https://forums.developer.nvidia.com/t/mttcan-does-not-work-on-orin-jetpack-6-0-dp/283756>
- NVIDIA Developer Forums, "Jetson Orin Nano Dev Kit Power supply": <https://forums.developer.nvidia.com/t/jetson-orin-nano-dev-kit-power-supply/312715>
- Actuator ratings, connector and protections: `spec/rebot-arm-b601-rs-clone.md` and its sources
