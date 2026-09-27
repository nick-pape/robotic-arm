# UR5e audit — ten parallel reviews, 2026-09-26

Ten agents audited the printed arm against UR5e photographs: one per joint
(J1-J6) and one per part (base_link+link1, link2, link3, link4+link5+link6).
Each rendered its subject, looked at the images, and measured in world
coordinates. Their findings are merged below, with duplicates collapsed and
conflicts called out rather than averaged.

The striking result is how much of this the passing test suite could not see.
Three of the archetype-wide defects are invisible to every check in `tests/`:
a fillet that silently does nothing, a motor that cannot be inserted, and a
tube that walks out the end of its own barrel.

## Archetype-wide

### A1. 49 of 50 fillets silently failed — FIXED

`cobot.break_edges` gathered its edge list once from the incoming part, then
iterated it while reassigning `part`. build123d infers a fillet's parent shape
from the edge, so from the second edge onward it re-filleted the *original*
and overwrote the previous result; the bare `except` hid it. Every part ended
up with exactly one fillet.

Found independently by three agents (link2, link3, J3), each reporting "1
TORUS face in the whole solid". This is most of why the arm read as raw
cylinders jammed together. Fixed by filleting the set in one call, with a
re-querying fallback. link2 went 1 -> 50 fillets, link3 1 -> 46.

### A2. No blend where a tube meets a barrel — FIXED

A cylinder meeting a cylinder off-axis produces a **BSPLINE** edge, and
`break_edges` only ever selected `CIRCLE`. So no junction on the arm was ever
a fillet candidate, and `STYLE.shoulder_fillet` was dead code. Added
`urlink._blend_junctions`, applied to the solid union before the cavities are
cut — afterwards the junction curve is interrupted by the shell openings and
OCCT refuses it. Radius raised 2.0 -> 8.0; radii up to 14 take cleanly.

### A3. The motors cannot be inserted — OPEN

`link.intersect(housed_actuator(link))` is **2,551.5 mm3** for motor_5 in
link4 and *identically* 2,551.5 for motor_6 in link5: the 2.0 mm mouth-cap
annulus lies inside the motor's O57 body, overlapping 2.60 mm axially. Worse,
each barrel is capped at both ends and its largest opening is the mouth bore
(O37.64 for an RS00), so an O57 actuator can never go in at all.

Found independently at J5 and J6, so it applies to all six housings.

An earlier check of mine reported "every vendor actuator fits" with 1.5 mm
radial margin. That compared the actuator's vertices against the barrel's
interior extents — containment, not insertability — and never did the
boolean, so it missed the cap clash completely.

Two fixes proposed: seat the motor a wall behind the cap
(`default_housing_length` -> `motor_depth - hub_protrusion + 3*wall`), or move
the cap outboard (`mouth = joint_plane - axis*(wall - hub_protrusion)`,
length -> `motor_depth + hub_protrusion + 2*wall`). Either way the back cap
needs a service opening >= O58.5, or to become a separate printed lid.

### A4. No guard catches a tube walking out of its barrel — OPEN

`build_ur_link` checks whether a tube is too *fat* for the barrel it meets
(`span > drum.length`) but never whether `tube_end_offset` pushes its end
station past the far cap. Missing condition:
`abs(tube_end_offset) + span/2 > drum.length/2 - wall`.

### A5. The degenerate-tangency guard is one wall too loose — OPEN

On both link2 and link3 the first tube diameter exactly equals
`FLANGE_LENGTH - 2*wall`, so the tube grazes the cavity end planes with
0.00 mm margin, removes zero area, and leaves a 2 mm knife rim. The guard
tests `span > drum.length`; it should test `span > drum.length - 4*wall`.

### A6. The loft sags below its own spec — OPEN

link2 measures O43.95 where the design says 44.0 minimum; link3 measures
O43.50 against 44. `lofted_tube` needs `ruled=True`.

## Per joint

| joint | verdict |
|---|---|
| J1 | see base_link / link1 below |
| J2 | **role inverted — see B1** |
| J3 | clean: one continuous O94 barrel 107.1 mm, L/D 1.14, uniform seam, 0 mm3 interference, both links straight (2.6 mm / 0.63 deg, 2.5 mm / 0.60 deg) |
| J4 | clean: one O64 barrel 98 mm, uniform 1.00 mm seam, 0.000 mm lateral offset, 0 mm3 interference |
| J5 | motor side now correct (+Z, matching stock motor_5 at z 265.7-320.8); barrels no longer share space |
| J6 | link6 undersize — see C3 |

### B1. J2's role is inverted, and the archetype cannot express it

`mounts` measures **joint2: link1 = rotor, link2 = stator** — stock link2
carries the stators of *both* J2 and J3, which is why the model ships one
`motor_2_3` mesh on link2. `designed_motor_side` overrides this to force a
uniform "parent holds the stator" convention. That override is wrong: the
motor belongs in the rotating side, bolting back to the vertical column, which
is what the UR5e does.

Removing the override alone only mirrors the problem, because `build_ur_link`
hardcodes flange-at-own-joint / housing-at-child-joint. The correct parts are
**link2 = two housings, no flange** and **link1 = two flanges, no housing**.
End kind must become role-driven, and bolt rings must follow the end kind
rather than its position.

Corrected recurrence, with `o[n] = +1` when the child carries joint n's
housing:

    side[n+1] = -o[n] * o[n+1] * side[n]          (parallel axes only)

The current code is this with `o` hardcoded to -1. Substituting the measured
roles leaves `side[3]` and `side[4]` numerically unchanged, so links 3-6 do
not move; only link1 and link2 rebuild.

## Per part

### C1. link4 is the worst part

* Its tube protrudes **11.39 mm / 825 mm3** past the barrel's far cap, and the
  tube's cavity punches an **open O24 hole into the link's interior** — the
  protrusion and the "open end" are one defect. Cause: `TUBE_END_OFFSET = 30.0`
  against a barrel half-length of 27.5, plus 8.89 mm of rake on the end disc.
  Max safe is ~18.6.
* It **clips link5's flange rim at every J5 angle**: swept over the full range,
  overlap is a constant 1.206 mm3 at -0.04 mm minimum radial clearance, and it
  is rotationally invariant, so no angle clears it. This had been dismissed as
  tessellation noise on the strength of a single pose.
* **Not a wrist.** J4->J5 is 104 mm = 1.63 D with **55.4 mm of exposed O26-28
  diagonal rod** between the barrel surfaces. A UR5e has no such thing.

**Unresolved conflict.** J4 and the wrist-parts agent want the tube *fatter*
(O36-40) for UR proportions; J5 finds the only combination clearing both the
protrusion and link5's rim is *thinner* — `(24, 20, 16)` at offset 24,
verified at 20.8 mm3 protrusion and 0.000 mm3 overlap. Neither fixes the
exposed rod, which needs the neck shortened or faired. This is the one place
the archetype itself looks wrong rather than mis-parameterised.

### C2. Tube/barrel proportions are half, not three-quarters

Measured ratios: link2 0.49/0.47/0.49, link3 0.51/0.47/0.69, link4 0.44.
UR5e photographs read 0.70-0.75. The cause is structural: the span guard caps
tube diameter at the *barrel length*, so O52 is today's ceiling on a O94
barrel. Raising `FLANGE_LENGTH` to 62 on link2 and link3 lifts it; room exists
(link2's Z extent is 55.1 mm against a stock 65.6).

link2's last tube diameter (46.0) also disagrees with link3's first (48.0)
across the J3 seam — a 2.0 mm step.

### C3. link6 is not a flange

O57.0 against link5's O64.0 barrel mouth leaves a **3.5 mm annular ledge** of
open housing rim exposed all round, plus a 1.00 mm axial seam of which 0.60 mm
is bare motor; 665 mm2 of link5's mouth face is uncovered. It should be a real
rotor flange at O64 with a short collar and a O58.5 skirt relief.

It also has a **sub-wall violation**: `bore_to_counterbore` is 1.25 mm against
a 2.0 mm structural wall. `BORE_DIAMETER 18.0 -> 16.5`.

### C4. link1 has a 0.65 mm ligament

Between J2 stator hole 0 and the tube bore: the hole centre sits 15.00 mm from
the tube axis, exactly on the O30 tube surface. The tube axis grazes the O82
BCD at 16.7 mm, so **no clocking fixes it**. The tube also eats 8.3% of the
stator seat annulus and swallows 3 of 6 service ports.

link1 is also "two lumps on a stick": its barrels overlap by 164 mm3 — 0.10%
of the flange, effectively tangent — with 26.4 mm of exposed tube between.

Much of this is moot once B1 lands, since link1 loses its J2 housing entirely.

### C5. base_link is not a UR base

A plain O94 x 65 cylinder on a 140 x 200 rectangle. Proposed: rounded-square
footprint 138 x 138 R20 x 5.0 thick, bolts at (+/-52, +/-52), a truncated cone
O124 at z=5 rising to O94 at z=45 (20.6 deg from vertical, self-supporting),
then a O94 collar to z=72.9. Fits inside the 140 x 200 x 75 stock envelope.

The 8 mm plate is **234.7 g of the part's 282.6 g** while the whole shell is
47.9 g. Thinning to 5.0 mm with six radial ribs saves ~135 g at equal
stiffness, and the part is at 0.24x stock mass, so there is room to spend it
on the cone instead.

## Printability

* Walls are exactly 2.0 mm everywhere, with zero margin, on every link.
* **link2 is 330 mm long and will not fit a 256 mm printer.**
* link1 has no orientation that makes both barrels axial; its axes are 90
  degrees apart.
* Several sealed cavities have unsupported flat ceilings reachable only
  through a bolt bore, so support would be unremovable.
