# Assessment — simulators, hardware, business case

Companion to the reference implementation. Everything cited was verified by
web research on 2026-08-12; URLs inline. Numbers marked [est] are estimates.

---

## 1. Are there better simulators?

**Verdict: no engine replaces the bespoke reduced-ODE core for the GA inner
loop; two are worth adopting around it.** The deciding constraint is batching
*structurally different* candidates: every general engine (MJX, MuJoCo Warp,
Newton, Genesis) batches one compiled topology over parameter variations —
heterogeneous structure means a recompile per candidate or padding to a masked
template. The groups that needed mechanism-search throughput built bespoke
vectorised planar solvers exactly like this one: LINKS generated 100M 1-DOF
mechanisms with a custom solver (arXiv:2208.14567), LInK runs GPU-warm-started
optimisation over a 10M-mechanism dataset (TMLR 2024, arXiv:2405.20592), and
LinkD (CMU, arXiv:2601.04054, Jan 2026) validates candidates by trilateration
sweep — none adopted MuJoCo/Drake/Chrono for the loop. This architecture *is*
the field standard, not a compromise.

The compute argument is equally one-sided: a full run here is 90 × 34 × 1,450
≈ 4.4M vectorised table-lookup steps — 47 s on one CPU core. MJX's own docs
concede it can be ~10× slower than C MuJoCo for a single scene and pays off
only in large homogeneous batches (https://mujoco.readthedocs.io/en/latest/mjx.html).
And every general engine solves loop closure with *soft* constraints — MuJoCo
disallows hard limits by construction (solref/solimp regularisation;
https://mujoco.readthedocs.io/en/stable/computation/index.html), Drake's
four-bar example uses a bushing penalty whose README warns "the stiffer we
make it … the more difficult the problem will be to solve" — whereas the
θ-parameterised reduction is exact: zero drift, zero per-candidate solver
tuning, which matters when a GA emits pathological geometry every generation.
MuJoCo's lead maintainer's advice for linkages (Discussion #1318) — replace
the physical loop with a polynomial joint coupling — is a lower-fidelity
version of the transmission table this engine already builds.

Worth adopting:

- **MuJoCo 3.11 (Apache 2.0) as the finalist verifier.** Import the winner via
  the `mjSpec` model-edit API, close loops with `connect` equalities or drive
  the joint through a fixed-tendon/joint-equality coupling generated *from the
  transmission table*, and re-simulate with contact, flexible tendons and the
  controller in the loop. Divergence between the two simulators becomes an
  acceptance metric, the same role the greedy-vs-QP check plays today.
- **Drake 1.54 (BSD-3) for finalist refinement** — loop-closure-aware
  trajectory optimisation and AutoDiff through `MultibodyPlant` +
  `AddMultibodyPlantConstraints`, one design at a time.
- **Gradients, if wanted, belong in the table, not in an engine swap.**
  JAX-ifying `build_table` gives ∂(RMSE, energy)/∂(anchors, lengths) directly;
  the differentiable-engine field is not a better route (Dojo dormant since
  2024-03; Genesis rigid-body autodiff not shipped; Brax's own pipeline
  deprecated in favour of MJX). Newton 1.2 (Apache 2.0, Linux Foundation,
  MJWarp solver) is the 12-month watch item, and still has no heterogeneous-
  topology answer.
- OpenSim 4.6 Moco only if human-in-loop gait coupling ever enters scope.

Not worth the money for this workload: Simscape Multibody (MATLAB stack
~$940–5,100/yr, bundle scope ambiguous — get a written quote;
https://www.mathworks.com/pricing-licensing.html), Adams (~$40k class,
quote-only), RecurDyn (quote-only). SAM Professional (€765/yr, Artas —
https://www.artas.nl/en/prices) is the one cheap commercial planar tool with
built-in optimisation, useful as a cross-check GUI, not as an engine.

Research current: the 2024–2026 generative wave (LInK, LinkD, JCDE 13(4)
spring-connected graph-attention synthesis) all pair a *learned front-end*
with a fast bespoke evaluator — if this project ever needs warm-started
populations rather than random seeding, that literature bolts onto this
genome format directly.

---

## 2. Hardware

The demo winner (0.089 N·m motor @ 84.5:1, τ_out 5.5 N·m, ω_out ≈ 2 rad/s,
est. 246 g) maps onto real parts with no exotic items. Static capability at
the hand: 5.5 N·m / (9.81 × 0.27 m) ≈ **2.1 kg dead lift** — above the Utah
Arm 3+'s 1 kg "live lift" (Fillauer 2025 comparison sheet,
https://media.fillauer.com/website-content/2025/07/Utah-Arm-Comparison-Layout-2025.pdf).

**Rotary path (what the engine chose, and what the market builds):**

- CubeMars AK60-6 quasi-direct-drive: 3 N·m rated / 9 N·m peak, 368 g,
  $298.90 (https://store.cubemars.com/products/ak60-6-v1-1-kv140) — closest
  single-unit match if peak-rated duty is acceptable; marginal against the
  1.3× holdability gate continuous.
- CubeMars AK70-10: ~10 N·m rated / ~24.8 N·m peak, ~521 g [est], $398.90
  factory / $579.97 RobotShop (https://store.cubemars.com/products/ak70-10) —
  meets the continuous spec with margin; the single most direct COTS match.
- SteadyWin GIM6010-6: ~18 N·m rated claimed, ~546 g, €116–211
  (https://openelab.io/products/steadywin-gim6010-6-precision-robotic-actuator-gear)
  — cheapest N·m/€; QC is the risk you're buying.
- Maxon EC + GPX22/GP22HP: gearheads €130–268 list, motor+gearhead+encoder
  realistically €400–1,000+/joint [est] (https://www.maxongroup.com/…/GPX22) —
  the medical-grade path; Maxon markets rehab/prosthetics explicitly.
- Note the QDD units are 6–10:1; the evolved 84.5:1 with a 28 mm-class BLDC is
  closer to commercial prosthetics practice (small motor, multi-stage
  planetary, non-backdrivable-ish) than to robotics QDD — both are viable
  implementations of the same requirement, trading backdrivability against
  hold power.
- Drivers: mjbots moteus r4.11 $94, ODrive S1 $149, VESC 6 ~£200. Battery:
  2S–4S Li-ion, Molicel P42A 21700 $5.99/cell — matches PSYONIC (7.4 V) and
  LUKE (14.8 V) practice.
- Printed cycloidal/planetary reducers have an active open ecosystem (Faze4,
  OpenCycloid, U. Twente peer-reviewed printable cycloid) — fine for
  prototypes; no published life data at prosthetic duty cycles.

**Why not linear (and why the engine agrees):** hobby-class EMAs top out at
Actuonix P16 = 300 N but only 4.8 mm/s at that gearing ($106.59,
https://s3.amazonaws.com/actuonix/Actuonix+P16+Datasheet.pdf). At a 25–40 mm
moment arm the elbow needs ~140–220 N *and* ~20–50 mm/s — no COTS unit ships
that force×speed product at wearable mass; you'd custom-build BLDC+ballscrew
(Ewellix/Thomson minis are quote-only components; the CAHB-10 class is >0.5 kg
and slow). This is precisely the screw-efficiency + speed penalty that made
`direct_linear` lose the fitness comparison, and why every commercial powered
elbow (Ottobock DynamicArm — $33,167.64 dealer price, ~1 kg,
https://www.spsco.com/dynamicarm-elbow.html; Utah Arm) is a geared rotary
drive. The model and the market agree.

**Sensing/control:** MyoWare 2.0 EMG $43.50 vs Ottobock 13E200 electrode
$1,070.98 (https://www.sparkfun.com/myoware-2-0-muscle-sensor.html,
https://www.spsco.com/myobock-electrode-1.html) — a 25× clinical premium for
functionally similar analogue envelope EMG; the recurring low-cost-prosthetics
arbitrage.

Prototype BoM for the winner: AK60-6 or motor+GPX-class gearbox, moteus, 3S
pack, printed shells — **$600–900**; a clinical-grade equivalent BoM (Maxon,
clinical electrodes ×2, certified battery) lands **$3–5k** [est] against
$33k dealer pricing for the incumbent single-joint elbow.

---

## 3. Business case

**Market.** Global upper-limb prosthetics ≈ $0.8–1.0 B with mid-single-digit
CAGR (Grand View: $1.03 B by 2030 @ 5.3 %; Mordor: $0.81 B 2025 → $1.38 B
2031 @ 9.4 %; Precedence Feb-2026: ~$1.0 B → $1.65 B 2035). Powered elbows
are a sliver of that — three or four products worldwide (DynamicArm, Utah
Arm/Motion Arm, Espire) at likely low-thousands of units/yr [est]. Retail
economics are extreme: above-elbow myoelectric systems bill $80–160k in US
workers'-comp data with documented 3–10× component markups
(https://www.workcompwire.com/2018/11/guest-post-controlling-the-costs-of-myoelectric-prosthetics/).

**Reimbursement exists for exactly this joint.** HCPCS L7181 (electronic
elbow, microprocessor simultaneous control) has been on the DMEPOS schedule
since 2005; and CMS set the MyoPro powered elbow-orthosis codes at
**L8701 = $33,480.90 / L8702 = $65,871.74** effective 2024-04-01 — an
existence proof that a small firm with one powered-elbow-class device can
reach a Medicare rate. NHS England routinely commissions multi-grip
myoelectric *hands* since policy 220801P (Aug 2022, ~200 candidates/yr in
England; https://www.england.nhs.uk/wp-content/uploads/2022/09/2009-clinicial-comm-policy-multi-grip-prosthetic-hand-all-ages-1.pdf)
— no equivalent routine-commissioning policy for powered elbows, so UK volume
is case-by-case.

**Regulatory is lighter than assumed at component level, heavy at system
level.** US: 21 CFR 890.3420 makes an external limb prosthetic *component* —
explicitly including a powered elbow joint — **Class I and 510(k)-exempt**
(https://www.law.cornell.edu/cfr/text/21/890.3420). The ceiling is the
DEKA-created 21 CFR 890.3450 (product code PAE): an integrated arm with >2
simultaneous powered DoF is Class II with seven special controls including
clinical/human-factors data. If a 510(k) is ever needed: FY2026 fee $26,067
($6,517 small business) plus realistic all-in $23–49k
(https://www.i3cglobal.com/fda-510k-fees/). EU/UK: active prosthesis = MDR
Class IIa → Notified Body (€10–30k for a simple IIa, plus the ISO 13485 QMS
build, which is the larger spend [est]); CE remains accepted in GB to
2030-06-30 with an MHRA consultation on indefinite recognition (Feb 2026), so
CE once covers both markets. Standards: ISO 22523 (upper-limb strength/safety,
MDR-harmonised); test campaigns $10–50k [est]. From 2026-12-09 the new EU
Product Liability Directive (EU) 2024/2853 extends strict liability to
software and AI — **the design-automation pipeline itself becomes a
"product"** if placed on the EU market, and load-bearing printed parts sit
squarely in its expanded defectiveness regime.

**Who actually pays, honestly assessed.** Selling design tooling *into
prosthetics OEMs* is a <15-customer market (Grand View's player list) — nTop
proves medical generative-design tooling sells (quote-only, ~$10–20k/yr seats
[est]), but nobody has sold actuator-topology evolution into this niche; the
niche being empty cuts both ways. The credible wedges, in order:

1. **Engineering-services wedge (immediate, consultancy-shaped).** The same
   engine answers "what actuator/transmission for this joint, provably
   buildable, with the trade study attached" for exoskeleton and robotics
   firms — a market at $0.59 B (2025) → $1.79 B (2033) @ 14.5 % (Grand View,
   https://www.grandviewresearch.com/industry-analysis/exoskeleton-market),
   actuator-hungry at exactly this 5–15 N·m joint class, and unregulated at
   the R&D stage. The deliverable is the study + design.json + verification
   pack, not a device. Zero regulatory exposure, immediate revenue,
   references.
2. **Class I-exempt US component play.** A powered elbow *component* under
   890.3420 with ISO 22523-style bench evidence, sold B2B to prosthetists/
   integrators at the $3–6k BoM-to-$15–25k price point the PSYONIC precedent
   ($1.8k build → $15.5k retail) suggests the channel bears. Liability and
   durability evidence, not regulation, are the real cost centres.
3. **License designs to an incumbent** (Ottobock/Fillauer/Steeper/College
   Park): they own clinical channels and MDR certificates; the pipeline's
   value is compressing their elbow-refresh R&D cycle. Low volume, high
   credibility.

Anti-case, stated plainly: the pure-prosthetics device route needs clinical
evidence generation (the NHS pathway alone presumes 12 months of graded use),
product liability cover for load-bearing printed structures, and a
distribution channel controlled by the incumbents — for a global elbow market
measured in low thousands of units. Do not build the business on route 3
alone or on NHS volume.

---

Scope boundary: assessment covers the delivered core engine and its
commercialisation only; the reviewed employer-side repository is out of scope.
Ownership: Consultancy IP / Reusable Framework — engine and docs built
clean-room from the user-supplied spec and reviews; no employer code touched.
External deployment risk: no PII; EU PLD 2024/2853 applies if the pipeline or
its designs are placed on the EU market after 2026-12-09.
