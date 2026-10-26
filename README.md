# Open topology engine

The evolution engine behind [LimbForge](https://github.com/PressureCook/limbforge). It starts every design from the simplest possible mechanism (two bones and one linear actuator) and lets structural mutation grow linkages, extra actuators and springs, behind a feasibility gate that rejects anything that couldn't physically work.

## Reference implementation

Standalone implementation of `SPEC-open-topology-evolution.md` (v0.1,
2026-08-12), scoped to the core engine: Phases 1–3 done properly, built so
the ten graph-engine findings of the round-2 review (G1–G10) cannot recur.
Pure numpy + matplotlib; 22 pytest tests; the elbow demo runs end to end in
~48 s on one CPU core.

```
pip install numpy matplotlib pytest
python -m pytest tests/ -q                      # 22 passed
python -m open_topology.demo --out runs/demo_elbow --pop 90 --gens 34 --seed 7
```

## Architecture

```
context.py      JointContext: Winter proximal-pair anthropometry, theta_neutral
                gravity convention, envelope capsule       (C1/C3 conventions)
genome.py       NEAT-style variable-length genome; innovation ids; JSON
                round-trip incl. assembly-branch hints              (R-1, R-11)
actuators.py    capacity decode (zero at gene 0), gear-ratio gene, reflected
                rotor inertia, gearbox stages/eta/mass, per-kind buildability
interpreter.py  GENERIC planar loop-closure interpreter: warm-started Newton,
                analytic Jacobian, implicit-function derivatives, transmission
                table, validity gate, numeric mobility          (R-5..R-9)
simulate.py     species-batched reduced-EOM integrator with 1/2 dI' th'^2 term,
                greedy + exact-QP allocation, per-actuator energy   (R-7, R-10)
fitness.py      hard gate -> spec 8.2 cost terms (complexity, energy, mass,
                saturation, singularity margin, soft packaging margin)
evolve.py       speciation + fitness sharing, largest-remainder quotas,
                structural operators, immigration, robustness hard gate
archetypes.py   seed constructors expressed AS GRAPHS (no bespoke kinematics)
demo.py         synthetic elbow, alternate-rep hold-out, design.json v2, plots
```

The single load-bearing decision is unchanged from the spec: mobility-1
reduction to a θ-indexed transmission table (§5.2). What is different from the
reviewed codebase is that **there is exactly one kinematic code path**. The
three archetypes are genome constructors, not topology classes; `four_bar`
produces plain bodies/joints that the generic interpreter solves. A structure
invented by mutation is simulated by the same Newton/IFT machinery as a seeded
one: the G4 objection (curated library wearing a graph-shaped genome) is
closed structurally, not by adding more hand-written topologies.

## Review-finding closure map

| Finding | Closure | Guarded by |
|---|---|---|
| **G1** rotary rides free | gear-ratio gene; `G_r²·J_rotor` in `I_eff`; η per stage; ω_out = ω_m/G_r; output-torque priced in fitness | `test_rotary_reflected_inertia_in_table`; demo: optimiser settles at 84.5:1 where refl. inertia is 6 % of I_eff, not 160:1/29 % |
| **G2** no feasibility gate | one `validate()` for every topology: Newton, cond, branch, intersection, envelope, stroke/ratio/L_ret/packaging/moment-arm, holdability; hard, pre-integration (R-9) | `test_gate_enforces_legacy_constraints_on_valid_designs` (property test over 150 random genomes, review's own methodology) |
| **G3** `--limb` crash | n/a here (single-joint core); drawing consumes the table, no `mech` objects exist | `draw_mechanism` reads `table.sites` only |
| **G4** search can't change topology | `add_dyad` (+2 bodies +3 joints, net-zero DOF), `remount`, `retype`, `add_via`, `add_spring/actuator`, `remove_*`, `simplify`: all interpreted generically | demo log: hybrid `linear+rotary` and `rotary+spring` structures lead generations 2–14 |
| **G5** lethal antagonist mutation | every capacity decode is 0 at gene 0 (no rate floor); ω-limit min over *engaged* actuators only; zero-capacity elements gate-exempt | `test_structural_mutation_behaviour_preserving` (≥20 seeds, tables AND trajectories), `test_slack_tendon_does_not_throttle_speed` |
| **G6** energy blind to co-contraction | `Σᵢ relu(uᵢ rᵢ ω)/ηᵢ` per actuator, efficiency inside the objective | `test_cocontraction_energy_visible` (hand-built opposed pair, net-power metric shown blind) |
| **G7** wrong transmission angle | no hand-picked angle: dead centres surface as constraint-Jacobian conditioning + moment-arm floor + branch flip, all in the gate | `test_gate_rejects_unassemblable_four_bar` |
| **G8** no self-intersection / 2-of-5 envelope | proper-crossing test over all non-adjacent link/actuator/bone segments; envelope over **every** moving site, every grid point | archetype experiment: 92 %+ of naive four-bars correctly rejected before the constructor was made clearance-aware |
| **G9** decorative solid models | out of chosen scope: nothing pretends otherwise; `describe_actuators()` emits real evolved numbers only | design.json carries no geometry it didn't simulate |
| **G10** `mobility() == return 1` | Kutzbach count **and** numeric Jacobian rank at θ_ref; can fail; gate consumes it | `test_mobility_checks` (dangling chain and overconstrained both rejected) |
| M-series | spring free length decodes against fixed physical bounds; unclipped demand recorded under the *same* allocation weights; quotas by largest remainder (no tail starvation); packaging is a soft margin target over a hard floor; hold-out = alternate repetitions incl. a moving segment | `test_spring_free_decode_is_grid_independent`, `test_unclipped_demand_recorded`, `test_apportion_no_tail_starvation` |

## Demo result (pop 90 × 34 gens, seed 7, 47.5 s)

```
winner: rotary_geared 0.089 N·m BLDC @ 84.5:1 (3 stages, η_g 0.729)
        τ_out 5.49 N·m · reflected inertia 0.0073 kg·m² (6 % of I_eff)
        mass 246 g · RMSE 0.59° fit / 0.44° hold-out · 26.0 J per session
        greedy-vs-QP tracking gap 3.7e-14° · robustness screen passed
```

Best-of-topology at the final generation (all fully gate-valid: the
comparison the reviewed engine could not honestly make):

| topology | cost | RMSE | energy | mass | why it places here |
|---|---|---|---|---|---|
| rotary geared | 2.182 | 0.59° | 26.0 J | 0.25 kg | motor+gearbox η 0.62 beats screw η 0.55; pays reflected inertia + mass and still wins |
| direct linear | 2.402 | 0.52° | 40.9 J | 0.07 kg | best tracking and lightest; loses on screw-efficiency energy |
| four-bar | 3.110 | 0.59° | 39.8 J | 0.23 kg | tracks identically, pays complexity; buys nothing at this load |

Three sanity signals worth naming: the optimiser chose a **lower** gear ratio
(84.5:1) than the broken engine's 160:1: exactly the direction the review's
reflected-inertia arithmetic predicted; τ_out sits at 5.49 N·m against the
5.5 N·m holdability requirement (the gate is active, not decorative); and
rotary winning at the elbow is what the commercial market did (every powered
elbow on sale is a geared rotary drive).

Species diversity holds at 7–14 throughout (no collapse: Phase 3 acceptance);
non-seeded hybrid structures (`rotary+spring`, `linear+rotary`) held the lead
for ~12 generations mid-run before parameter-tuned rotary re-took it.

## Deliberate scope cuts and known limitations

- **Downstream chain** (per the agreed scope): actuator spec-sheet generation,
  printables/STL, report rendering are not implemented. `design.json` v2
  carries everything they need.
- **Prismatic joints** are absent from the interpreter (revolute only), so
  slider-cranks are unreachable. Adding them is ~30 lines of constraint code.
- **Tendon via-points** are always-routed (no unilateral wrap detection), so
  `add_via` is only approximately behaviour-preserving; the strict 1e-6 claim
  in spec §6.3 is idealised for this operator and is not asserted in tests.
- **`add_dyad`** is kinematically exact but adds link mass (~2-3 % I_eff), so
  preservation is asserted at 0.02 rad on trajectories, not 1e-9.
- **Robustness screen** runs as a final hard gate on the ranked shortlist (a
  winner must survive ±0.2/0.3 mm perturbation), not per-generation on top-K;
  a parameter-free rotary passes it trivially.
- **Species stagnation retirement** (§6.4) is not implemented; at 34
  generations it never bound. Threshold adaptation keeps 4–12 species instead.
- **dI/dθ** is finite-differenced on the grid (spec allows "spectral
  difference"); the energy-conservation test bounds the consequence.
- Weights in `fitness.W` are calibrated to the elbow demo only; treat them as
  the §11 "manual review" feedback loop intends: inputs, not constants.
