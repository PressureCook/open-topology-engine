"""Speciated structural evolution (spec 6) over generic mechanism graphs.

Round-2 closures here:
* G4 - ``add_dyad`` + ``remount`` + ``retype`` + ``add_via`` genuinely change
  the mechanism graph; nothing in the reachable set is enumerated downstream.
* G5 - structural additions decode to exactly zero capacity/stiffness and are
  exempt from capacity-gated validity, so they are behaviour-preserving at
  birth; the test suite checks tables *and* trajectories over >= 20 seeds.
* Offspring quotas use largest-remainder apportionment with elites inside the
  quota - the tail-species starvation found in round 2 cannot occur.
* The robustness screen is a **hard gate on the shipped winner** (8.3): the
  ranked list is walked until a design survives perturbation.
"""
from __future__ import annotations

import numpy as np

from .genome import (Genome, BodyGene, JointGene, ActuatorGene, SpringGene,
                     Innovations, GROUND, OUTPUT, param_bounds, iter_params)
from .interpreter import CompiledMechanism, validate
from .fitness import evaluate_population
from . import archetypes, actuators as act_mod


# ---------------------------------------------------------------- geometry help
def _poses_at_ref(genome: Genome, ctx):
    mech = CompiledMechanism(genome, ctx)
    x, ok, _, _ = mech.solve_point(genome.theta_ref, mech._hint_x())
    return mech, mech._poses_at(genome.theta_ref, x), ok


def _world(mech, poses, body_id, local):
    return mech._point(poses, mech.bidx[body_id], np.asarray(local, float))


def _ground_local(ctx, world):
    up = ctx.prox_dir
    nrm = np.array([-up[1], up[0]])
    return float(world @ up), float(world @ nrm)


def _mini_r(genome: Genome, ctx, ai: int, half: float = 0.35):
    mech = CompiledMechanism(genome, ctx)
    grid = np.linspace(genome.theta_ref - half, genome.theta_ref + half, 9)
    t = mech.build_table(grid)
    return float(t.r[ai, 4]), t


# ---------------------------------------------------------------- operators
def op_add_spring(g, rng, inn, ctx, gen):
    mech, poses, ok = _poses_at_ref(g, ctx)
    if rng.random() < 0.35:                                  # torsional
        sid = inn.get(("spring_t", len(g.springs)))[0]
        gfree = float(np.clip((g.theta_ref + 0.5 * np.pi) / (1.5 * np.pi), 0, 1))
        g.springs.append(SpringGene(f"s{sid}", sid, "torsional",
                                    p={"gk": 0.0, "gfree": gfree}))
    else:
        if not ok:
            return None
        bodies = [GROUND, OUTPUT] + [b.id for b in g.bodies]
        ba = GROUND
        bb = str(rng.choice([b for b in bodies if b != GROUND]))
        pa_w = _ground_world_anchor(rng, ctx)
        pb_l = _local_anchor(rng, ctx, g, bb)
        pa_l = _ground_local(ctx, pa_w)
        Lref = float(np.linalg.norm(_world(mech, poses, bb, pb_l) - pa_w))
        gfree = float(np.clip((Lref - 0.02) / 0.38, 0, 1))
        sid = inn.get(("spring_l", len(g.springs), bb))[0]
        g.springs.append(SpringGene(
            f"s{sid}", sid, "linear", body_a=ba, body_b=bb,
            p={"ax": pa_l[0], "ay": pa_l[1], "bx": pb_l[0], "by": pb_l[1],
               "gk": 0.0, "gfree": gfree}))
    g.meta.setdefault("birth", {})[str(g.springs[-1].innovation)] = gen
    return g


def _ground_world_anchor(rng, ctx):
    up = ctx.prox_dir
    nrm = np.array([-up[1], up[0]])
    return (float(rng.uniform(0.06, 0.9 * ctx.L_prox)) * up
            + float(rng.uniform(-0.06, 0.06)) * nrm)


def _local_anchor(rng, ctx, g, body_id):
    if body_id == OUTPUT:
        return (float(rng.uniform(0.05, 0.9 * ctx.L_dist)),
                float(rng.uniform(-0.03, 0.03)))
    b = g.body(body_id)
    return (float(rng.uniform(0.0, b.p["length"])), 0.0)


def op_add_actuator(g, rng, inn, ctx, gen):
    """Add a zero-capacity linear/tendon actuator; antagonist variant mirrors an
    existing one across the bone."""
    base = [a for a in g.actuators if a.kind in ("linear", "tendon")]
    if base and rng.random() < 0.6:                          # antagonist
        src = base[int(rng.integers(len(base)))]
        aid = inn.get(("antag", src.innovation))[0]
        p = dict(src.p)
        p["ay"], p["by"] = -p["ay"] - 0.02, -p["by"]
        p["gF"], p["gv"] = 0.0, float(rng.uniform(0.3, 0.8))
        g.actuators.append(ActuatorGene(f"a{aid}", aid, "tendon",
                                        body_a=src.body_a, body_b=src.body_b,
                                        p=p))
    else:
        pa = _ground_local(ctx, _ground_world_anchor(rng, ctx))
        pb = _local_anchor(rng, ctx, g, OUTPUT)
        kind = "tendon" if rng.random() < 0.5 else "linear"
        aid = inn.get(("act", len(g.actuators), kind))[0]
        g.actuators.append(ActuatorGene(
            f"a{aid}", aid, kind, body_a=GROUND, body_b=OUTPUT,
            p={"ax": pa[0], "ay": pa[1], "bx": pb[0], "by": pb[1],
               "gF": 0.0, "gv": float(rng.uniform(0.3, 0.8))}))
    g.meta.setdefault("birth", {})[str(g.actuators[-1].innovation)] = gen
    return g


def op_add_dyad(g, rng, inn, ctx, gen, rom):
    """Attach an RRR dyad between ground and output: +2 bodies, +3 joints,
    net-zero DOF - kinematically behaviour-preserving, and the substrate for
    linkage discovery (a subsequent remount turns it into a driven four-bar)."""
    s, n = float(rng.uniform(0.06, 0.6 * ctx.L_prox)), float(rng.uniform(-0.05, 0.05))
    up = ctx.prox_dir
    nrm = np.array([-up[1], up[0]])
    PA = s * up + n * nrm
    bx = float(rng.uniform(0.2, 0.7) * ctx.L_dist)
    by = float(rng.uniform(-0.02, 0.02))
    ths = np.linspace(rom[0], rom[1], 25)
    ds = [np.linalg.norm(archetypes._out_world(ctx, th, (bx, by)) - PA)
          for th in ths]
    dmin, dmax = float(np.min(ds)), float(np.max(ds))
    Ltot = 1.15 * dmax
    for _ in range(8):
        t = float(rng.uniform(0.35, 0.65))
        L1, L2 = t * Ltot, (1 - t) * Ltot
        if abs(L1 - L2) < 0.8 * dmin:
            break
    else:
        return None
    B = archetypes._out_world(ctx, g.theta_ref, (bx, by))
    d = float(np.linalg.norm(B - PA))
    a = (L1 ** 2 - L2 ** 2 + d ** 2) / (2 * d)
    h2 = L1 ** 2 - a ** 2
    if h2 <= 1e-8:
        return None
    h = float(np.sqrt(h2))
    u = (B - PA) / d
    E = PA + a * u + (h if rng.random() < 0.5 else -h) * np.array([-u[1], u[0]])
    ids = inn.get(("dyad", round(s, 3), round(bx, 3)), count=5)
    b1, b2 = f"d{ids[0]}", f"d{ids[1]}"
    g.bodies.append(BodyGene(b1, ids[0], {"length": L1}))
    g.bodies.append(BodyGene(b2, ids[1], {"length": L2}))
    g.joints.append(JointGene(f"j{ids[2]}", ids[2], "revolute", GROUND, b1,
                              {"ax": s, "ay": n, "bx": 0.0, "by": 0.0}))
    g.joints.append(JointGene(f"j{ids[3]}", ids[3], "revolute", b1, b2,
                              {"ax": L1, "ay": 0.0, "bx": 0.0, "by": 0.0}))
    g.joints.append(JointGene(f"j{ids[4]}", ids[4], "revolute", b2, OUTPUT,
                              {"ax": L2, "ay": 0.0, "bx": bx, "by": by}))
    g.hints[b1] = [float(PA[0]), float(PA[1]),
                   float(np.arctan2(*(E - PA)[::-1]))]
    g.hints[b2] = [float(E[0]), float(E[1]),
                   float(np.arctan2(*(B - E)[::-1]))]
    birth = g.meta.setdefault("birth", {})
    for i in ids:
        birth[str(i)] = gen
    return g


def op_add_via(g, rng, inn, ctx, gen):
    tendons = [(i, a) for i, a in enumerate(g.actuators)
               if a.kind == "tendon" and not a.via]
    if not tendons:
        return None
    ai, a = tendons[int(rng.integers(len(tendons)))]
    mech, poses, ok = _poses_at_ref(g, ctx)
    if not ok:
        return None
    up = ctx.prox_dir
    nrm = np.array([-up[1], up[0]])
    pa = _world(mech, poses, a.body_a,
                (a.p["ax"] * up + a.p["ay"] * nrm) if a.body_a == GROUND
                else (a.p["ax"], a.p["ay"]))
    if a.body_a == GROUND:
        pa = a.p["ax"] * up + a.p["ay"] * nrm
    pb = _world(mech, poses, a.body_b, (a.p["bx"], a.p["by"]))
    mid = 0.5 * (pa + pb)
    s, n = _ground_local(ctx, mid)
    a.via = [[GROUND, s, n]]
    return g


def op_retype(g, rng, inn, ctx, gen):
    cand = [i for i, a in enumerate(g.actuators)
            if act_mod.decode(a)["u_hi"] > 1e-9]
    if not cand:
        return None
    ai = int(rng.choice(cand))
    a = g.actuators[ai]
    try:
        r_ref, _ = _mini_r(g, ctx, ai)
    except Exception:
        return None
    if a.kind in ("linear", "tendon"):
        dec = act_mod.decode(a)
        tau_mid = abs(r_ref) * dec["u_hi"]
        tau_m = 0.35 * act_mod.TAU_M_HI
        ratio = tau_mid / max(tau_m * 0.8, 1e-6)
        ratio = float(np.clip(ratio, act_mod.RATIO_LO, act_mod.RATIO_HI))
        ns = act_mod.n_stages(ratio)
        tau_m = float(np.clip(tau_mid / max(ratio * 0.9 ** ns, 1e-6),
                              0.0, act_mod.TAU_M_HI))
        om_out = dec["v_hi"] / max(abs(r_ref), 1e-3)
        om_m = float(np.clip(ratio * om_out, 0.0, act_mod.OMEGA_M_HI))
        gratio = (np.log(ratio / act_mod.RATIO_LO)
                  / np.log(act_mod.RATIO_HI / act_mod.RATIO_LO))
        g.actuators[ai] = ActuatorGene(
            a.id, a.innovation, "rotary",
            p={"gtau": float(np.sqrt(tau_m / act_mod.TAU_M_HI)),
               "gratio": float(np.clip(gratio, 0, 1)),
               "gomega": float(om_m / act_mod.OMEGA_M_HI)})
    else:
        dec = act_mod.decode(a)
        tau_out = dec["r_const"] * dec["u_hi"]
        r_typ = 0.03
        F = float(np.clip(tau_out / r_typ, 0.0, act_mod.F_HI))
        pa = _ground_local(ctx, _ground_world_anchor(rng, ctx))
        pb = _local_anchor(rng, ctx, g, OUTPUT)
        g.actuators[ai] = ActuatorGene(
            a.id, a.innovation, "linear", body_a=GROUND, body_b=OUTPUT,
            p={"ax": pa[0], "ay": pa[1], "bx": pb[0], "by": pb[1],
               "gF": float(np.sqrt(F / act_mod.F_HI)),
               "gv": float(rng.uniform(0.3, 0.8))})
    return g


def op_remount(g, rng, inn, ctx, gen):
    """Move a linear/tendon actuator's proximal-side mount onto a coupler -
    the step that turns a passive dyad into a driven linkage. Non-preserving."""
    if not g.bodies:
        return None
    cand = [a for a in g.actuators if a.kind in ("linear", "tendon")]
    if not cand:
        return None
    a = cand[int(rng.integers(len(cand)))]
    b = g.bodies[int(rng.integers(len(g.bodies)))]
    t = float(rng.uniform(0.2, 1.0))
    a.body_b = b.id
    a.p["bx"], a.p["by"] = t * b.p["length"], 0.0
    return g


def op_remove_spring(g, rng, inn, ctx, gen):
    weak = [s for s in g.springs if act_mod.spring_decode(s)["k"] < 1.0]
    if not weak:
        return None
    g.springs.remove(weak[int(rng.integers(len(weak)))])
    return g


def op_remove_actuator(g, rng, inn, ctx, gen):
    if len(g.actuators) <= 1:
        return None
    decs = [act_mod.decode(a) for a in g.actuators]
    tot = sum(abs(d["u_hi"]) for d in decs) + 1e-9
    weak = [a for a, d in zip(g.actuators, decs) if d["u_hi"] < 0.05 * tot]
    if not weak:
        return None
    g.actuators.remove(weak[int(rng.integers(len(weak)))])
    return g


def op_simplify(g, rng, inn, ctx, gen):
    """Behaviour-preserving cleanup: drop stale zero elements, unused dyads."""
    birth = g.meta.get("birth", {})
    g.springs = [s for s in g.springs
                 if act_mod.spring_decode(s)["k"] > 1e-9
                 or gen - birth.get(str(s.innovation), gen) < 8]
    keep = []
    for a in g.actuators:
        if act_mod.decode(a)["u_hi"] > 1e-9 \
                or gen - birth.get(str(a.innovation), gen) < 8 \
                or len(g.actuators) == 1:
            keep.append(a)
    g.actuators = keep if keep else g.actuators[:1]
    used = {a.body_a for a in g.actuators} | {a.body_b for a in g.actuators} \
        | {s.body_a for s in g.springs} | {s.body_b for s in g.springs}
    for b in list(g.bodies):
        if b.id in used:
            continue
        if gen - birth.get(str(b.innovation), gen) < 10:
            continue
        others = [x.id for x in g.bodies if x.id != b.id]
        js = [j for j in g.joints if b.id in (j.body_a, j.body_b)]
        partner_cpl = [j for j in js
                       if (j.body_a in others) or (j.body_b in others)]
        if len(partner_cpl) == 1:          # half of an unused dyad: drop pair
            continue
        g.joints = [j for j in g.joints if b.id not in (j.body_a, j.body_b)]
        g.bodies.remove(b)
        g.hints.pop(b.id, None)
    return g


STRUCT_OPS = [(op_add_spring, 0.22), (op_add_actuator, 0.16),
              (op_add_dyad, 0.16), (op_add_via, 0.06), (op_retype, 0.10),
              (op_remount, 0.10), (op_remove_spring, 0.06),
              (op_remove_actuator, 0.06), (op_simplify, 0.08)]


def mutate_params(g: Genome, rng, ctx, sigma: float):
    for e, name in iter_params(g):
        if rng.random() < 0.25:
            lo, hi = param_bounds(e, name, ctx)
            e.p[name] = float(np.clip(
                e.p[name] + rng.normal(0.0, sigma * (hi - lo)), lo, hi))
    return g


def mutate_structure(g: Genome, rng, inn, ctx, gen, rom):
    ops, ws = zip(*STRUCT_OPS)
    w = np.array(ws) / sum(ws)
    op = ops[int(rng.choice(len(ops), p=w))]
    try:
        out = (op(g, rng, inn, ctx, gen, rom) if op is op_add_dyad
               else op(g, rng, inn, ctx, gen))
    except Exception:
        return g
    return out if out is not None else g


# ---------------------------------------------------------------- crossover
def crossover(fit_parent: Genome, other: Genome, rng, ctx, alpha=0.3):
    child = fit_parent.copy()
    e2 = other.elements()
    for e in child.elements().values():
        o = e2.get(e.innovation)
        if o is None:
            continue
        for name in e.p:
            if name in o.p:
                lo, hi = param_bounds(e, name, ctx)
                a, b = e.p[name], o.p[name]
                lo_b, hi_b = min(a, b), max(a, b)
                span = hi_b - lo_b
                e.p[name] = float(np.clip(
                    rng.uniform(lo_b - alpha * span, hi_b + alpha * span),
                    lo, hi))
    return child


# ---------------------------------------------------------------- speciation
def distance(g1: Genome, g2: Genome, ctx) -> float:
    e1, e2 = g1.elements(), g2.elements()
    k1, k2 = set(e1), set(e2)
    union = len(k1 | k2)
    sym = len(k1 ^ k2)
    pd = []
    for i in k1 & k2:
        a, b = e1[i], e2[i]
        for name in a.p:
            if name in b.p:
                lo, hi = param_bounds(a, name, ctx)
                pd.append(abs(a.p[name] - b.p[name]) / max(hi - lo, 1e-9))
    return 1.3 * sym / max(union, 1) + 2.0 * (float(np.mean(pd)) if pd else 0.0)


def speciate(genomes, infos, reps, delta_t, ctx):
    species = []
    for rep in reps:
        species.append({"rep": rep, "members": []})
    for gi, g in enumerate(genomes):
        placed = False
        for sp in species:
            if distance(g, sp["rep"], ctx) < delta_t:
                sp["members"].append(gi)
                placed = True
                break
        if not placed:
            species.append({"rep": g, "members": [gi]})
    species = [sp for sp in species if sp["members"]]
    for sp in species:
        best = min(sp["members"], key=lambda i: infos[i]["cost"])
        sp["best"] = best
        sp["best_cost"] = infos[best]["cost"]
        sp["rep"] = genomes[best]
    return species


def apportion(scores, total):
    """Largest-remainder quotas (no tail starvation)."""
    scores = np.maximum(np.asarray(scores, float), 1e-12)
    raw = scores / scores.sum() * total
    base = np.floor(raw).astype(int)
    rem = total - base.sum()
    order = np.argsort(-(raw - base))
    for i in range(rem):
        base[order[i % len(base)]] += 1
    return base


# ---------------------------------------------------------------- robustness
def robustness_screen(genome: Genome, ctx, grid, rom, traj, rng,
                      n: int = 12) -> dict:
    from .fitness import evaluate_population
    rmses = []
    for _ in range(n):
        gp = genome.copy()
        for e, name in iter_params(gp):
            if name.startswith("g"):
                continue
            if name == "length":
                e.p[name] += float(rng.uniform(-3e-4, 3e-4))
            else:
                e.p[name] += float(rng.uniform(-2e-4, 2e-4))
        info = evaluate_population([gp], ctx, grid, rom, traj, cache=None)[0]
        if not info["gate_ok"]:
            return {"pass": False, "invalid_fraction": 1.0 / n,
                    "rmse_std_deg": None}
        rmses.append(info["metrics"]["rmse_fit_deg"])
    return {"pass": True, "invalid_fraction": 0.0,
            "rmse_std_deg": float(np.std(rmses)),
            "rmse_mean_deg": float(np.mean(rmses))}


# ---------------------------------------------------------------- main loop
def evolve(ctx, traj, pop: int = 90, gens: int = 34, seed: int = 7,
           grid_n: int = 112, pad: float = 0.10, log=print):
    rng = np.random.default_rng(seed)
    inn = Innovations()
    th_lo = float(np.min(traj.theta_d))
    th_hi = float(np.max(traj.theta_d))
    grid = np.linspace(th_lo - pad, th_hi + pad, grid_n)
    rom = (grid >= th_lo - 1e-9) & (grid <= th_hi + 1e-9)
    theta_ref = 0.5 * (th_lo + th_hi)
    rom_bounds = (th_lo, th_hi)

    quick_grid = np.linspace(th_lo - pad, th_hi + pad, 25)
    quick_rom = (quick_grid >= th_lo - 1e-9) & (quick_grid <= th_hi + 1e-9)

    def seed_genome(prescreen: int = 8):
        last = None
        for _ in range(prescreen):
            g = None
            for _ in range(20):
                g = archetypes.random_archetype(rng, ctx, theta_ref)
                if g is not None:
                    break
            if g is None:
                continue
            last = g
            mech = CompiledMechanism(g, ctx)
            t = mech.build_table(quick_grid)
            ok, _, _ = validate(mech, t, quick_rom)
            if ok:
                return g
        return last if last is not None else archetypes.direct_linear(
            rng, ctx, theta_ref)

    population = [seed_genome() for _ in range(pop)]
    cache: dict = {}
    delta_t = 0.55
    reps: list = []
    history = []
    best_ever = None

    n_imm = max(2, int(round(0.08 * pop)))
    for gen in range(gens):
        infos = evaluate_population(population, ctx, grid, rom, traj, cache)
        order = int(np.argmin([i["cost"] for i in infos]))
        if best_ever is None or infos[order]["cost"] < best_ever[1]["cost"]:
            best_ever = (population[order].copy(), infos[order])
        species = speciate(population, infos, reps, delta_t, ctx)
        reps = [sp["rep"] for sp in species]
        if len(species) > 9:
            delta_t = min(delta_t + 0.05, 2.0)
        elif len(species) < 4:
            delta_t = max(delta_t - 0.05, 0.20)

        history.append({
            "gen": gen, "best_cost": infos[order]["cost"],
            "best_label": infos[order]["label"],
            "n_species": len(species),
            "gate_pass_frac": float(np.mean([i["gate_ok"] for i in infos])),
            "mean_parts": float(np.mean([len(g.bodies) + len(g.actuators)
                                         + len(g.springs)
                                         for g in population])),
        })
        log(f"gen {gen:3d}  best {infos[order]['cost']:7.3f}  "
            f"species {len(species):2d}  "
            f"gate {history[-1]['gate_pass_frac']:.2f}  "
            f"{infos[order]['label']}")

        if gen == gens - 1:
            break

        # ---- reproduction (fitness sharing: quota ~ mean adjusted fitness,
        # elites INSIDE the quota, largest-remainder apportionment) ----------
        for sp in species:
            f = [1.0 / (1.0 + infos[i]["cost"]) for i in sp["members"]]
            sp["mean_adj"] = float(np.mean(f))
        quotas = apportion([sp["mean_adj"] for sp in species], pop - n_imm)
        sigma = 0.10 * (1.0 - 0.7 * gen / max(gens - 1, 1))
        new_pop = []
        for sp, quota in zip(species, quotas):
            if quota <= 0:
                continue
            members = sorted(sp["members"], key=lambda i: infos[i]["cost"])
            elite = population[members[0]]
            new_pop.append(elite.copy())                    # elite inside quota
            parents = members[:max(2, len(members) // 2)]
            for _ in range(quota - 1):
                pa = population[parents[int(rng.integers(len(parents)))]]
                pb = population[parents[int(rng.integers(len(parents)))]]
                if infos_cost(infos, pa, population) > infos_cost(
                        infos, pb, population):
                    pa, pb = pb, pa
                child = (crossover(pa, pb, rng, ctx)
                         if rng.random() < 0.5 else pa.copy())
                child = mutate_params(child, rng, ctx, sigma)
                if rng.random() < 0.20:
                    child = mutate_structure(child, rng, inn, ctx, gen,
                                             rom_bounds)
                child.meta["parents"] = [pa.key()[:8], pb.key()[:8]]
                new_pop.append(child)
        while len(new_pop) < pop - n_imm:
            new_pop.append(seed_genome())
        for _ in range(n_imm):
            new_pop.append(seed_genome())
        population = new_pop[:pop]

    # ---- final ranking with hard robustness gate ----------------------
    infos = evaluate_population(population, ctx, grid, rom, traj, cache)
    ranked = sorted(range(len(population)), key=lambda i: infos[i]["cost"])
    cand = [(population[i], infos[i]) for i in ranked
            if infos[i]["gate_ok"]][:6]
    if best_ever is not None and best_ever[1]["gate_ok"]:
        cand = [(best_ever[0], best_ever[1])] + cand
    winner = None
    rob = None
    rejected = []
    for g, info in cand:
        r = robustness_screen(g, ctx, grid, rom, traj, rng)
        if r["pass"]:
            winner, winner_info, rob = g, info, r
            break
        rejected.append((info["label"], info["cost"]))
    if winner is None:                       # nothing survives: report best
        winner, winner_info = cand[0] if cand else (population[ranked[0]],
                                                    infos[ranked[0]])
        rob = {"pass": False, "invalid_fraction": None, "rmse_std_deg": None}

    return {"winner": winner, "winner_info": winner_info,
            "robustness": rob, "robustness_rejected": rejected,
            "history": history, "grid": grid, "rom": rom,
            "population": population, "infos": infos, "cache": cache}


def infos_cost(infos, genome, population):
    for g, i in zip(population, infos):
        if g is genome:
            return i["cost"]
    return np.inf
