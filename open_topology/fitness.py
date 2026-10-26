"""Fitness: hard validity gate, then simulation cost with the spec 8.2 terms.

Cost structure (spec 8.1 retained, 8.2 added):
tracking + capacity + rate + energy (per-actuator, efficiency-weighted)
+ system mass + complexity + unclipped saturation + singularity margin
+ moment-arm margin + soft packaging margin (M6: the margin now means something).

Gate failures never simulate (R-9); they receive a large constant plus a
coverage gradient so the search can climb toward feasibility.
"""
from __future__ import annotations

import numpy as np

from . import actuators as act_mod
from .interpreter import CompiledMechanism, validate
from .simulate import simulate_batch

W = {
    "track": 1.0,          # per deg RMSE on the fit segment
    "cap": 0.8,
    "rate": 0.25,
    "energy": 0.03,        # per J
    "mass": 1.2,           # per kg
    "body": 0.10, "joint": 0.05, "act": 0.30, "spring": 0.05,
    "sat": 2.5,
    "sing": 0.6,
    "rmarg": 0.8,
    "pack": 0.4,
    "gate_fail": 400.0,
}
R_TARGET = 0.012          # soft moment-arm target (hard floor is 8 mm)
PACK_TARGET = 0.005       # soft packaging margin target (m)


def capacity_norm(dec: dict) -> float:
    if dec["kind"] == "rotary":
        # price the OUTPUT torque class (u_hi is gearbox-output-referred motor
        # torque; joint torque = ratio * u_hi) - G1's lesson applies to cost
        # terms too: rotary must not ride free anywhere.
        return dec["u_hi"] * (dec["r_const"] or 1.0) / 40.0
    return dec["u_hi"] / 800.0


def rate_norm(dec: dict) -> float:
    if dec["kind"] == "rotary":
        ratio = dec["r_const"] or 1.0
        return (dec["v_hi"] / max(ratio, 1e-9)) / 6.0
    return dec["v_hi"] / 0.30


def effective_counts(genome) -> tuple:
    n_act = sum(1 for a, d in ((a, act_mod.decode(a)) for a in genome.actuators)
                if d["u_hi"] > 1e-9)
    n_spr = sum(1 for s in genome.springs
                if act_mod.spring_decode(s)["k"] > 1e-9)
    return n_act, n_spr


def cost_terms(genome, mech, table, met, rom) -> dict:
    decs = table.decs
    terms = {"track": W["track"] * met["rmse_fit_deg"]}
    terms["cap"] = W["cap"] * sum(capacity_norm(d) for d in decs)
    terms["rate"] = W["rate"] * sum(rate_norm(d) for d in decs
                                    if d["u_hi"] > 1e-9)
    terms["energy"] = W["energy"] * met["energy_total_J"]
    terms["mass"] = W["mass"] * table.mass_total
    n_act, n_spr = effective_counts(genome)
    terms["complexity"] = (W["body"] * len(genome.bodies)
                           + W["joint"] * len(genome.joints)
                           + W["act"] * n_act + W["spring"] * n_spr)
    terms["sat"] = W["sat"] * min(max(met["sat_excess"], 0.0), 3.0)
    # conditioning margin: cond(J) mixes metres and radians, so a healthy
    # linkage sits at cond ~ 20-200; penalise only the approach to the gate
    inv_cond = 1.0 / np.maximum(table.cond[rom], 1.0)
    terms["sing"] = W["sing"] * max(0.0, (2e-3 - float(np.min(inv_cond)))) / 2e-3
    rm = [float(np.min(np.abs(ri[rom])))
          for ri, d, a in zip(table.r, decs, genome.actuators)
          if d["u_hi"] > 1e-9 and a.kind != "rotary"]
    terms["rmarg"] = (W["rmarg"] * max(0.0, R_TARGET - min(rm)) / R_TARGET
                      if rm else 0.0)
    pmarg = min((act_mod.packaging_margin(a, d, qi, rom)
                 for a, d, qi in zip(genome.actuators, decs, table.q)),
                default=np.inf)
    terms["pack"] = (W["pack"] * max(0.0, PACK_TARGET - pmarg) / PACK_TARGET
                     if np.isfinite(pmarg) else 0.0)
    return terms


def evaluate_population(genomes: list, ctx, grid, rom, traj,
                        cache: dict | None = None) -> list:
    """Returns per-genome info dicts with 'cost'; batches sims by signature."""
    infos = [None] * len(genomes)
    pending = {}
    for gi, g in enumerate(genomes):
        key = g.key()
        if cache is not None and key in cache:
            infos[gi] = cache[key]
            continue
        mech = CompiledMechanism(g, ctx)
        table = mech.build_table(grid)
        ok, reasons, coverage = validate(mech, table, rom)
        if not ok:
            info = {"cost": W["gate_fail"] + 80.0 * (1.0 - coverage)
                    + 8.0 * len(reasons),
                    "gate_ok": False, "reasons": reasons,
                    "coverage": coverage, "label": g.structure_label()}
            infos[gi] = info
            if cache is not None:
                cache[key] = info
            continue
        pending.setdefault(g.signature(), []).append((gi, g, mech, table))

    for sig, items in pending.items():
        mets = simulate_batch([it[3] for it in items], traj, ctx)
        for (gi, g, mech, table), met in zip(items, mets):
            terms = cost_terms(g, mech, table, met, rom)
            info = {"cost": float(sum(terms.values())), "terms": terms,
                    "gate_ok": True, "metrics": met,
                    "label": g.structure_label(),
                    "mass": table.mass_total}
            infos[gi] = info
            if cache is not None:
                cache[g.key()] = info
    return infos
