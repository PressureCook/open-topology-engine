"""Synthetic elbow demo: 3 curls + hold, alternate-repetition hold-out (8.4).

Run:  python -m open_topology.demo --out runs/demo_elbow --pop 90 --gens 34
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .context import demo_elbow
from .genome import Genome
from .interpreter import CompiledMechanism, validate
from .simulate import Trajectory, simulate_one
from .evolve import evolve
from . import actuators as act_mod


def elbow_trajectory(dt: float = 0.01) -> Trajectory:
    """rest 0.5 s | three curls 10->110 deg (4 s each) | hold 90 deg 2 s.
    Fit = curls 1+3 + first half of hold; hold-out = curl 2 + rest of hold."""
    segs = []

    def smooth(a, b, T, n):
        s = np.linspace(0, 1, n, endpoint=False)
        s = s * s * (3 - 2 * s)
        return a + (b - a) * s

    rest = np.full(int(0.5 / dt), np.deg2rad(10.0))
    segs.append(rest)
    for _ in range(3):
        up = smooth(np.deg2rad(10), np.deg2rad(110), 2.0, int(2.0 / dt))
        dn = smooth(np.deg2rad(110), np.deg2rad(10), 2.0, int(2.0 / dt))
        segs.append(np.concatenate([up, dn]))
    ramp = smooth(np.deg2rad(10), np.deg2rad(90), 0.8, int(0.8 / dt))
    hold = np.full(int(1.2 / dt), np.deg2rad(90.0))
    segs.append(np.concatenate([ramp, hold]))
    theta_d = np.concatenate(segs)
    t = np.arange(len(theta_d)) * dt
    omega_d = np.gradient(theta_d, dt)

    fit = np.zeros(len(t), bool)
    hold_m = np.zeros(len(t), bool)
    n_rest = len(rest)
    n_curl = int(4.0 / dt)
    c0 = n_rest
    fit[:n_rest] = True
    fit[c0:c0 + n_curl] = True                       # curl 1
    hold_m[c0 + n_curl:c0 + 2 * n_curl] = True       # curl 2 (hold-out)
    fit[c0 + 2 * n_curl:c0 + 3 * n_curl] = True      # curl 3
    h0 = c0 + 3 * n_curl
    h_mid = h0 + (len(t) - h0) // 2
    fit[h0:h_mid] = True
    hold_m[h_mid:] = True
    return Trajectory(t, theta_d, omega_d, dt, fit, hold_m)


def transmission_summary(table, rom, genome) -> dict:
    out = {"r_range_mm": {}, "I_eff_range_kg_m2":
           [float(np.min(table.I_eff[rom])), float(np.max(table.I_eff[rom]))],
           "min_reciprocal_condition":
           float(np.min(1.0 / np.maximum(table.cond[rom], 1.0)))}
    for a, ri in zip(genome.actuators, table.r):
        rr = ri[rom]
        if a.kind == "rotary":
            out["r_range_mm"][a.id] = f"ratio {float(rr[0]):.1f}:1"
        else:
            out["r_range_mm"][a.id] = [float(np.min(np.abs(rr)) * 1000),
                                       float(np.max(np.abs(rr)) * 1000)]
    return out


def describe_actuators(genome) -> list:
    out = []
    for a in genome.actuators:
        d = act_mod.decode(a)
        e = {"id": a.id, "kind": a.kind}
        if a.kind == "rotary":
            e.update({"tau_motor_Nm": round(d["tau_m"], 3),
                      "gear_ratio": round(d["ratio"], 1),
                      "stages": d["stages"],
                      "eta_gearbox": round(d["eta_gear"], 3),
                      "tau_out_Nm": round(d["u_hi"] * d["ratio"], 2),
                      "reflected_inertia_kg_m2": round(d["refl_J"], 5),
                      "mass_kg": round(d["mass"], 3)})
        else:
            e.update({"F_max_N": round(d["u_hi"], 1),
                      "v_max_m_s": round(d["v_hi"], 3),
                      "mass_kg": round(d["mass"], 3)})
        out.append(e)
    return out


def draw_mechanism(ax, mech, table, k, ctx):
    P = table.sites[k]
    ax.plot([P[mech.s_origin, 0], P[mech.s_prox_end, 0]],
            [P[mech.s_origin, 1], P[mech.s_prox_end, 1]], "-", lw=6,
            color="#b0b0b0", solid_capstyle="round", zorder=1)
    ax.plot([P[mech.s_origin, 0], P[mech.s_dist_end, 0]],
            [P[mech.s_origin, 1], P[mech.s_dist_end, 1]], "-", lw=6,
            color="#808080", solid_capstyle="round", zorder=1)
    for sa, sb in mech.link_segs:
        ax.plot([P[sa, 0], P[sb, 0]], [P[sa, 1], P[sb, 1]], "-", lw=3,
                color="#2c7fb8", zorder=3)
        ax.plot([P[sa, 0], P[sb, 0]], [P[sa, 1], P[sb, 1]], "o", ms=4,
                color="#2c7fb8", zorder=4)
    for a, path, dec in zip(mech.genome.actuators, mech.act_paths, mech.decs):
        if a.kind == "rotary":
            th = np.linspace(0, 2 * np.pi, 40)
            rr = 0.018 + 0.012 * min(dec["tau_m"] / act_mod.TAU_M_HI, 1.0)
            ax.plot(rr * np.cos(th), rr * np.sin(th), "-", lw=2.5,
                    color="#d95f02", zorder=5)
            continue
        pts = P[path]
        style = "--" if a.kind == "tendon" else "-"
        ax.plot(pts[:, 0], pts[:, 1], style, lw=2.5, color="#d95f02", zorder=5)
        ax.plot(pts[:, 0], pts[:, 1], "s", ms=3.5, color="#d95f02", zorder=6)
    for s, path in zip(mech.genome.springs, mech.spr_paths):
        if not path:
            continue
        p0, p1 = P[path[0]], P[path[1]]
        tvec = p1 - p0
        L = np.linalg.norm(tvec)
        if L < 1e-6:
            continue
        u = tvec / L
        n = np.array([-u[1], u[0]])
        s_par = np.linspace(0, 1, 24)
        zz = np.where(np.arange(24) % 2 == 0, 1, -1) * 0.006
        zz[:2] = zz[-2:] = 0
        pts = p0[None, :] + s_par[:, None] * tvec[None, :] + zz[:, None] * n
        ax.plot(pts[:, 0], pts[:, 1], "-", lw=1.5, color="#1b9e77", zorder=5)
    ax.set_aspect("equal")
    ax.axis("off")


def run_demo(out_dir: str, pop: int, gens: int, seed: int, log=print):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    ctx = demo_elbow()
    traj = elbow_trajectory()
    t0 = time.time()
    lines = []

    def tee(msg):
        lines.append(str(msg))
        log(msg)

    res = evolve(ctx, traj, pop=pop, gens=gens, seed=seed, log=tee)
    wall = time.time() - t0
    tee(f"wall clock {wall:.1f} s")

    winner: Genome = res["winner"]
    grid, rom = res["grid"], res["rom"]
    mech = CompiledMechanism(winner, ctx)
    table = mech.build_table(grid)
    ok, reasons, cov = validate(mech, table, rom)

    met_g = simulate_one(table, traj, ctx, "greedy", return_traces=True)
    met_q = simulate_one(table, traj, ctx, "qp", return_traces=True)
    qp_gap_deg = float(np.degrees(np.sqrt(np.mean(
        (met_g["theta_trace"] - met_q["theta_trace"]) ** 2))))
    tee(f"winner {winner.structure_label()}  gate_ok={ok}  "
        f"rmse_fit {met_g['rmse_fit_deg']:.2f} deg  "
        f"rmse_hold {met_g['rmse_hold_deg']:.2f} deg  "
        f"greedy-vs-QP {qp_gap_deg:.3e} deg RMS")

    design = {
        "schema_version": 2,
        "topology": winner.to_dict(),
        "actuators": describe_actuators(winner),
        "transmission_summary": transmission_summary(table, rom, winner),
        "performance": {
            "rmse_deg": met_g["rmse_fit_deg"],
            "rmse_holdout_deg": met_g["rmse_hold_deg"],
            "energy_J": met_g["energy_total_J"],
            "energy_per_actuator_J": met_g["energy_J"],
            "effort_peak": met_g["u_peak"],
            "effort_demand_peak": met_g["u_demand_peak"],
            "sat_excess": met_g["sat_excess"],
            "greedy_vs_qp_rmse_deg": qp_gap_deg,
            "system_mass_kg": table.mass_total,
        },
        "evolution": {"label": winner.structure_label(),
                      "seed": seed, "pop": pop, "gens": gens,
                      "wall_clock_s": wall},
        "robustness": res["robustness"],
        "robustness_rejected": res["robustness_rejected"],
        "gate": {"ok": ok, "reasons": reasons, "coverage": cov},
    }
    with open(os.path.join(out_dir, "design.json"), "w") as f:
        json.dump(design, f, indent=2)

    hist = res["history"]
    with open(os.path.join(out_dir, "convergence.csv"), "w") as f:
        f.write("gen,best_cost,n_species,gate_pass_frac,mean_parts,best_label\n")
        for h in hist:
            f.write(f"{h['gen']},{h['best_cost']:.4f},{h['n_species']},"
                    f"{h['gate_pass_frac']:.3f},{h['mean_parts']:.2f},"
                    f"{h['best_label']}\n")

    # ---- plots ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    ks = [int(np.argmin(np.abs(grid - v))) for v in
          (np.min(traj.theta_d), 0.5 * (np.min(traj.theta_d)
                                        + np.max(traj.theta_d)),
           np.max(traj.theta_d))]
    for ax, k in zip(axes, ks):
        draw_mechanism(ax, mech, table, k, ctx)
        ax.set_title(f"theta = {np.degrees(grid[k]):.0f} deg", fontsize=9)
    fig.suptitle(f"winner: {winner.structure_label()}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "mechanism.png"), dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    ax = axes[0, 0]
    ax.plot([h["gen"] for h in hist], [h["best_cost"] for h in hist], "-o",
            ms=2.5)
    ax.set_xlabel("generation")
    ax.set_ylabel("best cost")
    ax.set_title("convergence")
    ax2 = ax.twinx()
    ax2.plot([h["gen"] for h in hist], [h["n_species"] for h in hist], "-",
             color="#d95f02", alpha=0.6)
    ax2.set_ylabel("species", color="#d95f02")

    ax = axes[0, 1]
    ax.plot(traj.t, np.degrees(traj.theta_d), "k--", lw=1, label="desired")
    ax.plot(traj.t, np.degrees(met_g["theta_trace"]), lw=1.2, label="achieved")
    for m, c, lab in ((traj.fit_mask, "#1b9e77", "fit"),
                      (traj.hold_mask, "#d95f02", "hold-out")):
        seg = np.where(m)[0]
        if len(seg):
            ax.fill_between(traj.t, 0, 1, where=m, alpha=0.08, color=c,
                            transform=ax.get_xaxis_transform(), label=lab)
    ax.legend(fontsize=7)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("theta (deg)")
    ax.set_title(f"tracking: fit {met_g['rmse_fit_deg']:.2f} deg, "
                 f"hold-out {met_g['rmse_hold_deg']:.2f} deg")

    ax = axes[1, 0]
    for a, ri in zip(winner.actuators, table.r):
        scale = 1000.0 if a.kind != "rotary" else 1.0
        lab = f"{a.id} ({a.kind})" + ("" if a.kind != "rotary" else " ratio")
        ax.plot(np.degrees(grid[rom]), (ri * scale)[rom], label=lab)
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("r (mm | ratio)")
    ax.legend(fontsize=7)
    ax.set_title("transmission ratios")

    ax = axes[1, 1]
    ax.plot(np.degrees(grid[rom]), table.I_eff[rom], label="I_eff (kg m^2)")
    axb = ax.twinx()
    axb.plot(np.degrees(grid[rom]), table.G[rom] + table.S[rom], "--",
             color="#d95f02", label="G+S (N m)")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("I_eff")
    axb.set_ylabel("G+S (N m)", color="#d95f02")
    ax.set_title("reduced dynamics tables")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "analysis.png"), dpi=140)
    plt.close(fig)

    with open(os.path.join(out_dir, "log.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return design, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/demo_elbow")
    ap.add_argument("--pop", type=int, default=90)
    ap.add_argument("--gens", type=int, default=34)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    run_demo(args.out, args.pop, args.gens, args.seed)


if __name__ == "__main__":
    main()
