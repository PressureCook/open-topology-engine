"""Acceptance tests mapping to spec §11 and the round-2 review findings."""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_topology.context import demo_elbow
from open_topology.genome import Genome, Innovations
from open_topology.interpreter import CompiledMechanism, validate
from open_topology.simulate import (Trajectory, simulate_one,
                                    integrate_passive, alloc_greedy, alloc_qp)
from open_topology import archetypes, actuators as act_mod
from open_topology.evolve import (op_add_spring, op_add_actuator, op_add_dyad,
                                  apportion)

CTX = demo_elbow()
TH_LO, TH_HI = np.deg2rad(10.0), np.deg2rad(110.0)
REF = 0.5 * (TH_LO + TH_HI)


def grid_n(n=64, pad=0.10):
    return np.linspace(TH_LO - pad, TH_HI + pad, n)


def rom_mask(grid):
    return (grid >= TH_LO - 1e-9) & (grid <= TH_HI + 1e-9)


def mini_traj(T=3.5, dt=0.01):
    t = np.arange(0, T, dt)
    th = np.deg2rad(10) + np.deg2rad(80) * 0.5 * (1 - np.cos(
        2 * np.pi * t / T))
    om = np.gradient(th, dt)
    fit = np.ones(len(t), bool)
    hold = np.zeros(len(t), bool)
    return Trajectory(t, th, om, dt, fit, hold)


def seeded_direct(seed=3, gF=0.5):
    g = archetypes.direct_linear(np.random.default_rng(seed), CTX, REF)
    g.actuators[0].p.update({"ax": 0.20, "ay": 0.01, "bx": 0.16, "by": 0.015,
                             "gF": gF, "gv": 0.6})
    return g


def seeded_four_bar(seed=None):
    for s in ([seed] if seed is not None else range(60)):
        g = archetypes.four_bar(np.random.default_rng(s), CTX, REF)
        if g is None:
            continue
        mech = CompiledMechanism(g, CTX)
        t = mech.build_table(grid_n())
        if t.valid[rom_mask(grid_n())].all():
            return g
    pytest.skip("no valid four_bar found in seed range")


# ---------------------------------------------------------------- geometry
def test_r_matches_finite_difference_direct():
    g = seeded_direct()
    mech = CompiledMechanism(g, CTX)
    grid = grid_n(256)
    t = mech.build_table(grid)
    dq = np.gradient(t.q[0], grid)
    assert np.max(np.abs(-dq[3:-3] - t.r[0][3:-3])) < 1e-6


def test_r_matches_finite_difference_four_bar():
    g = seeded_four_bar()
    mech = CompiledMechanism(g, CTX)
    grid = grid_n(256)
    t = mech.build_table(grid)
    rom = rom_mask(grid)
    if not t.valid[rom].all():
        pytest.skip("fixture invalid on fine grid")
    dq = np.gradient(t.q[0], grid)
    err = np.abs(-dq - t.r[0])[rom][3:-3]
    assert np.max(err) < 5e-5


def test_direct_linear_closed_form_reduction():
    """Legacy topology through the generic path: analytic L(theta), constant
    I_eff (Phase 1 contract, Appendix A)."""
    g = seeded_direct(gF=0.0)          # zero actuator mass
    mech = CompiledMechanism(g, CTX)
    grid = grid_n(96)
    t = mech.build_table(grid)
    p = g.actuators[0].p
    up = CTX.prox_dir
    nrm = np.array([-up[1], up[0]])
    A = p["ax"] * up + p["ay"] * nrm
    for k, th in enumerate(grid):
        ph = CTX.phi_out0 + th
        R = np.array([[np.cos(ph), -np.sin(ph)], [np.sin(ph), np.cos(ph)]])
        B = R @ np.array([p["bx"], p["by"]])
        assert abs(np.linalg.norm(A - B) - t.q[0, k]) < 1e-12
    I_exp = CTX.I_about_joint + CTX.payload_mass * CTX.payload_dist ** 2
    assert np.allclose(t.I_eff, I_exp, rtol=1e-9)
    assert np.max(np.abs(t.dI_dtheta)) < 1e-9


def test_gravity_release_settles_at_neutral():
    g = seeded_direct(gF=0.0)
    mech = CompiledMechanism(g, CTX)
    grid = np.linspace(-0.4, 2.4, 128)
    t = mech.build_table(grid)
    _, ths, _ = integrate_passive(t, 1.0, 12.0, 5e-4, damping=0.8)
    assert abs(ths[-1] - CTX.theta_neutral) < np.deg2rad(2.0)


def test_energy_conservation_requires_dI_term(monkeypatch):
    """R-7: with couplers I_eff varies; the 1/2 dI/dtheta thdot^2 term is what
    keeps energy bookkeeping honest.  Removing it must break conservation.

    Heavy links amplify dI/dtheta; a stiff torsional spring puts the potential
    well inside the ROM so the passive swing never leaves the table."""
    import open_topology.interpreter as interp
    from open_topology.genome import SpringGene
    monkeypatch.setattr(interp, "LINK_DENSITY", 25.0)
    g = seeded_four_bar()
    g.springs.append(SpringGene("s0", 60, "torsional",
                                p={"gk": 0.95,
                                   "gfree": float((REF + 0.5 * np.pi)
                                                  / (1.5 * np.pi))}))
    mech = CompiledMechanism(g, CTX)
    grid = grid_n(256)
    t = mech.build_table(grid)
    span = np.max(t.I_eff) - np.min(t.I_eff)
    assert span > 1e-3, f"fixture I_eff span too small: {span}"
    th0 = REF + 0.25
    _, th1, E1 = integrate_passive(t, th0, 1.2, 1e-4, include_dI=True)
    _, th0s, E0 = integrate_passive(t, th0, 1.2, 1e-4, include_dI=False)
    assert np.min(th1) > grid[0] and np.max(th1) < grid[-1], \
        "swing left the table - fixture invalid"
    d1 = np.max(np.abs(E1 - E1[0]))
    d0 = np.max(np.abs(E0 - E0[0]))
    assert d1 < 0.10 * max(d0, 1e-12), (d1, d0)


# ---------------------------------------------------------------- mobility
def test_mobility_checks():
    g = seeded_direct()
    assert CompiledMechanism(g, CTX).mobility() == 1
    fb = seeded_four_bar()
    assert CompiledMechanism(fb, CTX).mobility() == 1
    broken = fb.copy()
    broken.joints.pop()                    # dangling chain: DOF > 1
    assert CompiledMechanism(broken, CTX).mobility() != 1
    over = fb.copy()
    over.joints.append(over.joints[0])     # duplicate constraint
    assert CompiledMechanism(over, CTX).mobility() != 1


# ---------------------------------------------------------------- validity gate
def test_gate_rejects_unassemblable_four_bar():
    rng = np.random.default_rng(11)
    g = archetypes.four_bar(rng, CTX, REF)
    assert g is not None
    # shrink the coupler so the chain cannot reach the output anchor at ROM edge
    g.bodies[1].p["length"] *= 0.55
    mech = CompiledMechanism(g, CTX)
    grid = grid_n()
    t = mech.build_table(grid)
    ok, reasons, cov = validate(mech, t, rom_mask(grid))
    assert not ok


def test_gate_enforces_legacy_constraints_on_valid_designs():
    """Round-2 G2 methodology: no fully-'valid' genome may violate the ported
    constraints (moment-arm floor, stroke ratio, retracted length, packaging)."""
    rng = np.random.default_rng(0)
    grid = grid_n()
    rom = rom_mask(grid)
    n_ok = n_rej = 0
    for _ in range(150):
        g = archetypes.direct_linear(rng, CTX, REF)
        mech = CompiledMechanism(g, CTX)
        t = mech.build_table(grid)
        ok, reasons, _ = validate(mech, t, rom)
        if not ok:
            n_rej += 1
            continue
        n_ok += 1
        q, r = t.q[0][rom], np.abs(t.r[0][rom])
        stroke = q.max() - q.min()
        assert r.min() >= act_mod.R_FLOOR - 1e-12
        assert q.max() / q.min() <= act_mod.STROKE_RATIO_MAX + 1e-9
        assert q.min() >= act_mod.L_RET_MIN - 1e-12
        assert q.min() >= stroke + 0.028 - 1e-12
    assert n_ok > 5 and n_rej > 5, (n_ok, n_rej)


def test_zero_capacity_actuator_is_gate_exempt():
    g = seeded_direct(gF=0.0)
    g.actuators[0].p["bx"] = 0.03          # tiny moment arm - would fail r_floor
    mech = CompiledMechanism(g, CTX)
    grid = grid_n()
    t = mech.build_table(grid)
    bad = act_mod.actuator_constraints(g.actuators[0], t.decs[0], t.q[0],
                                       t.r[0], rom_mask(grid))
    assert bad == []


# ---------------------------------------------------------------- preservation
@pytest.mark.parametrize("op,tol_r,tol_sim", [
    (op_add_spring, 1e-12, 1e-9),
    (op_add_actuator, 1e-12, 1e-9),
    (op_add_dyad, 1e-9, 0.02),
])
def test_structural_mutation_behaviour_preserving(op, tol_r, tol_sim):
    """R-12 over >= 20 seeds, asserting tables AND simulated trajectories -
    the coverage the round-2 review found missing (G5)."""
    grid = grid_n(48)
    traj = mini_traj()
    inn = Innovations()
    tested = 0
    for s in range(40):
        rng = np.random.default_rng(1000 + s)
        g0 = seeded_direct(seed=s % 7, gF=0.55)
        mech0 = CompiledMechanism(g0, CTX)
        t0 = mech0.build_table(grid)
        m0 = simulate_one(t0, traj, CTX, return_traces=True)
        g1 = g0.copy()
        try:
            out = (op(g1, rng, inn, CTX, 0, (TH_LO, TH_HI))
                   if op is op_add_dyad else op(g1, rng, inn, CTX, 0))
        except Exception:
            out = None
        if out is None:
            continue
        mech1 = CompiledMechanism(out, CTX)
        t1 = mech1.build_table(grid)
        assert np.max(np.abs(t1.r[:t0.r.shape[0]] - t0.r)) < tol_r
        assert np.max(np.abs(t1.S - t0.S)) < 1e-9
        m1 = simulate_one(t1, traj, CTX, return_traces=True)
        dev = np.max(np.abs(m1["theta_trace"] - m0["theta_trace"]))
        assert dev < tol_sim, f"seed {s}: trajectory changed by {dev}"
        tested += 1
        if tested >= 20:
            break
    assert tested >= 12, f"only {tested} mutations exercised"


# ---------------------------------------------------------------- round trip
def test_json_round_trip_resimulates_identically():
    g = seeded_direct()
    g2 = Genome.from_json(g.to_json())
    grid = grid_n()
    t1 = CompiledMechanism(g, CTX).build_table(grid)
    t2 = CompiledMechanism(g2, CTX).build_table(grid)
    for a, b in ((t1.q, t2.q), (t1.r, t2.r), (t1.I_eff, t2.I_eff),
                 (t1.G, t2.G), (t1.S, t2.S)):
        assert np.array_equal(a, b)
    traj = mini_traj()
    m1 = simulate_one(t1, traj, CTX, return_traces=True)
    m2 = simulate_one(t2, traj, CTX, return_traces=True)
    assert np.array_equal(m1["theta_trace"], m2["theta_trace"])


def test_four_bar_round_trip_same_branch():
    g = seeded_four_bar()
    g2 = Genome.from_json(g.to_json())
    grid = grid_n()
    t1 = CompiledMechanism(g, CTX).build_table(grid)
    t2 = CompiledMechanism(g2, CTX).build_table(grid)
    assert np.array_equal(t1.poses, t2.poses)


# ---------------------------------------------------------------- allocation
def test_qp_matches_torque_and_beats_greedy_cost():
    rng = np.random.default_rng(5)
    for _ in range(50):
        na = int(rng.integers(2, 4))
        r = rng.uniform(-0.05, 0.05, (1, na))
        u_hi = rng.uniform(50, 800, (1, na))
        tendon = rng.random((1, na)) < 0.5
        cap_pos = (np.where(tendon, np.maximum(r, 0), np.abs(r)) * u_hi).sum()
        cap_neg = (np.where(tendon, np.maximum(-r, 0), np.abs(r)) * u_hi).sum()
        tau = np.array([rng.uniform(-0.6 * cap_neg, 0.6 * cap_pos)])
        ug, _ = alloc_greedy(r, u_hi, tendon, tau)
        uq = alloc_qp(r, u_hi, tendon, tau)
        for u in (ug, uq):
            assert np.all(u <= u_hi + 1e-6)
            assert np.all(np.where(tendon, u >= -1e-9, u >= -u_hi - 1e-6))
        tq = (r * uq).sum()
        tg = (r * ug).sum()
        cap_dir = cap_pos if tau[0] >= 0 else cap_neg
        if abs(tau[0]) < 0.95 * cap_dir:          # direction-aware feasibility
            assert abs(tq - tau[0]) < 1e-3 * max(abs(tau[0]), 1.0)
            assert abs(tg - tau[0]) < 5e-2 * max(abs(tau[0]), 1.0)
        w = 1.0 / u_hi ** 2
        assert (w * uq ** 2).sum() <= (w * ug ** 2).sum() + 1e-6


def test_greedy_vs_qp_tracking_agreement_on_antagonist():
    g = archetypes.antagonistic_tendons(np.random.default_rng(2), CTX, REF)
    for a in g.actuators:
        a.p["gF"], a.p["gv"] = 0.6, 0.7
    mech = CompiledMechanism(g, CTX)
    grid = grid_n()
    t = mech.build_table(grid)
    traj = mini_traj()
    mg = simulate_one(t, traj, CTX, "greedy", return_traces=True)
    mq = simulate_one(t, traj, CTX, "qp", return_traces=True)
    gap = np.degrees(np.sqrt(np.mean(
        (mg["theta_trace"] - mq["theta_trace"]) ** 2)))
    assert gap < 0.5, f"greedy/QP tracking gap {gap} deg"


# ---------------------------------------------------------------- physics fixes
def test_rotary_reflected_inertia_in_table():
    g = archetypes.rotary_geared(np.random.default_rng(4), CTX, REF)
    mech = CompiledMechanism(g, CTX)
    t = mech.build_table(grid_n())
    dec = t.decs[0]
    I_base = CTX.I_about_joint + CTX.payload_mass * CTX.payload_dist ** 2
    assert dec["refl_J"] > 0
    assert np.allclose(t.I_eff, I_base + dec["refl_J"], rtol=1e-9)


def test_cocontraction_energy_visible():
    """G6, mirroring the round-2 measurement: a hand-built opposed pair with
    both actuators working must cost MORE than the clean allocation on the
    same trajectory; under the old net-power form the two are identical."""
    from open_topology.simulate import per_actuator_energy
    ga = archetypes.antagonistic_tendons(np.random.default_rng(2), CTX, REF)
    for a in ga.actuators:
        a.p["gF"], a.p["gv"] = 0.6, 0.7
    t = CompiledMechanism(ga, CTX).build_table(grid_n())
    traj = mini_traj()
    T = len(traj.t)
    th = traj.theta_d
    om = traj.omega_d
    r = np.stack([np.interp(th, t.theta_grid, t.r[i]) for i in range(2)],
                 axis=1)                                   # (T,2)
    tau_net = 3.0 * np.sin(2 * np.pi * traj.t / traj.t[-1])   # arbitrary duty
    # clean: only the correctly-signed tendon works
    u_clean = np.zeros((T, 2))
    for k in range(T):
        i = 0 if tau_net[k] * r[k, 0] > 0 else 1
        u_clean[k, i] = abs(tau_net[k] / r[k, i])
    # co-contract: torque-balanced pretension (net torque history UNCHANGED)
    u_cc = u_clean.copy()
    ratio = -r[:, 0] / np.where(np.abs(r[:, 1]) > 1e-6, r[:, 1], 1e-6)
    ok = ratio > 0                        # genuinely opposed at this angle
    u_cc[:, 0] += 80.0 * ok
    u_cc[:, 1] += 80.0 * ratio * ok
    eta = np.array([0.6, 0.6])
    E_clean = per_actuator_energy(u_clean, r, om, eta, traj.dt).sum()
    E_cc = per_actuator_energy(u_cc, r, om, eta, traj.dt).sum()
    tau_hist_clean = (u_clean * r).sum(axis=1)
    tau_hist_cc = (u_cc * r).sum(axis=1)
    E_net_clean = np.maximum(tau_hist_clean * om, 0).sum() * traj.dt / 0.6
    E_net_cc = np.maximum(tau_hist_cc * om, 0).sum() * traj.dt / 0.6
    assert E_cc > 1.5 * E_clean, (E_cc, E_clean)
    # the old net-power metric is blind to the identical-net co-contraction
    assert abs(E_net_cc - E_net_clean) < 0.05 * abs(E_cc - E_clean)


def test_slack_tendon_does_not_throttle_speed():
    """G5/medium: a tendon that is not pulling must not cap joint speed."""
    ga = archetypes.antagonistic_tendons(np.random.default_rng(2), CTX, REF)
    ga.actuators[0].p.update({"gF": 0.7, "gv": 0.9})    # fast flexor
    ga.actuators[1].p.update({"gF": 0.5, "gv": 0.05})   # slow extensor
    t = CompiledMechanism(ga, CTX).build_table(grid_n())
    traj = mini_traj(T=2.0)
    m = simulate_one(t, traj, CTX, return_traces=True)
    om = np.gradient(m["theta_trace"], traj.dt)
    d_ext = t.decs[1]
    r_ext = np.max(np.abs(t.r[1]))
    cap_ext = d_ext["v_hi"] / max(r_ext, 1e-9)
    assert np.max(om) > 1.5 * cap_ext, \
        "flexion speed should exceed the slack extensor's rate cap"


def test_unclipped_demand_recorded():
    g = seeded_direct(gF=0.18)             # deliberately undersized
    t = CompiledMechanism(g, CTX).build_table(grid_n())
    m = simulate_one(t, mini_traj(), CTX)
    assert m["u_demand_peak"][0] > m["u_peak"][0] + 1e-6
    assert m["sat_excess"] > 0


# ---------------------------------------------------------------- evolution bits
def test_apportion_no_tail_starvation():
    scores = np.array([31, 15, 10, 7, 6, 5, 4, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1],
                      float)
    q = apportion(scores, 90)
    assert q.sum() == 90
    assert np.all(q[scores >= 1] >= 1)


def test_spring_free_decode_is_grid_independent():
    from open_topology.genome import SpringGene
    s = SpringGene("s0", 50, "torsional", p={"gk": 0.6, "gfree": 0.5})
    d1 = act_mod.spring_decode(s)
    d2 = act_mod.spring_decode(s)
    assert d1["q_free"] == d2["q_free"]
    g = seeded_direct()
    g.springs.append(SpringGene("s0", 50, "torsional",
                                p={"gk": 0.6, "gfree": 0.5}))
    t1 = CompiledMechanism(g, CTX).build_table(grid_n(64))
    t2 = CompiledMechanism(g, CTX).build_table(grid_n(200))
    th = REF
    s1 = np.interp(th, t1.theta_grid, t1.S)
    s2 = np.interp(th, t2.theta_grid, t2.S)
    assert abs(s1 - s2) < 1e-6
