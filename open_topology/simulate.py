"""Reduced-EOM simulation, vectorised over signature-batched sub-populations.

Integrates exactly (R-7):

    I_eff(th) th'' + 1/2 dI/dth th'^2 = G(th) + S(th) + sum_i r_i(th) u_i - b th'

Controller: PD + gravity feed-forward (unchanged from the legacy engine).

Round-2 fixes:
* G5  - the joint-speed limit is a min over *engaged* actuators only: zero
        capacity actuators and slack tendons cannot throttle the joint.
* G6  - energy is accumulated **per actuator** with that actuator's
        efficiency: sum_i relu(u_i r_i omega)/eta_i.  Co-contraction between
        opposed actuators is paid for, not cancelled.
* M5  - the unclipped per-actuator demand is recorded, under the *same*
        allocation weights as the applied effort.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .context import JointContext


@dataclass
class Trajectory:
    t: np.ndarray
    theta_d: np.ndarray
    omega_d: np.ndarray
    dt: float
    fit_mask: np.ndarray
    hold_mask: np.ndarray


def controller_gains(ctx: JointContext, wn: float = 14.0, zeta: float = 0.9):
    I_nom = ctx.I_about_joint + ctx.payload_mass * ctx.payload_dist ** 2
    return I_nom * wn ** 2, 2.0 * zeta * I_nom * wn


# ------------------------------------------------------------------ allocation
def alloc_greedy(r, u_hi, tendon, tau):
    """Capacity/headroom-weighted greedy allocation (spec 5.6 population policy).

    r (P,na), u_hi (P,na), tendon (P,na) bool, tau (P,) -> (u, u_dem)."""
    lo = np.where(tendon, 0.0, -u_hi)
    hi = u_hi
    u = np.zeros_like(r)
    u_dem = None
    for it in range(2):
        resid = tau - (r * u).sum(axis=1)
        pos = resid[:, None] >= 0
        head = np.where(pos,
                        np.where(r > 0, (hi - u) * r, (lo - u) * r),
                        np.where(r > 0, (u - lo) * r, (u - hi) * r))
        head = np.maximum(head, 0.0)
        tot = head.sum(axis=1)
        frac = head / np.maximum(tot[:, None], 1e-12)
        dtau = resid[:, None] * frac
        du = np.where(np.abs(r) > 1e-9, dtau / np.where(np.abs(r) > 1e-9, r, 1.0),
                      0.0)
        if it == 0:
            u_dem = du.copy()             # unclipped demand, same weights (M5)
        u = np.clip(u + du, lo, hi)
    return u, u_dem


def alloc_qp(r, u_hi, tendon, tau, iters: int = 70):
    """Exact box-constrained QP via bisection on the KKT multiplier (spec 5.6).

    minimise sum w_i u_i^2, w_i = 1/u_hi^2, s.t. sum r_i u_i = tau, bounds."""
    lo = np.where(tendon, 0.0, -u_hi)
    hi = u_hi
    w = 1.0 / (np.maximum(u_hi, 1e-9) ** 2)
    lam_lo = np.full(tau.shape, -1e7)
    lam_hi = np.full(tau.shape, 1e7)
    for _ in range(iters):
        lam = 0.5 * (lam_lo + lam_hi)
        u = np.clip(lam[:, None] * r / (2.0 * w), lo, hi)
        g = (r * u).sum(axis=1) - tau
        lam_hi = np.where(g > 0, lam, lam_hi)
        lam_lo = np.where(g > 0, lam_lo, lam)
    lam = 0.5 * (lam_lo + lam_hi)
    return np.clip(lam[:, None] * r / (2.0 * w), lo, hi)


def per_actuator_energy(u, r, omega, eta, dt) -> np.ndarray:
    """E_i = int relu(u_i r_i omega)/eta_i dt  over a (T,na) effort history.

    This is the G6 fix in isolation: internal work between opposed actuators
    does NOT cancel, unlike the net-power form relu(tau_net*omega)."""
    P = u * r * (omega[:, None] if omega.ndim == 1 else omega)
    return (np.maximum(P, 0.0) / eta).sum(axis=0) * dt


# ------------------------------------------------------------------ batched sim
def simulate_batch(tables: list, traj: Trajectory, ctx: JointContext,
                   allocation: str = "greedy", return_traces: bool = False):
    """Simulate candidates sharing an actuator-kind signature on a common grid.

    ``tables`` is a list of TransmissionTable built on the SAME theta grid."""
    P = len(tables)
    grid = tables[0].theta_grid
    g0, h, N = grid[0], grid[1] - grid[0], len(grid)
    na = tables[0].q.shape[0]

    R = np.stack([tb.r for tb in tables])            # (P,na,N)
    I = np.stack([tb.I_eff for tb in tables])        # (P,N)
    dI = np.stack([tb.dI_dtheta for tb in tables])
    Gt = np.stack([tb.G for tb in tables])
    St = np.stack([tb.S for tb in tables])
    if na:
        u_hi = np.stack([[d["u_hi"] for d in tb.decs] for tb in tables])
        v_hi = np.stack([[d["v_hi"] for d in tb.decs] for tb in tables])
        eta = np.stack([[d["eta"] for d in tb.decs] for tb in tables])
        tendon = np.stack([[d["tendon"] for d in tb.decs] for tb in tables])
    else:
        u_hi = v_hi = eta = np.zeros((P, 0))
        tendon = np.zeros((P, 0), bool)

    Kp, Kd = controller_gains(ctx)
    b = ctx.damping
    dt = traj.dt
    T = len(traj.t)
    idxP = np.arange(P)

    th = np.full(P, traj.theta_d[0])
    om = np.zeros(P)
    se_fit = np.zeros(P)
    se_hold = np.zeros(P)
    E = np.zeros((P, na))
    u_peak = np.zeros((P, na))
    dem_peak = np.zeros((P, na))
    sat = np.zeros((P, na))
    traces = np.zeros((T, P)) if return_traces else None
    u_traces = np.zeros((T, P, na)) if return_traces else None

    lo_edge, hi_edge = grid[0], grid[-1]
    for k in range(T):
        pos = np.clip((th - g0) / h, 0.0, N - 1.001)
        i0 = pos.astype(np.int64)
        w1 = pos - i0

        def gather2(A):                              # (P,N) -> (P,)
            return A[idxP, i0] * (1 - w1) + A[idxP, i0 + 1] * w1

        rk = (R[idxP[:, None], np.arange(na)[None, :], i0[:, None]]
              * (1 - w1)[:, None]
              + R[idxP[:, None], np.arange(na)[None, :], i0[:, None] + 1]
              * w1[:, None]) if na else np.zeros((P, 0))
        Ik, dIk = gather2(I), gather2(dI)
        Gk, Sk = gather2(Gt), gather2(St)

        e = traj.theta_d[k] - th
        de = traj.omega_d[k] - om
        tau_cmd = Kp * e + Kd * de - Gk              # gravity feed-forward
        if na:
            if allocation == "qp":
                u = alloc_qp(rk, u_hi, tendon, tau_cmd)
                _, u_dem = alloc_greedy(rk, u_hi, tendon, tau_cmd)
            else:
                u, u_dem = alloc_greedy(rk, u_hi, tendon, tau_cmd)
            tau_act = (rk * u).sum(axis=1)
        else:
            u = u_dem = np.zeros((P, 0))
            tau_act = np.zeros(P)

        acc = (Gk + Sk + tau_act - b * om - 0.5 * dIk * om ** 2) / Ik
        om = om + dt * acc

        if na:
            # engaged-actuator speed limit (G5)
            engaged = (u_hi > 1e-9) & (~tendon | (u > 1e-9))
            with np.errstate(divide="ignore"):
                om_cap = np.where(engaged,
                                  v_hi / np.maximum(np.abs(rk), 1e-9), np.inf)
            om_lim = om_cap.min(axis=1)
            om = np.clip(om, -om_lim, om_lim)

            Pmech = u * rk * om[:, None]
            E += np.maximum(Pmech, 0.0) / eta * dt
            u_peak = np.maximum(u_peak, np.abs(u))
            dem_peak = np.maximum(dem_peak, np.abs(u_dem))
            sat = np.maximum(sat, (np.abs(u_dem) - u_hi)
                             / np.maximum(u_hi, 1e-9))

        th = np.clip(th + dt * om, lo_edge, hi_edge)
        if return_traces:
            traces[k] = th
            u_traces[k] = u

        err2 = (traj.theta_d[k] - th) ** 2
        se_fit += err2 * traj.fit_mask[k]
        se_hold += err2 * traj.hold_mask[k]

    n_fit = max(int(traj.fit_mask.sum()), 1)
    n_hold = max(int(traj.hold_mask.sum()), 1)
    out = []
    for p in range(P):
        m = {
            "rmse_fit_deg": float(np.degrees(np.sqrt(se_fit[p] / n_fit))),
            "rmse_hold_deg": float(np.degrees(np.sqrt(se_hold[p] / n_hold))),
            "energy_J": [float(x) for x in E[p]],
            "energy_total_J": float(E[p].sum()),
            "u_peak": [float(x) for x in u_peak[p]],
            "u_demand_peak": [float(x) for x in dem_peak[p]],
            "sat_excess": float(np.max(sat[p]) if na else 0.0),
        }
        if return_traces:
            m["theta_trace"] = traces[:, p].copy()
            m["u_trace"] = u_traces[:, p].copy()
        out.append(m)
    return out


def simulate_one(table, traj, ctx, allocation="greedy", return_traces=False):
    return simulate_batch([table], traj, ctx, allocation, return_traces)[0]


# ------------------------------------------------------------------ passive
def integrate_passive(table, theta0: float, t_end: float, dt: float,
                      include_dI: bool = True, damping: float = 0.0):
    """Unactuated integration - used by the energy-conservation test (R-7)."""
    grid = table.theta_grid
    V = -np.concatenate([[0.0], np.cumsum(
        0.5 * (table.G[1:] + table.S[1:] + table.G[:-1] + table.S[:-1])
        * np.diff(grid))])
    def acc(th_, om_):
        Ik = np.interp(th_, grid, table.I_eff)
        dIk = np.interp(th_, grid, table.dI_dtheta) if include_dI else 0.0
        Gk = np.interp(th_, grid, table.G)
        Sk = np.interp(th_, grid, table.S)
        return (Gk + Sk - damping * om_ - 0.5 * dIk * om_ ** 2) / Ik

    steps = int(t_end / dt)
    th, om = theta0, 0.0
    ts, ths, Es = [], [], []
    for k in range(steps):
        Ik = np.interp(th, grid, table.I_eff)
        Vk = np.interp(th, grid, V)
        ts.append(k * dt)
        ths.append(th)
        Es.append(0.5 * Ik * om ** 2 + Vk)
        k1t, k1o = om, acc(th, om)                       # RK4
        k2t, k2o = om + 0.5 * dt * k1o, acc(th + 0.5 * dt * k1t,
                                            om + 0.5 * dt * k1o)
        k3t, k3o = om + 0.5 * dt * k2o, acc(th + 0.5 * dt * k2t,
                                            om + 0.5 * dt * k2o)
        k4t, k4o = om + dt * k3o, acc(th + dt * k3t, om + dt * k3o)
        th += dt / 6.0 * (k1t + 2 * k2t + 2 * k3t + k4t)
        om += dt / 6.0 * (k1o + 2 * k2o + 2 * k3o + k4o)
        th = float(np.clip(th, grid[0], grid[-1]))
    return np.array(ts), np.array(ths), np.array(Es)
