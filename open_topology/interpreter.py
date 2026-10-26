"""Generic planar mechanism interpreter.

Reads ``genome.bodies`` / ``genome.joints`` directly - there are **no
hand-written per-topology kinematics** (closes G4: what a mutation builds, this
module can simulate).  Loop closure is solved by warm-started Newton with an
analytic Jacobian; derivatives come from the implicit function theorem
(dx/dtheta = -Jx^-1 Jth), per spec 5.3.

The validity gate (spec 5.5) is real for every topology - closing G2/G7/G8/G10:
Newton convergence, Jacobian conditioning (the true dead-centre signal, not a
hand-picked transmission angle), grid-scaled branch-flip detection,
link/actuator/bone self-intersection, limb-envelope packaging for every moving
point, per-actuator buildability (stroke, retracted length, stroke ratio,
packaging, moment-arm floor) and static holdability.  ``mobility()`` is a
Kutzbach count *plus* a numeric Jacobian rank check and can actually fail.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .context import JointContext, G_ACC, LINK_DENSITY
from .genome import Genome, GROUND, OUTPUT
from . import actuators as act_mod

COND_MAX = 1e8
NEWTON_TOL = 1e-10
NEWTON_OK = 1e-9


def rot(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s], [s, c]])


@dataclass
class TransmissionTable:
    theta_grid: np.ndarray          # (N,)
    q: np.ndarray                   # (na, N)
    r: np.ndarray                   # (na, N)  r_i = -dq_i/dtheta
    I_eff: np.ndarray               # (N,)
    dI_dtheta: np.ndarray           # (N,)
    G: np.ndarray                   # (N,)
    S: np.ndarray                   # (N,)
    valid: np.ndarray               # (N,) bool
    cond: np.ndarray                # (N,)
    poses: np.ndarray               # (N, nb, 3) for drawing / round-trip checks
    sites: np.ndarray               # (N, ns, 2)
    decs: list = field(default_factory=list)   # per-actuator decode dicts
    spring_S: np.ndarray | None = None
    mass_total: float = 0.0
    fail: dict = field(default_factory=dict)   # per-point failure diagnostics


class CompiledMechanism:
    def __init__(self, genome: Genome, ctx: JointContext):
        self.genome, self.ctx = genome, ctx
        self.body_ids = [GROUND, OUTPUT] + [b.id for b in genome.bodies]
        self.bidx = {b: i for i, b in enumerate(self.body_ids)}
        self.nc = len(genome.bodies)
        self.nx = 3 * self.nc
        self.phi0 = ctx.phi_out0

        # --- constraints (output revolute is implicit in the output pose) ----
        self.cons = []
        for j in genome.joints:
            self.cons.append((self.bidx[j.body_a],
                              np.array([j.p["ax"], j.p["ay"]]),
                              self.bidx[j.body_b],
                              np.array([j.p["bx"], j.p["by"]])))
        self.m = 2 * len(self.cons)

        # --- site registry --------------------------------------------------
        self._sites: list = []          # (body_idx, local(2,))
        self._site_of: dict = {}

        def site(body_id: str, lx: float, ly: float) -> int:
            k = (body_id, round(lx, 9), round(ly, 9))
            if k not in self._site_of:
                self._site_of[k] = len(self._sites)
                self._sites.append((self.bidx[body_id], np.array([lx, ly])))
            return self._site_of[k]

        up = ctx.prox_dir

        def ground_xy(s, n):
            nrm = np.array([-up[1], up[0]])
            v = s * up + n * nrm
            return v[0], v[1]

        # bones
        self.s_origin = site(GROUND, 0.0, 0.0)
        self.s_prox_end = site(GROUND, *(ctx.L_prox * up))
        self.s_dist_end = site(OUTPUT, ctx.L_dist, 0.0)

        # coupler link endpoints (rod spans (0,0)->(L,0))
        self.link_segs = []             # (site_a, site_b) per coupler
        for b in genome.bodies:
            self.link_segs.append((site(b.id, 0.0, 0.0),
                                   site(b.id, b.p["length"], 0.0)))

        # joint anchor sites (for adjacency grouping)
        self.joint_sites = []
        for j in genome.joints:
            pa = (ground_xy(j.p["ax"], j.p["ay"]) if j.body_a == GROUND
                  else (j.p["ax"], j.p["ay"]))
            pb = (ground_xy(j.p["bx"], j.p["by"]) if j.body_b == GROUND
                  else (j.p["bx"], j.p["by"]))
            self.joint_sites.append((site(j.body_a, *pa), site(j.body_b, *pb)))
        # rebuild constraint anchors in body-local terms consistent with sites
        self.cons = []
        for j, (sa, sb) in zip(genome.joints, self.joint_sites):
            ia, la = self._sites[sa]
            ib, lb = self._sites[sb]
            self.cons.append((ia, la, ib, lb))
        self.m = 2 * len(self.cons)

        # actuator paths
        self.decs = [act_mod.decode(a) for a in genome.actuators]
        self.act_paths = []             # list[list[site]] ; [] for rotary
        for a in genome.actuators:
            if a.kind == "rotary":
                self.act_paths.append([])
                continue
            pa = (ground_xy(a.p["ax"], a.p["ay"]) if a.body_a == GROUND
                  else (a.p["ax"], a.p["ay"]))
            pb = (ground_xy(a.p["bx"], a.p["by"]) if a.body_b == GROUND
                  else (a.p["bx"], a.p["by"]))
            path = [site(a.body_a, *pa)]
            for v in a.via:
                vb, vx, vy = v[0], v[1], v[2]
                if vb == GROUND:
                    vx, vy = ground_xy(vx, vy)
                path.append(site(vb, vx, vy))
            path.append(site(a.body_b, *pb))
            self.act_paths.append(path)

        # spring paths
        self.spr_decs = [act_mod.spring_decode(s) for s in genome.springs]
        self.spr_paths = []
        for s in genome.springs:
            if s.kind == "torsional":
                self.spr_paths.append([])
                continue
            pa = (ground_xy(s.p["ax"], s.p["ay"]) if s.body_a == GROUND
                  else (s.p["ax"], s.p["ay"]))
            pb = (ground_xy(s.p["bx"], s.p["by"]) if s.body_b == GROUND
                  else (s.p["bx"], s.p["by"]))
            self.spr_paths.append([site(s.body_a, *pa), site(s.body_b, *pb)])

        # point masses: (site, mass) ; body rotary inertias handled separately
        self.masses = [(site(OUTPUT, ctx.com, 0.0), ctx.mass)]
        if ctx.payload_mass > 0:
            self.masses.append((site(OUTPUT, ctx.payload_dist, 0.0),
                                ctx.payload_mass))
        for b in genome.bodies:
            L = b.p["length"]
            self.masses.append((site(b.id, 0.5 * L, 0.0), LINK_DENSITY * L))
        for a, dec, path in zip(genome.actuators, self.decs, self.act_paths):
            if dec["mass"] > 0 and path:
                self.masses.append((path[0], 0.5 * dec["mass"]))
                self.masses.append((path[-1], 0.5 * dec["mass"]))

        # body rotational inertias about own COM: (body_idx, I_com)
        self.body_I = [(1, ctx.I_com)]
        for b in genome.bodies:
            L = b.p["length"]
            self.body_I.append((self.bidx[b.id], LINK_DENSITY * L * L * L / 12.0))

        self.refl_J = sum(d["refl_J"] for d in self.decs)
        self.mass_total = (sum(d["mass"] for d in self.decs)
                           + sum(LINK_DENSITY * b.p["length"]
                                 for b in genome.bodies))
        self.ns = len(self._sites)

    # ---------------------------------------------------------------- poses
    def _poses_at(self, theta: float, x: np.ndarray) -> np.ndarray:
        poses = np.zeros((len(self.body_ids), 3))
        poses[1, 2] = self.phi0 + theta
        for i in range(self.nc):
            poses[2 + i] = x[3 * i: 3 * i + 3]
        return poses

    def _point(self, poses, bidx, local):
        x, y, ph = poses[bidx]
        c, s = np.cos(ph), np.sin(ph)
        return np.array([x + c * local[0] - s * local[1],
                         y + s * local[0] + c * local[1]])

    def _point_jac(self, poses, bidx, local):
        """d p / d pose  -> (2,3)."""
        _, _, ph = poses[bidx]
        c, s = np.cos(ph), np.sin(ph)
        return np.array([[1.0, 0.0, -(s * local[0] + c * local[1])],
                         [0.0, 1.0, (c * local[0] - s * local[1])]])

    def _residual(self, theta: float, x: np.ndarray):
        poses = self._poses_at(theta, x)
        F = np.zeros(self.m)
        Jx = np.zeros((self.m, self.nx))
        Jth = np.zeros(self.m)
        for k, (ia, la, ib, lb) in enumerate(self.cons):
            row = 2 * k
            pa = self._point(poses, ia, la)
            pb = self._point(poses, ib, lb)
            F[row:row + 2] = pa - pb
            for idx, loc, sgn in ((ia, la, 1.0), (ib, lb, -1.0)):
                J = self._point_jac(poses, idx, loc)
                if idx >= 2:
                    c0 = 3 * (idx - 2)
                    Jx[row:row + 2, c0:c0 + 3] += sgn * J
                elif idx == 1:
                    Jth[row:row + 2] += sgn * J[:, 2]
        return F, Jx, Jth

    def solve_point(self, theta: float, x0: np.ndarray):
        x = x0.copy()
        ok = self.nx == 0
        for _ in range(30):
            F, Jx, Jth = self._residual(theta, x)
            nF = np.linalg.norm(F) if self.m else 0.0
            if nF < NEWTON_TOL:
                ok = True
                break
            try:
                dx = np.linalg.solve(Jx, -F)
            except np.linalg.LinAlgError:
                return x, False, np.inf, np.zeros(self.nx)
            x = x + np.clip(dx, -0.25, 0.25)
        F, Jx, Jth = self._residual(theta, x)
        ok = (np.linalg.norm(F) if self.m else 0.0) <= NEWTON_OK
        if self.nx:
            cond = np.linalg.cond(Jx)
            try:
                dxdth = np.linalg.solve(Jx, -Jth)
            except np.linalg.LinAlgError:
                dxdth = np.zeros(self.nx)
                ok = False
        else:
            cond, dxdth = 1.0, np.zeros(0)
        return x, ok, cond, dxdth

    def mobility(self) -> int:
        """Kutzbach + numeric rank (R-5).  1 = usable single-DOF mechanism."""
        kutz = 3 * (1 + self.nc) - 2 * (1 + len(self.cons))
        if kutz != 1 or self.m != self.nx:
            return kutz
        if self.nx == 0:
            return 1
        x0 = self._hint_x()
        x, ok, cond, _ = self.solve_point(self.genome.theta_ref, x0)
        if not ok:
            return 0                      # does not assemble at reference
        _, Jx, _ = self._residual(self.genome.theta_ref, x)
        rank = np.linalg.matrix_rank(Jx, tol=1e-9)
        return 1 + (self.nx - rank)

    def _hint_x(self) -> np.ndarray:
        x = np.zeros(self.nx)
        for i, b in enumerate(self.genome.bodies):
            h = self.genome.hints.get(b.id)
            if h is not None:
                x[3 * i: 3 * i + 3] = h
        return x

    # ---------------------------------------------------------------- table
    def build_table(self, theta_grid: np.ndarray) -> TransmissionTable:
        N = len(theta_grid)
        na = len(self.genome.actuators)
        xs = np.zeros((N, self.nx))
        ok = np.zeros(N, bool)
        cond = np.ones(N)
        dxdths = np.zeros((N, self.nx))
        branch = np.zeros(N, bool)

        k0 = int(np.argmin(np.abs(theta_grid - self.genome.theta_ref)))
        order = list(range(k0, N)) + list(range(k0 - 1, -1, -1))
        xw = self._hint_x()
        prev = {}
        for k in order:
            if k == k0 - 1:                       # restart downward from ref
                xw = xs[k0].copy()
                prev = {"k": k0}
            start = xw
            if "k" in prev and self.nx:
                dth = theta_grid[k] - theta_grid[prev["k"]]
                start = xs[prev["k"]] + dxdths[prev["k"]] * dth
            x, o, c, dxdth = self.solve_point(theta_grid[k], start)
            xs[k], ok[k], cond[k], dxdths[k] = x, o, min(c, 1e12), dxdth
            if "k" in prev and self.nx and o:
                dth = abs(theta_grid[k] - theta_grid[prev["k"]])
                pred = xs[prev["k"]] + dxdths[prev["k"]] * (theta_grid[k]
                                                            - theta_grid[prev["k"]])
                tol = max(0.02, 3.0 * np.linalg.norm(dxdths[prev["k"]]) * dth)
                branch[k] = np.linalg.norm(x - pred) > tol
            xw = x
            prev = {"k": k}

        # ---- vectorised world kinematics over the grid ----------------------
        nb = len(self.body_ids)
        poses = np.zeros((N, nb, 3))
        poses[:, 1, 2] = self.phi0 + theta_grid
        dposes = np.zeros((N, nb, 3))
        dposes[:, 1, 2] = 1.0
        for i in range(self.nc):
            poses[:, 2 + i, :] = xs[:, 3 * i: 3 * i + 3]
            dposes[:, 2 + i, :] = dxdths[:, 3 * i: 3 * i + 3]

        P = np.zeros((N, self.ns, 2))
        dP = np.zeros((N, self.ns, 2))
        for si, (bi, loc) in enumerate(self._sites):
            ph = poses[:, bi, 2]
            c, s = np.cos(ph), np.sin(ph)
            rx = c * loc[0] - s * loc[1]
            ry = s * loc[0] + c * loc[1]
            P[:, si, 0] = poses[:, bi, 0] + rx
            P[:, si, 1] = poses[:, bi, 1] + ry
            dph = dposes[:, bi, 2]
            dP[:, si, 0] = dposes[:, bi, 0] - dph * ry
            dP[:, si, 1] = dposes[:, bi, 1] + dph * rx

        # actuator coordinates
        q = np.zeros((na, N))
        r = np.zeros((na, N))
        for ai, (a, dec, path) in enumerate(zip(self.genome.actuators,
                                                self.decs, self.act_paths)):
            if a.kind == "rotary":
                q[ai] = -dec["r_const"] * theta_grid
                r[ai] = dec["r_const"]
                continue
            qa = np.zeros(N)
            dqa = np.zeros(N)
            for s0, s1 in zip(path[:-1], path[1:]):
                d = P[:, s1] - P[:, s0]
                L = np.linalg.norm(d, axis=1)
                L = np.maximum(L, 1e-9)
                qa += L
                dd = dP[:, s1] - dP[:, s0]
                dqa += (d * dd).sum(axis=1) / L
            q[ai] = qa
            r[ai] = -dqa

        # I_eff and gravity
        I_eff = np.full(N, self.refl_J)
        Gv = np.zeros(N)
        for si, mass in self.masses:
            I_eff += mass * (dP[:, si] ** 2).sum(axis=1)
            Gv += -mass * G_ACC * dP[:, si, 1]
        for bi, Ic in self.body_I:
            I_eff += Ic * dposes[:, bi, 2] ** 2
        I_eff = np.maximum(I_eff, 1e-6)
        dI = np.gradient(I_eff, theta_grid)

        # springs
        S = np.zeros(N)
        for sg, dec, path in zip(self.genome.springs, self.spr_decs,
                                 self.spr_paths):
            if dec["k"] <= 0:
                continue
            if sg.kind == "torsional":
                S += -dec["k"] * (theta_grid - dec["q_free"])
            else:
                d = P[:, path[1]] - P[:, path[0]]
                L = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
                dd = dP[:, path[1]] - dP[:, path[0]]
                dq = (d * dd).sum(axis=1) / L
                S += dec["k"] * (L - dec["q_free"]) * (-dq)

        # geometric validity: envelope + self-intersection
        env_ok = self._envelope_ok(P)
        cross = self._self_intersect(P)

        valid = ok & (~branch) & (cond < COND_MAX) & env_ok & (~cross)
        fail = {"newton": ~ok, "branch": branch, "cond": cond >= COND_MAX,
                "envelope": ~env_ok, "intersect": cross}
        return TransmissionTable(theta_grid, q, r, I_eff, dI, Gv, S, valid,
                                 cond, poses, P, self.decs, S,
                                 self.mass_total, fail)

    # ------------------------------------------------------------- geometry
    def _segments(self):
        segs = [(self.s_origin, self.s_prox_end), (self.s_origin, self.s_dist_end)]
        segs += list(self.link_segs)
        for path in self.act_paths:
            segs += list(zip(path[:-1], path[1:]))
        return segs

    def _adjacency_groups(self, P0):
        """Union sites that coincide in world space at theta_ref (joints)."""
        group = list(range(self.ns))

        def find(i):
            while group[i] != i:
                group[i] = group[group[i]]
                i = group[i]
            return i

        for i in range(self.ns):
            for j in range(i + 1, self.ns):
                if np.linalg.norm(P0[i] - P0[j]) < 1.5e-3:
                    group[find(i)] = find(j)
        return [find(i) for i in range(self.ns)]

    def _self_intersect(self, P: np.ndarray) -> np.ndarray:
        """(N,) bool - any two non-adjacent segments properly cross (G8)."""
        N = P.shape[0]
        segs = self._segments()
        grp = self._adjacency_groups(P[N // 2])
        out = np.zeros(N, bool)
        for i in range(len(segs)):
            a0, a1 = segs[i]
            for j in range(i + 1, len(segs)):
                b0, b1 = segs[j]
                if len({grp[a0], grp[a1]} & {grp[b0], grp[b1]}) > 0:
                    continue
                p, pr = P[:, a0], P[:, a1] - P[:, a0]
                qq, qs = P[:, b0], P[:, b1] - P[:, b0]
                den = pr[:, 0] * qs[:, 1] - pr[:, 1] * qs[:, 0]
                dq = qq - p
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = (dq[:, 0] * qs[:, 1] - dq[:, 1] * qs[:, 0]) / den
                    u = (dq[:, 0] * pr[:, 1] - dq[:, 1] * pr[:, 0]) / den
                hit = (np.abs(den) > 1e-12) & (t > 0.02) & (t < 0.98) \
                    & (u > 0.02) & (u < 0.98)
                out |= np.nan_to_num(hit)
        return out

    def _envelope_ok(self, P: np.ndarray) -> np.ndarray:
        """Every moving site within a capsule around either bone (G8)."""
        N = P.shape[0]
        tol = self.ctx.envelope_r + 0.005
        a1 = P[:, self.s_origin]                       # origin (0,0)
        prox_e = P[:, self.s_prox_end]
        dist_e = P[:, self.s_dist_end]

        def dist_to_seg(pts, e0, e1):
            d = e1 - e0
            L2 = (d ** 2).sum(axis=1, keepdims=True)
            t = np.clip(((pts - e0) * d).sum(axis=1, keepdims=True) / L2, 0, 1)
            proj = e0 + t * d
            return np.linalg.norm(pts - proj, axis=1)

        ok = np.ones(N, bool)
        skip = {self.s_prox_end, self.s_dist_end}
        for si in range(self.ns):
            if si in skip:
                continue
            pts = P[:, si]
            d1 = dist_to_seg(pts, a1, prox_e)
            d2 = dist_to_seg(pts, a1, dist_e)
            ok &= (np.minimum(d1, d2) <= tol)
        return ok


# ------------------------------------------------------------------ validity gate
def validate(mech: CompiledMechanism, table: TransmissionTable,
             rom: np.ndarray) -> tuple:
    """Hard gate (R-9).  Returns (ok, reasons, coverage)."""
    reasons = []
    mob = mech.mobility()
    if mob != 1:
        reasons.append(f"mobility={mob}")
    v = table.valid[rom]
    coverage = float(np.mean(v)) if v.size else 0.0
    if not np.all(v):
        for name, mask in table.fail.items():
            if np.any(mask[rom]):
                reasons.append(name)
    for a, dec, qi, ri in zip(mech.genome.actuators, table.decs,
                              table.q, table.r):
        for c in act_mod.actuator_constraints(a, dec, qi, ri, rom):
            reasons.append(f"{a.id}:{c}")
    # static holdability with 1.3 margin (direction-aware, tendons unilateral)
    load = table.G[rom] + table.S[rom]
    cap_pos = np.zeros(rom.sum())
    cap_neg = np.zeros(rom.sum())
    for dec, ri in zip(table.decs, table.r):
        rr = ri[rom]
        if dec["tendon"]:
            cap_pos += np.maximum(rr, 0) * dec["u_hi"]
            cap_neg += np.maximum(-rr, 0) * dec["u_hi"]
        else:
            cap_pos += np.abs(rr) * dec["u_hi"]
            cap_neg += np.abs(rr) * dec["u_hi"]
    need = -load                          # torque actuators must supply to hold
    bad = np.where(need > 0, cap_pos < 1.3 * need, cap_neg < 1.3 * (-need))
    if np.any(bad):
        reasons.append("holdability")
    return (len(reasons) == 0), reasons, coverage
