"""Seed/immigrant genome constructors (spec 6.5).

These are *seeding conveniences only*: every archetype is expressed as a plain
mechanism graph and interpreted by the generic solver - there is no archetype
specific kinematic code anywhere downstream (the round-2 G4 objection).
Structural skeletons use fixed low innovation ids so seeded structures align
across the run; mutation-created elements draw ids >= 100.
"""
from __future__ import annotations

import numpy as np

from .genome import (Genome, BodyGene, JointGene, ActuatorGene,
                     GROUND, OUTPUT)

# fixed skeleton innovation ids
INN_DIRECT_ACT = 1
INN_FB_CRANK, INN_FB_COUPLER = 2, 3
INN_FB_J1, INN_FB_J2, INN_FB_J3 = 4, 5, 6
INN_FB_ACT = 7
INN_ROTARY_ACT = 8
INN_TENDON_FLEX, INN_TENDON_EXT = 9, 10


def _out_world(ctx, theta, local):
    ph = ctx.phi_out0 + theta
    c, s = np.cos(ph), np.sin(ph)
    return np.array([c * local[0] - s * local[1], s * local[0] + c * local[1]])


def _ground_world(ctx, s, n):
    up = ctx.prox_dir
    nrm = np.array([-up[1], up[0]])
    return s * up + n * nrm


def _cross(p0, p1, q0, q1) -> bool:
    """Proper interior segment crossing (same predicate as the gate)."""
    pr, qs = p1 - p0, q1 - q0
    den = pr[0] * qs[1] - pr[1] * qs[0]
    if abs(den) < 1e-12:
        return False
    dq = q0 - p0
    t = (dq[0] * qs[1] - dq[1] * qs[0]) / den
    u = (dq[0] * pr[1] - dq[1] * pr[0]) / den
    return 0.02 < t < 0.98 and 0.02 < u < 0.98


def direct_linear(rng, ctx, theta_ref) -> Genome:
    g = Genome(theta_ref=theta_ref, meta={"seed_topology": "direct_linear"})
    g.actuators.append(ActuatorGene(
        id="a0", innovation=INN_DIRECT_ACT, kind="linear",
        body_a=GROUND, body_b=OUTPUT,
        p={"ax": float(rng.uniform(0.10, 0.85 * ctx.L_prox)),
           "ay": float(rng.uniform(-0.05, 0.05)),
           "bx": float(rng.uniform(0.35, 0.85) * ctx.L_dist),
           "by": float(rng.uniform(-0.03, 0.03)),
           "gF": float(rng.uniform(0.30, 0.70)),
           "gv": float(rng.uniform(0.30, 0.80))}))
    return g


def rotary_geared(rng, ctx, theta_ref) -> Genome:
    g = Genome(theta_ref=theta_ref, meta={"seed_topology": "rotary_geared"})
    g.actuators.append(ActuatorGene(
        id="a0", innovation=INN_ROTARY_ACT, kind="rotary",
        p={"gtau": float(rng.uniform(0.35, 0.80)),
           "gratio": float(rng.uniform(0.45, 0.85)),
           "gomega": float(rng.uniform(0.30, 0.90))}))
    return g


def antagonistic_tendons(rng, ctx, theta_ref) -> Genome:
    g = Genome(theta_ref=theta_ref, meta={"seed_topology": "antagonist_tendon"})
    s = float(rng.uniform(0.12, 0.8 * ctx.L_prox))
    b = float(rng.uniform(0.30, 0.75) * ctx.L_dist)
    n = float(rng.uniform(0.025, 0.06))
    for inn, aid, side in ((INN_TENDON_FLEX, "a0", -1.0),
                           (INN_TENDON_EXT, "a1", +1.0)):
        # via point = joint-adjacent pulley (olecranon analogue): keeps the
        # tendon on its own side of the joint across the whole ROM, so the
        # moment-arm sign cannot flip mid-flexion.
        g.actuators.append(ActuatorGene(
            id=aid, innovation=inn, kind="tendon",
            body_a=GROUND, body_b=OUTPUT,
            via=[[GROUND, 0.025, side * 0.035]],
            p={"ax": s, "ay": side * n,
               "bx": b, "by": side * 0.02,
               "gF": float(rng.uniform(0.30, 0.70)),
               "gv": float(rng.uniform(0.35, 0.85))}))
    return g


def four_bar(rng, ctx, theta_ref, tries: int = 40,
             rom=(np.deg2rad(10), np.deg2rad(110))):
    """Linear-actuator-driven four-bar: ground pivot -> crank -> coupler -> output.

    Posterior-side placement (the side the distal bone never sweeps for a
    flexion joint) and constructive crank/coupler sizing from the anchor
    distance range, so the chain assembles across the whole ROM by design."""
    for _ in range(tries):
        # anterior wedge: the pocket between the bones that stays open across
        # flexion - the only in-plane region a joint-spanning linkage can
        # occupy without crossing a bone (the same pocket the legacy
        # direct-linear actuator uses).
        # cluster at the joint (polycentric-brace style): only there can the
        # moving pins stay inside the limb envelope across the whole ROM
        s0 = float(rng.uniform(0.030, 0.105))
        n0 = float(rng.uniform(-0.050, -0.015))       # anterior of prox bone
        P0 = _ground_world(ctx, s0, n0)
        bx = float(rng.uniform(0.08, 0.24) * ctx.L_dist)
        by = float(rng.uniform(0.004, 0.028))         # anterior, local frame
        ths = np.linspace(rom[0], rom[1], 25)
        ds = np.array([np.linalg.norm(_out_world(ctx, th, (bx, by)) - P0)
                       for th in ths])
        dmin, dmax = float(ds.min()), float(ds.max())
        # exact triangle-box solve: Lk - Lc = alpha*dmin, Lk + Lc = beta*dmax
        # keeps the chain away from both flat configurations across the ROM,
        # so neither assembly nor transmission can dead-point.
        alpha = float(rng.uniform(0.30, 0.75))
        beta = float(rng.uniform(1.10, 1.35))
        Lk = 0.5 * (alpha * dmin + beta * dmax)
        Lc = 0.5 * (beta * dmax - alpha * dmin)
        if Lc > 0.20 or Lk > 0.32 or Lc < 0.018:
            continue
        B = _out_world(ctx, theta_ref, (bx, by))
        d = float(np.linalg.norm(B - P0))
        if not (abs(Lc - Lk) + 0.005 < d < Lc + Lk - 0.005):
            continue

        # crank tip across the ROM (fixed branch) - drives actuator placement
        def tip(th, sgn):
            Bt = _out_world(ctx, th, (bx, by))
            dd = float(np.linalg.norm(Bt - P0))
            if not (abs(Lc - Lk) < dd < Lc + Lk):
                return None
            aa = (Lc ** 2 - Lk ** 2 + dd ** 2) / (2 * dd)
            hh2 = Lc ** 2 - aa ** 2
            if hh2 <= 1e-10:
                return None
            uu = (Bt - P0) / dd
            nn = np.array([-uu[1], uu[0]])
            return P0 + aa * uu + sgn * np.sqrt(hh2) * nn
        sgn = 1.0 if rng.random() < 0.5 else -1.0
        tips = [tip(th, sgn) for th in ths]
        if any(tp is None for tp in tips):
            continue
        tips = np.array(tips)
        ta = float(rng.uniform(0.6, 1.0))
        mounts = P0 + ta * (tips - P0)                 # actuator rides here
        mchord = mounts[-1] - mounts[0]
        if np.linalg.norm(mchord) < 0.018:             # stroke would be tiny
            continue
        up_v = ctx.prox_dir
        nrm_v = np.array([-up_v[1], up_v[0]])
        sA = float(rng.uniform(0.12, 0.90 * ctx.L_prox))
        nA = float(rng.uniform(-0.055, 0.055))          # inside the socket strip
        A = sA * up_v + nA * nrm_v
        qs = np.linalg.norm(A - mounts, axis=1)
        q_lo, q_hi = float(qs.min()), float(qs.max())
        stroke = q_hi - q_lo
        dth = ths[1] - ths[0]
        if (q_lo < 0.042 or stroke < 0.009 or q_hi / q_lo > 1.75
                or q_lo < stroke + 0.030
                or np.min(np.abs(np.diff(qs))) / dth < 0.009):
            continue                                    # legacy gate prechecks
        # clearance + envelope sweep: no bone crossing, every moving pin
        # inside the limb capsule, at every sampled angle
        prox0, prox1 = np.zeros(2), ctx.L_prox * ctx.prox_dir

        def seg_dist(p, e0, e1):
            dv = e1 - e0
            tt = float(np.clip((p - e0) @ dv / (dv @ dv), 0.0, 1.0))
            return float(np.linalg.norm(p - (e0 + tt * dv)))

        clear = True
        for th, Tt in zip(ths, tips):
            dist1 = _out_world(ctx, th, (ctx.L_dist, 0.0))
            Bt = _out_world(ctx, th, (bx, by))
            At = P0 + ta * (Tt - P0)
            for s0_, s1_ in ((P0, Tt), (Tt, Bt), (A, At)):
                if _cross(s0_, s1_, prox0, prox1) \
                        or _cross(s0_, s1_, np.zeros(2), dist1):
                    clear = False
                    break
            if clear:
                for pt in (Tt, At):
                    if min(seg_dist(pt, prox0, prox1),
                           seg_dist(pt, np.zeros(2), dist1)) > ctx.envelope_r:
                        clear = False
                        break
            if not clear:
                break
        if not clear:
            continue
        T = tip(theta_ref, sgn)
        g = Genome(theta_ref=theta_ref, meta={"seed_topology": "four_bar"})
        g.bodies.append(BodyGene("crank", INN_FB_CRANK, {"length": Lc}))
        g.bodies.append(BodyGene("coupler", INN_FB_COUPLER, {"length": Lk}))
        g.joints.append(JointGene("j1", INN_FB_J1, "revolute", GROUND, "crank",
                                  {"ax": s0, "ay": n0, "bx": 0.0, "by": 0.0}))
        g.joints.append(JointGene("j2", INN_FB_J2, "revolute", "crank",
                                  "coupler",
                                  {"ax": Lc, "ay": 0.0, "bx": 0.0, "by": 0.0}))
        g.joints.append(JointGene("j3", INN_FB_J3, "revolute", "coupler",
                                  OUTPUT,
                                  {"ax": Lk, "ay": 0.0, "bx": bx, "by": by}))
        g.actuators.append(ActuatorGene(
            id="a0", innovation=INN_FB_ACT, kind="linear",
            body_a=GROUND, body_b="crank",
            p={"ax": sA, "ay": nA,
               "bx": ta * Lc, "by": 0.0,
               "gF": float(rng.uniform(0.40, 0.85)),
               "gv": float(rng.uniform(0.30, 0.80))}))
        phc = float(np.arctan2(*(T - P0)[::-1]))
        phk = float(np.arctan2(*(B - T)[::-1]))
        g.hints["crank"] = [float(P0[0]), float(P0[1]), phc]
        g.hints["coupler"] = [float(T[0]), float(T[1]), phk]
        return g
    return None


SEED_MIX = [("direct_linear", 0.60, direct_linear),
            ("four_bar", 0.20, four_bar),
            ("rotary_geared", 0.10, rotary_geared),
            ("antagonist_tendon", 0.10, antagonistic_tendons)]


def random_archetype(rng, ctx, theta_ref):
    r = rng.random()
    acc = 0.0
    for _, w, fn in SEED_MIX:
        acc += w
        if r <= acc:
            return fn(rng, ctx, theta_ref)
    return direct_linear(rng, ctx, theta_ref)
