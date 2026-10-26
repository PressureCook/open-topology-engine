"""Capacity decoding, drivetrain physics and per-kind buildability constraints.

Round-2 fixes carried here:

* G1 - ``rotary`` has a gear-ratio gene; reflected rotor inertia
  ``G_r^2 * J_rotor`` enters ``I_eff``; gearbox efficiency, stage count, motor
  mass and speed limit ``omega_m / G_r`` all scale with the ratio.  A rotary
  drive no longer gets perfect transmission for free.
* G5 - every capacity decode is exactly 0 at gene 0 (quadratic maps, no
  floors), so newborn structural elements are physically inert.
* G6 - each kind carries an efficiency used by the per-actuator energy
  integral.
"""
from __future__ import annotations

import numpy as np

from .context import DEAD_LENGTH

# capacity ceilings
F_HI = 1600.0        # N        linear / tendon
V_HI = 0.45          # m/s      linear / tendon
TAU_M_HI = 1.5       # N*m      motor continuous torque
RATIO_LO, RATIO_HI = 3.0, 300.0
OMEGA_M_HI = 900.0   # rad/s    motor speed

ETA = {"linear": 0.55, "tendon": 0.60, "rotary": 0.85}   # rotary = motor only

# legacy hard limits (evolve.py of the reviewed repo), ported per G2
R_FLOOR = 0.008          # m   minimum |moment arm| for linear/tendon
STROKE_MIN = 0.008       # m
STROKE_RATIO_MAX = 1.8
L_RET_MIN = 0.040        # m
TENDON_STROKE_MAX = 0.090  # m reel capacity


def n_stages(ratio: float) -> int:
    return max(1, int(np.ceil(np.log(max(ratio, 1.001)) / np.log(6.5))))


def rotor_inertia(tau_m: float) -> float:
    """BLDC rotor inertia vs continuous torque; 0.1 N*m class ~ 1.2e-6 kg*m^2."""
    return 1.2e-6 * (tau_m / 0.1) ** 1.4 if tau_m > 0 else 0.0


def decode(act) -> dict:
    """ActuatorGene -> physical capacities.

    Returns u_hi (N or N*m at gearbox output), v_hi (m/s, or motor rad/s for
    rotary), r_const (rotary transmission ratio), refl_J, mass, eta, tendon flag.
    """
    p = act.p
    if act.kind == "rotary":
        tau_m = TAU_M_HI * p["gtau"] ** 2
        ratio = float(np.exp(np.log(RATIO_LO)
                             + p["gratio"] * np.log(RATIO_HI / RATIO_LO)))
        omega_m = OMEGA_M_HI * p["gomega"]
        ns = n_stages(ratio)
        eta_g = 0.90 ** ns
        u_hi = tau_m * eta_g                      # torque referred to gearbox output
        mass = (0.09 + 0.06 * ns) * (tau_m / 0.1) ** 0.8 if tau_m > 0 else 0.0
        return {"kind": "rotary", "u_hi": u_hi, "v_hi": omega_m,
                "r_const": ratio, "refl_J": ratio ** 2 * rotor_inertia(tau_m),
                "mass": mass, "eta": ETA["rotary"], "tendon": False,
                "tau_m": tau_m, "ratio": ratio, "stages": ns, "eta_gear": eta_g}
    F = F_HI * p["gF"] ** 2
    v = V_HI * p["gv"]
    mass = 1.2 * F / 1500.0                       # 0 at F = 0 (G5)
    return {"kind": act.kind, "u_hi": F, "v_hi": v, "r_const": None,
            "refl_J": 0.0, "mass": mass, "eta": ETA[act.kind],
            "tendon": act.kind == "tendon"}


def spring_decode(spr) -> dict:
    """SpringGene -> (k, q_free).  Free length/angle decodes against FIXED
    physical bounds (never the theta grid) - closes the round-2 medium finding
    where the evolved and final grids disagreed and flipped the spring torque."""
    p = spr.p
    if spr.kind == "torsional":
        k = 30.0 * p["gk"] ** 2                       # N*m/rad
        q_free = -0.5 * np.pi + p["gfree"] * 1.5 * np.pi   # [-90, 180] deg
    else:
        k = 20000.0 * p["gk"] ** 2                    # N/m
        q_free = 0.02 + p["gfree"] * 0.38             # [0.02, 0.40] m
    return {"k": k, "q_free": q_free}


def actuator_constraints(act, dec: dict, q: np.ndarray, r: np.ndarray,
                         rom: np.ndarray) -> list:
    """Per-actuator buildability violations inside the ROM (G2 port).

    Zero-capacity actuators are exempt: a behaviour-preserving newborn must not
    be gate-lethal."""
    bad = []
    if dec["u_hi"] < 1e-9:
        return bad
    if act.kind == "rotary":
        return bad                                    # torque gate is holdability
    qr, rr = q[rom], np.abs(r[rom])
    q_lo, q_hi = float(np.min(qr)), float(np.max(qr))
    stroke = q_hi - q_lo
    if np.min(rr) < R_FLOOR:
        bad.append("r_floor")
    if act.kind == "linear":
        if stroke < STROKE_MIN:
            bad.append("stroke_min")
        if q_lo < L_RET_MIN:
            bad.append("l_ret_min")
        if q_hi / max(q_lo, 1e-6) > STROKE_RATIO_MAX:
            bad.append("stroke_ratio")
        if q_lo < stroke + DEAD_LENGTH:
            bad.append("packaging")
    else:  # tendon
        if stroke > TENDON_STROKE_MAX:
            bad.append("tendon_reel")
    return bad


def packaging_margin(act, dec: dict, q: np.ndarray, rom: np.ndarray) -> float:
    """Signed packaging margin (m) for linear actuators; +inf otherwise."""
    if act.kind != "linear" or dec["u_hi"] < 1e-9:
        return np.inf
    qr = q[rom]
    return float(np.min(qr) - ((np.max(qr) - np.min(qr)) + DEAD_LENGTH))
