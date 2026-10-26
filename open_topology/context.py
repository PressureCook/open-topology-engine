"""Joint context: anthropometry, gravity convention, packaging envelope.

Winter (2009) *proximal-pair* values (self-consistent com_frac / k_frac about the
proximal joint), matching the round-2 review's verified C3 closure:
forearm+hand mass_frac 0.022, com_frac 0.682, k_frac 0.827.

Gravity convention (C1 closure): each joint carries ``theta_neutral``, the
gravity-stable flexion angle.  The output body's world orientation is
``phi_out = (theta - theta_neutral) - pi/2`` with gravity along -y, so the
generalised gravity torque of the output body reduces to
``-m g com sin(theta - theta_neutral)`` and every additional body picks up the
same convention for free through its own dy/dtheta.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

G_ACC = 9.81

# Linear density of a printed link (PETG/CF shell + pins + bearings), kg/m.
LINK_DENSITY = 0.9
# Dead (non-stroke) length of a linear EMA body, m — motor pod + nut + clevises.
DEAD_LENGTH = 0.028


@dataclass(frozen=True)
class JointContext:
    name: str
    L_dist: float            # joint -> distal end landmark (m)
    L_prox: float            # joint -> proximal landmark (m)
    mass: float              # distal segment mass incl. hand (kg)
    com: float               # joint -> segment COM (m)
    k_gyr: float             # radius of gyration about the joint (m)
    theta_neutral: float     # rad, gravity-stable flexion angle
    payload_mass: float = 0.0
    payload_dist: float = 0.0
    envelope_r: float = 0.06   # packaging capsule radius around each bone (m)
    damping: float = 0.06      # joint viscous damping (N*m*s/rad)

    @property
    def I_about_joint(self) -> float:
        """Distal segment inertia about the joint: m (k_frac L)^2."""
        return self.mass * self.k_gyr ** 2

    @property
    def I_com(self) -> float:
        """Distal segment inertia about its own COM (parallel axis removed)."""
        return max(self.mass * (self.k_gyr ** 2 - self.com ** 2), 1e-6)

    @property
    def phi_out0(self) -> float:
        """World orientation of the distal bone at theta = 0."""
        return -self.theta_neutral - 0.5 * np.pi

    @property
    def prox_dir(self) -> np.ndarray:
        """Unit vector of the proximal bone (opposite the fully-extended distal)."""
        a = self.phi_out0 + np.pi
        return np.array([np.cos(a), np.sin(a)])


def demo_elbow(payload: float = 0.5, body_mass: float = 75.0) -> JointContext:
    """Synthetic elbow: Winter forearm+hand proximal pair, 0.5 kg payload at 0.25 m."""
    L = 0.27
    m = 0.022 * body_mass
    return JointContext(
        name="elbow", L_dist=L, L_prox=0.30, mass=m,
        com=0.682 * L, k_gyr=0.827 * L, theta_neutral=0.0,
        payload_mass=payload, payload_dist=0.25,
    )
