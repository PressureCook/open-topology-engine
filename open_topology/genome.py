"""Mechanism-graph genome: variable-length, typed, innovation-tracked (R-1, R-11).

GROUND and OUTPUT are implicit (spec 6.1); ``genome.bodies`` holds coupler rods
only.  Anchors are stored as physical metres in each body's local frame:

* ground:   (s, n) = along-bone / normal-to-bone coordinates of the proximal bone
* output:   (x, y) with local +x along the distal bone
* coupler:  (x, y) with the rod spanning (0,0) -> (length, 0)

Capacity genes live in [0, 1] and decode through ``actuators.py``; every decode
is zero at gene 0 so structural additions are behaviour-preserving at birth
(R-12) - the round-2 G5 failure (a 0.02 m/s rate floor on a zero-force
actuator) is ruled out by construction.

``hints`` stores each coupler's pose at ``theta_ref`` so the assembly branch is
part of the genome: serialisation round-trips reproduce the identical branch
(R-8), and branch selection is deterministic.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

GROUND, OUTPUT = "ground", "output"


class Innovations:
    """Monotonic innovation counter with a per-run reuse cache (NEAT semantics)."""

    def __init__(self, start: int = 100):
        self._n = start
        self._cache: dict = {}

    def fresh(self) -> int:
        self._n += 1
        return self._n

    def get(self, key, count: int = 1):
        """Same structural op at the same site in the same run -> same ids."""
        if key not in self._cache:
            self._cache[key] = [self.fresh() for _ in range(count)]
        return self._cache[key]

    def state(self) -> int:
        return self._n


@dataclass
class BodyGene:
    id: str
    innovation: int
    p: dict  # {'length': m}


@dataclass
class JointGene:
    id: str
    innovation: int
    kind: str            # 'revolute'
    body_a: str
    body_b: str
    p: dict              # {'ax','ay','bx','by'} local metres


@dataclass
class ActuatorGene:
    id: str
    innovation: int
    kind: str            # 'linear' | 'rotary' | 'tendon'
    body_a: Optional[str] = None
    body_b: Optional[str] = None
    via: list = field(default_factory=list)   # [[body_id, x, y], ...]
    p: dict = field(default_factory=dict)
    # linear/tendon: {'ax','ay','bx','by','gF','gv'}
    # rotary:        {'gtau','gratio','gomega'}  (mounted at the output joint)


@dataclass
class SpringGene:
    id: str
    innovation: int
    kind: str            # 'linear' | 'torsional'
    body_a: Optional[str] = None
    body_b: Optional[str] = None
    p: dict = field(default_factory=dict)
    # linear:    {'ax','ay','bx','by','gk','gfree'}
    # torsional: {'gk','gfree'}   (at the output joint)


@dataclass
class Genome:
    bodies: list = field(default_factory=list)
    joints: list = field(default_factory=list)
    actuators: list = field(default_factory=list)
    springs: list = field(default_factory=list)
    hints: dict = field(default_factory=dict)     # body_id -> [x, y, phi] at theta_ref
    theta_ref: float = 0.0
    meta: dict = field(default_factory=dict)      # seed_topology, birth, parents...

    # ------------------------------------------------------------------ misc
    def body(self, bid: str):
        for b in self.bodies:
            if b.id == bid:
                return b
        return None

    def innovations(self) -> set:
        out = set()
        for lst in (self.bodies, self.joints, self.actuators, self.springs):
            out.update(e.innovation for e in lst)
        return out

    def elements(self) -> dict:
        return {e.innovation: e
                for lst in (self.bodies, self.joints, self.actuators, self.springs)
                for e in lst}

    def signature(self) -> tuple:
        """Simulation-batching signature: actuator kind sequence."""
        return tuple(a.kind for a in self.actuators)

    def structure_label(self) -> str:
        na = len(self.actuators)
        kinds = "+".join(sorted({a.kind for a in self.actuators})) or "none"
        return (f"{self.meta.get('seed_topology', '?')}"
                f"/{len(self.bodies)}cpl/{na}act({kinds})/{len(self.springs)}spr")

    # ------------------------------------------------------------- serialise
    def to_dict(self) -> dict:
        return {
            "schema_version": 2,
            "bodies": [asdict(b) for b in self.bodies],
            "joints": [asdict(j) for j in self.joints],
            "actuators": [asdict(a) for a in self.actuators],
            "springs": [asdict(s) for s in self.springs],
            "hints": {k: list(v) for k, v in self.hints.items()},
            "theta_ref": self.theta_ref,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict) -> "Genome":
        g = Genome(
            bodies=[BodyGene(**b) for b in d.get("bodies", [])],
            joints=[JointGene(**j) for j in d.get("joints", [])],
            actuators=[ActuatorGene(**a) for a in d.get("actuators", [])],
            springs=[SpringGene(**s) for s in d.get("springs", [])],
            hints={k: list(v) for k, v in d.get("hints", {}).items()},
            theta_ref=float(d.get("theta_ref", 0.0)),
            meta=dict(d.get("meta", {})),
        )
        return g

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @staticmethod
    def from_json(s: str) -> "Genome":
        return Genome.from_dict(json.loads(s))

    def key(self) -> str:
        return hashlib.sha1(self.to_json().encode()).hexdigest()

    def copy(self) -> "Genome":
        return Genome.from_dict(json.loads(self.to_json()))


# ---------------------------------------------------------------- parameter bounds
def param_bounds(elem, name: str, ctx) -> tuple:
    """(lo, hi) for a physical parameter; capacity genes are (0, 1)."""
    if name.startswith("g"):
        return (0.0, 1.0)
    if isinstance(elem, BodyGene):
        return (0.03, 0.32)
    body = None
    if name in ("ax", "ay"):
        body = getattr(elem, "body_a", None)
    elif name in ("bx", "by"):
        body = getattr(elem, "body_b", None)
    axis = name[-1]  # 'x' | 'y'  (ground stores s->'x' slot, n->'y' slot)
    # normal-offset bounds stay inside the envelope capsule (ctx.envelope_r)
    if body == GROUND:
        return (0.03, 0.95 * ctx.L_prox) if axis == "x" else (-0.055, 0.055)
    if body == OUTPUT:
        return (0.02, 0.92 * ctx.L_dist) if axis == "x" else (-0.05, 0.05)
    # coupler-local
    return (0.0, 0.32) if axis == "x" else (-0.03, 0.03)


def iter_params(genome: Genome):
    """Yield (element, param_name) for every continuous parameter."""
    for lst in (genome.bodies, genome.joints, genome.actuators, genome.springs):
        for e in lst:
            for name in e.p:
                yield e, name
