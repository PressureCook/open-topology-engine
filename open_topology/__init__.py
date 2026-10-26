"""Open-topology mechanism evolution - standalone reference implementation.

Spec: SPEC-open-topology-evolution.md (v0.1, 2026-08-12), scoped to the core
engine: mechanism-graph genome, generic interpreter + transmission table +
validity gate, reduced-EOM simulation, speciated structural search, demo.
"""
from .context import JointContext, demo_elbow
from .genome import Genome, Innovations
from .interpreter import CompiledMechanism, TransmissionTable, validate
from .simulate import Trajectory, simulate_one, simulate_batch
from .evolve import evolve

__all__ = ["JointContext", "demo_elbow", "Genome", "Innovations",
           "CompiledMechanism", "TransmissionTable", "validate",
           "Trajectory", "simulate_one", "simulate_batch", "evolve"]
