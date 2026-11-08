# Copyright 2026 Google LLC
"""Benchmark circuits for the Magicside suite (not part of the shipped package)."""

import numpy as np
import stim


def build_leakage_surface_code(distance, rounds, p_pauli=0.002, p_leak=0.004):
    """Rotated Z-memory surface code with leakage tags on every CX and measurement."""
    base = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p_pauli,
        before_measure_flip_probability=p_pauli,
        after_reset_flip_probability=p_pauli,
    ).flattened()
    mr_qubits = {t.value for inst in base if inst.name == "MR" for t in inst.targets_copy()}
    data_qubits = sorted(set(range(base.num_qubits)) - mr_qubits)
    transition_2 = f"LEAKAGE_TRANSITION_2: ({p_leak}, U_U-->2_D)"
    out = stim.Circuit()
    for inst in base:
        name, targets, args = inst.name, inst.targets_copy(), inst.gate_args_copy()
        if name == "CX":
            t_qubits = [targets[i] for i in range(1, len(targets), 2)]
            out.append("H", t_qubits)
            out.append("CZ", targets, [], tag="CONDITIONED_ON_PAIR: (U, U)")
            leak_targets = []
            for i in range(0, len(targets), 2):
                q0, q1 = targets[i].value, targets[i + 1].value
                swap = q0 in mr_qubits and q1 not in mr_qubits
                leak_targets.extend([q1, q0] if swap else [q0, q1])
            out.append("II_ERROR", leak_targets, [0.0], tag=transition_2)
            out.append("H", t_qubits)
        elif name == "MR":
            out.append("M", targets, args, tag="LEAKAGE_PROJECTION_Z: (1.0, 2)")
            out.append("R", targets)
            out.append("I", targets, [], tag="LEAKAGE_TRANSITION_1: (1.0, 2-->0)")
            out.append("I", data_qubits, [], tag="LEAKAGE_TRANSITION_1: (0.5, 2-->U)")
        elif name == "M":
            out.append("M", targets, args, tag="LEAKAGE_PROJECTION_Z: (1.0, 2)")
        else:
            out.append(inst)
    return out


def xmem_t(k, distance=3, rounds=3, p=0.002):
    """Rotated X-memory surface code with k T gates (tag 'T') on data qubits after first TICK."""
    base = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
    ).flattened()
    data = sorted({t.value for i in base if i.name == "MX" for t in i.targets_copy()})
    c, done = stim.Circuit(), False
    for inst in base:
        c.append(inst)
        if inst.name == "TICK" and not done:
            if k:
                c.append("I", data[:k], tag="T")
            done = True
    return c


def random_t_cx(n=10, layers=12, p_t=0.25, seed=7, p_noise=0.002):
    """Random layers of T / CX / depolarizing noise / H on n qubits, then measure all."""
    rng = np.random.default_rng(seed)
    c = stim.Circuit()
    c.append("R", range(n))
    c.append("H", range(n))
    for _ in range(layers):
        ts = [q for q in range(n) if rng.random() < p_t]
        if ts:
            c.append("I", ts, tag="T")
        perm = rng.permutation(n)
        c.append("CX", [int(x) for x in perm[: 2 * (n // 2)]])
        c.append("DEPOLARIZE1", range(n), [p_noise])
        hs = [q for q in range(n) if rng.random() < 0.3]
        if hs:
            c.append("H", hs)
    c.append("M", range(n))
    return c


def t_count(circuit):
    """Number of T gates (qubit targets of 'I' instructions tagged 'T')."""
    return sum(len(i.targets_copy()) for i in circuit if i.name == "I" and i.tag == "T")
