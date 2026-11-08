"""Bit-exact differential tests of `mbe_apply_block` against `apply_compiled_block`.

Oracle: today's per-shot `phase_qfe.apply_compiled_block(check_unitary=True)` on each group of
bit-identical shots, the result copied to the group (what `_flush_magic` loops over), grouped by
a pure-Python partition written here (NOT `simulator_magic._group_shots_by_state`, so a shared
bug cannot hide), followed by the `max_components` check. Subject: `LiveExpansions.apply_block`
(C++ `MagicBatchEngine`). Everything is compared as bytes (weights through a uint64 view, so
-0.0 != 0.0). Blocks come from the real compiler (`compile_open_block`) over random commuting
T / CCZ / Z8 terms, in both modes (pauli_sum, and sequential with P1, P2, N1 and N2 steps).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np
import pytest

import stimside._magic.phase_qfe as pq
from stimside._magic.channels import MagicPhaseTerm, MagicTermRecord
from stimside._magic.gates import CompiledPauliAxis
from stimside._magic.live_store import LiveExpansions, block_table
from stimside._magic.live_store_test import _first_difference, random_state
from stimside._magic.plan import compile_open_block
from stimside._magic.query import QueryBudgetExceededError

_ATOL = 1e-10
_N_CHI = [(n, chi) for n in (5, 64, 70, 130) for chi in (1, 2, 4, 64)]
_KINDS = ("plain", "d", "raw", "signed0", "zero", "plain")


def commuting_axes(rng: np.random.Generator, n: int, k: int) -> list[CompiledPauliAxis]:
    """k pairwise commuting Hermitian axes: Z-type words, or X-type words on disjoint qubits
    (any mix of Z-type and a shared-support X-type word would anticommute)."""
    out: list[CompiledPauliAxis] = []
    for i in range(k):
        bits = (rng.random(n) < float(rng.choice([0.05, 0.3, 0.8]))).astype(np.uint8)
        bits[int(rng.integers(0, n))] = 1
        w = pq.pack_bit_vector(bits, n)
        z = np.zeros_like(w)
        # Z-type: kappa 0 or 2 (sign). The first axis may instead be an X-type word: it commutes
        # with Z-type words only if their supports are disjoint, so keep X-type for k == 1.
        x_type = k == 1 and bool(rng.integers(0, 2))
        kap = 2 * int(rng.integers(0, 2))
        out.append(CompiledPauliAxis(w, z, kap) if x_type else CompiledPauliAxis(z, w, kap))
    return out


def y_axes(rng: np.random.Generator, n: int, k: int) -> list[CompiledPauliAxis]:
    """Up to k pairwise commuting Hermitian axes with X and Z parts that overlap (Y-type:
    <a, b> is odd on some qubits and kappa is then odd), the rows `commuting_axes` never makes."""
    out: list[CompiledPauliAxis] = []
    for _ in range(k):
        for _try in range(100):
            dens = float(rng.choice([0.05, 0.3, 0.8]))
            a = pq.pack_bit_vector((rng.random(n) < dens).astype(np.uint8), n)
            b = pq.pack_bit_vector((rng.random(n) < dens).astype(np.uint8), n)
            if not (a.any() or b.any()):
                continue
            kap = (int(pq.word_dot_parity(a, b)) + 2 * int(rng.integers(0, 2))) & 3
            ax = CompiledPauliAxis(a, b, kap)
            if all(
                int(pq.word_dot_parity(ax.a_words, o.b_words))
                ^ int(pq.word_dot_parity(ax.b_words, o.a_words)) == 0
                for o in out
            ):
                out.append(ax)
                break
    return out


AxesFn = Callable[[np.random.Generator, int, int], list[CompiledPauliAxis]]


def random_records(
    rng: np.random.Generator, n: int, axes_fn: AxesFn = commuting_axes
) -> list[MagicTermRecord]:
    axes = axes_fn(rng, n, int(rng.integers(1, 4)))
    k = len(axes)
    recs: list[MagicTermRecord] = []
    for _ in range(int(rng.integers(1, 4))):
        kind = str(rng.choice(["T", "T", "CCZ", "Z8_DIAG"]))
        if kind == "CCZ" and k < 3:
            kind = "T"
        if kind == "CCZ":
            use = tuple(axes[:3])
        else:
            use = (axes[int(rng.integers(0, k))],)
        eps = int(rng.choice([1, -1])) if kind == "T" else 1
        recs.append(MagicTermRecord(MagicPhaseTerm(use, kind, eps)))  # type: ignore[arg-type]
    return recs


def reference_groups(exps: Sequence[pq.ShotExpansion]) -> list[list[int]]:
    """Independent partition: pairwise byte comparison of every field, groups by lowest member."""
    groups: list[list[int]] = []
    for s, e in enumerate(exps):
        if e.chi == 0:
            continue
        for g in groups:
            if _first_difference(exps[g[0]], e) is None:
                g.append(s)
                break
        else:
            groups.append([s])
    return groups


def reference_flush(
    exps: list[pq.ShotExpansion], block: pq.CompiledPauliBlock, cap: int | None
) -> tuple[int, int, Exception | None]:
    """Today's `_flush_magic` loop body for one block: (groups, live shots, exception or None)."""
    groups = reference_groups(exps)
    live = sum(len(g) for g in groups)
    for members in groups:
        out = pq.apply_compiled_block(exps[members[0]], block, check_unitary=True, atol=_ATOL)
        for s in members[1:]:
            exps[s] = out.copy()
        exps[members[0]] = out
        if cap is not None and out.chi > cap:
            return len(groups), live, QueryBudgetExceededError(
                f"Expansion exceeded max_components={cap} (got chi={out.chi})"
            )
    return len(groups), live, None


def make_shots(rng: np.random.Generator, n: int, chi: int, copies: int) -> list[pq.ShotExpansion]:
    """A batch with repeated states (so groups merge) and chi == 0 / signed-zero shots."""
    base = [random_state(rng, n, chi, str(rng.choice(_KINDS))) for _ in range(copies)]
    order = rng.integers(0, copies, 3 * copies)
    return [base[i].copy() for i in order]


def run_case(
    rng: np.random.Generator, n: int, chi: int, cov: Counter[str], cap_mode: str,
    axes_fn: AxesFn = commuting_axes,
) -> None:
    shots = make_shots(rng, n, chi, 3)
    recs = compile_open_block(
        random_records(rng, n, axes_fn), n_qubits=n, budget=int(rng.choice([1, 64]))
    )
    for rec in recs:
        blk = rec.block  # type: ignore[attr-defined]
        table = block_table(blk, n)
        assert table is not None
        cov["odd_kappa"] += bool((table.axes_kappa & 1).any())
        cov["y_overlap"] += bool(((table.axes_a & table.axes_b) != 0).any())
        cov["cap_" + cap_mode] += 1
        for sh in shots:  # properties of the input states, counted per block
            cov["d_positive"] += sh.d > 0
            cov["raw_input"] += not sh._is_canonical_merged
            cov["log_mass_nonzero"] += sh.log_mass != 0.0
        oracle = [s.copy() for s in shots]
        cap = {"none": None, "tight": 1, "mid": 8, "loose": 10**6}[cap_mode]
        n_groups, live, exc = reference_flush(oracle, blk, cap)
        store = LiveExpansions(len(shots), n, _ATOL)
        for b, s in enumerate(shots):
            store[b] = s
        if exc is not None:
            with pytest.raises(QueryBudgetExceededError) as got:
                store.apply_block(table, cap)
            assert str(got.value) == str(exc)
            cov["cap_hit"] += 1
        else:
            assert store.apply_block(table, cap) == (n_groups, live)
        for b in range(len(shots)):
            diff = _first_difference(oracle[b], store[b])
            assert diff is None, f"n={n} chi={chi} mode={blk.mode} cap={cap} shot={b}: {diff}"
            w = oracle[b].weights
            cov["signed_zero_state"] += bool(np.any(np.signbit(w.real) | np.signbit(w.imag)))
            cov["chi0"] += oracle[b].chi == 0
        cov["blocks"] += 1
        cov[blk.mode] += 1
        cov["groups_merged"] += live - n_groups
        cov["steps_" + "_".join(s.kind for s in blk.steps)] += 1  # type: ignore[union-attr]
        store.close()


def test_apply_block_is_bit_identical_to_the_per_shot_function_on_2000_blocks() -> None:
    cov: Counter[str] = Counter()
    modes = ["none", "none", "loose", "mid", "tight"]
    for seed in range(4):
        for n, chi in _N_CHI:
            rng = np.random.default_rng([seed, n, chi, 9])
            for rep in range(36):
                run_case(rng, n, chi, cov, modes[(seed + rep) % len(modes)])
    plain_blocks = cov["blocks"]
    assert cov["odd_kappa"] == 0 and cov["y_overlap"] == 0, cov  # Z-type / disjoint X-type rows
    for n, chi in _N_CHI:  # Y-type rows: odd kappa, X and Z parts overlapping
        rng = np.random.default_rng([4, n, chi, 9])
        for rep in range(14):
            run_case(rng, n, chi, cov, modes[rep % len(modes)], axes_fn=y_axes)
    assert cov["blocks"] - plain_blocks >= 150, cov
    assert cov["odd_kappa"] >= 60 and cov["y_overlap"] >= 80, cov
    assert cov["d_positive"] >= 2000 and cov["raw_input"] >= 1000, cov  # d > 0, canon False
    assert cov["log_mass_nonzero"] >= 2000, cov
    assert min(cov["cap_none"], cov["cap_tight"], cov["cap_mid"], cov["cap_loose"]) >= 100, cov
    assert cov["blocks"] >= 2000, cov
    assert cov["pauli_sum"] >= 600 and cov["sequential"] >= 600, cov
    assert cov["groups_merged"] >= 1000 and cov["chi0"] >= 500, cov
    assert cov["signed_zero_state"] >= 500 and cov["cap_hit"] >= 100, cov
    for step_kind in ("P2", "N1", "N2", "P1"):
        assert any(k.startswith("steps_") and step_kind in k.split("_") for k in cov), step_kind


def test_a_pauli_sum_block_whose_coefficients_are_all_zero_raises_the_reference_error() -> None:
    """r == 0 after dropping zero coefficients: the result is the empty expansion, and the
    unitarity check against the non-zero input norm raises (the reference's text)."""
    n, rng = 70, np.random.default_rng(3)
    nw = pq.n_words_for_qubits(n)
    blk = pq.CompiledPauliBlock(
        n_qubits=n, mode="pauli_sum", coeffs=np.zeros(2, dtype=np.complex128),
        a_words=np.zeros((2, nw), dtype=np.uint64), b_words=np.zeros((2, nw), dtype=np.uint64),
        kappas=np.zeros(2, dtype=np.uint8),
    )
    shots = [random_state(rng, n, 4, "plain") for _ in range(3)]
    oracle = [s.copy() for s in shots]
    with pytest.raises(FloatingPointError) as want:
        pq.apply_compiled_block(oracle[0], blk, check_unitary=True, atol=_ATOL)
    store = LiveExpansions(3, n, _ATOL)
    for b, s in enumerate(shots):
        store[b] = s
    table = block_table(blk, n)
    assert table is not None
    with pytest.raises(FloatingPointError) as got:
        store.apply_block(table, None)
    assert str(got.value) == str(want.value) and "apply_compiled_block" in str(got.value)
    assert store.failed_shot == 0
    for b in range(3):  # the failing group (all three identical or not) is untouched
        assert _first_difference(shots[b], store[b]) is None
    store.close()


def _scaled(blk: pq.CompiledPauliBlock, scalar: complex) -> pq.CompiledPauliBlock:
    return pq.CompiledPauliBlock(
        n_qubits=blk.n_qubits, mode=blk.mode, coeffs=blk.coeffs, a_words=blk.a_words,
        b_words=blk.b_words, kappas=blk.kappas, steps=blk.steps, global_scalar=scalar,
        is_unitary=blk.is_unitary, is_identity=blk.is_identity,
    )


@pytest.mark.parametrize("scalar", [2.0 + 0j, 0.5 + 0j, 1.0000001 + 0j, 0.3 - 1.1j, 1e150 + 0j])
def test_a_non_unitary_block_raises_the_reference_error_with_the_reference_partial_state(
    scalar: complex,
) -> None:
    """A scalar that changes the norm: the unitarity check fires at the first group that fails;
    earlier groups were already updated (the same partial mutation as the per-shot loop)."""
    cov: Counter[str] = Counter()
    for seed in range(12):
        rng = np.random.default_rng([seed, 31])
        n = int(rng.choice([5, 70]))
        budget = int(rng.choice([1, 64]))
        recs = compile_open_block(random_records(rng, n), n_qubits=n, budget=budget)
        shots = make_shots(rng, n, 4, 3)
        for rec in recs:
            blk = _scaled(rec.block, scalar)  # type: ignore[attr-defined]
            oracle = [s.copy() for s in shots]
            try:
                reference_flush(oracle, blk, None)
                want_exc: Exception | None = None
            except (FloatingPointError, ValueError) as exc:  # unitarity / non-finite checks
                want_exc = exc
            store = LiveExpansions(len(shots), n, _ATOL)
            for b, s in enumerate(shots):
                store[b] = s
            table = block_table(blk, n)
            assert table is not None
            if want_exc is None:
                store.apply_block(table, None)
            else:
                with pytest.raises(type(want_exc)) as got:
                    store.apply_block(table, None)
                assert str(got.value) == str(want_exc)
                cov["raised"] += 1
            for b in range(len(shots)):
                assert _first_difference(oracle[b], store[b]) is None, (seed, b)
            store.close()
    assert cov["raised"] >= 3, cov


def _identity_block(n: int, scalar: complex) -> pq.CompiledPauliBlock:
    return pq.CompiledPauliBlock(
        n_qubits=n, mode="pauli_sum", global_scalar=scalar, is_unitary=True, is_identity=True
    )


def test_the_unitarity_check_accepts_an_exact_tie_and_rejects_the_next_float() -> None:
    """|post - pre| == atol * max(1, pre) exactly passes (`<=`); one ulp more raises. With
    atol = 2**-40 and scalar 1 + 2**-20 j every product and sum is exact: post - pre = 2**-40."""
    n, atol = 5, 2.0**-40
    tie = _identity_block(n, complex(1.0, 2.0**-20))
    over = _identity_block(n, complex(1.0, 2.0**-20 + 2.0**-33))  # im**2 - 2**-40 > 2**-52
    for blk, ok in ((tie, True), (over, False)):
        oracle = pq.ShotExpansion.zero(n, atol=atol)
        store = LiveExpansions(2, n, atol)
        store[0] = pq.ShotExpansion.zero(n, atol=atol)
        store[1] = pq.ShotExpansion.zero(n, atol=atol)
        table = block_table(blk, n)
        assert table is not None
        if ok:
            pq.apply_compiled_block(oracle, blk, check_unitary=True, atol=atol)  # reference ok
            store.apply_block(table, None)
        else:
            with pytest.raises(FloatingPointError) as want:
                pq.apply_compiled_block(oracle, blk, check_unitary=True, atol=atol)
            with pytest.raises(FloatingPointError) as got:
                store.apply_block(table, None)
            assert str(got.value) == str(want.value)
        store.close()


@pytest.mark.parametrize("scalar", [1.0 + 0j, 1j, -1.0 + 0j, 0.6 - 0.8j])
def test_a_hand_built_pauli_sum_with_zero_coefficients_and_a_global_scalar_matches(
    scalar: complex,
) -> None:
    """The compiler never emits zero coefficients or a scalar other than 1 in a pauli_sum, but
    `apply_compiled_block` handles both: zero rows are dropped and the scalar multiplies the
    surviving coefficients (as numpy's `g_sc * coeffs[nz]`). The scalars are unit modulus and the
    coefficients form the unitary (I + iZ)/sqrt(2), so the block passes the check."""
    n = 70
    rng = np.random.default_rng([int(scalar.real * 8) + 9, 5])
    nw = pq.n_words_for_qubits(n)
    z = _z_word(n, 3)
    a = np.zeros((4, nw), dtype=np.uint64)
    b = np.zeros((4, nw), dtype=np.uint64)
    b[2] = z  # rows: identity, (zero), Z, (zero): the quarter turn (I + iZ)/sqrt(2), unitary
    c = 1.0 / np.sqrt(2.0)
    blk = pq.CompiledPauliBlock(
        n_qubits=n, mode="pauli_sum", global_scalar=scalar,
        coeffs=np.array([c, 0.0, 1j * c, 0.0], dtype=np.complex128), a_words=a, b_words=b,
        kappas=np.zeros(4, dtype=np.uint8),
    )
    shots = make_shots(rng, n, 4, 3)
    oracle = [s.copy() for s in shots]
    reference_flush(oracle, blk, None)
    store = LiveExpansions(len(shots), n, _ATOL)
    for i, s in enumerate(shots):
        store[i] = s
    table = block_table(blk, n)
    assert table is not None and table.axes_a.shape[0] == 2  # zero rows were dropped
    store.apply_block(table, None)
    for i in range(len(shots)):
        assert _first_difference(oracle[i], store[i]) is None, i
    store.close()


def _z_word(n: int, qubit: int) -> np.ndarray:
    return pq.pack_bit_vector((np.arange(n) == qubit).astype(np.uint8), n)
