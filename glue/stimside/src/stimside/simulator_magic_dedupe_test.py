"""Block dedupe in `MagicsideSimulator._flush_magic`: apply a block once per distinct exact state.

The private seam `stimside.simulator_magic._group_shots_by_state` is the on/off switch: replacing it
with identity grouping (one group per live shot) gives the pre-dedupe per-shot behaviour.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest
import stim  # type: ignore[import-untyped]

import stimside.simulator_magic as sm
from stimside import golden_test as gt
from stimside._magic.phase_qfe import ShotExpansion, apply_pauli
from stimside.simulator_magic import MagicsideSimulator

_ARRAY_FIELDS = ("weights", "h_words", "ell", "A_words", "a_union", "Gamma", "pivots", "ell_odd")
# d = 2, chi = 8 on every shot (deterministic: the Clifford fault fires with probability 1).
_D2_CIRCUIT = "R 0 1 2\nH 0 1 2\nI[T] 0 1\nI[CLIFFORD_FAULT: (1.0, CZ)] 0 1\nI[CCZ] 0 1 2\nI[T] 2"
# Noise between T blocks gives a mix of equal and different shots at every flush.
_NOISY_CIRCUIT = """R 0 1 2
H 0 1 2
I[T] 0 1
Z_ERROR(0.5) 0 1 2
H 0
I[T] 0 2
X_ERROR(0.5) 1
H 1 2
I[T] 1 2
Y_ERROR(0.3) 0 2
H 0 2
I[T] 0 1 2
M 0 1 2"""
_MAGIC_PAIRS = [
    (name, mode, has_leak, text)
    for name, modes, has_leak, text in gt.CORPUS
    for mode in modes
    if mode.startswith("magic_")
]
_ORIG_RUN = MagicsideSimulator.run
_EXPECTED = json.loads(gt.DIGESTS_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def list_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the reference list store: these tests spy on the per-shot grouping seam or edit the
    stored expansions in place, which a store that lives in the C++ engine does not have (the
    engine path is compared with the reference in `simulator_magic_dedupe_engine_test.py`)."""
    monkeypatch.setattr(sm, "_LIVE_STORE_ENGINE", False)


def _identity_grouping(exps: Sequence[ShotExpansion]) -> list[list[int]]:
    return [[s] for s, e in enumerate(exps) if e.chi > 0]


def _exp_digest(e: ShotExpansion) -> str:
    h = hashlib.sha256()
    for name in _ARRAY_FIELDS:
        a = getattr(e, name)
        h.update(a.tobytes() + str(a.shape).encode())
    h.update(repr((e.n_qubits, e.d, e.chi, e.log_mass, e.atol, e._is_canonical_merged)).encode())
    return h.hexdigest()


def _state_digest(sim: MagicsideSimulator) -> str:
    """Every live expansion, the branch masses and every RNG state of the simulator."""
    h = hashlib.sha256()
    for e in sim._live_exps:
        h.update(_exp_digest(e).encode())
    h.update(np.asarray(sim._log_branch_masses).tobytes())
    h.update(repr(sim.np_rng.bit_generator.state).encode())
    for rng in sim._shot_np_rngs or []:
        h.update(repr(rng.bit_generator.state).encode())
    return h.hexdigest()


def _run_case_recording(
    monkeypatch: pytest.MonkeyPatch, case: tuple[str, str, bool, str], grouping: Callable[..., Any]
) -> tuple[str, list[str], list[tuple[int, int]]]:
    """Golden digest, final-state digest per run(), and (groups, shots) per grouping call."""
    name, mode, has_leak, text = case
    finals: list[str] = []
    sizes: list[tuple[int, int]] = []
    def run(self: MagicsideSimulator) -> Any:
        res = _ORIG_RUN(self)
        finals.append(_state_digest(self))
        return res

    def counting(exps: Sequence[ShotExpansion]) -> list[list[int]]:
        groups = grouping(exps)
        sizes.append((len(groups), sum(len(m) for m in groups)))
        return groups

    monkeypatch.setattr(MagicsideSimulator, "run", run)
    monkeypatch.setattr(sm, "_group_shots_by_state", counting)
    mb = name == "c16_sparse_multibatch"
    digest = gt._run_case(
        stim.Circuit(text), mode, has_leak, batch=70 if mb else gt.BATCH, multibatch_packed=mb
    )
    return digest, finals, sizes


@pytest.mark.parametrize("case", _MAGIC_PAIRS, ids=[f"{c[0]}-{c[1]}" for c in _MAGIC_PAIRS])
def test_dedupe_on_off_identical_on_every_magic_golden_pair(
    case: tuple[str, str, bool, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Records, final live states, branch masses and RNG states are identical with dedupe on/off."""
    on = _run_case_recording(monkeypatch, case, sm._group_shots_by_state)
    off = _run_case_recording(monkeypatch, case, _identity_grouping)
    assert on[:2] == off[:2]
    assert on[0] == _EXPECTED[case[0]][case[1]]  # and both equal the committed golden digest


@pytest.mark.usefixtures("list_store")
def test_dedupe_actually_merges_shots_on_the_magic_golden_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards the on/off differential against being vacuous: many golden flushes do merge shots."""
    n_groups = n_shots = 0
    for case in _MAGIC_PAIRS:
        if case[1] == "magic_ds":
            sizes = _run_case_recording(monkeypatch, case, sm._group_shots_by_state)[2]
            n_groups += sum(g for g, _ in sizes)
            n_shots += sum(s for _, s in sizes)
    assert n_shots > 0 and n_groups < n_shots


def _run_noisy(seed: int, batch: int) -> tuple[str, str]:
    sim = MagicsideSimulator(stim.Circuit(_NOISY_CIRCUIT), batch_size=batch, seed=seed)
    sim.run()
    recs = hashlib.sha256(sim.get_final_measurement_records().tobytes()).hexdigest()
    return _state_digest(sim), recs


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.usefixtures("list_store")
def test_dedupe_matches_per_shot_on_noisy_circuit_with_mixed_group_sizes(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equal and unequal shots mixed at every flush: dedupe on == identity grouping."""
    seen: list[list[int]] = []
    real = sm._group_shots_by_state

    def spy(exps: Sequence[ShotExpansion]) -> list[list[int]]:
        groups = real(exps)
        seen.append([len(m) for m in groups])
        return groups

    monkeypatch.setattr(sm, "_group_shots_by_state", spy)
    on = _run_noisy(seed, 16)
    assert any(len(sizes) > 1 and max(sizes) > 1 for sizes in seen), seen
    monkeypatch.setattr(sm, "_group_shots_by_state", _identity_grouping)
    assert on == _run_noisy(seed, 16)


def _queued_d2_sim(
    batch: int = 6, perturb: Callable[[ShotExpansion], None] | None = None,
    shots: Sequence[int] = (1, 4),
) -> MagicsideSimulator:
    """Identical d=2 states on every shot (`perturb` applied to `shots`) and one pending block."""
    circ = stim.Circuit(_D2_CIRCUIT)
    sim = MagicsideSimulator(circ, batch_size=batch, seed=1)
    sim._do(circ)
    sim._flush_magic()
    sim._do(stim.Circuit("I[T] 1"))
    assert len(sim._pending_magic) == 1
    assert {(e.d, e.chi) for e in sim._live_exps} == {(2, 8)}
    if perturb is not None:
        for s in shots:
            perturb(sim._live_exps[s])
    return sim


def _pauli_x0_in_place(e: ShotExpansion) -> None:
    x = np.zeros(e.n_qubits, dtype=np.uint8)
    x[0] = 1
    apply_pauli(e, x, np.zeros(e.n_qubits, dtype=np.uint8), 0, in_place=True)


@pytest.mark.usefixtures("list_store")
def test_fan_out_gives_each_shot_its_own_expansion_with_no_shared_memory() -> None:
    sim = _queued_d2_sim()
    assert len(sm._group_shots_by_state(sim._live_exps)) == 1  # all six shots share one group
    sim._flush_magic()
    exps = sim._live_exps
    assert len({id(e) for e in exps}) == len(exps)
    for a, b in itertools.combinations(exps, 2):
        assert _exp_digest(a) == _exp_digest(b)
        for name in _ARRAY_FIELDS:
            assert not np.shares_memory(getattr(a, name), getattr(b, name)), name


@pytest.mark.usefixtures("list_store")
def test_in_place_edit_of_one_shot_after_dedupe_leaves_other_shots_unchanged() -> None:
    sim = _queued_d2_sim()
    sim._flush_magic()
    before = [_exp_digest(e) for e in sim._live_exps]
    _pauli_x0_in_place(sim._live_exps[2])  # E3-style in-place update: rewrites shot 2 only
    sim._live_exps[4].weights *= 2.0  # raw in-place edit of the geometry-sharing neighbour
    sim._live_exps[5].Gamma ^= np.uint8(1)
    after = [_exp_digest(e) for e in sim._live_exps]
    changed = [s for s in range(6) if before[s] != after[s]]
    assert changed == [2, 4, 5]


def _perturbations() -> dict[str, Callable[[ShotExpansion], None]]:
    def flip_ell(e: ShotExpansion) -> None:
        e.ell[1, 0] ^= np.uint8(1)

    def negate_weight(e: ShotExpansion) -> None:
        e.weights[0] *= -1.0

    def set_log_mass(e: ShotExpansion) -> None:
        e.log_mass = -0.75

    def clear_canon(e: ShotExpansion) -> None:
        e._is_canonical_merged = False

    def x_on_qubit0(e: ShotExpansion) -> None:
        _pauli_x0_in_place(e)

    return {
        "ell": flip_ell, "weights": negate_weight, "log_mass": set_log_mass,
        "canon_flag": clear_canon, "h_words": x_on_qubit0,
    }


@pytest.mark.parametrize("which", sorted(_perturbations()))
@pytest.mark.usefixtures("list_store")
def test_flush_matches_per_shot_when_two_shots_differ_in_exactly_one_field(
    which: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through `_flush_magic`: shots differing in exactly one field are not merged with the rest."""
    perturb = _perturbations()[which]
    sim = _queued_d2_sim(perturb=perturb)
    assert sm._group_shots_by_state(sim._live_exps) == [[0, 2, 3, 5], [1, 4]]
    sim._flush_magic()
    got = [_exp_digest(e) for e in sim._live_exps]
    monkeypatch.setattr(sm, "_group_shots_by_state", _identity_grouping)
    ref = _queued_d2_sim(perturb=perturb)
    ref._flush_magic()
    assert got == [_exp_digest(e) for e in ref._live_exps]
    if which != "canon_flag":  # the block canonicalises, which erases only the flag difference
        assert got[0] != got[1]  # the perturbation survived the block: groups really differ


@pytest.mark.usefixtures("list_store")
def test_flush_keeps_each_shots_own_log_mass_through_a_merged_group() -> None:
    """Identical geometry/weights but per-shot log_mass: each shot keeps its own value."""
    sim = _queued_d2_sim()
    want = [-0.25 * s for s in range(6)]
    for e, lm in zip(sim._live_exps, want):
        e.log_mass = lm
    assert len(sm._group_shots_by_state(sim._live_exps)) == 6  # log_mass is part of the key
    sim._flush_magic()
    assert [e.log_mass for e in sim._live_exps] == want
    assert len({_exp_digest(e) for e in sim._live_exps}) == 6


def _base_expansion() -> ShotExpansion:
    return ShotExpansion(
        5, 2, np.array([[1], [2]], np.uint64), np.array([3], np.uint64),
        np.array([[0, 1], [1, 0]], np.uint8), np.array([0, 1], np.int32),
        np.array([1 + 0j, 0.5j]), np.array([[4], [8]], np.uint64),
        np.array([[0, 1], [2, 3]], np.uint8), np.array([0, 1], np.uint8), 0.0, 1e-12, True,
    )


def _key_perturbations() -> dict[str, Callable[[ShotExpansion], None]]:
    def put(name: str, idx: Any, val: Any) -> Callable[[ShotExpansion], None]:
        return lambda e: getattr(e, name).__setitem__(idx, val)

    return {
        "weights": put("weights", 0, 1 + 1e-9 + 0j),
        "h_words": put("h_words", (0, 0), np.uint64(5)),
        "ell": put("ell", (1, 1), np.uint8(1)),
        "A_words": put("A_words", (0, 0), np.uint64(7)),
        "Gamma": put("Gamma", (0, 1), np.uint8(0)),
        "pivots": put("pivots", 0, np.int32(3)),
        "ell_odd": put("ell_odd", 0, np.uint8(1)),
        "a_union": put("a_union", 0, np.uint64(1)),
        "canon_flag": lambda e: setattr(e, "_is_canonical_merged", False),
        "atol": lambda e: setattr(e, "atol", 1e-9),
        "log_mass": lambda e: setattr(e, "log_mass", -3.0),
        "n_qubits": lambda e: setattr(e, "n_qubits", 6),
    }


@pytest.mark.parametrize("field", sorted(_key_perturbations()))
def test_grouping_separates_states_differing_in_exactly_one_field(field: str) -> None:
    base, other = _base_expansion(), _base_expansion()
    _key_perturbations()[field](other)
    assert sm._group_shots_by_state([base, other]) == [[0], [1]]
    assert sm._group_shots_by_state([base, other.copy(), base.copy(), other]) == [[0, 2], [1, 3]]


def test_grouping_separates_floats_differing_in_the_last_bit() -> None:
    """Exact-state means bit-exact: no rounding of log_mass, weights or atol in the key."""
    for field in ("log_mass", "atol"):
        base, other = _base_expansion(), _base_expansion()
        setattr(base, field, 0.5)
        setattr(other, field, float(np.nextafter(0.5, 1.0)))
        assert sm._group_shots_by_state([base, other]) == [[0], [1]], field
    base, other = _base_expansion(), _base_expansion()
    other.weights[1] = np.complex128(complex(np.nextafter(0.0, 1.0), 0.5))
    assert sm._group_shots_by_state([base, other]) == [[0], [1]]
    other = _base_expansion()
    other.weights[1] = np.complex128(complex(0.0, np.nextafter(0.5, 1.0)))
    assert sm._group_shots_by_state([base, other]) == [[0], [1]]


def test_grouping_separates_different_d_and_merges_equal_copies() -> None:
    base = _base_expansion()
    d1 = ShotExpansion(
        5, 1, np.array([[1]], np.uint64), np.array([1], np.uint64), np.array([[0]], np.uint8),
        np.array([0], np.int32), np.array([1 + 0j, 0.5j]), np.array([[4], [8]], np.uint64),
        np.array([[0], [1]], np.uint8), np.array([0, 1], np.uint8), 0.0, 1e-12, True,
    )
    assert base.d != d1.d
    zero = ShotExpansion.zero(5)  # d = 0, chi = 1: a real state, its own group
    assert sm._group_shots_by_state([base, d1, zero, base.copy()]) == [[0, 3], [1], [2]]


def test_grouping_skips_empty_expansions_and_orders_groups_by_lowest_member() -> None:
    a, b = _base_expansion(), _base_expansion()
    b.log_mass = -1.0
    empty = ShotExpansion.zero(5)
    empty.weights = np.zeros(0, dtype=np.complex128)
    assert empty.chi == 0
    assert sm._group_shots_by_state([empty, b, a, empty, b.copy(), a.copy()]) == [[1, 4], [2, 5]]


def test_block_path_draws_no_random_numbers_and_leaves_branch_masses_alone() -> None:
    sim = _queued_d2_sim()
    rng_before = repr(sim.np_rng.bit_generator.state)
    masses = np.array(sim._log_branch_masses)
    sim._flush_magic()
    assert repr(sim.np_rng.bit_generator.state) == rng_before
    assert np.array_equal(sim._log_branch_masses, masses)


# --- header layout (chi, d) and the cheap-fingerprint prefilter -------------------------------


def _array_bytes(e: ShotExpansion) -> bytes:
    """The array part of the exact key (no header), in key order."""
    assert e.ell_odd is not None
    arrays = (e.weights, e.h_words, e.ell, e.A_words, e.a_union, e.Gamma, e.pivots, e.ell_odd)
    return b"".join(a.tobytes() for a in arrays)


def _header_collision_pair() -> tuple[ShotExpansion, ShotExpansion]:
    """(chi=4, d=0) and (chi=1, d=4) whose array bytes concatenate to the same 104 bytes."""
    one = ShotExpansion(
        5, 4, np.arange(1, 5, dtype=np.uint64).reshape(4, 1), np.array([15], np.uint64),
        np.eye(4, dtype=np.uint8), np.arange(4, dtype=np.int32), np.array([1 + 0j]),
        np.array([[3]], np.uint64), np.array([[0, 1, 0, 1]], np.uint8),
        np.array([0, 1, 0, 1], np.uint8), 0.0, 1e-12, True,
    )
    raw = _array_bytes(one)
    four = ShotExpansion(
        5, 0, np.zeros((0, 1), np.uint64), np.frombuffer(raw[96:104], np.uint64).copy(),
        np.zeros((0, 0), np.uint8), np.zeros(0, np.int32),
        np.frombuffer(raw[:64], np.complex128).copy(),
        np.frombuffer(raw[64:96], np.uint64).copy().reshape(4, 1),
        np.zeros((4, 0), np.uint8), np.zeros(0, np.uint8), 0.0, 1e-12, True,
    )
    return four, one


def test_states_with_equal_array_bytes_but_different_chi_and_d_do_not_merge() -> None:
    """(chi=4, d=0) vs (chi=1, d=4): byte-identical array parts; the header must keep them apart."""
    four, one = _header_collision_pair()
    assert (four.chi, four.d, one.chi, one.d) == (4, 0, 1, 4)
    assert _array_bytes(four) == _array_bytes(one) and len(_array_bytes(one)) == 104
    assert sm._state_key(four) != sm._state_key(one)
    assert sm._group_shots_by_state([four, one, four.copy()]) == [[0, 2], [1]]


def test_state_key_changes_with_d_alone_and_with_chi_alone() -> None:
    """d and chi each reach the key (equal-length array parts cannot hide a change of either)."""
    base = _base_expansion()
    other_d = base.copy()
    other_d.d = base.d + 1
    other_chi = base.copy()
    for name in ("weights", "h_words", "ell"):  # chi 2 -> 3, shapes kept consistent
        arr = getattr(base, name)
        setattr(other_chi, name, np.concatenate([arr, arr[:1]]))
    assert sm._state_key(base) != sm._state_key(other_d)
    assert sm._state_key(base) != sm._state_key(other_chi)
    assert sm._group_shots_by_state([base, other_d, other_chi]) == [[0], [1], [2]]


def _same_state(a: ShotExpansion, b: ShotExpansion) -> bool:
    """Exact equality written independently of `_state_key`: shape + raw bytes of every field."""
    scalars = (a.n_qubits, a.d, a._is_canonical_merged) == (b.n_qubits, b.d, b._is_canonical_merged)
    floats = all(
        np.float64(getattr(a, f)).tobytes() == np.float64(getattr(b, f)).tobytes()
        for f in ("atol", "log_mass")
    )
    arrays = all(
        getattr(a, f).shape == getattr(b, f).shape
        and getattr(a, f).tobytes() == getattr(b, f).tobytes()
        for f in _ARRAY_FIELDS
    )
    return scalars and floats and arrays


def _reference_partition(exps: Sequence[ShotExpansion]) -> list[list[int]]:
    """Pairwise-comparison partition of the chi > 0 shots, ordered by lowest member."""
    groups: list[list[int]] = []
    for s, e in enumerate(exps):
        if e.chi == 0:
            continue
        for g in groups:
            if _same_state(exps[g[0]], e):
                g.append(s)
                break
        else:
            groups.append([s])
    return groups


def _random_state(rng: np.random.Generator, chi: int, d: int, nw: int) -> ShotExpansion:
    return ShotExpansion(
        5 + 64 * (nw - 1), d, rng.integers(0, 2**63, (d, nw)).astype(np.uint64),
        rng.integers(0, 2**63, nw).astype(np.uint64), rng.integers(0, 2, (d, d)).astype(np.uint8),
        rng.integers(0, 9, d).astype(np.int32), rng.normal(size=chi) + 1j * rng.normal(size=chi),
        rng.integers(0, 2**63, (chi, nw)).astype(np.uint64),
        rng.integers(0, 2, (chi, d)).astype(np.uint8), rng.integers(0, 2, d).astype(np.uint8),
        float(rng.normal()), 1e-12, True,
    )


def _mutate_once(rng: np.random.Generator, e: ShotExpansion) -> None:
    """Change exactly one bit/ulp of one field (rows of chi-sized arrays chosen uniformly)."""
    field = (*_ARRAY_FIELDS, "log_mass", "atol", "canon")[int(rng.integers(0, 11))]
    if field == "log_mass":
        e.log_mass = float(np.nextafter(e.log_mass, np.inf))
    elif field == "atol":
        e.atol = float(np.nextafter(e.atol, np.inf))
    elif field == "canon":
        e._is_canonical_merged = not e._is_canonical_merged
    else:
        arr = getattr(e, field)
        flat = arr.view(np.float64) if field == "weights" else arr.reshape(-1)
        if flat.size == 0:
            e.log_mass = float(np.nextafter(e.log_mass, -np.inf))
        elif field == "weights":
            i = int(rng.integers(0, flat.size))
            flat[i] = np.nextafter(flat[i], np.inf)
        else:
            flat[int(rng.integers(0, flat.size))] ^= 1


def _random_flush_input(rng: np.random.Generator) -> list[ShotExpansion]:
    chi, d, nw = (int(rng.choice(c)) for c in ((1, 2, 3, 8, 17, 64), (0, 1, 3), (1, 2)))
    bases = [_random_state(rng, chi, d, nw) for _ in range(int(rng.integers(1, 4)))]
    shots = []
    for _ in range(int(rng.integers(2, 13))):
        e = bases[int(rng.integers(0, len(bases)))].copy()
        for _ in range(int(rng.integers(0, 3)) if rng.random() < 0.6 else 0):
            _mutate_once(rng, e)
        shots.append(e)
    if rng.random() < 0.3:
        empty = ShotExpansion.zero(5)
        empty.weights = np.zeros(0, dtype=np.complex128)
        shots.insert(int(rng.integers(0, len(shots) + 1)), empty)
    return shots


def test_prefilter_partition_equals_independent_full_comparison_on_random_flushes() -> None:
    """500 random flush inputs incl. one-ulp / one-bit near-collisions: same partition exactly."""
    near_collisions = 0
    for seed in range(500):
        exps = _random_flush_input(np.random.default_rng(seed))
        got = sm._group_shots_by_state(exps)
        assert got == _reference_partition(exps), seed
        live = [e for e in exps if e.chi > 0]
        near_collisions += any(
            sm._state_fingerprint(a) == sm._state_fingerprint(b) and not _same_state(a, b)
            for a, b in itertools.combinations(live, 2)
        )
    assert near_collisions >= 20  # the prefilter really faces same-fingerprint, different-state


@pytest.mark.parametrize("field", ["weights", "h_words", "ell"])
def test_states_differing_only_in_an_unsampled_row_share_a_fingerprint_but_do_not_merge(
    field: str,
) -> None:
    """Same fingerprint, different later bytes: the exact key must still split them."""
    base = _random_state(np.random.default_rng(7), 64, 3, 1)
    other = base.copy()
    if field == "weights":
        other.weights[1] = np.complex128(complex(other.weights[1].real, np.nextafter(
            other.weights[1].imag, np.inf)))
    else:
        getattr(other, field)[1, 0] ^= np.uint8(1) if field == "ell" else np.uint64(1)
    assert sm._state_fingerprint(base) == sm._state_fingerprint(other)  # row 1 is not sampled
    assert not _same_state(base, other)
    assert sm._group_shots_by_state([base, other, base.copy()]) == [[0, 2], [1]]


def test_full_key_is_built_only_for_shots_that_share_a_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All-different shots (log_mass) build no full key; coinciding shots do; singletons never."""
    calls: list[int] = []
    real = sm._state_key
    monkeypatch.setattr(sm, "_state_key", lambda e: calls.append(1) or real(e))
    base = _base_expansion()
    distinct = []
    for s in range(6):
        e = base.copy()
        e.log_mass = -0.5 * s
        distinct.append(e)
    assert sm._group_shots_by_state(distinct) == [[s] for s in range(6)]
    assert calls == []
    assert sm._group_shots_by_state([base.copy() for _ in range(5)]) == [[0, 1, 2, 3, 4]]
    assert len(calls) == 5
    calls.clear()
    pair = [distinct[0], distinct[1], distinct[2], distinct[2].copy(), distinct[3]]
    assert sm._group_shots_by_state(pair) == [[0], [1], [2, 3], [4]]
    assert len(calls) == 2  # only the coinciding pair reaches the exact key
