"""Tests for the compiled-seed derivation of the Sinter sampler bases (WP-22 item 2)."""

from __future__ import annotations

import copyreg
import io
import pickle

import pytest
import sinter  # type: ignore[import-untyped]
import stim  # type: ignore[import-untyped]

from stimside import (
    CosetsideSampler,
    FlipsideSampler,
    MagicsideSampler,
    TablesideSampler,
)
from stimside._sampling import next_compiled_seed
from stimside.dem_generators.leakage_decoder import BaseDecoder
from stimside.op_handlers.leakage_handlers.leakage_uint8_coset import LeakageUint8Coset
from stimside.op_handlers.leakage_handlers.leakage_uint8_flip import LeakageUint8Flip
from stimside.op_handlers.leakage_handlers.leakage_uint8_tableau import (
    LeakageUint8Tableau,
)


def _samplers(seed: int):
    return (
        MagicsideSampler(seed=seed),
        FlipsideSampler(LeakageUint8Flip(), BaseDecoder(), seed=seed),
        TablesideSampler(LeakageUint8Tableau(), BaseDecoder(), seed=seed),
        CosetsideSampler(LeakageUint8Coset(), BaseDecoder(), seed=seed),
    )


def _clone(sampler):
    return pickle.loads(pickle.dumps(sampler))


@pytest.mark.parametrize("which", range(4))
def test_repickled_copies_never_share_seed_streams(which: int) -> None:
    root = _samplers(777)[which]
    c1, c2 = _clone(root), _clone(root)
    copies = [c1, c2, _clone(c1), _clone(c1), _clone(c2), _clone(_clone(c1))]
    streams = [tuple(c._next_compiled_seed() for _ in range(3)) for c in copies]
    assert len(set(streams)) == len(streams)
    # Copy 0 of the root keeps the raw seed at its first compile; everything else is distinct.
    flat = [s for st in streams[2:] for s in st] + list(streams[1])
    assert len(set(flat)) == len(flat)
    # Reproducible: the same pickling order from an identical sampler gives the same streams.
    c1_again = _clone(_samplers(777)[which])
    assert tuple(c1_again._next_compiled_seed() for _ in range(3)) == streams[0]


def test_compiled_seeds_use_63_bits_and_all_engines_accept_them() -> None:
    root = MagicsideSampler(seed=1)
    seeds = [_clone(root)._next_compiled_seed() for _ in range(300)]
    assert max(seeds) > 2**40 and all(0 <= s < 2**63 for s in seeds)
    assert len(set(seeds)) == len(seeds)
    circ = stim.Circuit("R 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    task = sinter.Task(circuit=circ)
    for sampler in _samplers(2**63 + 5):
        copy = _clone(_clone(sampler))
        copy._compile_count = 1  # skip the raw-seed slot: use a derived 63-bit seed
        assert copy.compiled_sampler_for_task(task).sample(8).shots >= 8


def test_next_compiled_seed_spawn_keys_are_collision_free() -> None:
    assert next_compiled_seed(None, 3, (1, 2)) is None
    assert next_compiled_seed(42, 0, 0) == 42 and next_compiled_seed(42, 0, (0,)) == 42
    keys = [(1,), (1, 0), (0, 1), (0, 0), (2,), (1, 1)]
    vals = {next_compiled_seed(42, i, k) for k in keys for i in (0, 1)}
    assert len(vals) == len(keys) * 2


def test_derived_seeds_depend_on_the_root_seed() -> None:
    for key in ((1,), (0, 1), (3, 1, 4)):
        for idx in (1, 2, 7):
            assert next_compiled_seed(5, idx, key) != next_compiled_seed(6, idx, key)
    a, b = (_clone(MagicsideSampler(seed=s)) for s in (11, 12))
    for sampler in (a, b):
        sampler._compile_count = 1  # skip the raw-seed slot
    assert a._next_compiled_seed() != b._next_compiled_seed()


def test_derived_seeds_fill_exactly_63_bits() -> None:
    seeds = [next_compiled_seed(1, idx, (1,)) for idx in range(1, 257)]
    assert all(0 <= s < 2**63 for s in seeds)  # bit 63 is never set (no 64-bit seeds)
    # Bit 62 is set for about half of the seeds (a 62-bit mask would never set it);
    # this sample is deterministic: SeedSequence output is stable.
    assert 64 < sum(1 for s in seeds if s >> 62) < 192
    assert len(set(seeds)) == len(seeds)


def _old_style_clone(sampler, worker_index):
    """Pickle `sampler` as it was before spawn keys: state has `_worker_index`, no `_spawn_key`."""
    state = sampler.__dict__.copy()
    state.pop("_spawn_key")
    state["_worker_index"] = worker_index

    class OldPickler(pickle.Pickler):
        def reducer_override(self, obj):
            if obj is sampler:
                return copyreg.__newobj__, (type(sampler),), state
            return NotImplemented

    buf = io.BytesIO()
    OldPickler(buf).dump(sampler)
    return pickle.loads(buf.getvalue()), state


@pytest.mark.parametrize("which", range(4))
def test_sampler_pickled_before_spawn_keys_still_unpickles_and_derives_seeds(which: int) -> None:
    root = _samplers(777)[which]
    old, old_state = _old_style_clone(root, 3)
    assert "_spawn_key" not in old_state
    assert old._spawn_key == (3,)  # its copy index becomes the key
    assert old._next_compiled_seed() == next_compiled_seed(777, 0, 3)  # not the raw seed: key (3,)
    assert old._next_compiled_seed() == next_compiled_seed(777, 1, (3,))
    child = pickle.loads(pickle.dumps(old))  # __getstate__ works and extends the key
    assert child._spawn_key == (3, 0)
    # Old copy 0 behaves like a spawn-key-(0,) copy: it uses the raw seed first.
    first, _ = _old_style_clone(root, 0)
    assert first._next_compiled_seed() == 777
    # A pre-spawn-key *root* state (no index at all) is process-salted like a fresh sampler.
    bare, _ = _old_style_clone(root, None)
    assert bare._spawn_key == ()
    assert bare._next_compiled_seed() == 777
