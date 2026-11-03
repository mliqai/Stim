"""Graph-like DEM decomposition and generated-DEM decoding tests."""

import numpy as np
import pytest
import sinter  # type: ignore[import-untyped]
import stim  # type: ignore[import-untyped]

from stimside.dem_generators import dem_decoding
from stimside.dem_generators.dem_decoding import decode_with_generated_dems
from stimside.dem_generators.dem_generator_marginal import MarginalLeakageDemGenerator
from stimside.dem_generators.dem_generator_marginal_test import (
    _make_surface_code_leakage_circuit,
)
from stimside.dem_generators.graphlike import (
    decompose_dem_graphlike,
    dem_has_hyperedge,
    raw_inst_components,
)


def _xor_symptom(inst: stim.DemInstruction):
    dets: set[int] = set()
    obs: set[int] = set()
    for d, o in raw_inst_components(inst):
        if d:
            dets ^= set(d)
            obs ^= set(o)
    return dets, obs


def _assert_valid_graphlike_decomposition(
    dem: stim.DetectorErrorModel, base_dem: stim.DetectorErrorModel | None = None
) -> None:
    out = decompose_dem_graphlike(dem, base_dem)
    assert not dem_has_hyperedge(out)
    assert decompose_dem_graphlike(out) is out

    flat_in = [i for i in dem.flattened() if i.type == "error"]
    flat_out = [i for i in out.flattened() if i.type == "error"]
    for inst in flat_out:
        p = float(inst.args_copy()[0])
        if p > 0:
            for dets, _ in raw_inst_components(inst):
                assert 1 <= len(dets) <= 2

    # When hyperedges exist, every detector-bearing error preserves net XOR detector/observable parity.
    if dem_has_hyperedge(dem):
        in_kept = []
        for inst in flat_in:
            dets, obs = _xor_symptom(inst)
            if any(d for d, _ in raw_inst_components(inst)):
                in_kept.append((float(inst.args_copy()[0]), dets, obs))
        assert len(flat_out) == len(in_kept)
        for out_inst, (p_in, dets_in, obs_in) in zip(flat_out, in_kept):
            dets_out, obs_out = _xor_symptom(out_inst)
            assert float(out_inst.args_copy()[0]) == pytest.approx(p_in)
            assert dets_out == dets_in
            assert obs_out == obs_in


def _random_dem_text(rng, num_dets, num_obs, num_lines, max_comp_dets):
    lines = [f"detector({d % 7}, {d // 7}) D{d}" for d in range(0, num_dets, 3)]
    lines.append(f"logical_observable L{num_obs - 1}")
    for _ in range(num_lines):
        comps = []
        for _ in range(int(rng.choice([1, 1, 1, 2, 3]))):
            k = int(rng.integers(0, min(max_comp_dets, num_dets) + 1))
            dets = rng.choice(num_dets, size=k, replace=bool(rng.random() < 0.2))
            obs = [o for o in rng.choice(num_obs, size=2) if rng.random() < 0.3]
            targets = [f"D{d}" for d in dets] + [f"L{o}" for o in obs]
            if targets:
                comps.append(" ".join(targets))
        if not comps:
            continue
        p = float(rng.choice([0.0, 0.5, rng.random() * 0.1, rng.random() * 1e-6]))
        tag = "[some_tag]" if rng.random() < 0.2 else ""
        lines.append(f"error{tag}({p!r}) " + " ^ ".join(comps))
    return "\n".join(lines)


@pytest.mark.parametrize("seed", range(40))
def test_random_dems_decompose_to_valid_graphlike(seed):
    rng = np.random.default_rng(seed)
    num_dets = int(rng.integers(3, 40))
    dem = stim.DetectorErrorModel(
        _random_dem_text(rng, num_dets, int(rng.integers(1, 90)), 60, 7)
    )
    _assert_valid_graphlike_decomposition(dem)
    base = stim.DetectorErrorModel(_random_dem_text(rng, num_dets, 3, 20, 2))
    _assert_valid_graphlike_decomposition(dem, base)


def test_repeat_blocks_shift_detectors_and_coords_decompose():
    dem = stim.DetectorErrorModel(
        """
        detector(1, 2) D0
        error(0.1) D0 D1
        error(0.01) D1
        repeat 3 {
            error(0.125) D0 D1 D2 D3 L0
            error(0.2) D2 D3
            detector(0, 0, 1) D2
            shift_detectors(0, 0, 1) 2
        }
        error(0.3) D0 D1 D2 L1 ^ D5
        """
    )
    _assert_valid_graphlike_decomposition(dem)
    _assert_valid_graphlike_decomposition(
        dem, stim.DetectorErrorModel("error(0.1) D0 D2 L0\nerror(0.1) D1 D3")
    )


def test_empty_known_set_and_observable_only_components():
    _assert_valid_graphlike_decomposition(
        stim.DetectorErrorModel("error(0.1) D0 D1 D2 D3 D4 L3")
    )
    out = decompose_dem_graphlike(
        stim.DetectorErrorModel(
            "error(0.1) D0 D1 D2 ^ L2\nerror(0.2) L1\nerror(0) D5 D6 D7 D8"
        )
    )
    assert not dem_has_hyperedge(out)


@pytest.mark.parametrize("n", [17, 20, 64, 70])
def test_large_hyperedges_decompose(n):
    lines = [f"error(0.01) D{d} D{d + 1}" for d in range(0, n - 1, 3)]
    lines += [f"error(0.01) D{d}" for d in range(1, n, 5)]
    lines.append("error(0.1) " + " ".join(f"D{d}" for d in range(n)) + " L0")
    lines.append("error(0.1) " + " ".join(f"D{d}" for d in range(0, n, 2)) + " L100")
    _assert_valid_graphlike_decomposition(stim.DetectorErrorModel("\n".join(lines)))


def test_cover_search_path_decomposes():
    for n in (3, 5, 9, 16):
        lines = [f"error(0.01) D{d} D{d + 1} L{d % 2}" for d in range(0, n - 1, 2)]
        lines.append("error(0.1) " + " ".join(f"D{d}" for d in range(n)) + " L7")
        _assert_valid_graphlike_decomposition(stim.DetectorErrorModel("\n".join(lines)))


@pytest.mark.parametrize("reweight_only", [False, True])
def test_surface_code_per_shot_dems_decompose(reweight_only):
    circuit = _make_surface_code_leakage_circuit(distance=3, rounds=3)
    gen = MarginalLeakageDemGenerator(reweight_only=reweight_only)
    rng = np.random.default_rng(5)
    records = rng.random((24, circuit.num_measurements)) < 0.05
    base = gen.base_dem(circuit)
    dems = gen(circuit, records)
    for dem in {id(d): d for d in dems}.values():
        _assert_valid_graphlike_decomposition(dem)
        _assert_valid_graphlike_decomposition(dem, base)
    _assert_valid_graphlike_decomposition(base)


def test_decode_with_generated_dems_skips_redecomposing_generator_dems(monkeypatch):
    circuit = _make_surface_code_leakage_circuit(distance=3, rounds=2)
    gen = MarginalLeakageDemGenerator()
    rng = np.random.default_rng(1)
    records = rng.random((16, circuit.num_measurements)) < 0.05
    dets = np.packbits(
        rng.random((16, circuit.num_detectors)) < 0.05, axis=1, bitorder="little"
    )
    decoder = sinter.BUILT_IN_DECODERS["pymatching"]
    dems = gen(circuit, records)

    # Raw (unregistered) copies must call decompose_dem_graphlike and produce identical predictions.
    raw_copies = [d.copy() for d in dems]
    want = decode_with_generated_dems(decoder, raw_copies, dets, decompose_errors=True)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Already-decomposed generator DEMs should not be re-decomposed")

    monkeypatch.setattr(dem_decoding, "decompose_dem_graphlike", fail_if_called)
    got = decode_with_generated_dems(decoder, dems, dets, decompose_errors=True)
    np.testing.assert_array_equal(got, want)


def test_dem_without_hyperedges_is_returned_unchanged():
    dem = stim.DetectorErrorModel("error(0.1) D0 D1\nerror(0.1) D2 L0")
    assert decompose_dem_graphlike(dem) is dem


def test_reweighted_dem_preserves_observables_and_detectors_when_edges_zeroed():
    import pymatching  # type: ignore[import-untyped]
    from stimside.dem_generators.leakage_decoder import MarginalDecoder

    circuit = stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        R 0 1 2
        RX 2
        I[LEAKAGE_TRANSITION_1: (0.05, 2-->U)] 2 1
        I[LEAKAGE_TRANSITION_1: (0.1, U-->3) (0.3, U-->2) (0.1, 2<->3)] 1 2
        CZ 1 2
        I_ERROR[LEAKAGE_TRANSITION_1: (0.3, 2-->3) (0.05, U-->3) (0.2, 3-->U)] 1 2
        TICK
        SQRT_Y 0 2
        I[LEAKAGE_TRANSITION_1: (0.2, U-->3) (0.2, 3-->U)] 1
        R 0
        I_ERROR[LEAKAGE_TRANSITION_1: (0.1, U-->2) (0.3, U-->3)] 1 0 2
        MPAD[LEAKAGE_MEASUREMENT: (0.85, 3): 0] 0
        X_ERROR(0.05) 2 1 0
        TICK
        CZ 2 0
        MPAD[LEAKAGE_MEASUREMENT: (0.1, 0) (0.2, 1) (0.9, 2): 0 2 1] 0 0 0
        DETECTOR rec[-3]
        DEPOLARIZE2(0.05) 0 2
        Y_ERROR(0.2) 0 1
        DEPOLARIZE2(0.45) 1 2
        PAULI_CHANNEL_1(0.033, 0.07, 0.079) 2 0 1
        TICK
        MPAD[LEAKAGE_MEASUREMENT: (0.9, 2): 2 1] 0 0
        DETECTOR rec[-1] rec[-5]
        M[LEAKAGE_PROJECTION_Z: (0.9, 2)] 0 1 2
        DETECTOR rec[-3] rec[-6]
        DETECTOR rec[-3] rec[-5]
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-3]
        """
    )

    class _NonPMDecoder(sinter.Decoder):
        def compile_decoder_for_dem(self, *, dem: stim.DetectorErrorModel):
            return sinter.BUILT_IN_DECODERS["pymatching"].compile_decoder_for_dem(dem=dem)

    seed, shots = 1053, 12
    recs = circuit.compile_sampler(seed=seed).sample(shots)
    recs = recs ^ (np.random.default_rng(seed).random(recs.shape) < 0.1)
    dets = np.packbits(
        circuit.compile_detector_sampler(seed=seed).sample(shots), axis=1, bitorder="little"
    )
    for dec in ("pymatching", _NonPMDecoder()):
        compiled = MarginalDecoder(
            decoder=dec, decompose_errors=True, reweight_only=True
        ).compile_for_task(sinter.Task(circuit=circuit))
        pred = compiled.decode_shots_bit_packed(
            bit_packed_detection_event_data=dets, records=recs
        )
        assert pred.shape == (shots, 1)

    # Coordinate-less highest detector & observable both preserved when last edge reweights to p=0.
    base = stim.DetectorErrorModel(
        "\n".join([f"error(0.1) D{i}" for i in range(8)] + ["error(0.1) D8 L0"])
    )
    edges = dem_decoding._extract_graphlike_edges(base)
    rebuilt = dem_decoding._apply_edge_reweights_to_dem(
        base, edges, np.array([[8, -1, 1e9]], dtype=np.float64)
    )
    assert rebuilt.num_detectors == 9
    assert rebuilt.num_observables == 1
    assert decode_with_generated_dems(
        _NonPMDecoder(),
        [np.array([[8, -1, 1e9]], dtype=np.float64)],
        np.zeros((1, 2), dtype=np.uint8),
        base_dem=base,
    ).shape == (1, 1)

    # Plain DEM with Matching raises TypeError; metadata-less reweight_only without base_dem raises ValueError.
    matcher = pymatching.Matching.from_detector_error_model(base)
    with pytest.raises(TypeError, match="pymatching.Matching"):
        decode_with_generated_dems(
            matcher,
            base,
            np.zeros((1, 2), dtype=np.uint8),
            decompose_errors=True,
            reweight_only=True,
            base_dem=base,
        )
    out_dem = decode_with_generated_dems(
        "pymatching", base, None, decompose_errors=True, reweight_only=True, base_dem=base
    )
    assert isinstance(out_dem, stim.DetectorErrorModel)

    upd = stim.DetectorErrorModel("error(0.2) D1")
    for bad_in in (upd, [upd]):
        with pytest.raises(ValueError, match="base_dem"):
            decode_with_generated_dems(
                "pymatching", bad_in, np.zeros((1, 1), dtype=np.uint8), reweight_only=True
            )


def test_generator_metadata_takes_precedence_over_fallback_base_dem():
    circuit = _make_surface_code_leakage_circuit(distance=3, rounds=2)
    gen = MarginalLeakageDemGenerator()
    recs = np.zeros((2, circuit.num_measurements), dtype=bool)
    recs[1, gen._get_analysis(circuit, False).flag_records[0]] = True
    dets = np.zeros((2, (circuit.num_detectors + 7) // 8), dtype=np.uint8)
    other_base = circuit.detector_error_model()

    for dec, rew in ((False, True), (True, True)):
        dems = gen(circuit, recs, decompose_errors=dec, reweight_only=rew)
        want = decode_with_generated_dems("pymatching", dems, dets)
        got = decode_with_generated_dems("pymatching", dems, dets, base_dem=other_base)
        np.testing.assert_array_equal(got, want)
        want_single = decode_with_generated_dems("pymatching", dems[1], dets[1:])
        got_single = decode_with_generated_dems(
            "pymatching", dems[1], dets[1:], base_dem=other_base
        )
        np.testing.assert_array_equal(got_single, want_single)


def test_fork_reweight_only_explicit_base_dem_adds_missing_edge() -> None:
    import inspect
    import pymatching  # type: ignore[import-untyped]

    if "edge_reweights" not in inspect.signature(pymatching.Matching.decode_batch).parameters:
        pytest.skip("needs the PyMatching version with per-shot edge_reweights")

    explicit_base = stim.DetectorErrorModel("error(0.01) D0 L0\nerror(0.01) D1\n")
    rw = np.array([[0, 1, 0.1]], dtype=np.float64)
    dets = np.array([[0b11]], dtype=np.uint8)
    pred = decode_with_generated_dems(
        "pymatching", [rw], dets, reweight_only=True, base_dem=explicit_base
    )
    np.testing.assert_array_equal(pred, np.array([[0]], dtype=np.uint8))


