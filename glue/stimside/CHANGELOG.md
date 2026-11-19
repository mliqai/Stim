# Changelog

All notable user-visible changes to `stimside` are documented in this file.

## Unreleased — Magicside, Cosetside, Kernel Loader, and Circuit-IR Updates

- **Independent `copy.copy(MagicsideSimulator)` in both engine and list modes:** Rewrote
  `MagicsideSimulator.__copy__` and added `_clone_op_handler` in `src/stimside/simulator_magic.py`
  so `copy.copy(sim)` returns an independent simulator in both engine-backed (`LiveExpansions`
  via `LiveExpansions.copy`) and Python-list (`ShotExpansion.copy`, whether `_LIVE_STORE_ENGINE`
  is `False` or a compiled run left a `list`) modes, cloning all mutable simulator state
  (`_pending_magic`, `_measurement_columns`, `_cached_m2d`, `_log_branch_masses`, `_corr_occurred`,
  `_live_ref_tab`, `_live_frame_events`, `_shot_tabs`, `_shot_sel_bits`, `qubit_coords`,
  `qubit_tags`, `coords_shifts`, `_unleaked_to_leaked_events`, `_leakage_events`,
  `_ref_sample_cache`, `np_rng` / `_shot_np_rngs`, `_tab_rngs` / `_tab_rngs_x`, and
  `_compiled_op_handler` / `compiled_op_handler`).
- **`MagicsideSimulator.run()` refusal after a failed first instruction:** Added `_started`
  (reset to `False` in `clear()` and checked/set to `True` in `run()` before the first
  instruction runs) so an exception inside instruction `0` makes subsequent `run()` calls raise
  `ValueError` until `clear()` is called (without adding an aborted-call latch).
- **Magicside tests & ASan+UBSan hostile-engine coverage:**
  - Retargeted the copy-isolation tests in `src/stimside/simulator_magic_live_store_test.py` and
    `src/stimside/simulator_magic_memo_test.py` for independent `copy.copy` semantics across both
    engine modes, and added boundary-copy, pending-magic copy, custom-handler copy, and
    first-instruction failure retry tests in `src/stimside/simulator_magic_live_store_test.py`.
  - Added `else:` list-mode fallback assertions to the default engine checks in
    `src/stimside/simulator_magic_live_store_test.py` and
    `src/stimside/simulator_magic_block_engine_test.py`.
  - Enabled ASan+UBSan execution (prepending `fi_so` in `LD_PRELOAD` with
    `verify_asan_link_order=0`, skipping only under `_under_tsan()` or when `c++` is missing) in
    `src/stimside/_native/magic_engine_oom_test.py`,
    `src/stimside/_native/magic_engine_born_hostile_test.py`, and
    `src/stimside/_native/magic_engine_measure_hostile_test.py`.
- **Magicside & Cosetside documentation corrections:** Updated `README.md` to clarify that
  `MagicsideSimulator` requires `libmagic_kernels.so` (Python list fallback applies only when
  `engine_numerics_available()` / `numerics_match_numpy()` returns `False`), note when compiled
  stream/query paths produce a Python `list` or per-call fallbacks occur, document that
  `_LIVE_STORE_ENGINE` is checked on every `clear()`, and describe independent `copy.copy(sim)`
  semantics and the updated kernel-loader policy; replaced the stale `"zero overhead on the healthy
  path"` claim in `CHANGELOG.md` with the paired-worker `run()` ratios (`0.989x–1.008x`) and
  pre-optimization `interactive_do` timings (`~1.18-1.24x` slower).
- **Faster `CosetsideSimulator.interactive_do` whole-circuit replay:** Cached the unrolled
  instruction sequence and step-match verdict in `_compute_unrolled_ops`, added a `whole_match`
  fast path in `interactive_do` for the whole circuit at time 0, and a per-instruction fast
  branch in validation that is taken only while the instruction equals the reference step at the
  step cursor (after a direct step call such as `do_conditional_clifford` between
  `interactive_do` calls, the next `interactive_do` raises `ValueError` as before instead of
  replaying against the wrong steps; the step check during execution is kept),
  deferred `rec[-k]` measurement counting in `_update_and_check_meas_count`,
  `_validate_interactive_slow_op`, and `_validate_interactive_item` via `num_measurements` and
  `_REC_FEEDBACK_OPS`, removed unreachable empty/non-positive checks, and restored the detailed
  Pauli product rotation (`SPP`/`SPP_DAG`) guidance error message (`0.832x–0.941x` of the
  pre-validation baseline on plain surface-code `interactive_do` timings).
- **Ordinary-subclass contract for `CosetsideSimulator`:** Added construction-time behavioural
  latch verification via `_ensure_subclass_latch` and `_SUBCLASS_VERDICTS` (raising `TypeError`
  for unsupported subclasses whose latched classes cannot be built or class-swapped), and removed
  the instance-`__dict__` fallback latch, `abc.ABCMeta.__new__` special-casing, per-class latch
  attribute caching, and redundant property guards on `unleaked_to_leaked_records` and
  `active_shots`.
- **Callback-flag unwind & docstring clarifications:** Moved `_override_packed_words` and
  `_in_group_callback = True` assignments inside `try:` across `_eval_group_condition`,
  `_dcc_sequential_py`, `_dcc_sequential_cpp`, and `_dcc_run_unitary`, restored `prev_cb` at
  `_dcc_run_unitary` exit, cleared both flags in `do_conditional_clifford`'s outer
  `except BaseException:` and `_clear_body`, and updated the `CosetsideSimulator` docstring
  with post-completion exception non-latching, latched `type(sim)` / read-only count behavior,
  and unsupported-subclass `TypeError` behavior.
- **Cosetside safety & sync test coverage:** Added tests in
  `src/stimside/simulator_coset_r8_safety_test.py` covering outer `run()` / `interactive_do`
  handler/interrupt latching, `rec[-k]` and `REPEAT` replay boundaries, whole-match /
  mismatched `compiled_ref` rejection, `condition_group_fn` flag-unwind windows, direct
  seam/mutator interrupt latching, `BaseException` `__str__` formatting, and non-vacuous
  interrupt-count assertions; restored C++-to-Python mirror and pre-clear state comparisons in
  `src/stimside/simulator_coset_sync_test.py`; tightened the `t == 0` interrupt assertion in
  `src/stimside/simulator_coset_r7_safety_test.py`; and retargeted the exotic-subclass tests in
  `src/stimside/simulator_coset_r8_safety_test.py` to expect `TypeError` at construction.
- **Kernel loader & packaging hardening (`src/stimside/_native/_kernel_loader.py`,
  `src/stimside/_native/packaging_build_test.py`):**
  - Increased the default `_native` directory `flock` wait from `30` s (`_LOCK_MAX_RETRIES = 600`)
    to `600` s (`12000` retries at `0.05` s), configurable via `STIMSIDE_BUILD_LOCK_TIMEOUT_S`
    (emitting a `RuntimeWarning` and falling back to the default on invalid values).
  - Caught `RecursionError` alongside `(OSError, ValueError)` in `_read_build_record` when parsing
    deeply nested `.build_id` JSON.
  - Kept `_plan_source_build` pure and moved sidecar adoption (`_adopt_build_record`) under
    `_acquire_dir_lock`, using `os.link` and a locked re-read so adoption never replaces an existing
    valid v2 record while still upgrading legacy/invalid records and swallowing `OSError`.
  - Switched lock-timeout and once-per-library failed-rebuild warnings from `sys.stderr.write` to
    `warnings.warn(..., RuntimeWarning)` so callers can filter them with `warnings` /
    `PYTHONWARNINGS`.
  - Recorded `"machine"` (`_current_machine()`) and a `/proc/cpuinfo` `"cpu_flags"` digest
    (`_cpu_flags_digest()`) in `"native"` `.build_id` sidecars (including `setup.py` in-tree sync
    during wheel and editable builds), triggering a native rebuild if either changes.
  - Allocated temporary library and build-record files under unique random names inside
    `_native`, created with the umask applied to modes `0777` / `0666` (as the compiler would), so
    built libraries and records stay readable by other users; and excluded `pip install --target`
    trees containing `stimside*.dist-info` in `_is_source_checkout`.
  - Expanded `src/stimside/_native/packaging_build_test.py` (`+16` test cases) covering
    configurable lock timeouts, `PYTHONWARNINGS` filtering, `RecursionError` records, locked
    non-replacing adoption, native machine/CPU-flags mismatch rebuilds, `.dist-info` vs
    `.egg-info` checkout detection, failed-compile temp-file cleanup, `site-packages` forced
    recompile isolation, and `setup.py` wheel/editable sync and `inplace` error handling.

- **Zero-target fusion in `unrolled_to_flattened_indices` and loss-oracle DEM generation:**
  Updated `unrolled_to_flattened_indices` (`src/stimside/util/leakage_events.py`,
  `_U2F_NON_FUSING`) so consecutive identical zero-target instructions fused by
  `circuit.flattened()` across `REPEAT` and `SHIFT_COORDS` boundaries map cleanly to accurate
  `flattened()` indices, superseding the previous `ValueError` limitation noted below so
  `MarginalLeakageDemGenerator(loss_oracle=True)` and `Builder` accept zero-target-fusing
  circuits while corrupted alignments with `validate=True` still raise `ValueError`.
- **`CircuitIR` performance, lazy alignment, tag equality, and verification updates:**
  - Memoized `classify_category` and `classify_untagged` (`stim.gate_data is _REAL_GATE_DATA`,
    `op_memo`), made `_FlatAlignment` (`flat_index`, `flat_offset`, `flat_ops`) evaluate lazily
    on first alignment access, removed `_tag_token` in favor of direct `op.tag == flat_op.tag`
    comparison, and switched `IRInstr` and `CircuitIR` to `@dataclass(frozen=True, eq=False)`
    for Python 3.11+ compatibility in `src/stimside/util/circuit_ir.py` and
    `src/stimside/util/circuit_ir_test.py`.
  - Added lazy `circuit_ir` imports inside `w12_ir_fields` and `w12b_tableau_layout` plus 14
    additional parity and zero-target/gate-semantics witness circuits in
    `src/stimside/ir_oracle_test.py` without modifying existing oracle entries, and expanded
    `tools/ir_mutants/run_required_mutants.py`, `tools/ir_mutants/required_mutants.tsv`, and
    `tools/ir_mutants/README.md`.
- **Internal `CircuitIR` migrations with unchanged simulator and oracle outputs:**
  - Migrated `src/stimside/util/known_states.py` (`compute_known_states_uint8(..., ir=...)`,
    `_forward_known_states`, `_backward_known_states`, rejecting conflicting `unrolled_ops` and
    `ir`) onto `CircuitIR`, with unit tests in `src/stimside/util/known_states_ir_test.py`.
  - Migrated `src/stimside/_flip/sync_reference.py` (`compute_sync_reference(..., ir=...)`,
    `_record_op_ref_z`, `_step_mpad`, `_step_product_meas`, `_step_controlled_pauli`,
    `_resolve_ir`) and `src/stimside/simulator_flip.py` (`FlipsideSimulator` building `self._ir`
    once in `__init__`) onto `CircuitIR`, with unit tests in
    `src/stimside/_flip/sync_reference_ir_test.py`.
  - Migrated `src/stimside/_coset/reference_compile.py`
    (`CompiledReferenceCircuit(..., ir=None)` validating `ir.num_qubits` and
    `ir.num_measurements` before `ReferenceCHPTableau` allocation and dispatching via
    `_compile_on` without recursive `REPEAT` walking) onto `CircuitIR`, with unit tests in
    `src/stimside/_coset/reference_compile_ir_test.py`.

## Run 8 — Stage A (Magicside Batch Engine Default ON)

- **Magicside native batch engine ON by default (`P3`, `D-35`):** Flipped `_LIVE_STORE_ENGINE`
  to `True` in `src/stimside/simulator_magic.py`, so `MagicsideSimulator` live streaming
  (`fault_mode="DIRECT"`, `backend="STREAM"`) keeps per-shot expansions in `MagicBatchEngine` /
  `LiveExpansions` (`libmagic_kernels.so`) by default (~2.4–2.5x faster on leaky SC d=3,
  ~3.0–3.1x on leaky SC d=5, ~4.3–4.4x on X-memory k=6 vs the Python list path).
- **Automatic Python fallback:** If `libmagic_kernels.so` is unavailable or
  `engine_numerics_available()` (`numerics_match_numpy()`) returns `False`, `MagicsideSimulator`
  falls back automatically to the Python reference list path.
- **Documentation & benchmarks:** Documented the default engine path, automatic fallback,
  operational limits (`FNF-31` not async-signal-safe, `FNF-32` cross-engine lock order, `FNF-34`
  `os.fork()` behaviour), and the ~10% `STIMSIDE_NATIVE_BUILD=1` speedup on amplitude-heavy
  circuits in `README.md` and `src/stimside/_native/magic_engine.py`; updated
  `benchmarks/README.md` to the portable-build Run 8 baseline table.

## Run 8 — Stage B (Cosetside Safety, Sticky Loader & Tests)

- **Interrupted `interactive_do` at `circuit_time == 0` (`R7-F1`):** `interactive_do` in
  `src/stimside/simulator_coset.py` now snapshots `_mutation_epoch`, `_step_cursor`,
  `_circuit_time`, `len(_compiled_ref.steps)`, and `_clear_epoch` at entry; `_used_interactively`
  and `_unrolled_ops` are restored only when all four state fields are unchanged at
  `circuit_time == 0`, otherwise `_record_aborted_call` latches the simulator.
- **Aborted-call latch & subclass-safe class swap (`D-36`, `R7-F4`):** Replaced the static failed
  class with dynamic subclass builders `_get_failed_class` and `_get_aborted_class` (keeping
  `_FailedCosetsideSimulator` and pre-building `_AbortedCosetsideSimulator`), preserving
  `isinstance` and `__slots__` layouts on user subclasses, bypassing user `__init_subclass__`
  hooks/required kwargs via `_SuppressInitSubclass` and custom metaclass `__new__` kwargs via
  `abc.ABCMeta.__new__` / `type.__new__`, and keeping `_FAILED_CLASSES` / `_ABORTED_CLASSES`
  GC-collectible (`weakref.WeakKeyDictionary` with `weakref.ReferenceType` values plus local
  class attributes). `_clear_aborted_latch` restores `__class__` first and re-latches on
  interrupt during cleanup, `_raise_engine_failed` / `_raise_aborted_call` use safe fallback
  messages without `assert`, and `_record_engine_failed` / `_check_not_engine_failed` perform
  idempotent class swaps with fallback protection on build errors. All 2 stateful properties
  (`active_shots`, `unleaked_to_leaked_records`) and 34 stateful/seam methods (including
  `_do_bare_instruction`, `_next_step_meta`, `_record_unleaked_to_leaked_counts`,
  `_record_leakage_event`, and `sample_pauli_noise_sync`) raise `RuntimeError` after an aborted
  mutating call (until `clear()` resets the simulator) or permanently after a native engine
  failure. In paired-worker benchmarks against the pre-latch tree (`run()` throughput ratios),
  plain SC `d=3, B=64` measured 0.991x (0.984-0.997, 4,086 shots/s), `d=5, B=64` measured
  0.989x (0.980-1.000, 1,591-1,604 shots/s), `d=5, B=256` measured 1.008x (0.991-1.016,
  4,387-4,408 shots/s), and dense 0/1 `d=5, B=256` measured 1.005x (0.992-1.025, 602-607
  shots/s), while `interactive_do(whole circuit)` on plain SC was ~1.18-1.24x slower before
  the subsequent validation optimization. In the Stage B delta round (`R8-B-C2-M1`),
  hardened latching against broken `__str__()` and heap exhaustion via `_fmt_latch_msg` with
  pre-allocated fallbacks (`_FAILED_FB`, `_ABORT_FB`), pre-warmed `_get_failed_class` /
  `_get_aborted_class` in `CosetsideSimulator.__init__` for subclasses, validated cached classes
  in `_get_latch_class` with `issubclass`, and added an instance-`__dict__` fallback in
  `_apply_latch` / `_restore_dict_latch` (covering `_BLOCKED_METHODS` and `_BLOCKED_ALL`, plus
  failed/aborted guards in `active_shots` and `unleaked_to_leaked_records`) when `__class__` swap
  raises `TypeError`.
- **Atomic `interactive_do` & noise-op validation (`R7-F3`, `R7-F3b`):**
  - `_validate_interactive_item` validates full `Circuit` and `CircuitRepeatBlock` inputs before
    applying the first instruction (including `_validate_mpp_op` via
    `reduce_hermitian_pauli_product` for anti-Hermitian `MPP` past end, conservative static
    `rec[-k]` feedback validation via `_update_and_check_meas_count` for standard handlers
    `_TrivialCompiledOpHandler` and `CompiledLeakageUint8Coset`, and O(1) short-circuiting of
    nested empty or zero-count `REPEAT` blocks in both `_validate_interactive_item` and `_do`
    via `_has_any_instruction`). Setting `_unrolled_ops` and `_used_interactively` in
    `interactive_do` runs inside the `try` restore block.
  - `do_conditional_pauli_noise` restricts operations to `_PAULI_NOISE_OPS` (the 14 Pauli noise
    and identity gates), atomically raising `ValueError` for non-noise instructions.
- **Native engine lifetime across explicit `__del__()` (`R7-F5`):** Added `_EngineOwner` with its
  own finalizer and retained a local `_cpp_owner` reference across every native call site so an
  explicit `sim.__del__()` during an in-flight kernel cannot destroy the engine in use.
- **Export/clear re-entry, zero-alloc RNG reset & circuit-change detection (`R7-F6`):**
  - `_verify_and_commit_export` checks epochs after `np.array_equal` and again after clearing
    `_cpp_dirty`; `_abort_export_sync` unconditionally sets `_cpp_dirty = True`; mutating native
    kernel sites (`_dcc_disjoint_cpp_v3`, `_project_pauli_cpp_exec`, `_postselect_cpp_exec`,
    `_reset_z_cpp`) also set `_cpp_dirty = True` after kernel return.
  - `active_shots` and `_clear_body` verify `_mutation_epoch` at exit and raise `RuntimeError` on
    mid-call mutation.
  - `engine_set_rng_states` in `src/stimside/_native/coset_kernels.cpp` copies caller RNG states
    into `shot_rng_states_owned` in place with zero heap allocations.
  - `_ensure_run_ops` stores `_run_ops_circuit_snap` and rebuilds `_run_ops` when `circuit`
    changes (documented in `CosetsideSimulator.run` that replacements must remain compatible with
    `_compiled_ref.steps`).
- **Sticky build flags & loader hardening (`D-42`, `R7-F7`):**
  - Upgraded `.lib<name>.build_id` sidecars in `src/stimside/_native/_kernel_loader.py` and
    `setup.py` to v2 single-line JSON separating `source_hash` from build `mode`
    (`_compute_source_hash`, `_read_build_record`, `_write_build_record`, `_plan_source_build`).
    In a writable source checkout, unset env vars never trigger a rebuild, source-triggered
    rebuilds keep the recorded mode (defaulting to portable without a record), missing/legacy
    sidecars are adopted as `unknown` without rebuilding until an explicit
    `STIMSIDE_NATIVE_BUILD=1` or `STIMSIDE_PORTABLE_BUILD=1` request, and `-march=native`
    fallback in `_compile_so_variants` records the requested mode so later starts do not loop.
    `_write_build_record` `OSError` failures after `os.replace` are isolated with a once-per-path
    warning (`_warn_failed_build_id_once`) so newly built native libraries are not overwritten by
    the portable fallback.
  - `_is_source_checkout` now requires the `src/stimside/_native` layout alongside `setup.py`
    (and excludes `site-packages` / `dist-packages`).
  - `_compile_so_variants` catches `(OSError, subprocess.SubprocessError)` (not broad
    `Exception`) and `_warn_failed_rebuild_once` emits a once-per-library warning on `stderr`
    when falling back to an existing `.so`.
  - `_acquire_dir_lock` uses non-blocking `flock` with bounded retries (`_LOCK_MAX_RETRIES`,
    `_LOCK_RETRY_DELAY_S`), guards against double-closing `lock_fd`, and warns once on `stderr`
    on timeout before building locklessly.
- **Test coverage & repairs (`R7-F8`):** Added `src/stimside/simulator_coset_r8_safety_test.py`
  (covering `R7-F1`, `D-36`, `R7-F4`, `R7-F3`/`R7-F3b`, `R7-F5` mid-kernel destroy check, `R7-F6`,
  and Stage B Round 2 regressions/mutant killers); expanded
  `src/stimside/simulator_coset_r7_safety_test.py`,
  `src/stimside/_native/coset_kernels_test.py` (all 25 guarded exports raw + wrapped, sentinel
  rebuild isolation, safe `RLIMIT_AS` OOM), and `src/stimside/_native/packaging_build_test.py`
  (recording mocks, dirty-copy sdist/wheel builds, `D-42` table coverage, strict
  `_is_offline_uv_prereq_error` classifier that fails rather than skips on build-backend/compile
  errors, `AssertionError` propagation, `EBADF` fast-break, and build-record/lock-close tests);
  repaired `_fingerprint` in `src/stimside/simulator_coset_mask_shape_test.py` to read live
  `engine_get_rng_states`; enabled all 14 `src/stimside/_native/coset_engine_oom_test.py` cases
  under ASan.

## Run 8 — Stage C (WP-27 Step 0 — Internal Circuit-IR Pilot Test Net)

- **Strictly additive test net (`D-46`, `R8-D9`, `R8-D10`):** Updated prior to `C6` Gate / `CP-C`
  to record verified `C1`–`C5` deliverables and Stage C Gate Round 2 fixes. Zero existing
  production, test, or config modules were modified, no consumers are wired in Run 8,
  `src/stimside/__init__.py` `__all__` stays at 38 names, and `src/stimside/layering_test.py` is
  unmodified.
- **Tag-agnostic circuit IR core (`src/stimside/util/circuit_ir.py`):** Added immutable, hashable
  IR primitives and pure derived passes with zero tag parsing or cross-layer imports: enums
  `Category` and `UntaggedFilter`; frozen dataclasses `IRInstr`, `CircuitIR`, `StepMap`,
  `TableauStepLayout`, and `CopyMap`; private gate-class sets `_ANNOTATION`, `_HERALDED`,
  `_CORRELATED`, `_PAULI_TERMS`, `_NON_FUSING`, and `_SHIFTED_ARGS_GATES`; alignment helpers
  `_tag_token`, `_op_matches_flat`, `_try_fuse_zero`, and `_align_flat_ops`; and functions
  `classify_category`, `classify_untagged`, `build_ir`, `step_map`, `tableau_step_layout`,
  `split_copies`, and `unrolled_to_flat`.
  - **Zero-target fusion & full alignment validation (`R8-C-REV-M1` / `R8-C-C6-M1`):** `build_ir`
    accurately maps empty-target fusing instructions across all 73/80 fusing Stim gates (matching
    `stim.Circuit.flattened()` across `SHIFT_COORDS`, `REPEAT`, and empty `REPEAT` boundaries
    while preserving non-fusion on the 7 `_NON_FUSING` gates). Validates op name, opaque escaped
    tag via `_tag_token` (preserving RFC `I6c` single `op.tag` attribute load inside `IRInstr`),
    target slices, overshoot, `gate_args_copy()` (except coordinate-shifted `DETECTOR` and
    `QUBIT_COORDS`), non-fusing 1-to-1 target equality, and trailing extra `flat_ops` — raising
    loud `ValueError` on any mismatch.
  - **Alias & case canonicalization (`R8-C-C5-L1`):** `classify_category` and `classify_untagged`
    canonicalize input names via `stim.gate_data(name)` (`gd.name`), resolving `"CORRELATED_ERROR"`
    and all lowercase Stim gate names identically to canonical uppercase names.
  - **Input preconditions (`R8-C-REV-L3` / `R8-C-C6-L3`):** `tableau_step_layout` raises
    `ValueError` when `has_bare_op` is shorter than `step_to_unrolled`; `build_ir` raises
    `TypeError` when a sequence input contains `stim.CircuitRepeatBlock`; `split_copies` documents
    its `from_op_list` empty `CopyMap` behavior and verbatim `copy_len` recording.
- **Per-consumer lazy tag memo (`src/stimside/op_handlers/ir_tags.py`):** Added `TagMemo`
  (method `params`, private type var `_T`) plus key-policy functions `BUILDER_KEY`
  (`(op.name, op.tag)`) and `VALUE_KEY` (instruction value) so tag parsing stays outside
  `src/stimside/util/` and lazy per consumer without caching raised exceptions.
- **Invariant test suite (`src/stimside/util/circuit_ir_test.py`):** Added 39 self-contained
  tests covering rev-2 invariants `I1`–`I19`, rev-3 additions (`I18b` both-direction memo
  differential, `I20`–`I23`, `I6w` recursive purity `test_i6w_recursive_util_and_ir_tags_purity`
  over `src/stimside/util/` and `src/stimside/op_handlers/ir_tags.py`), tightened AST purity
  scanners (`_scan_circuit_ir_i6`, `_scan_tree_i6w`, `_scan_common_evasions`,
  `test_i6_and_i6w_extended_evasion_sensitivity` catching `exec`, `eval`, `getattr`, `pydoc`,
  `builtins`, `sys.modules`, `TYPE_CHECKING` rebinding, and cross-layer absolute/relative imports
  inside function bodies), linear-time complexity guard (`test_i19_build_ir_linear_complexity`,
  using `time.process_time()` to avoid same-core wall-clock flakiness plus fused-`REPEAT` guards),
  zero-target fusion & validation tests (`test_zero_target_fusion_witnesses_and_offsets`,
  `test_zero_target_fusion_all_80_gates`, `test_build_ir_alignment_validation_errors`),
  live `A4_u2f_validate_off` pin (`test_legacy_u2f_validate_true_kills_a4_witnesses`),
  precondition & alias tests (`test_precondition_guards_and_negative_copy_len`,
  `test_rf_alias_and_case_canonicalization`), fresh novel-mutant detectors
  (`test_rf_oplist_uc_generator_and_aliasing`, `test_rf_first_measurement_after_mry_and_heralded`,
  `test_rf_tableau_layout_numpy_bool_and_int`,
  `test_rf_tag_memo_falsy_values_and_value_key_targets`), and blind-mutant detectors
  (`test_rf_enum_str_mixin_contract`, `test_rf_classify_untagged_multi_qubit_fallback`,
  `test_rf_build_ir_flattened_too_few_ops_raises`, `test_i3_has_repeat_and_no_repeat_identity`).
- **Structural oracle (`src/stimside/ir_oracle_test.py`):** Added 6 self-contained oracle tests
  pinning `FROZEN_DIGESTS` across 21 `WALKERS` × 123 circuits (2,583 total entries, 193 raising:
  53 `NotImplementedError`, 140 `ValueError`). `FROZEN_DIGESTS` was generated once from `CP-B`
  (identical across Python 3.13, `-BO`, and 3.11) and remains untouched post-Round-2 (frozen
  post-Stage-C-gate per `D-46`, like `src/stimside/golden_digests.json`).
- **Required-mutant runner (`tools/ir_mutants/`):** Added outside `src/` (excluded from sdist,
  wheel, and `pytest` collection per `D-46`): `tools/ir_mutants/run_required_mutants.py`
  (`load_mutants`, `check_pattern_counts`, `MutantSpec`, `DetectorResult`, `MutantResult`,
  `GROUPS`, `_classify_pytest_output`, `_aggregate_verdict`, `_summarize_and_write`),
  `tools/ir_mutants/required_mutants.tsv`, and `tools/ir_mutants/README.md`.
  - Flipped `A4_u2f_validate_off` (`src/stimside/util/leakage_events.py`, `validate` guard) from
    `equivalent` to `kill` after live zero-target fusion witnesses (`W12`/`W16` through `Builder`
    and `MarginalLeakageDemGenerator`) refuted RFC §7.2's fuzz-only argument: covers **50/50**
    non-equivalent mutants KILLED(test) (100%, 0 equivalent survivors, 0 timeouts, 0 setup errors).
  - Hardened `_classify_pytest_output` (parses pytest `E` exception lines via `_TB_E_EXC_RE` and
    `ERROR` lines via `_ERROR_RE` on exit code 1), added distinct `ERROR` status for pytest
    fixture/setup errors (exit code 1 with non-zero `n_error` and zero `failed_tests`), and
    updated `_summarize_and_write` to exit nonzero when `n_eq_killed` or `n_setup_err` is nonzero.
- **Blind fresh-mutant round (`C5`):** Independently generated 283 unique blind mutants across
  `src/stimside/util/circuit_ir.py` (262) and `src/stimside/op_handlers/ir_tags.py` (21): 40
  hand-crafted + 243 mechanical AST; Round 1 raised score from 277/283 to 283/283 (100%).
  Round 2 re-verified all applicable blind mutants plus a fresh mechanical AST & hand-crafted
  mutant sweep covering every new/edited branch in `src/stimside/util/circuit_ir.py`.
- **Carry-forward note (`R8-CF-U2F-ZEROTARGET`, pre-existing `CP-B`, untouched per `D-46`):**
  Existing `unrolled_to_flattened_indices` in `src/stimside/util/leakage_events.py` does not fuse
  consecutive zero-target ops across `REPEAT` or `SHIFT_COORDS`, but fails loudly with `ValueError`
  when `validate` is `True` (not silent corruption). Left untouched in Run 8 per `D-46` and carried
  forward to Run 9; `test_legacy_u2f_validate_true_kills_a4_witnesses` pins `validate` checking.

## Run 7 — Stage B (Packaging & Loader, `11b42b5`)

- **Portable builds by default (`D-39`):** `get_cpp_compile_and_link_args` in
  `src/stimside/_native/_kernel_loader.py` builds portable shared libraries without
  `-march=native` (with `-ffp-contract=off`) unless `STIMSIDE_NATIVE_BUILD=1` is set;
  `STIMSIDE_PORTABLE_BUILD=1` always forces portable builds.
- **Cross-Python wheel tag & headers:** `StimsideBdistWheel` in `setup.py` tags wheels
  `py3-none-<platform>` (compatible across Python 3.11+ on the same platform) and packages
  `qfe_core.hpp` and `magic_engine.hpp` alongside the C++ sources.
- **Zero-leftover directory locking (`C-8`):** `_acquire_dir_lock` locks the `_native` directory
  with `flock` instead of creating `.stimside_build_*.lock` files in `$TMPDIR`.
- **Source-checkout `.build_id` sidecars:** `resolve_so` and `_sync_extensions_to_source_dir`
  write `.lib<name>.build_id` sidecars in writable source checkouts only (excluded from sdists
  and wheels).

## Run 7 — Stage A″ (Cosetside Safety Closeout)

- **Bounded `active_shots` C++ export (`R6-F1`):** `_sync_cpp_to_py()` now calls the capacity-checked `engine_export_shot_v3` kernel and detects mid-export mutations via epoch and active-set checks (re-entry via GC callbacks, profile hooks, or proxies), raising `RuntimeError` instead of writing past exported buffers (remaining late-export/copy-loop window closed in Run 8 Stage B).
- **Failed-engine semantics & `clear()` ordering (`D-36` fallback, `R7-D5`):**
  - `clear()` now marks `_cpp_dirty = True` before calling `engine_clear` so an interrupted `clear()` cannot expose stale cached Python shots.
  - Caught C++ exceptions (`MemoryError` or C++ exception codes) permanently fail the native engine: subsequent state-dependent calls, getters, and `clear()` raise `RuntimeError` naming `MemoryError` / the C++ exception and advising creating a new simulator.
  - Pre-mutation rejections (wrong shape, type, or bounds) remain atomic; mid-call interrupts/exceptions after mutation begins did not latch in Run 7 (`L-ABORTED-CALL` documented limit; closed by `D-36` in Run 8 Stage B).
- **Atomic `do_conditional_pauli_noise` (`R6-F2`) and `interactive_do` (`R6-F7`):**
  - `do_conditional_pauli_noise` now converts `condition_met_mask` by truth value and validates its exact shape before advancing `_step_cursor` or drawing random numbers; previously accepted shapes (`(batch_size,)`, `(1, batch_size)`, `(rows, 1)`, scalars, `(rows+1, batch_size)`, `(0, batch_size)`) and string/bytes dtypes now raise `ValueError`/`TypeError`.
  - Integer masks for `E` / `CORRELATED_ERROR` / `ELSE_CORRELATED_ERROR` changed from bitwise `& 1` (e.g. `2` = not met) to truth value (`!= 0`, so `2` = met).
  - A rejected first `interactive_do` at `circuit_time == 0` restores `_used_interactively` and `_unrolled_ops` so `run()` can still succeed.
- **Operation caching & RNG ABI (`R7`):**
  - `run()` caches unrolled operations in `_run_ops` via `_ensure_run_ops` on first run (extended in Run 8 Stage B to rebuild when `circuit` changes).
  - `engine_set_rng_states` return type changed from `void` to `int` and copies caller RNG states into `shot_rng_states_owned` (switched to in-place zero-allocation copy in Run 8 Stage B).
- **Documented contract (`R6-F5`, `D-31` amended):** Added a `Contract:` section to `CosetsideSimulator` covering operand normalization, `active_shots` copy semantics, supported `condition_group_fn` re-entry, failed-engine behavior, `L-THREADS` (one simulator per thread), and `_use_cpp` usage; documented `order` on `do_unmatched_noisy_reset_z`.

## Run 6 (`8c7e33e`)

- **Single ownership and read-only `active_shots` copies (`D-32`):** In C++ mode, `BatchedMRQFEngine` is the sole owner of active shot states; `active_shots` returns a `MappingProxyType` of fresh `MRQFState` copies materialized at read time, so mutating returned objects does not alter simulator state.
- **Ordered Z reset C++ fix (`R6-F6`):** `do_unmatched_noisy_reset_z(..., pauli="Z", order=...)` now routes through `engine_project_pauli` in C++ mode so `order` (including partial and empty orders) is honored identically to the Python path.
- **Truth-value masks (`D-34`):** All mask operands (`bool`, numeric `!= 0`, and `object` via `bool()`) are converted once at entry to private C-ordered `np.bool_` arrays; unsupported dtypes (e.g. strings, structured dtypes) raise `TypeError`.
- **Stricter operand checks (`D-40`):**
  - `measurement_column` and `overwrite_measurement_column` require non-bool integer `index` (no `bool` or `slice`).
  - `overwrite_measurement_column` requires a 1-D `(batch_size,)` mask (no scalars or wrong lengths).
  - `apply_pauli` requires exact `(num_qubits, batch_size)` mask shape and non-bool, non-NaN real `p`.
  - `postselect_z_batch` requires non-bool `q` in `[0, num_qubits)`, exact `(batch_size,)` `flip_mask`, and integer/bool `desired_bits` (no floats).
  - `do_unmatched_noisy_reset_z` validates `order` eagerly (raising `IndexError` for out-of-range qubits before state changes).
- **C++ exception guard (`A-1`):** Native `coset_kernels` catches `std::bad_alloc` and other C++ exceptions across exports and raises `MemoryError` or `RuntimeError` via `_CosetKernelsLib`.
