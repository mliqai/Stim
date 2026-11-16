# Changelog

All notable user-visible changes to `stimside` are documented in this file.

## Run 7 — Stage A″ (Cosetside Safety Closeout)

- **Bounded `active_shots` C++ export (`R6-F1`):** `_sync_cpp_to_py()` now calls the capacity-checked `engine_export_shot_v3` kernel and detects mid-read mutations (re-entry via GC callbacks, profile hooks, or proxies), raising `RuntimeError` instead of writing past exported buffers.
- **Failed-engine semantics & `clear()` ordering (`D-36` fallback, `R7-D5`):**
  - `clear()` now marks `_cpp_dirty = True` before calling `engine_clear` so an interrupted `clear()` cannot expose stale cached Python shots.
  - Caught C++ exceptions (`MemoryError` or C++ exception codes) permanently fail the native engine: subsequent state-dependent calls, getters, and `clear()` raise `RuntimeError` naming `MemoryError` / the C++ exception and advising creating a new simulator.
  - Pre-mutation rejections (wrong shape, type, or bounds) remain atomic; mid-call interrupts/exceptions after mutation begins do not latch (`L-ABORTED-CALL` documented limit; `clear()` resets).
- **Atomic `do_conditional_pauli_noise` (`R6-F2`) and `interactive_do` (`R6-F7`):**
  - `do_conditional_pauli_noise` now converts `condition_met_mask` by truth value and validates its exact shape before advancing `_step_cursor` or drawing random numbers.
  - A rejected first `interactive_do` at `circuit_time == 0` restores `_used_interactively` and `_unrolled_ops` so `run()` can still succeed.
- **Documented contract (`R6-F5`, `D-31` amended):** Added a `Contract:` section to `CosetsideSimulator` covering operand normalization, `active_shots` copy semantics, supported `condition_group_fn` re-entry, failed-engine / `L-ABORTED-CALL` behavior, `L-THREADS` (one simulator per thread), and `_use_cpp` usage; documented `order` on `do_unmatched_noisy_reset_z`.

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
