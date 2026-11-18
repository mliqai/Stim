# WP-27 Step 0 Required-Mutant Suite (`tools/ir_mutants/`)

Self-contained mutation testing harness for the WP-27 circuit-IR safety net
(RFC §7.2, §7.3, §7.5; Plan v8 §2.4 `D-46` and §4 Stage C4).

This folder intentionally lives **outside `src/`** (under `glue/stimside/tools/`):
- **Not collected by `pytest`**: `pyproject.toml` sets `testpaths = ["src"]`.
- **Not shipped in sdist or wheel**: `MANIFEST.in` and `[tool.setuptools.packages.find]`
  include only `src/` and explicit root metadata files.

## Files

- `required_mutants.tsv` — **50** required non-equivalent mutants (**`0`** equivalent):
  - **`rev2_head` (9):** 9 non-equivalent HEAD-code mutants (`R1`–`R5`, `A10`, `C1b`,
    `A4_u2f_validate_off`, `A1`).
  - **`rev2_ir` (26):** IR core (`util/circuit_ir.py`) and tag memo (`op_handlers/ir_tags.py`)
    mutants (`A2`, `A3`, `A5`–`A9`, `A11`, `A12`, `B2`, `B5`, `B6a`–`B6c`, `B10`–`B14`,
    `T1`–`T3`), adapted verbatim to C1's typed/formatted source (`PATTERN-COUNT == 1`).
  - **`rev3_K1` (2):** `H01` (`BUILDER_KEY` drops `op.name`), `H02` (`VALUE_KEY` coarsened
    to `(name, tag)`).
  - **`rev3_K2` (5):** `G119`, `G120`, `G132`, `G137`, `G141` (`frozen=False` on all 5
    IR dataclasses).
  - **`rev3_K3` (3):** `G144`, `G145` (empty op-list `num_qubits`), `H05` (op-list
    `num_qubits` ignoring `QUBIT_COORDS`).
  - **`rev3_K4` (1):** `H04` (`MEASURE_RESET` hardcoded without `MRY`).
  - **`rev3_K5` (4):** `EO1`–`EO4` (`Builder.__init__` error-order mutants in
    `dem_generators/marginal_builder.py`).
- `run_required_mutants.py` — isolated-snapshot CLI runner implementing the mandatory
  **E6 runner-hygiene rule**.

## Usage

Run from any directory (prefer a scratch `--work` directory outside the repository):

```bash
python tools/ir_mutants/run_required_mutants.py \
  --pkg /path/to/Stim/glue/stimside \
  --work /tmp/stimside_scratch/ir_mutants_run \
  --jobs 4 \
  --cores 0-7 \
  --out /tmp/stimside_scratch/ir_mutants_run/results
```

Writes `<out>.tsv`, `<out>.md`, and `<out>.json`. All temporary mutant worktrees and
pytest basetemps are cleaned up automatically in `try/finally` (pass `--keep-work` to
inspect them).

### Optional Flags

- `--only ID1,ID2` — run a comma-separated subset of mutant IDs.
- `--python /path/to/python` — interpreter used to invoke `pytest` subprocesses.
- `--no-owner-tests` — skip `GOLD` and per-mutant owner test groups (`TAB`, `MAGIC`,
  `DEM`, `UTIL`) during mutant runs (baseline still runs all detectors per E6).
- `--include-e2e-circuits [DIR]` — optionally also compare the 8 external
  `e2e_tests/circuits/*.stim` files across all 21 `ir_oracle_test.WALKERS` against
  baseline digests (omitted by default so the runner has zero dependency on files
  outside `glue/stimside`).
- `--timeout SECONDS` — per-detector subprocess timeout (default `900`).

## E6 Runner-Hygiene Rule (RFC §7.3 item 6)

1. **Unique-match guard**: Every selected row MUST match `PATTERN-COUNT == 1` against
   the unmutated baseline snapshot, else the runner aborts immediately (`exit 2`).
2. **Baseline-first execution**: ALL 7 detectors run on the unmutated snapshot FIRST:
   `INV` (`util/circuit_ir_test.py`), `ORACLE` (`ir_oracle_test.py`), `GOLD`
   (`golden_test.py`), `TAB`, `MAGIC`, `DEM`, and `UTIL` (excluding `circuit_ir_test.py`
   so `UTIL` owner tests are not double-counted with `INV`). If ANY baseline detector
   fails or crashes, the runner aborts immediately (`exit 3`).
3. **Live-tree isolation**: The runner NEVER edits the live `PKG` tree. Every mutant
   runs inside a fresh scratch copy with `PYTHONDONTWRITEBYTECODE=1`,
   `-p no:cacheprovider`, and `--basetemp` inside `--work`.
4. **Transparent kill classification**:
   - **`KILLED(test)`**: pytest exited with `rc=1` and at least one assertion failure
     (`AssertionError` / `Failed` from `pytest.raises` or `pytest.fail`).
   - **`KILLED(crash)`**: non-1 nonzero exit (syntax/import/collection/internal crash,
     signal), or `rc=1` with failed tests caused solely by uncaught non-assertion exceptions.
   - **`ERROR`**: pytest fixture/setup error (`rc=1` with `n_error > 0` and `n_failed == 0`,
     NOT collection crashes like `rc=2` `SyntaxError`); fails runner with nonzero exit.
   - **`TIMEOUT`**: detector exceeded `--timeout` (causes `exit 1`).
   - **`SURVIVED(justified_equiv)`**: justified equivalent mutant (`expected=equivalent`)
     remained silent across all detectors. If an `expected=equivalent` control is killed
     (`equiv_killed > 0`), the runner exits nonzero (`exit 1`).
   - **`SURVIVED`**: unexpected non-equivalent survivor (causes `exit 1`).

## Live Disproof of `A4_u2f_validate_off` & Carry-Forward `R8-CF-U2F-ZEROTARGET`

RFC §7.2 previously treated `A4_u2f_validate_off` (`util/leakage_events.py:121`:
`if validate:` -> `if False:`) as justified-equivalent based on a 2,108-circuit fuzz
containing no zero-target fusing ops. Stage C Gate Round 1 refuted that claim empirically:
- Stim `1.16` fuses consecutive identical zero-target instructions in `flattened()` across
  **`73/80`** non-`REPEAT` gates (only `DETECTOR`, `E`, `ELSE_CORRELATED_ERROR`,
  `OBSERVABLE_INCLUDE`, `QUBIT_COORDS`, `SHIFT_COORDS`, and `TICK` do not fuse), even
  without `REPEAT` blocks (across `SHIFT_COORDS`, e.g., `W12`/`W16`).
- On `W12` (`H 0; SHIFT_COORDS(1); H; X 1`) and `W16`, untouched
  `unrolled_to_flattened_indices(c, validate=True)`, live
  `Builder(w16, True).build(oracle=True)`, and the public API
  `MarginalLeakageDemGenerator(loss_oracle=True).base_dem(w16)` all **raise `ValueError`**
  loudly, whereas mutant `A4_u2f_validate_off` (`validate=False`) **succeeds silently**
  with a corrupted `flat_index` map.
- Accordingly, `A4_u2f_validate_off` is now marked `expected=kill` and killed by
  `src/stimside/util/circuit_ir_test.py`.
- **Carry-forward (`R8-CF-U2F-ZEROTARGET`):** Per Plan v8 §2.4 (`D-46`), Stage C does NOT
  modify existing `src/stimside/util/leakage_events.py`. Its zero-target fusion limitation
  fails loudly today (`ValueError`, not silent corruption) and is carried forward to Run 9.
  Meanwhile, the new `src/stimside/util/circuit_ir.py` (`build_ir`) handles zero-target
  fusion accurately across all 80 Stim gates with full alignment validation.

## Target Score

**100% of listed non-equivalent mutants killed (`50/50` `KILLED(test)`)**, `0` equivalent
survivors, `0` unexpected survivors, `0` setup errors, `0` timeouts.
