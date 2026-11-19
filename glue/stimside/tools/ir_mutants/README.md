# Required-Mutant Suite (`tools/ir_mutants/`)

Self-contained mutation testing harness for the stimside circuit-IR and HEAD safety net.

This folder intentionally lives **outside `src/`** (under `glue/stimside/tools/`):
- **Not collected by `pytest`**: `pyproject.toml` sets `testpaths = ["src"]`.
- **Not shipped in sdist or wheel**: `MANIFEST.in` and `[tool.setuptools.packages.find]`
  include only `src/` and explicit root metadata files.

## Files

- `required_mutants.tsv` — **143** mutants (**141** required non-equivalent `kill` mutants and
  **2** justified `equivalent` controls):
  - **Pre-IR HEAD contracts (9):** 9 non-equivalent HEAD-code mutants (`R1`–`R5`, `A10`, `C1b`,
    `A4_u2f_validate_off`, `A1`).
  - **IR core & tag memo (37):**
    - 26 IR core (`util/circuit_ir.py`) and tag-key (`op_handlers/ir_tags.py`) mutants (`A2`,
      `A3`, `A5`–`A9`, `A11`, `A12`, `B2`, `B5`, `B6a`–`B6c`, `B10`–`B14`, `T1`–`T3`).
    - 2 key-policy mutants (`H01` `BUILDER_KEY` drops `op.name`, `H02` `VALUE_KEY` coarsened
      to `(name, tag)`).
    - 5 dataclass-immutability mutants (`G119`, `G120`, `G132`, `G137`, `G141`).
    - 3 op-list `num_qubits` mutants (`G144`, `G145`, `H05`).
    - 1 `MEASURE_RESET` classification mutant (`H04`).
  - **`Builder.__init__` error-order contracts (4):** `EO1`–`EO4` in
    `dem_generators/marginal_builder.py`.
  - **Simulator & oracle walkers (93):**
    - `util/known_states.py` (`KS01`–`KS30`, 29 `kill` + `KS21eq` `equivalent`).
    - `_flip/sync_reference.py` (`SR01`–`SR32`, 31 `kill` + `SR30eq` `equivalent`).
    - `_coset/reference_compile.py` (`RC01`–`RC31`, 31 `kill`).
- `run_required_mutants.py` — isolated-snapshot CLI runner implementing the mandatory
  baseline-first runner hygiene rule.

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
  `DEM`, `UTIL`, `FLIP`, `COSET`) during mutant runs (baseline still runs all detectors).
- `--include-e2e-circuits [DIR]` — optionally also compare the 8 external
  `e2e_tests/circuits/*.stim` files across all 21 `ir_oracle_test.WALKERS` against
  baseline digests (omitted by default so the runner has zero dependency on files
  outside `glue/stimside`).
- `--timeout SECONDS` — per-detector subprocess timeout (default `900`).

## Baseline-First Runner Hygiene Rule

1. **Unique-match guard**: Every selected row MUST match `PATTERN-COUNT == 1` against
   the unmutated baseline snapshot, else the runner aborts immediately (`exit 2`).
2. **Baseline-first execution**: ALL 9 detectors run on the unmutated snapshot FIRST:
   `INV` (`util/circuit_ir_test.py`), `ORACLE` (`ir_oracle_test.py`), `GOLD`
   (`golden_test.py`), `TAB`, `MAGIC`, `DEM`, `UTIL` (excluding `circuit_ir_test.py` so
   `UTIL` owner tests are not double-counted with `INV`), `FLIP`, and `COSET`. If ANY
   baseline detector fails or crashes, the runner aborts immediately (`exit 3`).
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

## Zero-Target Fusion & `A4_u2f_validate_off`

Stim `1.16` fuses consecutive identical zero-target instructions in `flattened()` across
non-`REPEAT` gates (only `DETECTOR`, `E`, `ELSE_CORRELATED_ERROR`, `OBSERVABLE_INCLUDE`,
`QUBIT_COORDS`, `SHIFT_COORDS`, and `TICK` do not fuse), even without `REPEAT` blocks
(for example across `SHIFT_COORDS`).
- `unrolled_to_flattened_indices` in `src/stimside/util/leakage_events.py` handles
  zero-target fusion explicitly by checking equality with the current flattened op so
  valid zero-target-fusing circuits succeed with accurate `flat_index` maps while genuine
  corrupted alignment still raises `ValueError` when `validate=True`.
- `A4_u2f_validate_off` (`if validate:` -> `if False:`) remains marked `expected=kill`
  and is killed by a corrupted-alignment witness in `src/stimside/util/circuit_ir_test.py`.
- `build_ir` in `src/stimside/util/circuit_ir.py` handles zero-target fusion across all
  Stim gates with full alignment validation.

## Target Score

**100% of listed non-equivalent mutants killed (`141/141` `KILLED(test)`)**, **`2/2`**
justified equivalent controls silent (`SURVIVED(justified_equiv)`), `0` equivalent
controls killed, `0` unexpected survivors, `0` setup errors, `0` timeouts.
