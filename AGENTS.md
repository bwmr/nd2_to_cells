# nd2_to_cells — Agent Instructions

## What this repo does

Python CLI pipeline: Nikon ND2 microscopy files → per-cell HDF5 files for timelapse analysis. Drop-in replacement for MATLAB SuperSegger (export, align, track steps). Omnipose segmentation runs separately between `align` and `track` (external conda environment, not managed here).

## Package layout

```
nd2_to_cells/      # Python package
  cli.py           # Click entrypoint — thin wrappers only
  export.py        # run_export(): ND2 → per-position TIFFs
  align.py         # run_align(): phase-cross-correlation drift correction
  assemble.py      # run_assemble(): Z-slice TIFFs → per-timepoint ZYX stacks (alternative to align)
  track.py         # run_track(): SuperSegger linking port → cell*.h5 files
  link.py          #   driver, errorRez, cell marker (trackOptiLinkCellMulti etc.)
  assign.py        #   frame-to-frame assignment (multiAssignmentSparse)
  mask_edit.py     #   merges and splits (merge2Regions, missingSeg2to1)
  imreg.py         #   affine mask registration (MATLAB imregtform / ITK v3 path)
dev/               # parity scripts against SuperSegger output (not packaged)
tests/             # pytest: merge events (linking report issues A and B)
presets/
  100XEc.toml      # E. coli defaults (default preset)
  100XPa.toml      # P. aeruginosa defaults
```

No CI. No Makefile. `dev/` holds parity scripts that compare `track` with SuperSegger output (see `docs/supersegger_port.md`). `tests/` holds pytest tests for the merge issues of the 260429 linking report (see `docs/linking_report_tests.md`).

## Install

```bash
pip install -e .          # runtime only
pip install -e ".[dev]"   # + ruff, pytest
```

Requires Python ≥ 3.11 (uses stdlib `tomllib`).

## Lint and format (only dev tools)

```bash
ruff check nd2_to_cells/ tests/ dev/   # E, F, I rules; line-length=88
ruff format nd2_to_cells/ tests/ dev/
```

## Tests

```bash
pytest                                          # fast tests (seconds)
ND2C_TEST_DATA=<260429 folder> pytest -k "baseline or combined"   # + real data, ~7 min per config
```

- The tests import `nd2_to_cells` from this checkout (`pythonpath` in `pyproject.toml`), and `tests/conftest.py` refuses to run against another copy, such as an editable install of the main checkout.
- A linking change made for the report is added as a configuration in `CONFIGS` (`tests/conftest.py`). The cases it resolves are listed in `RESOLVED` (`tests/test_report_cases.py`). The baseline must keep passing `test_baseline_event`.

## Pipeline commands (must run in order)

```bash
nd2_to_cells export --input experiment.nd2 --output /data/exp/ --basename 260430
nd2_to_cells align  --data /data/exp/ --workers 4
# <run Omnipose externally to produce masks/ PNGs>
nd2_to_cells track  --data /data/exp/ --preset 100XEc --workers 4
```

Z-stack data: `export --export-z-slices` → `assemble` (instead of `align`). No segmentation or tracking for Z-stack (3D+T) data.
`--basename` is optional for `align` and `assemble` when `raw_im/` holds a single export.

## Output layout

```
/data/exp/
  raw_im/        exported TIFFs (written by export; input to align/assemble)
  xy01/
    phase/       aligned phase TIFFs
    fluor1/      aligned fluor channel 1
    masks/       Omnipose PNG masks (external step)
    cell/        per-cell HDF5 files (written by track)
  xy02/ ...
```

## Architecture notes

- **Memory design**: `align.py` holds at most 2 frames; `assemble.py` holds one timepoint's Z-stack. `track.py` holds 3 label frames (previous, current, next) while linking, plus scalar region tables for all frames and compressed copies of edited masks; the writing step keeps crop buffers for all tracks until the end, so memory grows with movie length × cell count. With the label-collision quirk (the baseline), some tracks span distant cells and their crops approach the full frame: up to 10 GB per position on 260429, against 1–2 GB with the `label_collision` fix. Do not load full stacks.
- **Linking port**: `link.py`/`assign.py`/`mask_edit.py`/`imreg.py` reproduce SuperSegger bug-for-bug (MATLAB quirks are kept and marked "MATLAB quirk"). Labels are 1-based: per-region arrays have an unused index 0. Check changes with the `dev/` parity scripts.
- **Parallelism**: `ProcessPoolExecutor` per xy position. Worker entry points are module-level picklable wrappers (`_align_position_wrapper`, `_assemble_position_wrapper`, `_track_position_wrapper`).
- **Preset resolution**: `load_preset()` first tries the argument as a path; if not found, looks in `presets/{name}.toml`. Custom presets can be passed as a file path.
- **Cell naming**: `cell{ID:07d}.h5` (lowercase) = partial observation; `Cell{ID:07d}.h5` (uppercase) = complete cell cycle (SuperSegger stat0 = 2: born from a division, divides, no linking error, divides ≥ `min_cell_age` frames after birth).
- **Filename convention**: TIFFs are `{basename}_t{T}xy{P}c{C}.tif`; Z-slice TIFFs (written by `export --export-z-slices`, read by `assemble`) are `{basename}_t{T}xy{P}z{Z}c{C}.tif`; masks follow Omnipose pattern `*cp_masks.png`.
- **Output replacement**: `align`/`assemble` replace `xy*/phase/` and `xy*/fluor*/`; `track` replaces its own `*.h5` output in `cell/`; `masks/` is never modified.

## Linking port: design decisions

Details and parity numbers are in `docs/supersegger_port.md`.

- **Scope**: only `trackOpti` steps that decide lineages are ported: linking (step 4), cell marker (step 5), and the lineage fields of the cell files (step 10). Fluorescence, `CellA` geometry, foci and `clist` (steps 6–9) are not computed or saved, because downstream reads only per-cell masks, birth, death, motherID, daughterID, sisterID and divide.
- **Bug-for-bug**: MATLAB quirks are kept on purpose and marked "MATLAB quirk". Examples: ghost (zero-area) labels, the label-collision quirk in merges, stale `death` (last frame + 1), the CellMarker stale index. Do not "fix" them without asking. Where MATLAB would error, the port raises `RuntimeError`. The equivalent port is the baseline: quirk fixes and the ITK v4 registration upgrade come later, each tested as a separate divergence against it.
- **Opt-in fixes**: each departure from MATLAB is named in `track.FIXES` and switched on with the preset's `fixes` list. Numeric departures, such as `division_lookahead`, are `TrackingParams` keys whose default is SuperSegger's value. With an empty list and the defaults, the code paths are the baseline's. `docs/matlab_quirks.md` lists the quirks, the fixes and their results.
- **Reference run**: `superSeggerGui` on Windows with preset 100XEc, Minimum Cell Age 3, Remove Stray off and `CONST.ignoreerror` unset (3 passes per frame). The presets hold these GUI-effective values, not the raw SuperSegger constants files.
- **Registration**: `imreg.py` re-implements MATLAB's `imregtform`, which runs ITK's v3 framework. SimpleITK (ITK v4) is used only with the opt-in `itk_v4` fix; it is an optional dependency (`pip install -e ".[itk]"`). The reference is single-threaded `imregtform`, because Windows runs it on one thread; on macOS, use `maxNumCompThreads(1)` in MATLAB oracles.
- **Splits are chaotic**: on the coarse pyramid levels, 1e-7 differences change the final transform. MATLAB with 10 threads does not reproduce MATLAB with 1 thread either. Judge split parity by outcome agreement rate, not exact masks. Everything except splits matches the err files exactly.
- **Windows instead of full frames**: registration and the split watershed run on windows around the cells. Full-frame coordinates, centre, optimizer scales and pixel count are kept, so results do not depend on frame size.
- **Behaviour MATLAB does not have**:
  - After 20 passes on one frame (`SAFETY_MAX_PASSES`), one last pass runs without mask edits, and a warning is printed. MATLAB can loop forever there.
  - When a watershed plateau is at least 100 000 px (`PLATEAU_FAST`), only the cells plus a 2-pixel ring are flooded; the rest is filled from the nearest label.
- **Masks and output**:
  - Edited masks are kept compressed in memory. No `masksOS/` is written.
  - Ghost cells are written with empty masks, as MATLAB does.
  - `birth`/`death` are frame numbers rather than file indices; the two only differ when a mask frame is missing.
  - Progress is printed every 50 frames, because a 361-frame position takes 30–60 min.

## Behavioral guidelines

Tradeoff: These guidelines bias toward caution over speed. For trivial tasks, use judgment.
1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

    State your assumptions explicitly. If uncertain, ask.
    If multiple interpretations exist, present them - don't pick silently.
    If a simpler approach exists, say so. Push back when warranted.
    If something is unclear, stop. Name what's confusing. Ask.

2. Simplicity First

Minimum code that solves the problem. Nothing speculative.

    No features beyond what was asked.
    No abstractions for single-use code.
    No "flexibility" or "configurability" that wasn't requested.
    No error handling for impossible scenarios.
    If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.
3. Surgical Changes

Touch only what you must. Clean up only your own mess.

When editing existing code:

    Don't "improve" adjacent code, comments, or formatting.
    Don't refactor things that aren't broken.
    Match existing style, even if you'd do it differently.
    If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

    Remove imports/variables/functions that YOUR changes made unused.
    Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.
4. Goal-Driven Execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

    "Add validation" → "Write tests for invalid inputs, then make them pass"
    "Fix the bug" → "Write a test that reproduces it, then make it pass"
    "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
