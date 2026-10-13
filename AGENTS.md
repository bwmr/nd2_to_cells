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
  track.py         # run_track(): SuperSegger-derived linking → cell*.h5 files
  link.py          #   driver, errorRez, cell marker (trackOptiLinkCellMulti etc.)
  assign.py        #   frame-to-frame assignment (multiAssignmentSparse)
  mask_edit.py     #   merges and splits (merge2Regions, missingSeg2to1)
tests/             # pytest: merge events (linking report issues A and B)
presets/
  100XEc.toml      # E. coli defaults (default preset)
  100XPa.toml      # P. aeruginosa defaults
```

No CI. No Makefile. `tests/` holds pytest tests for the merge issues of the 260429 linking report (see `docs/linking.md`).

## Install

```bash
pip install -e .          # runtime only
pip install -e ".[dev]"   # + ruff, pytest
```

Requires Python ≥ 3.11 (uses stdlib `tomllib`).

## Lint and format (only dev tools)

```bash
ruff check nd2_to_cells/ tests/   # E, F, I rules; line-length=88
ruff format nd2_to_cells/ tests/
```

## Tests

```bash
pytest                                          # about 2.5 min; no data needed
```

- The tests import `nd2_to_cells` from this checkout (`pythonpath` in `pyproject.toml`), and `tests/conftest.py` refuses to run against another copy, such as an editable install of the main checkout.
- A linking change must keep all 10 real-data cases of `tests/test_report_cases.py` resolved (mask crops in `tests/data/`). Count merge events on whole positions with `python tests/merge_events.py <cell dir> ...` and compare with `docs/linking.md`.

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

- **Memory design**: `align.py` holds at most 2 frames; `assemble.py` holds one timepoint's Z-stack. `track.py` holds 5 label frames while linking (previous, current, next, and `division_lookahead` − 1 = 2 look-ahead frames), plus scalar region tables for all frames and compressed copies of edited masks; the writing step keeps crop buffers for all tracks until the end, so memory grows with movie length × cell count. On 260429 that is 1–2.5 GB per position. A track that jumps between distant cells has a crop close to the full frame: SuperSegger's label collision, which `mask_edit.py` avoids, took this to 7–10 GB. Do not load full stacks.
- **Linking**: `link.py`/`assign.py`/`mask_edit.py` follow SuperSegger, with seven changes marked "Not SuperSegger"; SuperSegger behaviour kept although it looks wrong is marked "MATLAB quirk". Labels are 1-based: per-region arrays have an unused index 0.
- **Parallelism**: `ProcessPoolExecutor` per xy position. Worker entry points are module-level picklable wrappers (`_align_position_wrapper`, `_assemble_position_wrapper`, `_track_position_wrapper`).
- **Preset resolution**: `load_preset()` first tries the argument as a path; if not found, looks in `presets/{name}.toml`. Custom presets can be passed as a file path.
- **Cell naming**: `cell{ID:07d}.h5` (lowercase) = partial observation; `Cell{ID:07d}.h5` (uppercase) = complete cell cycle (SuperSegger stat0 = 2: born from a division, divides, no linking error, spans ≥ `min_cycle_frames` frames).
- **Filename convention**: TIFFs are `{basename}_t{T}xy{P}c{C}.tif`; Z-slice TIFFs (written by `export --export-z-slices`, read by `assemble`) are `{basename}_t{T}xy{P}z{Z}c{C}.tif`; masks follow Omnipose pattern `*cp_masks.png`.
- **Output replacement**: `align`/`assemble` replace `xy*/phase/` and `xy*/fluor*/`; `track` replaces its own `*.h5` output in `cell/`; `masks/` is never modified.

## Linking: design decisions

Details, the evidence for each change and the tests are in `docs/linking.md`.

- **Scope**: only `trackOpti` steps that decide lineages are ported: linking (step 4), cell marker (step 5), and the lineage fields of the cell files (step 10). Fluorescence, `CellA` geometry, foci and `clist` (steps 6–9) are not computed or saved, because downstream reads only per-cell masks, birth, death, motherID, daughterID, sisterID and divide.
- **Not SuperSegger-faithful**: the code started as a bug-for-bug port, checked against SuperSegger's output on 260429, and now carries seven changes that cut merged cells by two thirds (`docs/linking.md`). No SuperSegger-equivalent mode is kept; that port, a switch per change and the parity scripts are in commit a40b03e. The assignment (`assign.py`) is still SuperSegger's, quirks included.
- **Reference run**: `superSeggerGui` on Windows with preset 100XEc, Minimum Cell Age 3, Remove Stray off and `CONST.ignoreerror` unset (3 passes per frame). The presets hold these GUI-effective values, plus `division_lookahead`. Preset keys use descriptive names (`fragment_merge_area`, `drop_single_frame_strays`, `min_cycle_frames`); the SuperSegger-style names still load with a deprecation warning.
- **Splits** (2 → 1): all previous cells are moved onto the fused region (centroid, then the best rotation within ±10°) and seed a watershed of its distance transform. SuperSegger's affine registration is not used: it rarely cuts the region, and is chaotic (MATLAB with 10 threads does not reproduce MATLAB with 1 thread).
- **Behaviour MATLAB does not have**:
  - After 20 passes on one frame (`SAFETY_MAX_PASSES`), one last pass runs without mask edits, and a warning is printed. MATLAB can loop forever there.
  - Where MATLAB would error, the code raises `RuntimeError`.
- **Masks and output**:
  - Edited masks are kept compressed in memory. No `masksOS/` is written.
  - Labels without pixels (ghosts) get no cell file, and `death` is the cell's last frame.
  - `birth`/`death` are frame numbers rather than file indices; the two only differ when a mask frame is missing.
  - Progress is printed every 50 frames, because a 361-frame position takes 10–35 min.

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
