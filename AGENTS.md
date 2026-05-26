# nd2_to_cells — Agent Instructions

## What this repo does

Python CLI pipeline: Nikon ND2 microscopy files → per-cell HDF5 files for timelapse analysis. Drop-in replacement for MATLAB SuperSegger (export, align, track steps). Omnipose segmentation runs separately between `align` and `track` (external conda environment, not managed here).

## Package layout

```
nd2_to_cells/      # Python package
  cli.py           # Click entrypoint — thin wrappers only
  export.py        # run_export(): ND2 → per-position TIFFs
  align.py         # run_align(): phase-cross-correlation drift correction
  track.py         # run_track(): IoU linker → cell*.h5 files
presets/
  100XEc.toml      # E. coli defaults (default preset)
  100XPa.toml      # P. aeruginosa defaults
```

No tests directory. No CI. No Makefile.

## Install

```bash
pip install -e .          # runtime only
pip install -e ".[dev]"   # + ruff
```

Requires Python ≥ 3.11 (uses stdlib `tomllib`).

## Lint and format (only dev tools)

```bash
ruff check nd2_to_cells/   # E, F, I rules; line-length=88
ruff format nd2_to_cells/
```

## Pipeline commands (must run in order)

```bash
nd2_to_cells export --input experiment.nd2 --output /data/exp/ --basename 260430
nd2_to_cells align  --data /data/exp/ --workers 4
# <run Omnipose externally to produce masks/ PNGs>
nd2_to_cells track  --data /data/exp/ --preset 100XEc --workers 4
```

## Output layout

```
/data/exp/
  raw_im/        pre-alignment TIFF copies (written by export)
  xy01/
    phase/       aligned phase TIFFs
    fluor1/      aligned fluor channel 1
    masks/       Omnipose PNG masks (external step)
    cell/        per-cell HDF5 files (written by track)
  xy02/ ...
```

## Architecture notes

- **Memory design**: `align.py` and `track.py` hold at most 2 frames in RAM at a time. Do not load full stacks.
- **`track.py` Region dataclass**: stores only scalars (label, area, centroid, bbox) — never mask arrays. IoU computed on-the-fly over bbox overlap only.
- **Parallelism**: `ProcessPoolExecutor` per xy position. Worker entry points are module-level picklable wrappers (`_align_position_wrapper`, `_track_position_wrapper`).
- **Preset resolution**: `load_preset()` first tries the argument as a path; if not found, looks in `presets/{name}.toml`. Custom presets can be passed as a file path.
- **Cell naming**: `cell{ID:07d}.h5` (lowercase) = partial observation; `Cell{ID:07d}.h5` (uppercase) = complete cell cycle (birth + division observed, length ≥ `min_cell_age`).
- **Filename convention**: TIFFs are `{basename}_t{T}xy{P}c{C}.tif`; masks follow Omnipose pattern `*cp_masks.png`.
- **`pandas` is a declared dependency but not currently imported** in any module — do not add a pandas import without a clear reason.

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
