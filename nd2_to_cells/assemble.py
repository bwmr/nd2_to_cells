"""Assemble per-Z-slice TIFFs (from export --export-z-slices) into ZYX stacks.

Reads per-slice TIFFs from raw_im/ that were produced by
`nd2_to_cells export --export-z-slices`, groups them by position ×
timepoint × channel, and writes one ZYX TIFF per group into the
appropriate channel subdirectory under xy{P}/.

This is the Z-stack equivalent of the align step and produces output in
the same locations (xy{P}/phase/, xy{P}/fluor1/, …). No drift correction is
applied, and segmentation/tracking of Z-stack data is not supported.

Input filename pattern (from export --export-z-slices):
    {basename}_t{T}xy{P}z{Z}c{C}.tif

Output filename pattern (one ZYX TIFF per timepoint × channel):
    xy{P}/{channel}/{basename}_t{T}xy{P}c{C}.tif

where channel = 'phase' for c=1, 'fluor1' for c=2, 'fluor2' for c=3, …

Existing xy{P}/phase/ and xy{P}/fluor*/ are removed before writing.

Memory: one timepoint's Z-stack (Z frames of Y×X) held at a time.
"""

import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from tqdm import tqdm

from .align import clear_channel_dirs, resolve_basename

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SLICE_RE = re.compile(
    r"^(?P<base>.+)_t(?P<t>\d+)xy(?P<p>\d+)z(?P<z>\d+)c(?P<c>\d+)\.tif$"
)


def _channel_subdir(c_suffix: int) -> str:
    """Map c-suffix (1-based) to output subdirectory name."""
    if c_suffix == 1:
        return "phase"
    return f"fluor{c_suffix - 1}"


def _assemble_position(
    xy_str: str,
    data_dir: Path,
    basename: str,
    slices_by_tc: dict,
) -> None:
    """Assemble all timepoints for one xy position.

    slices_by_tc: {(t_str, c_suffix): [(z_int, path), ...]}
    """
    xy_dir = data_dir / f"xy{xy_str}"
    # Clear previous output first so stale frames or channels don't survive
    if xy_dir.exists():
        clear_channel_dirs(xy_dir)

    # Collect unique (t_str, c_suffix) keys and sort for determinism.
    for (t_str, c_suffix), z_entries in sorted(slices_by_tc.items()):
        subdir = _channel_subdir(c_suffix)
        out_dir = xy_dir / subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Sort slices by Z index and load.
        z_entries_sorted = sorted(z_entries, key=lambda x: x[0])
        slices = [iio.imread(path) for _, path in z_entries_sorted]
        stack = np.stack(slices, axis=0)  # (Z, Y, X)

        fname = f"{basename}_t{t_str}xy{xy_str}c{c_suffix}.tif"
        iio.imwrite(out_dir / fname, stack)


def _assemble_position_wrapper(args: tuple) -> None:
    """Picklable top-level wrapper for ProcessPoolExecutor."""
    _assemble_position(*args)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_assemble(
    data_dir: str,
    basename: str | None = None,
    workers: int = 1,
) -> None:
    """Assemble per-Z-slice TIFFs into per-timepoint ZYX stacks.

    Args:
        data_dir: Root experiment directory containing raw_im/ and xy*/.
        basename: Filename prefix used during export (e.g. '260430'). If None,
                  raw_im/ must contain Z-slice TIFFs of exactly one basename.
        workers:  Number of parallel worker processes (one per xy position).
    """
    data_dir = Path(data_dir)
    raw_im_dir = data_dir / "raw_im"

    if not raw_im_dir.exists():
        raise FileNotFoundError(f"raw_im/ not found in {data_dir}")

    # Discover all per-slice TIFFs, then restrict to one basename.
    matches = [
        (path, m)
        for path in sorted(raw_im_dir.iterdir())
        if (m := _SLICE_RE.match(path.name))
    ]
    if not matches:
        raise FileNotFoundError(
            f"No z-slice TIFFs matching '*_t*xy*z*c*.tif' "
            f"found in {raw_im_dir}. "
            "Run 'nd2_to_cells export --export-z-slices' first."
        )
    basename = resolve_basename({m["base"] for _, m in matches}, basename, raw_im_dir)
    matches = [(path, m) for path, m in matches if m["base"] == basename]
    slice_files = [path for path, _ in matches]

    # Group files by xy position string, then by (t_str, c_suffix).
    # per_pos[xy_str][(t_str, c_suffix)] = [(z_int, path), ...]
    per_pos: dict[str, dict] = defaultdict(lambda: defaultdict(list))

    for path, m in matches:
        t_str = m.group("t")
        p_str = m.group("p")
        z_int = int(m.group("z"))
        c_suffix = int(m.group("c"))
        per_pos[p_str][(t_str, c_suffix)].append((z_int, path))

    positions = sorted(per_pos.keys())
    print(
        f"Found {len(slice_files)} slice TIFF(s) across {len(positions)} position(s)."
    )

    tasks = [
        (xy_str, data_dir, basename, dict(per_pos[xy_str])) for xy_str in positions
    ]

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(
                tqdm(
                    pool.map(_assemble_position_wrapper, tasks),
                    total=len(tasks),
                    desc="Assembling",
                    unit="pos",
                )
            )
    else:
        for task in tqdm(tasks, desc="Assembling", unit="pos"):
            _assemble_position_wrapper(task)

    print(f"\nAssemble complete → {data_dir}")
