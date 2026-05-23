"""Frame drift correction for all xy positions.

Algorithm:
    1. Load phase images in frame order and compute frame-to-frame shifts
       via phase_cross_correlation (upsample_factor=100, matching
       SuperSegger's precision=100).
    2. Clamp each frame-to-frame shift to max_shift_px to reject spurious
       large shifts caused by blurry or artifact frames.
    3. Accumulate clamped shifts to get the correction for each frame
       relative to frame 0.
    4. Compute the minimum padding canvas that fits all shifted frames.
    5. Shift every channel of every frame into the padded canvas and
       write aligned TIFFs back in place.

Memory: phase images for shift computation are loaded one pair at a time.
        Per-channel alignment loads one frame at a time.

Frame of reference: frame 0. All frames are brought to the coordinate
system of frame 0.
"""

import math
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from skimage.registration import phase_cross_correlation
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_frame_number(fname: str) -> int:
    """Extract the frame number from a TIFF filename."""
    m = re.search(r"_t(\d+)xy", fname)
    if m is None:
        raise ValueError(f"Cannot parse frame number from filename: {fname!r}")
    return int(m.group(1))


def _sorted_tifs(channel_dir: Path) -> list[Path]:
    """Return TIFF paths in a channel directory sorted by frame number."""
    return sorted(
        [f for f in channel_dir.iterdir()
         if f.suffix.lower() in (".tif", ".tiff")],
        key=lambda f: _parse_frame_number(f.name),
    )


def _shift_image(img: np.ndarray, dy: float, dx: float,
                 canvas_shape: tuple[int, int],
                 row_offset: int, col_offset: int,
                 fill_value: float) -> np.ndarray:
    """Place img, shifted by (dy, dx) corrections, into a padded canvas.

    The correction shifts are the cumulative values returned by
    phase_cross_correlation accumulated relative to frame 0.
    A positive dy means the frame drifted down relative to frame 0
    (correction shifts it back up), and vice versa.

    canvas[row_offset + r + dy, col_offset + c + dx] = img[r, c]
    for all valid (r, c).

    row_offset / col_offset are chosen so that all frames fit in the canvas.
    """
    canvas = np.full(canvas_shape, fill_value, dtype=img.dtype)

    dy_i = int(round(dy))
    dx_i = int(round(dx))
    H, W = img.shape[:2]

    # Source pixel (r, c) maps to canvas (row_offset + r + dy_i, col_offset + c + dx_i).
    # Valid source range: canvas destination must be in [0, canvas_shape).
    dst_r0 = row_offset + dy_i          # canvas row for src row 0
    dst_c0 = col_offset + dx_i          # canvas col for src col 0

    # Clamp to canvas bounds
    cr0 = max(0, dst_r0);  cr1 = min(canvas_shape[0], dst_r0 + H)
    cc0 = max(0, dst_c0);  cc1 = min(canvas_shape[1], dst_c0 + W)

    if cr1 <= cr0 or cc1 <= cc0:
        return canvas  # frame entirely outside canvas (shouldn't happen)

    # Corresponding source region
    sr0 = cr0 - dst_r0;  sr1 = sr0 + (cr1 - cr0)
    sc0 = cc0 - dst_c0;  sc1 = sc0 + (cc1 - cc0)

    canvas[cr0:cr1, cc0:cc1] = img[sr0:sr1, sc0:sc1]
    return canvas


# ---------------------------------------------------------------------------
# Per-position alignment
# ---------------------------------------------------------------------------

def _align_position(
    xy_dir: Path,
    align_channel: str,
    max_shift_px: float = 50.0,
) -> None:
    """Align all frames for one xy position.

    Args:
        xy_dir:        Path to xy*/ directory.
        align_channel: Name of the subdirectory used to compute shifts
                       (typically 'phase').
        max_shift_px:  Frame-to-frame shifts larger than this (in pixels)
                       are clamped to 0. Guards against spurious large shifts
                       from blurry or artifact frames.
    """
    align_dir = xy_dir / align_channel
    if not align_dir.exists():
        print(f"  [skip] {xy_dir.name}: no '{align_channel}/' directory")
        return

    tif_paths = _sorted_tifs(align_dir)
    n_frames = len(tif_paths)
    if n_frames < 2:
        print(f"  [skip] {xy_dir.name}: fewer than 2 frames")
        return

    # --- Step 1: compute frame-to-frame shifts, loading one pair at a time ---
    raw_shifts = np.zeros((n_frames, 2))
    prev = iio.imread(tif_paths[0]).astype(float)
    img_shape = prev.shape[:2]
    n_clamped = 0

    for i in range(1, n_frames):
        cur = iio.imread(tif_paths[i]).astype(float)
        shift, _, _ = phase_cross_correlation(prev, cur, upsample_factor=100)

        # Reject spurious large shifts (e.g. from blurry/artifact frames)
        if np.sqrt(shift[0]**2 + shift[1]**2) > max_shift_px:
            print(f"  {xy_dir.name} t{i}: shift {shift} exceeds {max_shift_px}px — clamped to 0")
            shift = np.array([0.0, 0.0])
            n_clamped += 1

        raw_shifts[i] = shift
        prev = cur  # advance: only two frames in memory at a time

    if n_clamped:
        print(f"  {xy_dir.name}: {n_clamped}/{n_frames-1} shifts clamped")

    # --- Step 2: accumulate to get corrections relative to frame 0 ---
    cum_shifts = np.cumsum(raw_shifts, axis=0)  # shape (n_frames, 2)

    row_shifts = cum_shifts[:, 0]
    col_shifts = cum_shifts[:, 1]

    # --- Step 3: compute padded canvas ---
    # frame i maps to canvas position (row_offset + cum_r[i], col_offset + cum_c[i])
    # row_offset = max(0, -min(cum_r)) ensures no negative canvas rows
    row_offset = int(math.ceil(max(0.0, -row_shifts.min())))
    col_offset = int(math.ceil(max(0.0, -col_shifts.min())))
    canvas_h = img_shape[0] + row_offset + int(math.ceil(max(0.0, row_shifts.max())))
    canvas_w = img_shape[1] + col_offset + int(math.ceil(max(0.0, col_shifts.max())))
    canvas_shape = (canvas_h, canvas_w)

    # --- Step 4: apply shifts to every channel, one frame at a time ---
    channel_dirs = sorted(
        [d for d in xy_dir.iterdir()
         if d.is_dir() and d.name not in ("masks", "cell", "seg", "cp_output")],
        key=lambda d: d.name,
    )

    for ch_dir in channel_dirs:
        ch_tifs = _sorted_tifs(ch_dir)
        if not ch_tifs:
            continue
        if len(ch_tifs) != n_frames:
            print(f"  {xy_dir.name}/{ch_dir.name}: "
                  f"{len(ch_tifs)} frames != {n_frames} (phase), skipping")
            continue

        for i, tif_path in enumerate(ch_tifs):
            img = iio.imread(tif_path)
            fill = float(np.mean(img))
            aligned = _shift_image(
                img,
                dy=cum_shifts[i, 0],
                dx=cum_shifts[i, 1],
                canvas_shape=canvas_shape,
                row_offset=row_offset,
                col_offset=col_offset,
                fill_value=fill,
            )
            iio.imwrite(tif_path, aligned.astype(img.dtype))

    print(f"  {xy_dir.name}: aligned {n_frames} frames "
          f"(canvas {canvas_h}x{canvas_w}, "
          f"drift row=[{row_shifts.min():.1f},{row_shifts.max():.1f}] "
          f"col=[{col_shifts.min():.1f},{col_shifts.max():.1f}])")


def _align_position_wrapper(args):
    """Top-level wrapper for ProcessPoolExecutor (must be picklable)."""
    xy_dir, align_channel, max_shift_px = args
    try:
        _align_position(Path(xy_dir), align_channel, max_shift_px)
    except Exception as exc:
        import traceback
        print(f"ERROR aligning {xy_dir}: {exc}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_align(
    data_dir: str,
    align_channel: str = "phase",
    workers: int = 1,
    max_shift_px: float = 50.0,
) -> None:
    """Drift-correct all xy positions in data_dir.

    Args:
        data_dir:      Directory containing xy*/ subdirectories.
        align_channel: Channel subdirectory used to compute shifts.
        workers:       Number of parallel worker processes.
        max_shift_px:  Frame-to-frame shifts larger than this are clamped
                       to 0 (outlier rejection for blurry/artifact frames).
    """
    data_dir = Path(data_dir)
    xy_dirs = sorted(
        [d for d in data_dir.iterdir()
         if d.is_dir() and re.match(r"xy\d+$", d.name)],
        key=lambda d: int(d.name[2:]),
    )

    if not xy_dirs:
        print(f"No xy*/ directories found in {data_dir}")
        return

    print(
        f"Aligning {len(xy_dirs)} position(s) using channel '{align_channel}' "
        f"(max_shift_px={max_shift_px})..."
    )

    args = [(str(d), align_channel, max_shift_px) for d in xy_dirs]

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(tqdm(
                pool.map(_align_position_wrapper, args),
                total=len(args),
                desc="Aligning positions",
                unit="pos",
            ))
    else:
        for xy_dir in tqdm(xy_dirs, desc="Aligning positions", unit="pos"):
            _align_position(xy_dir, align_channel, max_shift_px)

    print("Alignment complete.")
