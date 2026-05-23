"""Frame drift correction for all xy positions.

Two registration modes (controlled by align_to_first):

  align_to_first=True (default):
    Every frame is registered directly against frame 0. The shift for
    each frame is measured independently, so subpixel noise does not
    compound across frames. This is the most robust mode for long movies
    with slow, monotonic drift — it matches SuperSegger's AlignToFirst
    option. Recommended when drift is small relative to the frame interval.

  align_to_first=False (sequential):
    Each frame is registered against the previous frame (frame-to-frame).
    Useful when drift between consecutive frames is large (fast drift or
    slow frame rate) and a direct frame-0 comparison would be unreliable.
    Equivalent to SuperSegger's default (AlignToFirst=false).
    Clamping (max_shift_px) applies per step to reject outlier frames.

Algorithm:
    1. Load the reference and one target frame at a time.
    2. Compute shift via phase_cross_correlation (upsample_factor=100).
    3. Clamp shifts exceeding max_shift_px to 0 (outlier rejection).
    4. Compute the padded canvas size from all shifts.
    5. Apply the same shift to every channel for each frame, writing
       aligned TIFFs back in place. One frame loaded at a time.

Memory: at most 2 frames held in RAM simultaneously.
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
    align_to_first: bool = True,
) -> None:
    """Align all frames for one xy position.

    Args:
        xy_dir:          Path to xy*/ directory.
        align_channel:   Subdirectory used to compute shifts (typically 'phase').
        max_shift_px:    Shifts larger than this (pixels) are clamped to 0.
        align_to_first:  If True (default), register every frame against frame 0.
                         If False, use sequential frame-to-frame registration.
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

    # --- Step 1: compute shifts relative to frame 0 ---
    frame0 = iio.imread(tif_paths[0]).astype(float)
    img_shape = frame0.shape[:2]
    cum_shifts = np.zeros((n_frames, 2))
    n_clamped = 0

    if align_to_first:
        # Register every frame directly against frame 0.
        # Shifts are absolute — no compounding of subpixel errors.
        for i in range(1, n_frames):
            cur = iio.imread(tif_paths[i]).astype(float)
            shift, _, _ = phase_cross_correlation(frame0, cur, upsample_factor=100)
            if np.hypot(shift[0], shift[1]) > max_shift_px:
                print(f"  {xy_dir.name} t{i}: shift {shift} exceeds "
                      f"{max_shift_px}px — clamped to 0")
                shift = np.array([0.0, 0.0])
                n_clamped += 1
            cum_shifts[i] = shift
    else:
        # Sequential: each frame registered against the previous frame.
        # Cumulative sum gives absolute correction relative to frame 0.
        prev = frame0
        raw_shifts = np.zeros((n_frames, 2))
        for i in range(1, n_frames):
            cur = iio.imread(tif_paths[i]).astype(float)
            shift, _, _ = phase_cross_correlation(prev, cur, upsample_factor=100)
            if np.hypot(shift[0], shift[1]) > max_shift_px:
                print(f"  {xy_dir.name} t{i}: shift {shift} exceeds "
                      f"{max_shift_px}px — clamped to 0")
                shift = np.array([0.0, 0.0])
                n_clamped += 1
            raw_shifts[i] = shift
            prev = cur
        cum_shifts = np.cumsum(raw_shifts, axis=0)

    if n_clamped:
        print(f"  {xy_dir.name}: {n_clamped}/{n_frames-1} shifts clamped")

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
    xy_dir, align_channel, max_shift_px, align_to_first = args
    try:
        _align_position(Path(xy_dir), align_channel, max_shift_px, align_to_first)
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
    align_to_first: bool = True,
) -> None:
    """Drift-correct all xy positions in data_dir.

    Args:
        data_dir:        Directory containing xy*/ subdirectories.
        align_channel:   Channel subdirectory used to compute shifts.
        workers:         Number of parallel worker processes.
        max_shift_px:    Shifts larger than this (px) are clamped to 0.
        align_to_first:  If True (default), register each frame against
                         frame 0 — avoids compounding of subpixel errors
                         over long movies. If False, use sequential
                         frame-to-frame registration.
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

    mode = "align-to-first" if align_to_first else "sequential"
    print(
        f"Aligning {len(xy_dirs)} position(s) using channel '{align_channel}' "
        f"(mode={mode}, max_shift_px={max_shift_px})..."
    )

    args = [(str(d), align_channel, max_shift_px, align_to_first) for d in xy_dirs]

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
            _align_position(xy_dir, align_channel, max_shift_px, align_to_first)

    print("Alignment complete.")
