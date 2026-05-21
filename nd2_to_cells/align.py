"""Frame drift correction for all xy positions.

Algorithm:
    1. Load phase images in frame order.
    2. Compute frame-to-frame shifts via phase cross-correlation
       (skimage.registration.phase_cross_correlation, upsample_factor=100,
       matching SuperSegger's precision=100).
    3. Accumulate shifts to get absolute displacement of each frame
       relative to frame 0.
    4. Compute the minimum padding canvas that fits all shifted frames.
    5. Shift each frame of every channel into the padded canvas.
    6. Write aligned TIFFs back in place.

Originals are expected to already exist in raw_im/ (written by export).
"""

import re
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
    """Extract the frame number from a TIFF filename.

    Handles any digit count: '_t001xy...', '_t0001xy...', etc.
    """
    m = re.search(r"_t(\d+)xy", fname)
    if m is None:
        raise ValueError(f"Cannot parse frame number from filename: {fname!r}")
    return int(m.group(1))


def _load_channel_frames(channel_dir: Path) -> tuple[list[int], list[np.ndarray]]:
    """Load all TIFF frames from a channel directory, sorted by frame number."""
    tifs = sorted(
        [f for f in channel_dir.iterdir() if f.suffix.lower() in (".tif", ".tiff")],
        key=lambda f: _parse_frame_number(f.name),
    )
    if not tifs:
        return [], []
    frame_numbers = [_parse_frame_number(f.name) for f in tifs]
    images = [iio.imread(f) for f in tifs]
    return frame_numbers, images, tifs


def _shift_image(img: np.ndarray, dy: float, dx: float,
                 canvas_shape: tuple[int, int],
                 row_offset: int, col_offset: int,
                 fill_value: float) -> np.ndarray:
    """Place img (shifted by dy, dx) into a padded canvas.

    canvas_shape: (H, W) of the output canvas.
    row_offset, col_offset: where frame 0 sits in the canvas.
    fill_value: background value (mean of the image).
    """
    canvas = np.full(canvas_shape, fill_value, dtype=img.dtype)

    # Integer shift (subpixel component is discarded for simplicity;
    # for biological images at 1-min frame rates this is sufficient).
    row_shift = int(round(dy))
    col_shift = int(round(dx))

    # Source region in original image
    src_r0 = max(0, -row_shift)
    src_r1 = img.shape[0] + min(0, -row_shift)
    src_c0 = max(0, -col_shift)
    src_c1 = img.shape[1] + min(0, -col_shift)

    # Destination region in canvas
    dst_r0 = row_offset + row_shift + src_r0
    dst_r1 = row_offset + row_shift + src_r1
    dst_c0 = col_offset + col_shift + src_c0
    dst_c1 = col_offset + col_shift + src_c1

    # Clamp to canvas bounds
    cr0 = max(0, dst_r0); cr1 = min(canvas_shape[0], dst_r1)
    cc0 = max(0, dst_c0); cc1 = min(canvas_shape[1], dst_c1)
    # Corresponding source trim
    sr0 = src_r0 + (cr0 - dst_r0); sr1 = sr0 + (cr1 - cr0)
    sc0 = src_c0 + (cc0 - dst_c0); sc1 = sc0 + (cc1 - cc0)

    if cr1 > cr0 and cc1 > cc0:
        canvas[cr0:cr1, cc0:cc1] = img[sr0:sr1, sc0:sc1]
    return canvas


# ---------------------------------------------------------------------------
# Per-position alignment
# ---------------------------------------------------------------------------

def _align_position(xy_dir: Path, align_channel: str) -> None:
    """Align all frames for one xy position."""
    align_dir = xy_dir / align_channel
    if not align_dir.exists():
        print(f"  [skip] {xy_dir.name}: no '{align_channel}/' directory")
        return

    frame_numbers, images, tif_paths = _load_channel_frames(align_dir)
    if len(images) < 2:
        print(f"  [skip] {xy_dir.name}: fewer than 2 frames")
        return

    n_frames = len(images)
    img_shape = images[0].shape  # (H, W)

    # --- Step 1: compute frame-to-frame shifts ---
    # shift[i] = displacement of frame i relative to frame i-1, in (row, col)
    raw_shifts = np.zeros((n_frames, 2))
    for i in range(1, n_frames):
        shift, _, _ = phase_cross_correlation(
            images[i - 1].astype(float),
            images[i].astype(float),
            upsample_factor=100,
        )
        raw_shifts[i] = shift  # (row_shift, col_shift)

    # --- Step 2: accumulate to get absolute shifts relative to frame 0 ---
    cum_shifts = np.cumsum(raw_shifts, axis=0)  # shape (n_frames, 2)

    # --- Step 3: compute padded canvas size ---
    row_shifts = cum_shifts[:, 0]
    col_shifts = cum_shifts[:, 1]

    pad_top = int(math.ceil(max(0, row_shifts.max())))
    pad_bottom = int(math.ceil(max(0, -row_shifts.min())))
    pad_left = int(math.ceil(max(0, col_shifts.max())))
    pad_right = int(math.ceil(max(0, -col_shifts.min())))

    canvas_h = img_shape[0] + pad_top + pad_bottom
    canvas_w = img_shape[1] + pad_left + pad_right
    canvas_shape = (canvas_h, canvas_w)

    # frame 0 sits at (pad_top, pad_left) in the canvas
    row_offset = pad_top
    col_offset = pad_left

    # --- Step 4: apply shifts to every channel ---
    channel_dirs = [d for d in xy_dir.iterdir()
                    if d.is_dir() and d.name not in ("masks", "cell", "seg")]

    for ch_dir in channel_dirs:
        _, ch_images, ch_tif_paths = _load_channel_frames(ch_dir)
        if not ch_images:
            continue

        for i, (img, tif_path) in enumerate(zip(ch_images, ch_tif_paths)):
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


def _align_position_wrapper(args):
    """Wrapper for ProcessPoolExecutor (must be top-level picklable)."""
    xy_dir, align_channel = args
    try:
        _align_position(Path(xy_dir), align_channel)
    except Exception as exc:
        print(f"ERROR aligning {xy_dir}: {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

import math  # noqa: E402 — needed by _align_position, imported here for clarity


def run_align(
    data_dir: str,
    align_channel: str = "phase",
    workers: int = 1,
) -> None:
    """Drift-correct all xy positions in data_dir.

    Args:
        data_dir:      Directory containing xy*/ subdirectories.
        align_channel: Subdirectory name to use for computing shifts.
        workers:       Number of parallel worker processes.
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

    print(f"Aligning {len(xy_dirs)} position(s) using channel '{align_channel}'...")

    args = [(str(d), align_channel) for d in xy_dirs]

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
            _align_position(xy_dir, align_channel)

    print("Alignment complete.")
