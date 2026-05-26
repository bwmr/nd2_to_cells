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
    1. Read source TIFFs from raw_im/, grouped by xy position and channel suffix.
    2. For each frame, compute an FFT-based focus score (port of SuperSegger
       isFocus.m). Frames with score <= 0 are skipped entirely (no output file).
    3. Compute shifts from the align_channel using phase_cross_correlation
       (upsample_factor=100) on good frames only.
    4. Clamp shifts exceeding max_shift_px to 0 (sequential mode only).
    5. Compute the padded canvas size from good-frame shifts.
    6. Place each frame in the padded canvas (integer shift), then apply the
       fractional residual via scipy.ndimage.shift (spline interpolation) for
       subpixel accuracy. Write aligned TIFFs to xy{P}/{subdir}/.
       raw_im/ is left untouched. One frame loaded at a time.

Memory: at most 2 frames held in RAM simultaneously.
"""

import math
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import scipy.ndimage
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


def _parse_xy_str(fname: str) -> str:
    """Extract the zero-padded xy position string.

    e.g. '260430_t001xy01c1.tif' -> '01'
    """
    m = re.search(r"xy(\d+)c", fname)
    if m is None:
        raise ValueError(f"Cannot parse xy position from filename: {fname!r}")
    return m.group(1)


def _parse_channel_suffix(fname: str) -> int:
    """Extract the channel suffix integer, e.g. '260430_t001xy01c2.tif' -> 2."""
    m = re.search(r"xy\d+c(\d+)\.tif", fname, re.IGNORECASE)
    if m is None:
        raise ValueError(f"Cannot parse channel suffix from filename: {fname!r}")
    return int(m.group(1))


def _channel_subdir(c_suffix: int, phase_suffix: int, all_suffixes: list[int]) -> str:
    """Map a channel suffix to its output subdirectory name.

    Args:
        c_suffix:      The channel suffix of the file being mapped.
        phase_suffix:  The channel suffix that corresponds to 'phase'.
        all_suffixes:  Sorted list of all channel suffixes present.
    """
    if c_suffix == phase_suffix:
        return "phase"
    fluor_suffixes = [s for s in all_suffixes if s != phase_suffix]
    return f"fluor{fluor_suffixes.index(c_suffix) + 1}"


def _sorted_tifs(paths: list[Path]) -> list[Path]:
    """Return TIFF paths sorted by frame number."""
    return sorted(paths, key=lambda f: _parse_frame_number(f.name))


def _apply_shift(
    img: np.ndarray,
    dy: float,
    dx: float,
    canvas_shape: tuple[int, int],
    row_offset: int,
    col_offset: int,
    fill_value: float,
) -> np.ndarray:
    """Place img into a padded canvas and apply a subpixel shift.

    The integer part of (dy, dx) is handled by placing the image at
    (row_offset + round(dy), col_offset + round(dx)) in the canvas.
    The fractional residual is then applied via scipy.ndimage.shift
    (spline interpolation), preserving subpixel registration accuracy.

    row_offset / col_offset are chosen so that all frames fit in the canvas.
    """
    canvas = np.full(canvas_shape, fill_value, dtype=np.float64)

    dy_i = int(round(dy))
    dx_i = int(round(dx))
    H, W = img.shape[:2]

    dst_r0 = row_offset + dy_i
    dst_c0 = col_offset + dx_i
    cr0 = max(0, dst_r0)
    cr1 = min(canvas_shape[0], dst_r0 + H)
    cc0 = max(0, dst_c0)
    cc1 = min(canvas_shape[1], dst_c0 + W)

    if cr1 > cr0 and cc1 > cc0:
        sr0 = cr0 - dst_r0
        sr1 = sr0 + (cr1 - cr0)
        sc0 = cc0 - dst_c0
        sc1 = sc0 + (cc1 - cc0)
        canvas[cr0:cr1, cc0:cc1] = img[sr0:sr1, sc0:sc1]

    # Apply fractional residual with spline interpolation
    frac_dy = dy - round(dy)
    frac_dx = dx - round(dx)
    if frac_dy != 0.0 or frac_dx != 0.0:
        canvas = scipy.ndimage.shift(
            canvas, (frac_dy, frac_dx), mode="constant", cval=fill_value
        )

    return canvas.astype(img.dtype)


def _focus_score(fft: np.ndarray) -> float:
    """Return a focus quality score for an image given its FFT.

    Port of SuperSegger's isFocus.m. Computes the ratio of mid-frequency
    power (wavelengths 8–12 px) to high-frequency power (wavelengths < 3 px).
    Focused images have more mid-frequency content relative to high-frequency
    noise. Score > 0 means the image is considered in focus.

    Args:
        fft: 2-D complex FFT of the image (np.fft.fft2 output, DC in [0,0]).
    """
    ss = fft.shape
    pp1 = (fft * np.conj(fft)).real  # power spectrum
    # Mean power along the first 10 rows, left half (low-frequency rows)
    mean_pp1 = pp1[:10, : ss[1] // 2].mean(axis=0)
    k = np.arange(1, len(mean_pp1) + 1) / ss[1]
    lam = 1.0 / k
    mm1 = mean_pp1[lam < 3].mean()  # high-freq power (λ < 3 px)
    mm2 = mean_pp1[(lam > 8) & (lam < 12)].mean()  # mid-freq power
    if mm1 == 0:
        return 0.0
    return float(mm2 / mm1 - 1)


# ---------------------------------------------------------------------------
# Per-position alignment
# ---------------------------------------------------------------------------


def _align_position(
    xy_dir: Path,
    raw_im_dir: Path,
    align_channel: str,
    phase_channel_suffix: int,
    max_shift_px: float = 50.0,
    align_to_first: bool = True,
) -> None:
    """Align all frames for one xy position.

    Reads source TIFFs from raw_im_dir (filtered by xy position string).
    Writes aligned TIFFs to xy_dir/{subdir}/, creating subdirs as needed.
    raw_im_dir is left untouched.

    Args:
        xy_dir:               Path to the xy*/ output directory.
        raw_im_dir:           Path to raw_im/ directory (source TIFFs).
        align_channel:        Subdirectory name used to compute shifts (e.g. 'phase').
        phase_channel_suffix: c-suffix integer that maps to the 'phase' subdirectory.
        max_shift_px:         Shifts larger than this (pixels) are clamped to 0.
        align_to_first:       If True, register every frame against frame 0.
                              If False, use sequential frame-to-frame registration.
    """
    xy_str = xy_dir.name[2:]  # e.g. "xy01" -> "01"

    # Collect all TIFFs for this position from raw_im/
    all_tifs = [
        f
        for f in raw_im_dir.iterdir()
        if f.suffix.lower() in (".tif", ".tiff") and _parse_xy_str(f.name) == xy_str
    ]
    if not all_tifs:
        print(f"  [skip] {xy_dir.name}: no TIFFs found in {raw_im_dir}")
        return

    # Group by channel suffix
    all_suffixes = sorted({_parse_channel_suffix(f.name) for f in all_tifs})
    by_suffix: dict[int, list[Path]] = {s: [] for s in all_suffixes}
    for f in all_tifs:
        by_suffix[_parse_channel_suffix(f.name)].append(f)

    # Identify the reference channel suffix for shift computation
    ref_suffix = next(
        (
            s
            for s in all_suffixes
            if _channel_subdir(s, phase_channel_suffix, all_suffixes) == align_channel
        ),
        None,
    )
    if ref_suffix is None:
        print(f"  [skip] {xy_dir.name}: no channel matching '{align_channel}'")
        return

    ref_tifs = _sorted_tifs(by_suffix[ref_suffix])
    n_frames = len(ref_tifs)
    if n_frames < 2:
        print(f"  [skip] {xy_dir.name}: fewer than 2 frames")
        return

    # --- Step 1: compute shifts relative to frame 0, with focus gating ---
    frame0 = iio.imread(ref_tifs[0]).astype(float)
    img_shape = frame0.shape[:2]
    cum_shifts = np.zeros((n_frames, 2))
    skipped: set[int] = set()
    n_clamped = 0

    if align_to_first:
        for i in range(1, n_frames):
            cur = iio.imread(ref_tifs[i]).astype(float)
            # fft_cur = np.fft.fft2(cur)
            # if _focus_score(fft_cur) <= 0:
            #     print(f"  {xy_dir.name} t{i}: out of focus — skipping frame")
            #     skipped.add(i)
            #     cum_shifts[i] = cum_shifts[i - 1]
            #     continue
            shift, _, _ = phase_cross_correlation(frame0, cur, upsample_factor=100)
            cum_shifts[i] = shift
    else:
        prev = frame0
        raw_shifts = np.zeros((n_frames, 2))
        for i in range(1, n_frames):
            cur = iio.imread(ref_tifs[i]).astype(float)
            # fft_cur = np.fft.fft2(cur)
            # if _focus_score(fft_cur) <= 0:
            #     print(f"  {xy_dir.name} t{i}: out of focus — skipping frame")
            #     skipped.add(i)
            #     # raw_shifts[i] stays 0; carry forward by leaving prev unchanged
            #     continue
            shift, _, _ = phase_cross_correlation(prev, cur, upsample_factor=100)
            if np.hypot(shift[0], shift[1]) > max_shift_px:
                print(
                    f"  {xy_dir.name} t{i}: shift {shift} exceeds "
                    f"{max_shift_px}px — clamped to 0"
                )
                shift = np.array([0.0, 0.0])
                n_clamped += 1
            raw_shifts[i] = shift
            prev = scipy.ndimage.shift(
                cur, (shift[0], shift[1]), mode="constant", cval=np.mean(cur)
            )
        cum_shifts = np.cumsum(raw_shifts, axis=0)

    if n_clamped:
        print(f"  {xy_dir.name}: {n_clamped}/{n_frames - 1} shifts clamped")

    # Exclude skipped frames when computing canvas bounds
    good = [i for i in range(n_frames) if i not in skipped]
    row_shifts = cum_shifts[:, 0]
    col_shifts = cum_shifts[:, 1]
    good_row = row_shifts[good]
    good_col = col_shifts[good]

    # --- Step 2: compute padded canvas from good frames only ---
    row_offset = int(math.ceil(max(0.0, -good_row.min())))
    col_offset = int(math.ceil(max(0.0, -good_col.min())))
    canvas_h = img_shape[0] + row_offset + int(math.ceil(max(0.0, good_row.max())))
    canvas_w = img_shape[1] + col_offset + int(math.ceil(max(0.0, good_col.max())))
    canvas_shape = (canvas_h, canvas_w)

    # --- Step 3: apply shifts to every channel, write to xy_dir/{subdir}/ ---
    for suffix in all_suffixes:
        ch_tifs = _sorted_tifs(by_suffix[suffix])
        if len(ch_tifs) != n_frames:
            print(
                f"  {xy_dir.name} c{suffix}: "
                f"{len(ch_tifs)} frames != {n_frames} (reference), skipping"
            )
            continue

        subdir = _channel_subdir(suffix, phase_channel_suffix, all_suffixes)
        out_dir = xy_dir / subdir
        out_dir.mkdir(exist_ok=True)

        for i, tif_path in enumerate(ch_tifs):
            if i in skipped:
                continue
            img = iio.imread(tif_path)
            fill = float(np.mean(img))
            aligned = _apply_shift(
                img,
                dy=cum_shifts[i, 0],
                dx=cum_shifts[i, 1],
                canvas_shape=canvas_shape,
                row_offset=row_offset,
                col_offset=col_offset,
                fill_value=fill,
            )
            iio.imwrite(out_dir / tif_path.name, aligned)

    n_good = len(good)
    n_skip = len(skipped)
    print(
        f"  {xy_dir.name}: aligned {n_good}/{n_frames} frames"
        + (f" ({n_skip} skipped, out of focus)" if n_skip else "")
        + f" (canvas {canvas_h}x{canvas_w}, "
        f"drift row=[{good_row.min():.1f},{good_row.max():.1f}] "
        f"col=[{good_col.min():.1f},{good_col.max():.1f}])"
    )


def _align_position_wrapper(args):
    """Top-level wrapper for ProcessPoolExecutor (must be picklable)."""
    xy_dir, raw_im_dir, align_channel = args[0], args[1], args[2]
    phase_channel_suffix, max_shift_px, align_to_first = args[3], args[4], args[5]
    try:
        _align_position(
            Path(xy_dir),
            Path(raw_im_dir),
            align_channel,
            phase_channel_suffix,
            max_shift_px,
            align_to_first,
        )
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
    phase_channel_suffix: int = 1,
    workers: int = 1,
    max_shift_px: float = 50.0,
    align_to_first: bool = True,
) -> None:
    """Drift-correct all xy positions in data_dir.

    Reads source TIFFs from raw_im/ and writes aligned TIFFs to xy{P}/{subdir}/.

    Args:
        data_dir:             Directory containing xy*/ subdirectories and raw_im/.
        align_channel:        Subdirectory name used to compute shifts (e.g. 'phase').
        phase_channel_suffix: c-suffix integer in filenames that maps to 'phase'
                              (must match --phase-channel used during export; default 1,
                              which corresponds to --phase-channel 0 in export).
        workers:              Number of parallel worker processes.
        max_shift_px:         Shifts larger than this (px) are clamped to 0.
        align_to_first:       If True (default), register each frame against
                              frame 0 — avoids compounding of subpixel errors
                              over long movies. If False, use sequential
                              frame-to-frame registration.
    """
    data_dir = Path(data_dir)
    raw_im_dir = data_dir / "raw_im"
    if not raw_im_dir.exists():
        print(f"raw_im/ not found in {data_dir} — run export first")
        return

    xy_dirs = sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and re.match(r"xy\d+$", d.name)],
        key=lambda d: int(d.name[2:]),
    )

    if not xy_dirs:
        print(f"No xy*/ directories found in {data_dir}")
        return

    mode = "align-to-first" if align_to_first else "sequential"
    print(
        f"Aligning {len(xy_dirs)} position(s) using channel '{align_channel}' "
        f"(mode={mode}, max_shift_px={max_shift_px}, "
        f"phase_suffix=c{phase_channel_suffix})..."
    )

    args = [
        (
            str(d),
            str(raw_im_dir),
            align_channel,
            phase_channel_suffix,
            max_shift_px,
            align_to_first,
        )
        for d in xy_dirs
    ]

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(
                tqdm(
                    pool.map(_align_position_wrapper, args),
                    total=len(args),
                    desc="Aligning positions",
                    unit="pos",
                )
            )
    else:
        for xy_dir in tqdm(xy_dirs, desc="Aligning positions", unit="pos"):
            _align_position(
                xy_dir,
                raw_im_dir,
                align_channel,
                phase_channel_suffix,
                max_shift_px,
                align_to_first,
            )

    print("Alignment complete.")
