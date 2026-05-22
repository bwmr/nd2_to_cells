"""Export ND2 files to per-position, per-channel TIFF files.

Output naming convention (matches timelapse_analysis expectations):
    {basename}_t{T:0Nd}xy{P:0Md}c{C}.tif

where:
    N = floor(log10(n_timepoints)) + 1   (time zero-padding width)
    M = floor(log10(n_positions))  + 1   (xy zero-padding width)
    C = 1 for phase, 2 for fluor1, 3 for fluor2, ...

Positions in ND2 files are stored as *scenes* (aicsimageio terminology).
They are accessed via img.set_scene(i) / img.scenes, NOT via an S
dimension in get_image_data.  Dims (T, C, Z) are re-read per scene
because they can differ between positions.
"""

import math
import shutil
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from aicsimageio import AICSImage
from tqdm import tqdm


def _pad_width(n: int) -> int:
    """Number of digits needed to zero-pad integers up to n."""
    return max(1, math.floor(math.log10(max(n, 1))) + 1)


def _z_project(stack: np.ndarray, method: str) -> np.ndarray:
    """Project a Z-stack (Z, Y, X) -> (Y, X)."""
    if method == "mean":
        return stack.mean(axis=0).astype(stack.dtype)
    elif method == "max":
        return stack.max(axis=0)
    else:
        raise ValueError(f"Unknown z_project method: {method!r}")


def run_export(
    nd2_path: str,
    output_dir: str,
    basename: str,
    phase_channel: int = 0,
    z_project: str = "mean",
) -> None:
    """Export an ND2 file to per-position TIFFs.

    Args:
        nd2_path:      Path to the ND2 file.
        output_dir:    Root output directory.
        basename:      Filename prefix (e.g. '260430').
        phase_channel: 0-based channel index in the ND2 for phase contrast.
                       All other channels become fluor1, fluor2, ... in order.
        z_project:     Z-projection method ('mean' or 'max') for Z-stacks.
    """
    nd2_path = Path(nd2_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_im_dir = output_dir / "raw_im"
    raw_im_dir.mkdir(exist_ok=True)

    img = AICSImage(nd2_path)

    # Positions in ND2 files are scenes, not an S array dimension.
    # img.scenes is a tuple of scene name strings, one per xy position.
    n_positions = len(img.scenes)
    p_pad = _pad_width(n_positions)

    # Read dims from scene 0 to determine T, C, Z for channel map.
    # (T and C are typically the same across all scenes; Z may vary but
    # we use scene 0 as representative for the channel map only.)
    img.set_scene(0)
    n_timepoints = img.dims.T
    n_channels = img.dims.C
    t_pad = _pad_width(n_timepoints)

    # Build channel map: phase excluded, all others become fluor1, fluor2, ...
    fluor_channels = [c for c in range(n_channels) if c != phase_channel]
    # channel_map[nd2_c_index] = (subdir_name, c_suffix_in_filename)
    channel_map = {phase_channel: ("phase", 1)}
    for fluor_idx, nd2_c in enumerate(fluor_channels, start=1):
        channel_map[nd2_c] = (f"fluor{fluor_idx}", fluor_idx + 1)

    print(
        f"ND2: {n_positions} position(s), {n_timepoints} timepoint(s), "
        f"{n_channels} channel(s)"
    )
    print(f"Scenes: {img.scenes}")
    print(
        f"Phase channel index: {phase_channel} → c1; "
        f"fluor channel indices: {fluor_channels}"
    )

    for p in tqdm(range(n_positions), desc="Positions", unit="pos"):
        # Select the scene (= xy position) before reading any data or dims.
        img.set_scene(p)

        # Re-read dims for this scene — T, C, Z may differ per scene.
        n_t = img.dims.T
        n_c = img.dims.C
        has_z = img.dims.Z > 1

        # Validate phase_channel index for this scene
        if phase_channel >= n_c:
            print(
                f"  [skip] scene {p} ({img.current_scene}): "
                f"phase_channel={phase_channel} >= n_channels={n_c}"
            )
            continue

        p_str = f"{p + 1:0{p_pad}d}"
        xy_dir = output_dir / f"xy{p_str}"

        # Create subdirectories (masks/ is created by Omnipose, not here)
        (xy_dir / "cell").mkdir(parents=True, exist_ok=True)
        for nd2_c, (subdir, _) in channel_map.items():
            if nd2_c < n_c:
                (xy_dir / subdir).mkdir(parents=True, exist_ok=True)

        for t in tqdm(range(n_t), desc=f"  xy{p_str}", unit="frame", leave=False):
            t_str = f"{t + 1:0{t_pad}d}"

            for nd2_c, (subdir, c_suffix) in channel_map.items():
                if nd2_c >= n_c:
                    continue  # this scene has fewer channels

                # get_image_data returns (Z, Y, X) for the current scene.
                # Do NOT pass S= here — scene is already selected via set_scene.
                frame_data = img.get_image_data("ZYX", T=t, C=nd2_c)

                if has_z and frame_data.shape[0] > 1:
                    frame_data = _z_project(frame_data, z_project)
                else:
                    frame_data = frame_data[0]  # drop Z dimension

                fname = f"{basename}_t{t_str}xy{p_str}c{c_suffix}.tif"
                dest = xy_dir / subdir / fname
                iio.imwrite(dest, frame_data)

                # Write copy to raw_im/
                shutil.copy2(dest, raw_im_dir / fname)

    print(f"\nExport complete → {output_dir}")
