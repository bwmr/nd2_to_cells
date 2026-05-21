"""Export ND2 files to per-position, per-channel TIFF files.

Output naming convention (matches timelapse_analysis expectations):
    {basename}_t{T:0Nd}xy{P:0Md}c{C}.tif

where:
    N = floor(log10(n_timepoints)) + 1   (time zero-padding width)
    M = floor(log10(n_positions))  + 1   (xy zero-padding width)
    C = 1 for phase, 2 for fluor1, 3 for fluor2, ...
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

    # Dimension order from aicsimageio: varies by file.
    # We always work through the public API to avoid dim-order assumptions.
    n_positions = img.dims.S if hasattr(img.dims, "S") else 1
    n_timepoints = img.dims.T
    n_channels = img.dims.C
    has_z = img.dims.Z > 1

    # Build ordered channel list: phase first, then fluors in ND2 order
    fluor_channels = [c for c in range(n_channels) if c != phase_channel]
    # channel_map[nd2_c] = (subdir_name, c_suffix)
    channel_map = {phase_channel: ("phase", 1)}
    for fluor_idx, nd2_c in enumerate(fluor_channels, start=1):
        channel_map[nd2_c] = (f"fluor{fluor_idx}", fluor_idx + 1)

    t_pad = _pad_width(n_timepoints)
    p_pad = _pad_width(n_positions)

    print(
        f"ND2: {n_positions} position(s), {n_timepoints} timepoint(s), "
        f"{n_channels} channel(s), Z={'yes' if has_z else 'no'}"
    )
    print(
        f"Phase channel index: {phase_channel} → c1; "
        f"fluor channels: {fluor_channels}"
    )

    for p in tqdm(range(n_positions), desc="Positions", unit="pos"):
        p_str = f"{p + 1:0{p_pad}d}"
        xy_dir = output_dir / f"xy{p_str}"

        # Create subdirectories
        (xy_dir / "masks").mkdir(parents=True, exist_ok=True)
        (xy_dir / "cell").mkdir(parents=True, exist_ok=True)
        for nd2_c, (subdir, _) in channel_map.items():
            (xy_dir / subdir).mkdir(parents=True, exist_ok=True)

        for t in tqdm(range(n_timepoints), desc=f"  xy{p_str} frames",
                      unit="frame", leave=False):
            t_str = f"{t + 1:0{t_pad}d}"

            for nd2_c, (subdir, c_suffix) in channel_map.items():
                # xarray with dims (T, Z, Y, X) or similar; use get_image_data
                # for positional access.  Scene = position index.
                frame_data = img.get_image_data(
                    "ZYX",
                    S=p,
                    T=t,
                    C=nd2_c,
                )  # shape: (Z, Y, X) even when Z=1

                if has_z and frame_data.shape[0] > 1:
                    frame_data = _z_project(frame_data, z_project)
                else:
                    frame_data = frame_data[0]  # drop Z dimension

                fname = f"{basename}_t{t_str}xy{p_str}c{c_suffix}.tif"
                dest = xy_dir / subdir / fname
                iio.imwrite(dest, frame_data)

                # Write copy to raw_im/
                shutil.copy2(dest, raw_im_dir / fname)

    print(f"Export complete → {output_dir}")
