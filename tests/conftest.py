"""Shared helpers: run `nd2_to_cells track` on a few label frames.

Linker configurations under test are in CONFIGS: "baseline" is the
bug-for-bug SuperSegger port. A change made for the linking report is added
as a further entry (a preset name or .toml path), so every test that takes
the `config` fixture runs on it too.
"""

import os
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

import nd2_to_cells
from nd2_to_cells.track import run_track
from tests.merge_events import Track, load_tracks

REPO = Path(__file__).resolve().parents[1]
if Path(nd2_to_cells.__file__).resolve().parents[1] != REPO:
    raise ImportError(
        f"nd2_to_cells is imported from {nd2_to_cells.__file__}, not from {REPO}"
    )

CONFIGS = {"baseline": "100XEc"}

# The 260429 experiment folder (with xy01 ... xy04), read-only.
DATA = Path(os.environ["ND2C_TEST_DATA"]) if "ND2C_TEST_DATA" in os.environ else None


@pytest.fixture(params=sorted(CONFIGS))
def config(request) -> str:
    return request.param


def link(
    out_dir: Path, frames: dict[int, np.ndarray | Path], config: str = "baseline"
) -> dict[int, Track]:
    """Track label frames (0-based frame -> label image or mask PNG path).

    The frames are written (or linked) into out_dir/xy01/masks/, and the
    tracks that `track` writes to out_dir/xy01/cell/ are returned.
    """
    masks = out_dir / "xy01" / "masks"
    masks.mkdir(parents=True)
    for t, src in frames.items():
        dst = masks / f"test_t{t + 1:03d}xy01c1_cp_masks.png"
        if isinstance(src, Path):
            dst.symlink_to(src)
        else:
            iio.imwrite(dst, src.astype(np.uint16))
    run_track(str(out_dir), preset=CONFIGS[config])
    return load_tracks(out_dir / "xy01" / "cell")
