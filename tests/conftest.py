"""Shared helpers: run `nd2_to_cells track` on a few label frames.

Linker configurations under test are in CONFIGS: the 100XEc preset with some
values replaced, such as a list of opt-in fixes (track.FIXES). "baseline" is
the bug-for-bug SuperSegger port. Every test that takes the `config` fixture
runs on all of them.
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

CONFIGS: dict[str, dict] = {
    "baseline": {},
    "label_collision": {"fixes": ["label_collision"]},
    "area_penalty": {"fixes": ["area_penalty"]},
    "split_needs_two": {"fixes": ["split_needs_two"]},
    "itk_v4": {"fixes": ["itk_v4"]},
    "lookahead3": {"division_lookahead": 3},
    "seeded_split": {"fixes": ["seeded_split"]},
    "end_on_merge": {"fixes": ["end_on_merge"]},
    "assign_bookkeeping": {
        "fixes": [
            "assign_rev",
            "assign_all_regions",
            "assign_robbed_area",
            "assign_exchange",
            "assign_nan",
        ]
    },
    "errorrez_bookkeeping": {
        "fixes": ["division_both_sisters", "f_no_match", "f_gate"]
    },
    # On top of label_collision, which removes the far events that every
    # extra mask edit otherwise adds through label collisions:
    "lc_itk_v4": {"fixes": ["label_collision", "itk_v4"]},
    "lc_seeded_split": {"fixes": ["label_collision", "seeded_split"]},
    "lc_merge_fallback": {"fixes": ["label_collision", "merge_fallback"]},
    "lc_lookahead3": {"fixes": ["label_collision"], "division_lookahead": 3},
    # The fixes that helped on 260429, together:
    "combined": {
        "fixes": [
            "label_collision",
            "seeded_split",
            "merge_fallback",
            "death_last_frame",
            "drop_ghosts",
            "marker_chain",
        ],
        "division_lookahead": 3,
    },
    "lc_seeded_orphan": {"fixes": ["label_collision", "seeded_split", "orphan_split"]},
    "combined_orphan": {
        "fixes": [
            "label_collision",
            "seeded_split",
            "merge_fallback",
            "orphan_split",
            "death_last_frame",
            "drop_ghosts",
            "marker_chain",
        ],
        "division_lookahead": 3,
    },
    "combined_end": {
        "fixes": [
            "label_collision",
            "seeded_split",
            "merge_fallback",
            "end_on_merge",
            "death_last_frame",
            "drop_ghosts",
            "marker_chain",
        ],
        "division_lookahead": 3,
    },
}


def preset_text(config: str) -> str:
    """The 100XEc preset with the values of CONFIGS[config] appended."""
    lines = [(REPO / "presets" / "100XEc.toml").read_text()]
    for key, value in CONFIGS[config].items():
        if isinstance(value, bool):
            value = str(value).lower()
        elif isinstance(value, list):
            value = "[" + ", ".join(f'"{v}"' for v in value) + "]"
        lines.append(f"{key} = {value}")
    return "\n".join(lines) + "\n"


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
    preset = out_dir / "preset.toml"
    preset.write_text(preset_text(config))
    run_track(str(out_dir), preset=str(preset))
    return load_tracks(out_dir / "xy01" / "cell")
