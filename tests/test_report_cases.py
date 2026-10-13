"""Real-data cases of issues A and B from the 260429 linking report.

Each case is a merge event of a SuperSegger-equivalent run on 260429: at
frame t, the track holding point T absorbed the track holding point X. Both
points are (y, x) in the full image and lie inside the two cells at t-1. Only
the window of frames start ... t+2 is linked.

The masks of each window are in tests/data/<case>.npz: the bounding box of
the two points plus 128 px (512 px for xy03_A), relabelled per frame in label
order, with its origin and the full frame shape. The test puts the crop back
into empty frames of that shape, because SuperSegger's split registration
depends on the frame size. So placed, every crop still makes the
SuperSegger-equivalent port (commit a40b03e) produce the case's event.

The test asks for the desired outcome: neither of the two cells absorbs a
track, or is absorbed, at t. Alternatively one track holds both at t-1
without having absorbed another track (a split not called a division).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import link
from tests.merge_events import Track, find_events

FIXTURES = Path(__file__).parent / "data"


@dataclass(frozen=True)
class Case:
    pos: str
    t: int  # 0-based frame of the merged object
    start: int  # first frame of the window
    point_t: tuple[int, int]  # inside the absorbing cell at t-1
    point_x: tuple[int, int]  # inside the absorbed cell at t-1
    kind: str  # event kind in the SuperSegger-equivalent run


CASES = {
    # Report figure A (IoU linker: daughters re-fuse at t=91). SuperSegger
    # picks the cell up into a track ~800 px away (label collision) at t=90.
    "report_fig_A": Case("xy04", 90, 80, (1824, 865), (1070, 536), "far"),
    # Report figure B (neighbour merged at t=340); SuperSegger's pair are sisters.
    "report_fig_B": Case("xy04", 340, 325, (693, 1689), (685, 1747), "A"),
    "xy01_A": Case("xy01", 191, 181, (1808, 844), (1784, 868), "A"),
    "xy02_A": Case("xy02", 133, 123, (1262, 648), (1251, 583), "A"),
    "xy03_A": Case("xy03", 268, 258, (1625, 785), (1594, 771), "A"),
    "xy04_A": Case("xy04", 302, 292, (639, 1728), (652, 1678), "A"),
    "xy01_B": Case("xy01", 313, 303, (1046, 1604), (1125, 1539), "B"),
    "xy02_B": Case("xy02", 108, 98, (439, 848), (438, 891), "B"),
    "xy03_B": Case("xy03", 139, 129, (1732, 746), (1661, 778), "B"),
    "xy04_B": Case("xy04", 198, 188, (689, 1663), (657, 1667), "B"),
}


def run_case(name: str, tmp_path) -> dict[int, Track]:
    case = CASES[name]
    with np.load(FIXTURES / f"{name}.npz") as d:
        crops, (y0, x0), shape = d["masks"], d["origin"], tuple(d["shape"])
    assert len(crops) == case.t + 3 - case.start
    frames = {}
    for t, crop in enumerate(crops, start=case.start):
        frame = np.zeros(shape, np.uint16)
        frame[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]] = crop
        frames[t] = frame
    return link(tmp_path, frames)


def track_at(tracks: dict[int, Track], t: int, point: tuple[int, int]):
    for tr in tracks.values():
        hits = [i for i, f in enumerate(tr.frames) if f == t]
        if hits:
            y, x = point[0] - tr.y0, point[1] - tr.x0
            h, w = tr.mask.shape[:2]
            if 0 <= y < h and 0 <= x < w and tr.mask[y, x, hits[0]]:
                return tr
    return None


def pair(tracks, case: Case):
    return (
        track_at(tracks, case.t - 1, case.point_t),
        track_at(tracks, case.t - 1, case.point_x),
    )


@pytest.mark.parametrize("name", sorted(CASES))
def test_merge_resolved(name, tmp_path):
    case = CASES[name]
    tracks = run_case(name, tmp_path)
    t, x = pair(tracks, case)
    assert t is not None and x is not None, "a cell is not tracked at t-1"
    events = find_events(tracks)
    if t is x:
        assert not [e for e in events if e.cell == t.key and e.frame <= case.t]
    else:
        pair_keys = {t.key, x.key}
        assert not [
            e for e in events if e.frame == case.t and pair_keys & {e.cell, e.absorbed}
        ]
