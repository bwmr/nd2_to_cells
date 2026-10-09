"""Real-data cases of issues A and B from the 260429 linking report.

Each case is a merge event of the baseline's full run on 260429. At frame t,
the track holding point T absorbs the track holding point X. Both points are
(y, x) in the full image and lie inside the two cells at t-1. Linking only
the window of frames start ... t+2 reproduces the full run's event.

- test_baseline_event: the baseline still produces each event (snapshot).
- test_merge_resolved: the desired outcome. The two cells are separate tracks
  at t-1 and neither absorbs the other at t. Configurations are expected to
  fail it (strict xfail) for every case not listed in RESOLVED.

Needs ND2C_TEST_DATA, the 260429 folder (read only). Takes about 7 min.
"""

from dataclasses import dataclass

import pytest

from tests.conftest import DATA, link
from tests.merge_events import Track, find_events

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(DATA is None, reason="ND2C_TEST_DATA is not set"),
]


@dataclass(frozen=True)
class Case:
    pos: str
    t: int  # 0-based frame of the merged object
    start: int  # first frame of the window
    point_t: tuple[int, int]  # inside the absorbing cell at t-1
    point_x: tuple[int, int]  # inside the absorbed cell at t-1
    kind: str  # event kind in the baseline


CASES = {
    # Report figure A (IoU linker: daughters re-fuse at t=91). The port picks
    # the cell up into a track ~800 px away (label collision) at t=90.
    "report_fig_A": Case("xy04", 90, 80, (1824, 865), (1070, 536), "far"),
    # Report figure B (neighbour merged at t=340); the port's pair are sisters.
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

# Cases each configuration is expected to resolve.
RESOLVED: dict[str, set[str]] = {"baseline": set()}

_runs: dict[tuple[str, str], dict[int, Track]] = {}


def run_case(name: str, config: str, tmp_path_factory) -> dict[int, Track]:
    if (name, config) not in _runs:
        case = CASES[name]
        masks = sorted((DATA / case.pos / "masks").glob("*cp_masks.png"))
        frames = {t: masks[t] for t in range(case.start, case.t + 3)}
        out = tmp_path_factory.mktemp(f"{name}_{config}")
        _runs[name, config] = link(out, frames, config)
    return _runs[name, config]


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
def test_baseline_event(name, tmp_path_factory):
    case = CASES[name]
    tracks = run_case(name, "baseline", tmp_path_factory)
    t, x = pair(tracks, case)
    assert t is not None and x is not None and t is not x
    assert any(
        (e.frame, e.cell, e.absorbed, e.kind) == (case.t, t.key, x.key, case.kind)
        for e in find_events(tracks)
    )


@pytest.mark.parametrize("name", sorted(CASES))
def test_merge_resolved(name, config, tmp_path_factory, request):
    if name not in RESOLVED[config]:
        mark = pytest.mark.xfail(strict=True, raises=AssertionError, reason="open")
        request.applymarker(mark)
    case = CASES[name]
    tracks = run_case(name, config, tmp_path_factory)
    t, x = pair(tracks, case)
    assert t is not None and x is not None, "a cell is not tracked at t-1"
    assert t is not x, "the two cells are one track at t-1"
    assert not [
        e
        for e in find_events(tracks)
        if e.frame == case.t and {e.cell, e.absorbed} == {t.key, x.key}
    ]
