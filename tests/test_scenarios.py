"""Synthetic movies of rod-shaped cells, linked with `nd2_to_cells track`.

Controls (growth, division) must come out right. The merge scenarios follow
issues A and B of the 260429 linking report: no track may absorb another one
(the merged label is split, or both tracks end).
"""

import numpy as np

from tests.conftest import link
from tests.merge_events import find_events

SHAPE = (160, 320)
WIDTH = 16  # px, about 1 um at 0.065 um/px


def rod(img: np.ndarray, label: int, y: float, x0: float, x1: float) -> None:
    """Draw a horizontal capsule from x0 to x1 (end-cap centres at x0 + r, x1 - r)."""
    r = WIDTH / 2
    yy, xx = np.mgrid[: img.shape[0], : img.shape[1]]
    cx = np.clip(xx, x0 + r, x1 - r)
    img[(yy - y) ** 2 + (xx - cx) ** 2 <= r**2] = label


def movie(cells_per_frame: list[list[tuple[float, float]]]) -> dict[int, np.ndarray]:
    """One label image per frame; each cell is (x0, x1) on the centre line."""
    frames = {}
    for t, cells in enumerate(cells_per_frame):
        img = np.zeros(SHAPE, np.uint16)
        for k, (x0, x1) in enumerate(cells, start=1):
            rod(img, k, SHAPE[0] / 2, x0, x1)
        frames[t] = img
    return frames


def growing(length: float, n: int, rate: float = 0.02) -> list[float]:
    return [length * (1 + rate) ** t for t in range(n)]


def dividing_cell(
    n_before: int, n_after: int, fuse_at: int | None = None, gap: float = 2
):
    """A cell that grows, divides after n_before frames; daughters may fuse."""
    frames = []
    centre = SHAPE[1] / 2
    for t, length in enumerate(growing(110, n_before + n_after)):
        if t < n_before or (fuse_at is not None and t >= fuse_at):
            frames.append([(centre - length / 2, centre + length / 2)])
        else:
            frames.append(
                [
                    (centre - length / 2, centre - gap / 2),
                    (centre + gap / 2, centre + length / 2),
                ]
            )
    return frames


def test_growth(tmp_path):
    frames = movie([[(100, 100 + length)] for length in growing(60, 8)])
    tracks = link(tmp_path, frames)
    assert [list(tr.frames) for tr in tracks.values()] == [list(range(8))]
    assert find_events(tracks) == []


def test_division(tmp_path):
    tracks = link(tmp_path, movie(dividing_cell(4, 5)))
    mother = [tr for tr in tracks.values() if tr.frames[0] == 0]
    daughters = [tr for tr in tracks.values() if tr.frames[0] == 4]
    assert len(mother) == 1 and list(mother[0].frames) == [0, 1, 2, 3]
    assert len(daughters) == 2 and len(tracks) == 3
    for d, other in zip(daughters, daughters[::-1]):
        assert list(d.frames) == list(range(4, 9))
        assert d.mother == mother[0].id and d.sister == other.id
    assert find_events(tracks) == []


def test_daughters_fuse_after_division(tmp_path):
    """Issue A: the daughters' labels fuse two frames after the division."""
    tracks = link(tmp_path, movie(dividing_cell(4, 6, fuse_at=6)))
    assert [e for e in find_events(tracks) if e.kind == "A"] == []


def test_neighbours_fuse(tmp_path):
    """Issue B: two cells lying end to end fuse into one label."""
    frames = []
    for t, length in enumerate(growing(70, 10)):
        left = (160 - 3 - length, 160 - 3)
        right = (160 + 3, 160 + 3 + length)
        frames.append([(left[0], right[1])] if t >= 5 else [left, right])
    tracks = link(tmp_path, movie(frames))
    assert [e for e in find_events(tracks) if e.kind == "B"] == []


def test_temporary_split(tmp_path):
    """Omnipose splits a cell for two frames: not a division (look-ahead)."""
    tracks = link(tmp_path, movie(dividing_cell(4, 6, fuse_at=6)))
    assert [list(tr.frames) for tr in tracks.values()] == [list(range(10))]


def test_unmergeable_split_keeps_mother(tmp_path):
    """A split too wide to merge back: the pieces must not lose their mother.

    A temporary split is merged only if the closed union is connected; a
    failed merge must not leave two motherless cells (SuperSegger does).
    """
    frames = movie(dividing_cell(4, 6, fuse_at=6, gap=5))
    tracks = link(tmp_path, frames)
    born = [tr for tr in tracks.values() if tr.frames[0] == 4]
    assert born and all(tr.mother for tr in born)


def test_small_neighbour_fuses(tmp_path):
    """Issue B: a small cell fuses with a large one, within the area tolerance.

    The assignment links the fused region to the large cell only, so
    SuperSegger's errorRez sees no 2 -> 1 and tries no split.
    """
    frames = []
    for t, length in enumerate(growing(150, 10)):
        big = (40, 40 + length)
        small = (40 + length + 3, 40 + length + 3 + 28)
        frames.append([(big[0], small[1])] if t >= 5 else [big, small])
    tracks = link(tmp_path, movie(frames))
    assert [e for e in find_events(tracks) if e.kind == "B"] == []
