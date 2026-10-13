"""The merge-event check (tests/merge_events.py) on hand-made tracks."""

import h5py
import numpy as np

from tests.merge_events import Track, find_events, load_tracks

SHAPE = (40, 200)


def box(y0, y1, x0, x1) -> np.ndarray:
    img = np.zeros(SHAPE, bool)
    img[y0:y1, x0:x1] = True
    return img


LEFT, RIGHT, BOTH = box(10, 20, 5, 35), box(10, 20, 40, 70), box(10, 20, 5, 70)


def track(tid: int, planes: dict[int, np.ndarray], mother=0, sister=0) -> Track:
    """A track stored like the HDF5 files: one crop box around all planes."""
    frames = sorted(planes)
    full = np.stack([planes[f] for f in frames], axis=2)
    ys, xs = np.nonzero(full.any(axis=2))
    y0, x0 = ys.min(), xs.min()
    mask = full[y0 : ys.max() + 1, x0 : xs.max() + 1]
    return Track(
        id=tid,
        key=f"cell{tid:07d}",
        frames=np.array(frames),
        mask=mask,
        x0=int(x0),
        y0=int(y0),
        mother=mother,
        sister=sister,
    )


def as_dict(*tracks: Track) -> dict[int, Track]:
    return {tr.id: tr for tr in tracks}


def test_neighbour_absorbed():
    t = track(1, {0: LEFT, 1: BOTH})
    x = track(2, {0: RIGHT})
    (e,) = find_events(as_dict(t, x))
    assert (e.kind, e.frame, e.cell, e.absorbed) == ("B", 1, t.key, x.key)
    assert e.absorbed_ends and e.area_after > 2 * e.area_before


def test_sister_absorbed():
    d1 = track(2, {1: LEFT, 2: BOTH}, mother=1, sister=3)
    d2 = track(3, {1: RIGHT}, mother=1, sister=2)
    (e,) = find_events(as_dict(track(1, {0: BOTH}), d1, d2))
    assert (e.kind, e.frame, e.cell, e.absorbed) == ("A", 2, d1.key, d2.key)


def test_larger_cell_keeps_its_id():
    """No area jump is needed: a big cell absorbing a smaller one counts."""
    big = box(10, 20, 5, 60)
    t = track(1, {0: big, 1: BOTH})
    x = track(2, {0: box(10, 20, 62, 70)})
    (e,) = find_events(as_dict(t, x))
    assert e.kind == "B" and e.area_after < 1.3 * e.area_before


def test_division_is_not_a_merge():
    d1 = track(2, {1: LEFT}, mother=1, sister=3)
    d2 = track(3, {1: RIGHT}, mother=1, sister=2)
    assert find_events(as_dict(track(1, {0: BOTH}), d1, d2)) == []


def test_moving_onto_a_neighbour_is_not_a_merge():
    t = track(1, {0: LEFT, 1: RIGHT})
    x = track(2, {0: RIGHT})
    assert find_events(as_dict(t, x)) == []


def test_small_overlap_is_not_a_merge():
    """Less than half of the neighbour is covered."""
    t = track(1, {0: LEFT, 1: box(10, 20, 5, 50)})
    x = track(2, {0: RIGHT})
    assert find_events(as_dict(t, x)) == []


def _write(g, tr: Track) -> None:
    g["frames"] = tr.frames
    g["mask"] = tr.mask
    g["r_offset"] = np.tile([tr.x0, tr.y0], (len(tr.frames), 1))
    g["motherID"] = np.int64(tr.mother)
    g["sisterID"] = np.int64(tr.sister)


def test_load_both_layouts_and_skip_ghosts(tmp_path):
    t = track(1, {0: LEFT, 1: BOTH})
    x = track(2, {0: RIGHT})
    ghost = track(3, {0: LEFT})
    ghost.mask[:] = False
    per_cell, consolidated = tmp_path / "a", tmp_path / "b"
    per_cell.mkdir()
    consolidated.mkdir()
    with h5py.File(consolidated / "cells.h5", "w") as f:
        for tr in (t, x, ghost):
            with h5py.File(per_cell / f"{tr.key}.h5", "w") as g:
                _write(g, tr)
            _write(f.create_group(tr.key), tr)
    for cell_dir in (per_cell, consolidated):
        tracks = load_tracks(cell_dir)
        assert sorted(tracks) == [1, 2]
        assert [e.kind for e in find_events(tracks)] == ["B"]


def test_distant_cell_absorbed():
    """The merged object spans cells far apart (MATLAB's label collision)."""
    far = box(10, 20, 150, 190)
    t = track(1, {0: LEFT, 1: LEFT | far})
    x = track(2, {0: far})
    (e,) = find_events(as_dict(t, x))
    assert e.kind == "far" and e.gap == 115
