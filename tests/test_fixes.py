"""Each opt-in fix (track.FIXES) against the MATLAB behaviour it replaces."""

import numpy as np

from nd2_to_cells.assign import assign
from nd2_to_cells.link import Frame
from nd2_to_cells.mask_edit import merge_regions


def test_label_collision():
    """Two merges in one pass: MATLAB writes both unions as num_regs + 1."""
    labels = np.zeros((20, 80), np.int32)
    labels[5:15, 2:10], labels[5:15, 11:19] = 1, 2  # pair 1, 1 px apart
    labels[5:15, 60:68], labels[5:15, 69:77] = 3, 4  # pair 2, far away
    for fixes, expect in (((), [5]), (("label_collision",), [5, 6])):
        frame = Frame(labels.copy())
        assert merge_regions(frame, [1, 2], fixes)
        assert merge_regions(frame, [3, 4], fixes)
        assert sorted(set(np.unique(frame.labels)) - {0}) == expect


def test_label_collision_keeps_other_cells():
    """MATLAB adds the new label onto a third cell inside the closed union."""
    labels = np.zeros((20, 40), np.int32)
    labels[5:15, 2:10], labels[5:15, 12:20] = 1, 2
    labels[9, 10:12] = 3  # a third cell in the gap
    base, fixed = Frame(labels.copy()), Frame(labels.copy())
    merge_regions(base, [1, 2])
    merge_regions(fixed, [1, 2], ("label_collision",))
    assert (base.labels == 3 + 4).any()  # label sum
    assert (fixed.labels == 3).sum() == 2


def test_area_penalty():
    """A link whose area changes by more than 60 %: penalty 50 vs 1000."""
    c = np.zeros((40, 80), np.int32)
    c[15:25, 10:30] = 1
    f = np.zeros_like(c)
    f[15:25, 10:70] = 1  # three times the area
    base = assign(Frame(c), Frame(f), True, -0.2, 0.3)
    fixed = assign(Frame(c), Frame(f), True, -0.2, 0.3, ("area_penalty",))
    assert np.nanmin(fixed.cost) - np.nanmin(base.cost) == 950


def test_split_needs_two(monkeypatch):
    """MATLAB accepts a 'split' whose result holds only label 2 (a relabel)."""
    import nd2_to_cells.mask_edit as mask_edit

    monkeypatch.setattr(mask_edit, "register_masks", lambda moving, fixed: moving)
    monkeypatch.setattr(
        mask_edit, "_create_mask", lambda mask_c, *a: 2 * (mask_c > 0).astype(int)
    )
    r = np.zeros((20, 40), np.int32)
    r[5:15, 2:18], r[5:15, 20:38] = 1, 2
    c = np.zeros_like(r)
    c[5:15, 2:38] = 1
    for fixes, expect in (((), True), (("split_needs_two",), False)):
        frame_c = Frame(c.copy())
        ok = mask_edit.missing_seg_2to1(frame_c, 1, Frame(r), [1, 2], None, [], fixes)
        assert ok is expect


def _states(death_last):
    """Cell 1 in frames 0 and 1 of three; its death field in frame 1 is given."""
    a = np.zeros((20, 40), np.int32)
    a[5:15, 2:18] = 1
    states = [Frame(a), Frame(a.copy()), Frame(np.zeros_like(a))]
    for i, fr in enumerate(states[:2]):
        fr.ID[1], fr.birth[1], fr.death[1] = 1, 1, i + 1
    states[1].death[1] = death_last
    return states


def test_death_last_frame():
    """MATLAB's death can be one frame after the cell's last frame."""
    from nd2_to_cells.track import _build_tracks

    base = _build_tracks(_states(death_last=3), [0, 1, 2])
    fixed = _build_tracks(_states(death_last=3), [0, 1, 2], ("death_last_frame",))
    assert base[1].death == 3 and fixed[1].death == 2


def test_drop_ghosts():
    from nd2_to_cells.track import _build_tracks

    states = _states(death_last=2)
    labels = states[1].labels.copy()
    labels[5:15, 30:38] = 3  # label 2 is unused: a ghost
    states[1] = Frame(labels)
    states[1].ID[1:4] = [1, 2, 3]
    states[1].birth[1:4] = 1
    states[1].death[1:4] = 2
    assert sorted(_build_tracks(states, [0, 1, 2])) == [1, 2, 3]
    assert sorted(_build_tracks(states, [0, 1, 2], ("drop_ghosts",))) == [1, 3]


def test_seeded_split_three_cells():
    """Three previous cells fused into one region: three pieces, not two."""
    from nd2_to_cells.mask_edit import missing_seg_2to1

    r = np.zeros((20, 70), np.int32)
    r[5:15, 2:20], r[5:15, 24:44], r[5:15, 48:68] = 1, 2, 3
    c = np.zeros_like(r)
    c[5:15, 2:68] = 1
    frame_c = Frame(c.copy())
    assert missing_seg_2to1(
        frame_c, 1, Frame(r), [1, 2, 3], None, [], ("seeded_split",)
    )
    assert sorted(set(np.unique(frame_c.labels)) - {0}) == [2, 3, 4]
