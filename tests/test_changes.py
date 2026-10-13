"""Where the linker departs from SuperSegger, on small hand-made frames."""

import numpy as np

from nd2_to_cells.link import Frame
from nd2_to_cells.mask_edit import merge_regions, missing_seg_2to1
from nd2_to_cells.track import _build_tracks


def test_merges_get_distinct_labels():
    """Two merges in one pass: SuperSegger writes both unions as num_regs + 1."""
    labels = np.zeros((20, 80), np.int32)
    labels[5:15, 2:10], labels[5:15, 11:19] = 1, 2  # pair 1, 1 px apart
    labels[5:15, 60:68], labels[5:15, 69:77] = 3, 4  # pair 2, far away
    frame = Frame(labels)
    assert merge_regions(frame, [1, 2])
    assert merge_regions(frame, [3, 4])
    assert sorted(set(np.unique(frame.labels)) - {0}) == [5, 6]


def test_merge_keeps_other_cells():
    """SuperSegger adds the new label onto a third cell inside the closed union."""
    labels = np.zeros((20, 40), np.int32)
    labels[5:15, 2:10], labels[5:15, 12:20] = 1, 2
    labels[9, 10:12] = 3  # a third cell in the gap
    frame = Frame(labels)
    merge_regions(frame, [1, 2])
    assert (frame.labels == 3).sum() == 2


def _states():
    """Cell 1 in frames 0 and 1 of three."""
    a = np.zeros((20, 40), np.int32)
    a[5:15, 2:18] = 1
    states = [Frame(a), Frame(a.copy()), Frame(np.zeros_like(a))]
    for fr in states[:2]:
        fr.ID[1], fr.birth[1] = 1, 1
    return states


def test_death_is_last_frame():
    """SuperSegger's death can be one frame after the cell's last frame."""
    assert _build_tracks(_states(), [0, 1, 2])[1].death == 2


def test_ghosts_are_dropped():
    """Labels without pixels get no cell file (SuperSegger writes them)."""
    states = _states()
    labels = states[1].labels.copy()
    labels[5:15, 30:38] = 3  # label 2 is unused: a ghost
    states[1] = Frame(labels)
    states[1].ID[1:4] = [1, 2, 3]
    states[1].birth[1:4] = 1
    assert sorted(_build_tracks(states, [0, 1, 2])) == [1, 3]


def test_split_three_cells():
    """Three previous cells fused into one region: three pieces, not two."""
    r = np.zeros((20, 70), np.int32)
    r[5:15, 2:20], r[5:15, 24:44], r[5:15, 48:68] = 1, 2, 3
    c = np.zeros_like(r)
    c[5:15, 2:68] = 1
    frame_c = Frame(c)
    assert missing_seg_2to1(frame_c, 1, Frame(r), [1, 2, 3])
    assert sorted(set(np.unique(frame_c.labels)) - {0}) == [2, 3, 4]


def test_split_follows_a_rotation():
    """Two cells end to end, fused and rotated by 8 degrees at t."""
    from scipy import ndimage

    from nd2_to_cells.mask_edit import _rigid_onto

    r = np.zeros((80, 140), np.int32)
    r[34:46, 20:68], r[34:46, 72:120] = 1, 2
    fused = ndimage.rotate((r > 0).astype(np.uint8), 8, reshape=False, order=0)
    fused = ndimage.binary_closing(fused > 0, np.ones((7, 7)))
    shifted = ndimage.shift(r, (2, 3), order=0)  # translation only
    moved = _rigid_onto(shifted, fused)
    assert np.count_nonzero((moved > 0) & fused) > 1.1 * np.count_nonzero(
        (r > 0) & fused
    )
    frame_c = Frame(fused.astype(np.int32))
    assert missing_seg_2to1(frame_c, 1, Frame(r), [1, 2])
    a, b = (np.count_nonzero(frame_c.labels == k) for k in (2, 3))
    assert 0.8 < a / b < 1.25
