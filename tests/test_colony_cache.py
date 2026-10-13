"""The colony cache of assign.colonies follows in-place edits of the labels."""

import numpy as np

from nd2_to_cells.assign import colonies
from nd2_to_cells.link import Frame, delete_regions
from nd2_to_cells.mask_edit import merge_regions


def fresh(labels):
    return colonies(Frame(labels.copy()))


def same(a, b):
    return np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1], equal_nan=True)


def test_matches_all_pixel_sums():
    labels = np.zeros((40, 50), np.int32)
    labels[5:12, 4:30] = 1
    labels[20:30, 10:12] = 2
    labels[22:26, 40:48] = 3
    lab, cent = fresh(labels)
    for k in (1, 2, 3):
        ys, xs = np.nonzero(lab == k)
        assert np.allclose(cent[k], (xs.mean() + 1, ys.mean() + 1))


def test_delete_regions_resets_cache():
    labels = np.zeros((40, 50), np.int32)
    labels[5:12, 4:30] = 1
    labels[20:30, 10:12] = 2
    frame = Frame(labels)
    before = colonies(frame)
    delete_regions(frame, [2])
    assert frame.colony is None
    after = colonies(frame)
    assert not same(before, after)
    assert same(after, fresh(frame.labels))


def test_merge_regions_resets_cache():
    labels = np.zeros((20, 30), np.int32)
    labels[5:10, 4:10] = 1
    labels[5:10, 11:17] = 2  # one background column between: closed by merge
    frame = Frame(labels)
    before = colonies(frame)
    assert merge_regions(frame, [1, 2])
    assert frame.colony is None
    after = colonies(frame)
    assert not same(before, after)
    assert same(after, fresh(frame.labels))
