"""The caches of assign.colonies and assign.neighbor_pairs follow in-place edits."""

import numpy as np
from scipy import ndimage

from nd2_to_cells.assign import colonies, find_neighbor_pairs, neighbor_pairs
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


def pairs_frame():
    labels = np.zeros((30, 40), np.int32)
    labels[5:10, 4:10] = 1
    labels[5:10, 11:17] = 2  # 1 px gap: within the 5x5 reach
    labels[20:25, 30:36] = 3
    return Frame(labels)


def test_neighbor_pairs_cached_and_reset():
    frame = pairs_frame()
    regs = [1, 2, 3]
    assert neighbor_pairs(frame, regs) == [(1, 2)]
    assert frame.pairs is not None
    delete_regions(frame, [2])
    assert frame.pairs is None
    assert neighbor_pairs(frame, regs) == []


def test_neighbor_pairs_match_binary_dilation():
    rng = np.random.default_rng(0)
    labels = np.zeros((60, 60), np.int32)
    for k in range(1, 25):
        y, x = rng.integers(0, 55, 2)
        labels[y : y + 4, x : x + 4][labels[y : y + 4, x : x + 4] == 0] = k
    frame = Frame(labels)
    regs = list(range(1, frame.num_regs + 1))
    expected = set()
    for k in regs:
        near = ndimage.binary_dilation(labels == k, structure=np.ones((5, 5), bool))
        expected |= {(k, int(u)) for u in np.unique(labels[near]) if u > k}
    assert set(find_neighbor_pairs(frame, regs, None)[0]) == expected
