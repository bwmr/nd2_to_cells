"""Mask edits made during linking: merges and splits of regions.

Derived from SuperSegger's merge2Regions and missingSeg2to1 (see
docs/linking.md). Both functions edit ``frame.labels`` in place and return
True when the mask changed; the frame is then re-run, as in SuperSegger. New
labels lie above every label in use. ``frame.num_regs`` is not refreshed.
"""

import numpy as np
from scipy import ndimage

from nd2_to_cells.assign import label_colmajor

_SQUARE3 = np.ones((3, 3), dtype=bool)
MASK_MARGIN = 16  # px around the region and the moved cells for the watershed
RIGID_ANGLES = (0, -2, 2, -4, 4, -6, 6, -8, 8, -10, 10)  # degrees, split search


def merge_regions(frame, list_merge: list[int]) -> bool:
    """merge2Regions: replace the regions by their closed union if connected.

    The union is closed with a 3x3 square; the merge happens only if the
    result is one 8-connected component. All listed regions are cleared, and
    the union gets a label above every label in use, written onto background
    only, so other cells keep their pixels. (SuperSegger writes num_regs + 1
    as a label sum, so two edits of one pass can give distant regions the
    same label.)
    """
    labels = frame.labels
    union = np.isin(labels, list_merge)
    if not union.any():
        return False
    # Work on the union's bounding box plus 2 px: the closing cannot reach
    # further.
    ys, xs = ndimage.find_objects(union.astype(np.int8))[0]
    y0, y1 = max(ys.start - 2, 0), min(ys.stop + 2, labels.shape[0])
    x0, x1 = max(xs.start - 2, 0), min(xs.stop + 2, labels.shape[1])
    crop = union[y0:y1, x0:x1]
    dilated = ndimage.binary_dilation(crop, structure=_SQUARE3)
    # imerode pads with +Inf: pixels at the image border survive.
    closed = ndimage.binary_erosion(dilated, structure=_SQUARE3, border_value=1)
    comp, n = label_colmajor(closed)
    if n != 1:
        return False
    if len(list_merge) < 2:
        raise RuntimeError("merge2Regions: MATLAB errors here (list_merge(2))")
    new = labels.max() + 1
    labels[union] = 0
    box = labels[y0:y1, x0:x1]
    box[(comp == 1) & (box == 0)] = new
    return True


def missing_seg_2to1(frame_c, reg_c: int, frame_r, regs_r: list[int]) -> bool:
    """Split region reg_c, into which the previous cells regs_r fused.

    Replaces missingSeg2to1, whose affine registration of the previous cells
    rarely cuts a fused region (docs/linking.md). Here all previous cells are
    moved onto the region together: their joint centroid onto the region's,
    rotated by the angle in RIGID_ANGLES at which they overlap it most. Their
    parts inside the region seed a watershed of its distance transform.
    Returns True if the mask changed.
    """
    new = _seeded_split(frame_c.labels == reg_c, frame_r.labels, regs_r)
    return new is not None and _write_split(frame_c, reg_c, new, int(new.max()))


def _seeded_split(region: np.ndarray, rl_r: np.ndarray, regs_r) -> np.ndarray | None:
    """Labels 1 ... n on ``region``, one per previous cell that seeds it.

    Returns None if fewer than two cells leave a seed inside the region.
    """
    from skimage.segmentation import watershed

    prev = np.isin(rl_r, regs_r)
    shift = np.round(
        np.array(ndimage.center_of_mass(region)) - ndimage.center_of_mass(prev)
    ).astype(int)
    # Room for the shift and for the ends of the rotated cells.
    extent = int(np.ptp(np.nonzero(prev), axis=1).max()) + 1
    margin = MASK_MARGIN + int(np.abs(shift).max())
    margin += int(np.ceil(extent * np.sin(np.deg2rad(max(RIGID_ANGLES)))))
    ys, xs = _window(region | prev, margin)
    reg = region[ys, xs]
    moved = _rigid_onto(np.where(prev[ys, xs], rl_r[ys, xs], 0), reg)
    seeds = np.zeros(reg.shape, np.int32)
    n = 0
    for k in regs_r:
        seed = (moved == k) & reg & (seeds == 0)
        if seed.any():
            n += 1
            seeds[seed] = n
    if n < 2:
        return None
    dist = ndimage.distance_transform_edt(reg)
    out = np.zeros(region.shape, np.int32)
    out[ys, xs] = watershed(-dist, seeds, mask=reg)
    return out


def _rigid_onto(labels: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Move ``labels`` onto ``region``: centroid onto centroid, best rotation.

    The rotation about the labels' joint centroid is chosen from RIGID_ANGLES
    by the overlap of the moved labels with the region; ties keep the smaller
    angle. Nearest-neighbour resampling keeps the labels.
    """
    c_in = np.array(ndimage.center_of_mass(labels > 0))
    c_out = np.array(ndimage.center_of_mass(region))
    best, best_overlap = None, -1
    for angle in RIGID_ANGLES:
        t = np.deg2rad(angle)
        rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        # output pixel o reads input pixel rot @ (o - c_out) + c_in
        moved = ndimage.affine_transform(
            labels, rot, offset=c_in - rot @ c_out, order=0
        )
        overlap = np.count_nonzero((moved > 0) & region)
        if overlap > best_overlap:
            best, best_overlap = moved, overlap
    return best


def _write_split(frame_c, reg_c: int, new: np.ndarray, n: int) -> bool:
    """Replace region reg_c by the pieces of ``new`` (labels 1 ... n)."""
    rl = frame_c.labels
    out = np.where(np.isin(new, np.arange(1, n + 1)), int(rl.max()) + new, 0)
    changed = rl.copy()
    changed[changed == reg_c] = 0
    changed += out.astype(changed.dtype)
    if np.array_equal(changed, rl):
        return False
    rl[...] = changed
    return True


def _window(support: np.ndarray, margin: int) -> tuple[slice, slice]:
    """Bounding box of ``support`` plus ``margin``, clipped to the image."""
    h, w = support.shape
    if not support.any():
        return slice(0, h), slice(0, w)
    ys, xs = ndimage.find_objects(support.astype(np.int8))[0]
    return (
        slice(max(ys.start - margin, 0), min(ys.stop + margin, h)),
        slice(max(xs.start - margin, 0), min(xs.stop + margin, w)),
    )
