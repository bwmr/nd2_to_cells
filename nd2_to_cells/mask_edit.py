"""Mask edits made during linking: ports of SuperSegger's merge2Regions and
missingSeg2to1 (bug-for-bug; see docs/supersegger_port.md).

Both functions edit ``frame.labels`` in place and return True when the mask
changed (MATLAB then re-runs the frame). ``frame.num_regs`` is deliberately
not refreshed, as in MATLAB.
"""

import numpy as np
from scipy import ndimage

from nd2_to_cells.assign import label_colmajor
from nd2_to_cells.imreg import register_masks, register_masks_v4

_SQUARE3 = np.ones((3, 3), dtype=bool)


def merge_regions(frame, list_merge: list[int], fixes=()) -> bool:
    """merge2Regions: replace the regions by their closed union if connected.

    The union is closed with a 3x3 square; the merge happens only if the
    result is one 8-connected component. It is written as num_regs + 1 with
    ``rl + labels``. MATLAB quirks kept: only the first two regions are
    cleared, so pixels of other labels inside the closed union become label
    sums, and num_regs is not refreshed between edits of one pass. Two edits
    in one pass can thus write the same label onto distant regions.

    With the "label_collision" fix, all listed regions are cleared and the
    union gets a label above every label in use; other cells keep their
    pixels.
    """
    labels = frame.labels
    union = np.isin(labels, list_merge)
    if not union.any():
        return False
    # Work on the union's bounding box plus 2 px: the closing cannot reach
    # further, so the result equals MATLAB's full-frame computation.
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
    rl = labels
    if "label_collision" in fixes:
        new = rl.max() + 1
        rl[union] = 0
        box = rl[y0:y1, x0:x1]
        box[(comp == 1) & (box == 0)] = new
        return True
    rl[rl == list_merge[0]] = 0
    rl[rl == list_merge[1]] = 0
    rl[y0:y1, x0:x1] += (comp == 1).astype(rl.dtype) * (frame.num_regs + 1)
    return True


def missing_seg_2to1(
    frame_c, reg_c, frame_r, regs_r, frame_f, regs_f, fixes=()
) -> bool:
    """missingSeg2to1: split a region that two previous cells collapsed into.

    The two previous cells (labelled 1 and 2) are registered onto the region;
    if the next frame also maps the region to two cells, those are registered
    too and the more consistent label pairing is used. The registered cells
    seed a watershed of the region. The region is replaced only if the result
    has maximum label 2 (MATLAB quirk: label 1 may be missing, in which case
    the region is just relabelled; with the "split_needs_two" fix that counts
    as a failure). The "seeded_split" fix replaces registration and createMask
    by _seeded_split. Returns True if the mask changed. The new
    labels are num_regs + 1 and + 2, not refreshed within a pass (MATLAB
    quirk); with the "label_collision" fix they lie above every label in use.
    """
    register = register_masks_v4 if "itk_v4" in fixes else register_masks
    rl = frame_c.labels
    mask_c = (rl == reg_c).astype(np.float64)
    if "seeded_split" in fixes:
        new = _seeded_split(mask_c > 0, frame_r.labels, regs_r)
        return new is not None and _write_split(
            frame_c, reg_c, new, fixes, int(new.max())
        )
    mask_r = (frame_r.labels == regs_r[0]) + 2.0 * (frame_r.labels == regs_r[1])
    reg_r = register(mask_r, mask_c)
    if len(regs_r) == 2 and len(regs_f) == 2:
        rlf = frame_f.labels
        mask_f = (rlf == regs_f[0]) + 2.0 * (rlf == regs_f[1])
        reg_rf = reg_r + 4 * register(mask_f, mask_c)
        n1 = np.count_nonzero(reg_rf == 2.5) + np.count_nonzero(reg_rf == 5)
        n2 = np.count_nonzero(reg_rf == 3) + np.count_nonzero(reg_rf == 4.5)
        if n1 > n2:
            new = _create_mask(mask_c, reg_rf, 2.5, 5)
        elif n2 > n1:
            new = _create_mask(mask_c, reg_rf, 3, 4.5)
        else:
            new = _create_mask(mask_c, reg_r, 0.5, 1)
    else:
        new = _create_mask(mask_c, reg_r, 0.5, 1)
    if new.max() != 2:
        return False
    if "split_needs_two" in fixes and not (new == 1).any():
        return False
    return _write_split(frame_c, reg_c, new, fixes)


def _seeded_split(region: np.ndarray, rl_r: np.ndarray, regs_r) -> np.ndarray | None:
    """Split region with the previous frame's cells as watershed seeds.

    Not MATLAB (the "seeded_split" fix). All previous cells mapped to the
    region (MATLAB uses the first two) are shifted together so that their
    joint centroid lies on the region's; the parts inside the region seed a
    watershed of its distance transform. Returns labels 1 ... n on the region
    (one per non-empty seed), or None if fewer than two seeds are left.
    """
    from skimage.segmentation import watershed

    prev = np.isin(rl_r, regs_r)
    shift = np.round(
        np.array(ndimage.center_of_mass(region)) - ndimage.center_of_mass(prev)
    ).astype(int)
    ys, xs = _window(region | prev, MASK_MARGIN + int(np.abs(shift).max()))
    reg = region[ys, xs]
    seeds = np.zeros(reg.shape, np.int32)
    n = 0
    for k in regs_r:
        cell = (rl_r[ys, xs] == k).astype(np.uint8)
        seed = (ndimage.shift(cell, shift, order=0) > 0) & reg & (seeds == 0)
        if seed.any():
            n += 1
            seeds[seed] = n
    if n < 2:
        return None
    dist = ndimage.distance_transform_edt(reg)
    out = np.zeros(region.shape, np.int32)
    out[ys, xs] = watershed(-dist, seeds, mask=reg)
    return out


def _write_split(frame_c, reg_c: int, new: np.ndarray, fixes, n: int = 2) -> bool:
    """Replace region reg_c by the pieces of ``new`` (labels 1 ... n)."""
    rl = frame_c.labels
    max_reg = int(rl.max()) if "label_collision" in fixes else frame_c.num_regs
    out = np.where(np.isin(new, np.arange(1, n + 1)), max_reg + new, 0)
    changed = rl.copy()
    changed[changed == reg_c] = 0
    changed += out.astype(changed.dtype)
    if np.array_equal(changed, rl):
        return False
    rl[...] = changed
    return True


MASK_MARGIN = 16  # px around the region and registered cells for the watershed


# ---------------------------------------------------------------------------
# createMask: watershed of the registered cells, with MATLAB's morphology
# ---------------------------------------------------------------------------


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


def _create_mask(mask_c, registered, num1, num2) -> np.ndarray:
    """createMask (missingSeg2to1.m): label mask_c by the two registered cells.

    Runs on a window around the region and both registered cells: the
    distance image is 0 outside the cells, so the morphology and the basins
    inside the region are local.
    """
    support = (mask_c > 0) | (registered == num1) | (registered == num2)
    return _create_mask_window(
        mask_c, registered, num1, num2, _window(support, MASK_MARGIN)
    )


def _create_mask_window(mask_c, registered, num1, num2, window) -> np.ndarray:
    ys, xs = window
    reg = registered[ys, xs]
    roi = mask_c[ys, xs] > 0
    sr12 = -(_bwdist_inside(reg == num1) + _bwdist_inside(reg == num2))
    mark = _imextendedmin(sr12, 2.0)
    dsr12 = _imimposemin(sr12, mark)
    out = np.zeros(mask_c.shape, dtype=np.int64)
    out[ys, xs] = _watershedpaw(dsr12, roi) * roi
    return out


def _bwdist_inside(bw: np.ndarray) -> np.ndarray:
    """bwdist(~bw) in single precision: distance to the nearest pixel not in bw."""
    out = np.zeros(bw.shape, dtype=np.float32)
    if not bw.any():
        return out
    if bw.all():
        out[:] = np.inf
        return out
    # Exact squared distances (integers), then a single-precision sqrt.
    ys, xs = ndimage.find_objects(bw.astype(np.int8))[0]
    y0, y1 = max(ys.start - 1, 0), min(ys.stop + 1, bw.shape[0])
    x0, x1 = max(xs.start - 1, 0), min(xs.stop + 1, bw.shape[1])
    crop = bw[y0:y1, x0:x1]
    if crop.all():  # nearest background lies outside the crop: use full frame
        y0, y1, x0, x1 = 0, bw.shape[0], 0, bw.shape[1]
        crop = bw
    idx = ndimage.distance_transform_edt(
        crop, return_distances=False, return_indices=True
    )
    rr, cc = np.indices(crop.shape)
    d2 = (idx[0] - rr).astype(np.int64) ** 2 + (idx[1] - cc).astype(np.int64) ** 2
    out[y0:y1, x0:x1] = np.sqrt(d2.astype(np.float32))
    return out


def _reconstruct(marker: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """imreconstruct (grayscale dilation, 8-connected), keeping float32."""
    from skimage.morphology import reconstruction

    return reconstruction(
        marker.astype(np.float64),
        mask.astype(np.float64),
        method="dilation",
        footprint=np.ones((3, 3)),
    ).astype(np.float32)


def _imextendedmin(img: np.ndarray, h: float) -> np.ndarray:
    """imextendedmin = imregionalmin(imhmin(img, h)) for single images."""
    comp = np.float32(1) - img  # imcomplement
    hmax = _reconstruct(comp - np.float32(h), comp)  # imhmax
    return _regional_min(np.float32(1) - hmax)


def _regional_min(img: np.ndarray) -> np.ndarray:
    """imregionalmin (8-connected) = imregionalmax(imcomplement(img)).

    The single-precision complement is taken first, as in the toolbox, since
    1 - x can merge values that differ in x.
    """
    from skimage.morphology import local_maxima

    comp = np.float32(1) - img
    if comp.min() == comp.max():
        # MATLAB: a constant image is one regional extremum; skimage finds none.
        return np.ones(img.shape, dtype=bool)
    return local_maxima(comp, connectivity=2, allow_borders=True)


def _imimposemin(img: np.ndarray, bw: np.ndarray) -> np.ndarray:
    """imimposemin(img, bw) for single images (toolbox algorithm)."""
    fm = np.where(bw, -np.inf, np.inf).astype(np.float32)
    rng = float(img.max()) - float(img.min())
    h = 0.1 if rng == 0 else rng * 0.001
    g = np.minimum(img + np.float32(h), fm)
    rec = _reconstruct(np.float32(1) - fm, np.float32(1) - g)
    return np.float32(1) - rec


PLATEAU_FAST = 100_000  # px of background plateau above which it is filled directly


def _watershed(img: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """MATLAB watershed (8-connected, ridge pixels 0) via scikit-image.

    scikit-image's ridge-line mode is very slow on large plateaus, which occur
    when the region is a label shared by distant blobs. Then only the cells
    and a 2-pixel ring of the background plateau are flooded; the rest of the
    plateau takes the label of the nearest labelled pixel (chessboard
    distance). That is what flooding a plateau does, except for where ridge
    pixels end up.
    """
    from skimage.segmentation import watershed

    plateau = (img == img.max()) & (markers == 0)
    if np.count_nonzero(plateau) < PLATEAU_FAST:
        return watershed(img, markers=markers, connectivity=2, watershed_line=True)
    core = ndimage.binary_dilation(~plateau, structure=np.ones((3, 3)), iterations=2)
    rl = watershed(img, markers=markers, connectivity=2, watershed_line=True, mask=core)
    seeds = rl > 0
    if seeds.any() and not core.all():
        iy, ix = ndimage.distance_transform_cdt(
            ~seeds, metric="chessboard", return_distances=False, return_indices=True
        )
        far = ~core
        rl[far] = rl[iy[far], ix[far]]
    return rl


def _watershedpaw(img: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """watershedpaw: watershed, then give each ridge pixel a neighbour's label.

    Only pixels in ``roi`` are filled (the caller multiplies by the region
    mask); each fill reads the unfilled watershed, so this equals MATLAB's
    full-frame loop there. Basins are numbered in column-major order of the
    regional minima, as MATLAB's watershed does.
    """
    minima, _ = label_colmajor(_regional_min(img))
    rl = _watershed(img, minima)
    out = rl.copy()
    h, w = img.shape
    rows, cols = np.nonzero((rl == 0) & roi)
    if rows.size == 0:
        return out
    d = np.array([-1, 0, 1])
    # 3x3 window in MATLAB order: rows yy_ = r + d, cols xx_ = c + d. MATLAB
    # quirk: index i is replaced by the centre in BOTH axes if either
    # yy_(i) or xx_(i) is out of bounds.
    yy = rows[:, None] + d[None, :]
    xx = cols[:, None] + d[None, :]
    oob = (yy < 0) | (yy >= h) | (xx < 0) | (xx >= w)
    yy = np.where(oob, rows[:, None], yy)
    xx = np.where(oob, cols[:, None], xx)
    # Column-major flatten of im(yy_, xx_): element j*3 + i is (yy[i], xx[j]).
    yi = np.tile(yy, (1, 3))
    xj = np.repeat(xx, 3, axis=1)
    vals = img[yi, xj]
    labs = rl[yi, xj]
    order = np.argsort(vals, axis=1, kind="stable")
    labs_sorted = np.take_along_axis(labs, order, axis=1)
    has = labs_sorted > 0
    if not has.any(axis=1).all():
        raise RuntimeError("watershedpaw: MATLAB errors here (no labelled neighbour)")
    first = np.argmax(has, axis=1)
    out[rows, cols] = labs_sorted[np.arange(rows.size), first]
    return out
