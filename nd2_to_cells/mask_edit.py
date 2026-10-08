"""Mask edits made during linking: ports of SuperSegger's merge2Regions and
missingSeg2to1 (bug-for-bug; see docs/supersegger_port.md).

Both functions edit ``frame.labels`` in place and return True when the mask
changed (MATLAB then re-runs the frame). ``frame.num_regs`` is deliberately
not refreshed, as in MATLAB.
"""

import numpy as np
from scipy import ndimage

from nd2_to_cells.assign import label_colmajor

_SQUARE3 = np.ones((3, 3), dtype=bool)


def merge_regions(frame, list_merge: list[int]) -> bool:
    """merge2Regions: replace the regions by their closed union if connected.

    The union is closed with a 3x3 square; the merge happens only if the
    result is one 8-connected component. It is written as num_regs + 1 with
    ``rl + labels``. MATLAB quirks kept: only the first two regions are
    cleared, so pixels of other labels inside the closed union become label
    sums, and num_regs is not refreshed between edits of one pass.
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
    rl[rl == list_merge[0]] = 0
    rl[rl == list_merge[1]] = 0
    rl[y0:y1, x0:x1] += (comp == 1).astype(rl.dtype) * (frame.num_regs + 1)
    return True


def missing_seg_2to1(frame_c, reg_c, frame_r, regs_r, frame_f, regs_f) -> bool:
    """missingSeg2to1: split a region that two previous cells collapsed into.

    The two previous cells (labelled 1 and 2) are registered onto the region;
    if the next frame also maps the region to two cells, those are registered
    too and the more consistent label pairing is used. The registered cells
    seed a watershed of the region. The region is replaced only if the result
    has maximum label 2 (MATLAB quirk: label 1 may be missing, in which case
    the region is just relabelled). Returns True if the mask changed.
    """
    rl = frame_c.labels
    mask_c = (rl == reg_c).astype(np.float64)
    mask_r = (frame_r.labels == regs_r[0]) + 2.0 * (frame_r.labels == regs_r[1])
    reg_r = register_masks(mask_r, mask_c)
    if len(regs_r) == 2 and len(regs_f) == 2:
        rlf = frame_f.labels
        mask_f = (rlf == regs_f[0]) + 2.0 * (rlf == regs_f[1])
        reg_rf = reg_r + 4 * register_masks(mask_f, mask_c)
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
    max_reg = frame_c.num_regs
    out = np.where(new == 1, max_reg + 1, np.where(new == 2, max_reg + 2, 0))
    changed = rl.copy()
    changed[changed == reg_c] = 0
    changed += out.astype(changed.dtype)
    if np.array_equal(changed, rl):
        return False
    rl[...] = changed
    return True


# ---------------------------------------------------------------------------
# registerMasks_monointensity (imregtform 'monomodal', affine) via SimpleITK
# ---------------------------------------------------------------------------

GAUSS_SIGMA = 1.013889
PYRAMID_SHRINK = (4, 2, 1)
PYRAMID_SIGMA = (2.0, 1.0, 0.0)
REG_MARGIN = 64  # px around both masks for registration
MASK_MARGIN = 16  # px around the region and registered cells for the watershed


def imgaussfilt(img: np.ndarray, sigma: float) -> np.ndarray:
    """imgaussfilt: kernel size 2*ceil(2*sigma)+1, replicate padding."""
    half = int(np.ceil(2 * sigma))
    x = np.arange(-half, half + 1, dtype=np.float64)
    k = np.exp(-(x**2) / (2 * sigma**2))
    k /= k.sum()
    out = ndimage.correlate1d(img, k, axis=0, mode="nearest")
    return ndimage.correlate1d(out, k, axis=1, mode="nearest")


def _window(support: np.ndarray, margin: int, align: int = 1) -> tuple[slice, slice]:
    """Bounding box of ``support`` plus ``margin``, edges on multiples of align."""
    h, w = support.shape
    if not support.any():
        return slice(0, h), slice(0, w)
    ys, xs = ndimage.find_objects(support.astype(np.int8))[0]
    y0 = max(ys.start - margin, 0) // align * align
    x0 = max(xs.start - margin, 0) // align * align
    y1 = min(-(-(ys.stop + margin) // align) * align, h)
    x1 = min(-(-(xs.stop + margin) // align) * align, w)
    return slice(y0, y1), slice(x0, x1)


def register_masks(moving: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """registerMasks_monointensity: affine-register ``moving`` onto ``fixed``.

    Returns the moving image, normalised to [0, 1], warped into the fixed
    frame with nearest-neighbour interpolation. Runs on a window around both
    masks (see _register); outside it the result is 0.
    """
    window = _window((moving != 0) | (fixed != 0), REG_MARGIN, align=4)
    return _register(moving, fixed, window)


def _register(moving: np.ndarray, fixed: np.ndarray, window) -> np.ndarray:
    """Registration on ``window`` that behaves like MATLAB's full-frame call.

    MATLAB registers the whole frame. On a window that keeps full-frame world
    coordinates and optimizer scales, starts on a multiple of the coarsest
    shrink factor and holds all non-zero pixels plus a margin, the optimizer
    takes the same steps: its step direction is normalised, and the gradient
    tolerance is rescaled because MeanSquares averages over all pixels.
    """
    import SimpleITK as sitk

    moving = moving.astype(np.float64)
    lo, hi = moving.min(), moving.max()
    moving = np.zeros_like(moving) if lo == hi else (moving - lo) / (hi - lo)
    h, w = fixed.shape
    max_translation = np.hypot(min(w, h), max(w, h))
    ys, xs = window
    n_crop = (ys.stop - ys.start) * (xs.stop - xs.start)

    def to_sitk(a):
        im = sitk.GetImageFromArray(np.ascontiguousarray(a, dtype=np.float64))
        # MATLAB world coordinates: pixel (1, 1) of the full frame is at (1, 1).
        im.SetOrigin((xs.start + 1.0, ys.start + 1.0))
        return im

    fixed_im = to_sitk(imgaussfilt(fixed[ys, xs].astype(np.float64), GAUSS_SIGMA))
    moving_im = to_sitk(imgaussfilt(moving[ys, xs], GAUSS_SIGMA))
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMeanSquares()
    reg.SetMetricSamplingStrategy(reg.NONE)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=6.25e-02,
        minStep=1e-05,
        numberOfIterations=100,
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-04 * (h * w) / n_crop,
        estimateLearningRate=reg.Never,
    )
    reg.SetOptimizerScales([1, 1, 1, 1, 1 / max_translation, 1 / max_translation])
    reg.SetShrinkFactorsPerLevel(list(PYRAMID_SHRINK))
    reg.SetSmoothingSigmasPerLevel(list(PYRAMID_SIGMA))
    reg.SetInitialTransform(sitk.AffineTransform(2), inPlace=False)
    tform = reg.Execute(fixed_im, moving_im)
    warped = sitk.Resample(
        to_sitk(moving[ys, xs]), fixed_im, tform, sitk.sitkNearestNeighbor, 0.0
    )
    out = np.zeros_like(moving)
    out[ys, xs] = sitk.GetArrayFromImage(warped)
    return out


# ---------------------------------------------------------------------------
# createMask: watershed of the registered cells, with MATLAB's morphology
# ---------------------------------------------------------------------------


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

    return local_maxima(np.float32(1) - img, connectivity=2, allow_borders=True)


def _imimposemin(img: np.ndarray, bw: np.ndarray) -> np.ndarray:
    """imimposemin(img, bw) for single images (toolbox algorithm)."""
    fm = np.where(bw, -np.inf, np.inf).astype(np.float32)
    rng = float(img.max()) - float(img.min())
    h = 0.1 if rng == 0 else rng * 0.001
    g = np.minimum(img + np.float32(h), fm)
    rec = _reconstruct(np.float32(1) - fm, np.float32(1) - g)
    return np.float32(1) - rec


def _watershedpaw(img: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """watershedpaw: watershed, then give each ridge pixel a neighbour's label.

    Only pixels in ``roi`` are filled (the caller multiplies by the region
    mask); each fill reads the unfilled watershed, so this equals MATLAB's
    full-frame loop there. Basins are numbered in column-major order of the
    regional minima, as MATLAB's watershed does.
    """
    from skimage.segmentation import watershed

    minima, _ = label_colmajor(_regional_min(img))
    rl = watershed(img, markers=minima, connectivity=2, watershed_line=True)
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
