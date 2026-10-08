"""Affine mask registration: port of SuperSegger's registerMasks_monointensity.

MATLAB's imregtform(..., 'affine', optimizer, metric, 'PyramidLevels', 3)
with the 'monomodal' configuration runs ITK's v3 registration framework
inside a builtin. This module re-implements that path; each piece was checked
against MATLAB R2026b on SuperSegger split calls:

- pyramid: ITK MultiResolutionPyramidImageFilter (discrete Gaussian of
  variance (f/2)^2, maximum error 0.1, on every level including f = 1; linear
  resampling at index (f-1)/2 + f*i)
- metric: mean squares over all fixed pixels whose mapped point lies inside
  the moving image; the moving-image gradient comes from a recursive Gaussian
  (sigma = pixel spacing, normalised across scale) looked up at the nearest
  pixel
- transform: ITK affine about the fixed image centre; MATLAB's tform is its
  inverse
- optimizer: regular-step gradient descent, scales applied once, with
  MATLAB's per-level step schedule (maximum step x f^2, minimum step x 10 per
  level above the finest)

MATLAB sums the metric over threads, so its own result depends slightly on
the thread count (SuperSegger runs on Windows use one thread). The metric is
only evaluated where the masks can contribute, with the full-frame pixel count,
so results do not depend on the frame size.
"""

import math

import numpy as np
from scipy import ndimage, special

GAUSS_SIGMA = 1.013889  # pre-filter in registerMasks_monointensity
MAX_STEP = 6.25e-02
MIN_STEP = 1.0e-05
MAX_ITER = 100
RELAXATION = 0.5
GRAD_TOL = 1.0e-04
PYRAMID = (4, 2, 1)  # shrink factor per level, coarse to fine
MARGIN = 64  # px of zero padding kept around each mask (a multiple of 4)


def register_masks(moving: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """registerMasks_monointensity(moving, fixed).RegisteredImage.

    ``moving`` is normalised to [0, 1], affinely registered onto ``fixed``
    and warped into the fixed frame with nearest-neighbour interpolation.
    """
    moving = moving.astype(np.float64)
    lo, hi = moving.min(), moving.max()
    moving = np.zeros_like(moving) if lo == hi else (moving - lo) / (hi - lo)
    params = register(moving, fixed.astype(np.float64))
    return warp(moving, params)


def register(
    moving: np.ndarray, fixed: np.ndarray, max_iter: int = MAX_ITER, trace=None
):
    """Optimise the ITK affine parameters [a00 a01 a10 a11 t0 t1].

    ``moving`` must already be normalised. If ``trace`` is a list, the metric
    value of every iteration is appended as (level, value).
    """
    h, w = fixed.shape
    centre = ((w + 1) / 2, (h + 1) / 2)
    mt = math.hypot(min(w, h), max(w, h))
    scales = np.array([1, 1, 1, 1, 1 / mt, 1 / mt])
    fwin = _window(fixed != 0, MARGIN)
    mwin = _window(moving != 0, MARGIN)
    fi = imgaussfilt(fixed[fwin], GAUSS_SIGMA)
    mi = imgaussfilt(moving[mwin], GAUSS_SIGMA)
    p = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    n_levels = len(PYRAMID)
    for level, f in enumerate(PYRAMID):
        fix = _Level(fi, fwin, f, (h, w))
        mov = _Level(mi, mwin, f, (h, w), gradient=True)
        max_step = MAX_STEP * f * f
        min_step = MIN_STEP * 10 ** (n_levels - 1 - level)
        p = _optimize(
            fix, mov, p, centre, scales, max_step, min_step, max_iter, trace, level
        )
    return p


def to_matlab_tform(p, shape) -> np.ndarray:
    """MATLAB affine2d.T (moving -> fixed) for ITK parameters ``p``."""
    h, w = shape
    a = np.array([[p[0], p[1]], [p[2], p[3]]])
    c = np.array([(w + 1) / 2, (h + 1) / 2])
    off = p[4:] + c - a @ c
    inv = np.linalg.inv(a)
    t = np.eye(3)
    t[:2, :2] = inv.T
    t[2, :2] = -(inv @ off)
    return t


def warp(moving: np.ndarray, p) -> np.ndarray:
    """imwarp(moving, tform, 'nearest', 'OutputView', fixed frame)."""
    h, w = moving.shape
    out = np.zeros_like(moving)
    support = moving != 0
    if not support.any():
        return out
    a, off = _affine(p, ((w + 1) / 2, (h + 1) / 2))
    ys, xs = ndimage.find_objects(support.astype(np.int8))[0]
    # Fixed pixels that map into the moving support (preimage bounding box).
    (r0, r1), (c0, c1) = _preimage(
        a, off, (xs.start, xs.stop + 1), (ys.start, ys.stop + 1)
    )
    r0, c0 = max(r0 - 2, 0), max(c0 - 2, 0)
    r1, c1 = min(r1 + 2, h), min(c1 + 2, w)
    if r0 >= r1 or c0 >= c1:
        return out
    yy, xx = np.meshgrid(
        np.arange(r0, r1) + 1.0, np.arange(c0, c1) + 1.0, indexing="ij"
    )
    sx = a[0, 0] * xx + a[0, 1] * yy + off[0]
    sy = a[1, 0] * xx + a[1, 1] * yy + off[1]
    ix = np.floor(sx + 0.5).astype(np.int64) - 1
    iy = np.floor(sy + 0.5).astype(np.int64) - 1
    ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    vals = np.zeros(sx.shape)
    vals[ok] = moving[iy[ok], ix[ok]]
    out[r0:r1, c0:c1] = vals
    return out


def imgaussfilt(img: np.ndarray, sigma: float) -> np.ndarray:
    """imgaussfilt: kernel size 2*ceil(2*sigma)+1, replicate padding."""
    half = int(np.ceil(2 * sigma))
    x = np.arange(-half, half + 1, dtype=np.float64)
    k = np.exp(-(x**2) / (2 * sigma**2))
    k /= k.sum()
    out = ndimage.correlate1d(img, k, axis=0, mode="nearest")
    return ndimage.correlate1d(out, k, axis=1, mode="nearest")


# ---------------------------------------------------------------------------
# Optimizer and metric
# ---------------------------------------------------------------------------


def _affine(p, centre):
    """ITK MatrixOffsetTransformBase: x' = A x + offset, offset = t + c - A c."""
    a = np.array([[p[0], p[1]], [p[2], p[3]]])
    off = np.array(
        [
            p[4] + centre[0] - a[0, 0] * centre[0] - a[0, 1] * centre[1],
            p[5] + centre[1] - a[1, 0] * centre[0] - a[1, 1] * centre[1],
        ]
    )
    return a, off


def _preimage(a, off, xr, yr):
    """Bounding box (rows, cols) of world pixels mapped into a world box.

    ``xr`` and ``yr`` are 0-based [start, stop) pixel ranges of the target;
    the result is a 0-based [start, stop) range of fixed pixels.
    """
    inv = np.linalg.inv(a)
    corners = np.array(
        [[x + 1.0, y + 1.0] for x in (xr[0] - 1, xr[1]) for y in (yr[0] - 1, yr[1])]
    )
    pts = (corners - off) @ inv.T
    c0, c1 = math.floor(pts[:, 0].min()) - 1, math.ceil(pts[:, 0].max())
    r0, r1 = math.floor(pts[:, 1].min()) - 1, math.ceil(pts[:, 1].max())
    return (r0, r1), (c0, c1)


def _optimize(fix, mov, p, centre, scales, max_step, min_step, max_iter, trace, level):
    """ITK RegularStepGradientDescentOptimizer (minimising)."""
    step = max_step
    grad = np.zeros(6)
    for _ in range(max_iter):
        prev = grad
        value, grad = _metric(fix, mov, p, centre)
        if trace is not None:
            trace.append((level, value))
        tg = grad / scales
        ptg = prev / scales
        mag = math.sqrt(sum(float(x) * float(x) for x in tg))
        if mag < GRAD_TOL:
            break
        if sum(float(x) * float(y) for x, y in zip(tg, ptg)) < 0:
            step *= RELAXATION
        if step < min_step:
            break
        factor = -1.0 * step / mag
        p = p + tg * factor
    return p


def _metric(fix, mov, p, centre):
    """MeanSquaresImageToImageMetric value and derivative at parameters p.

    Sums run over all fixed pixels that can have a non-zero difference: those
    in blocks holding fixed-image signal or mapping into the moving signal.
    Elsewhere both images are zero, so the skipped terms are exactly zero; the
    pixel count n is the full-frame one.
    """
    f = fix.f
    a, off = _affine(p, centre)
    win_x = (mov.j0 * f, (mov.j0 + mov.w) * f)
    win_y = (mov.i0 * f, (mov.i0 + mov.h) * f)
    (r0, r1), (c0, c1) = _preimage(a, off, win_x, win_y)
    i0 = max(min(fix.i0, (r0 - fix.f) // f), 0)
    i1 = min(max(fix.i0 + fix.h, r1 // f + 2), fix.H)
    j0 = max(min(fix.j0, (c0 - fix.f) // f), 0)
    j1 = min(max(fix.j0 + fix.w, c1 // f + 2), fix.W)
    o = fix.origin
    i, j = _active_pixels(fix, mov, a, off, i0, i1, j0, j1)
    yy = o + f * i.astype(np.float64)
    xx = o + f * j.astype(np.float64)
    tx = (a[0, 0] * xx + a[0, 1] * yy) + off[0]
    ty = (a[1, 0] * xx + a[1, 1] * yy) + off[1]
    ci = (ty - o) / f
    cj = (tx - o) / f
    inside = (cj >= -0.5) & (cj < mov.W - 0.5) & (ci >= -0.5) & (ci < mov.H - 0.5)
    fv = fix.lookup(fix.img, i[inside], j[inside])
    ci, cj = ci[inside], cj[inside]
    diff = mov.interpolate(ci, cj) - fv
    gy, gx = mov.gradient_at(np.floor(ci + 0.5), np.floor(cj + 0.5))
    xc = xx[inside] - centre[0]
    yc = yy[inside] - centre[1]
    n = _count_inside(a, off, fix, mov)
    d = 2.0 * diff
    deriv = np.array(
        [
            np.sum(d * xc * gx),
            np.sum(d * yc * gx),
            np.sum(d * xc * gy),
            np.sum(d * yc * gy),
            np.sum(d * gx),
            np.sum(d * gy),
        ]
    )
    return float(np.sum(diff * diff)) / n, deriv / n


BLOCK = 16  # level pixels per side of the blocks used to skip empty regions


def _active_pixels(fix, mov, a, off, i0, i1, j0, j1):
    """Level indices (i, j) of fixed pixels in blocks that can contribute."""
    f, o = fix.f, fix.origin
    bi = np.arange(i0, i1, BLOCK)
    bj = np.arange(j0, j1, BLOCK)
    bi0, bj0 = np.meshgrid(bi, bj, indexing="ij")
    bi1, bj1 = np.minimum(bi0 + BLOCK, i1), np.minimum(bj0 + BLOCK, j1)
    hit = fix.any_signal(bi0, bi1, bj0, bj1, dilate=0)
    # Moving-image index range covered by each block's mapped corners.
    cis, cjs = [], []
    for ri in (bi0, bi1 - 1):
        for rj in (bj0, bj1 - 1):
            y, x = o + f * ri, o + f * rj
            cis.append(((a[1, 0] * x + a[1, 1] * y) + off[1] - o) / f)
            cjs.append(((a[0, 0] * x + a[0, 1] * y) + off[0] - o) / f)
    cis, cjs = np.stack(cis), np.stack(cjs)
    m_i0 = np.floor(cis.min(axis=0)).astype(np.int64) - 1
    m_i1 = np.ceil(cis.max(axis=0)).astype(np.int64) + 2
    m_j0 = np.floor(cjs.min(axis=0)).astype(np.int64) - 1
    m_j1 = np.ceil(cjs.max(axis=0)).astype(np.int64) + 2
    hit |= mov.any_signal(m_i0, m_i1, m_j0, m_j1, dilate=2)
    full = np.repeat(np.repeat(hit, BLOCK, axis=0), BLOCK, axis=1)
    ii, jj = np.nonzero(full[: i1 - i0, : j1 - j0])
    return ii + i0, jj + j0


def _count_inside(a, off, fix, mov) -> int:
    """Number of full-frame fixed level pixels that map inside the moving image."""
    f, o = fix.f, fix.origin
    y = o + f * np.arange(fix.H, dtype=np.float64)
    lo = np.full(fix.H, -np.inf)
    hi = np.full(fix.H, np.inf)
    # Continuous moving index along each axis is linear in the column j:
    # c(j) = (a_r0 * (o + f j) + a_r1 * y + off_r - o) / f, constrained to
    # [-0.5, size - 0.5).
    for row, size in ((0, mov.W), (1, mov.H)):
        slope = a[row, 0]
        base = (a[row, 0] * o + a[row, 1] * y + off[row] - o) / f
        lim_lo, lim_hi = -0.5, size - 0.5
        if slope == 0:
            ok = (base >= lim_lo) & (base < lim_hi)
            lo = np.where(ok, lo, np.inf)
            continue
        j_a = (lim_lo - base) / slope
        j_b = (lim_hi - base) / slope
        if slope > 0:
            lo = np.maximum(lo, j_a)  # j >= j_a
            hi = np.minimum(hi, j_b)  # j < j_b
        else:
            lo = np.maximum(lo, np.nextafter(j_b, np.inf))  # j > j_b
            hi = np.minimum(hi, np.nextafter(j_a, np.inf))  # j <= j_a
    first = np.maximum(np.ceil(lo), 0)
    last = np.minimum(np.ceil(hi), fix.W)  # exclusive
    return int(np.sum(np.maximum(last - first, 0)))


class _Level:
    """One pyramid level of a windowed image, addressed in full-frame indices."""

    def __init__(self, img, win, f, shape, gradient=False):
        self.f = f
        self.H, self.W = shape[0] // f, shape[1] // f
        self.i0, self.j0 = win[0].start // f, win[1].start // f
        self.img = _pyramid_level(img, f)
        self.h, self.w = self.img.shape
        self.origin = 1 + (f - 1) / 2  # world coordinate of level index 0
        self._sat: dict[int, np.ndarray] = {}
        if gradient:
            self.gy, self.gx = _recursive_gradient(self.img, f)

    def any_signal(self, r0, r1, c0, c1, dilate: int) -> np.ndarray:
        """Whether each full-frame index box [r0, r1) x [c0, c1) holds signal.

        Signal is a non-zero pixel, dilated by ``dilate`` pixels (to cover
        the neighbours used by interpolation).
        """
        key = dilate
        if key not in self._sat:
            sig = self.img != 0
            if dilate:
                sig = ndimage.binary_dilation(sig, iterations=dilate)
            sat = np.zeros((self.h + 1, self.w + 1), dtype=np.int64)
            sat[1:, 1:] = np.cumsum(np.cumsum(sig, axis=0), axis=1)
            self._sat[key] = sat
        sat = self._sat[key]
        rr0 = np.clip(r0 - self.i0, 0, self.h)
        rr1 = np.clip(r1 - self.i0, 0, self.h)
        cc0 = np.clip(c0 - self.j0, 0, self.w)
        cc1 = np.clip(c1 - self.j0, 0, self.w)
        rr1 = np.maximum(rr1, rr0)
        cc1 = np.maximum(cc1, cc0)
        total = sat[rr1, cc1] - sat[rr0, cc1] - sat[rr1, cc0] + sat[rr0, cc0]
        return total > 0

    def lookup(self, arr, i, j):
        """arr at full-frame indices (i, j); 0 outside the window."""
        li, lj = i - self.i0, j - self.j0
        ok = (li >= 0) & (li < self.h) & (lj >= 0) & (lj < self.w)
        out = np.zeros(i.shape)
        out[ok] = arr[li[ok], lj[ok]]
        return out

    def interpolate(self, ci, cj) -> np.ndarray:
        """LinearInterpolateImageFunction (neighbours clamped to the image)."""
        i = np.floor(ci)
        j = np.floor(cj)
        di, dj = ci - i, cj - j
        i = i.astype(np.int64)
        j = j.astype(np.int64)
        i0, i1 = np.clip(i, 0, self.H - 1), np.clip(i + 1, 0, self.H - 1)
        j0, j1 = np.clip(j, 0, self.W - 1), np.clip(j + 1, 0, self.W - 1)
        v00 = self.lookup(self.img, i0, j0)
        v01 = self.lookup(self.img, i0, j1)
        v10 = self.lookup(self.img, i1, j0)
        v11 = self.lookup(self.img, i1, j1)
        top = v00 + (v01 - v00) * dj
        bottom = v10 + (v11 - v10) * dj
        return top + (bottom - top) * di

    def gradient_at(self, i, j):
        i = i.astype(np.int64)
        j = j.astype(np.int64)
        return self.lookup(self.gy, i, j), self.lookup(self.gx, i, j)


# ---------------------------------------------------------------------------
# Image filters
# ---------------------------------------------------------------------------


def _window(support: np.ndarray, margin: int) -> tuple[slice, slice]:
    """Bounding box of ``support`` plus ``margin``, edges on multiples of 4."""
    h, w = support.shape
    if not support.any():
        return slice(0, h), slice(0, w)
    ys, xs = ndimage.find_objects(support.astype(np.int8))[0]
    align = max(PYRAMID)
    y0 = max(ys.start - margin, 0) // align * align
    x0 = max(xs.start - margin, 0) // align * align
    y1 = min(-(-(ys.stop + margin) // align) * align, h)
    x1 = min(-(-(xs.stop + margin) // align) * align, w)
    return slice(y0, y1), slice(x0, x1)


def _gaussian_operator(variance: float, max_error: float = 0.1, max_width: int = 32):
    """itk::GaussianOperator coefficients (discrete Gaussian, Bessel based)."""
    et = math.exp(-variance)
    cap = 1.0 - max_error
    coeff = [et * special.i0(variance), et * special.i1(variance)]
    total = coeff[0] + 2.0 * coeff[1]
    i = 2
    while total < cap:
        coeff.append(coeff[i - 2] - 2 * (i - 1) * coeff[i - 1] / variance)
        total += 2.0 * coeff[i]
        if coeff[i] < total * np.finfo(float).eps or i > max_width:
            break
        i += 1
    c = np.array(coeff) / (coeff[0] + 2.0 * sum(coeff[1:]))
    return np.concatenate([c[:0:-1], c])


def _pyramid_level(img: np.ndarray, f: int) -> np.ndarray:
    """itk::MultiResolutionPyramidImageFilter output for shrink factor f."""
    k = _gaussian_operator((0.5 * f) ** 2)
    s = ndimage.correlate1d(img, k, axis=0, mode="nearest")
    s = ndimage.correlate1d(s, k, axis=1, mode="nearest")
    if f == 1:
        return s
    h, w = img.shape
    yy, xx = np.meshgrid(
        (f - 1) / 2 + f * np.arange(h // f), (f - 1) / 2 + f * np.arange(w // f),
        indexing="ij",
    )  # fmt: skip
    return ndimage.map_coordinates(s, [yy, xx], order=1, mode="nearest")


# itk::RecursiveGaussianImageFilter (Deriche) constants.
_A1 = (1.3530, -0.6724, -1.3563)
_B1 = (1.8151, -3.4327, 5.2318)
_W1, _L1 = 0.6681, -1.3932
_A2 = (-0.3531, 0.6724, 0.3446)
_B2 = (0.0902, 0.6100, -2.2355)
_W2, _L2 = 2.0787, -1.3732


def _recursive_coefficients(sigma: float, spacing: float, order: int):
    s = sigma / spacing
    cos1, cos2 = math.cos(_W1 / s), math.cos(_W2 / s)
    sin1, sin2 = math.sin(_W1 / s), math.sin(_W2 / s)
    e1, e2 = math.exp(_L1 / s), math.exp(_L2 / s)
    d4 = e1 * e1 * e2 * e2
    d3 = -2 * cos1 * e1 * e2 * e2
    d3 += -2 * cos2 * e2 * e1 * e1
    d2 = 4 * cos2 * cos1 * e1 * e2
    d2 += e1 * e1 + e2 * e2
    d1 = -2 * (e2 * cos2 + e1 * cos1)
    sd = 1.0 + d1 + d2 + d3 + d4
    dd = d1 + 2 * d2 + 3 * d3 + 4 * d4
    a1, b1, a2, b2 = _A1[order], _B1[order], _A2[order], _B2[order]
    n0 = a1 + a2
    n1 = e2 * (b2 * sin2 - (a2 + 2 * a1) * cos2)
    n1 += e1 * (b1 * sin1 - (a1 + 2 * a2) * cos1)
    n2 = (a1 + a2) * cos2 * cos1
    n2 -= b1 * cos2 * sin1 + b2 * cos1 * sin2
    n2 *= 2 * e1 * e2
    n2 += a2 * e1 * e1 + a1 * e2 * e2
    n3 = e2 * e1 * e1 * (b2 * sin2 - a2 * cos2)
    n3 += e1 * e2 * e2 * (b1 * sin1 - a1 * cos1)
    sn = n0 + n1 + n2 + n3
    dn = n1 + 2 * n2 + 3 * n3
    if order == 0:
        alpha = 2 * sn / sd - n0
        n = [x / alpha for x in (n0, n1, n2, n3)]
        m = [n[1] - d1 * n[0], n[2] - d2 * n[0], n[3] - d3 * n[0], -d4 * n[0]]
    else:  # first derivative, normalised across scale
        alpha = 2 * (sn * dd - dn * sd) / (sd * sd)
        n = [x * sigma / alpha for x in (n0, n1, n2, n3)]
        m = [-(n[1] - d1 * n[0]), -(n[2] - d2 * n[0]), -(n[3] - d3 * n[0]), d4 * n[0]]
    d = (d1, d2, d3, d4)
    bn = [x * sum(n) / sd for x in d]
    bm = [x * sum(m) / sd for x in d]
    return n, d, m, bn, bm


def _recursive_filter(x: np.ndarray, coeffs) -> np.ndarray:
    """RecursiveGaussianImageFilter::FilterDataArray along the last axis."""
    (n0, n1, n2, n3), (d1, d2, d3, d4), (m1, m2, m3, m4), bn, bm = coeffs
    ln = x.shape[-1]
    if ln < 4:
        raise ValueError("recursive Gaussian needs at least 4 pixels per line")
    sc = np.empty_like(x)
    v1 = x[:, 0]
    sc[:, 0] = v1 * n0 + v1 * n1 + v1 * n2 + v1 * n3
    sc[:, 1] = x[:, 1] * n0 + v1 * n1 + v1 * n2 + v1 * n3
    sc[:, 2] = x[:, 2] * n0 + x[:, 1] * n1 + v1 * n2 + v1 * n3
    sc[:, 3] = x[:, 3] * n0 + x[:, 2] * n1 + x[:, 1] * n2 + v1 * n3
    sc[:, 0] -= v1 * bn[0] + v1 * bn[1] + v1 * bn[2] + v1 * bn[3]
    sc[:, 1] -= sc[:, 0] * d1 + v1 * bn[1] + v1 * bn[2] + v1 * bn[3]
    sc[:, 2] -= sc[:, 1] * d1 + sc[:, 0] * d2 + v1 * bn[2] + v1 * bn[3]
    sc[:, 3] -= sc[:, 2] * d1 + sc[:, 1] * d2 + sc[:, 0] * d3 + v1 * bn[3]
    for i in range(4, ln):
        sc[:, i] = x[:, i] * n0 + x[:, i - 1] * n1 + x[:, i - 2] * n2 + x[:, i - 3] * n3
        sc[:, i] -= (
            sc[:, i - 1] * d1
            + sc[:, i - 2] * d2
            + sc[:, i - 3] * d3
            + sc[:, i - 4] * d4
        )
    out = sc.copy()
    v2 = x[:, ln - 1]
    sc[:, ln - 1] = v2 * m1 + v2 * m2 + v2 * m3 + v2 * m4
    sc[:, ln - 2] = x[:, ln - 1] * m1 + v2 * m2 + v2 * m3 + v2 * m4
    sc[:, ln - 3] = x[:, ln - 2] * m1 + x[:, ln - 1] * m2 + v2 * m3 + v2 * m4
    sc[:, ln - 4] = x[:, ln - 3] * m1 + x[:, ln - 2] * m2 + x[:, ln - 1] * m3 + v2 * m4
    sc[:, ln - 1] -= v2 * bm[0] + v2 * bm[1] + v2 * bm[2] + v2 * bm[3]
    sc[:, ln - 2] -= sc[:, ln - 1] * d1 + v2 * bm[1] + v2 * bm[2] + v2 * bm[3]
    sc[:, ln - 3] -= sc[:, ln - 2] * d1 + sc[:, ln - 1] * d2 + v2 * bm[2] + v2 * bm[3]
    sc[:, ln - 4] -= (
        sc[:, ln - 3] * d1 + sc[:, ln - 2] * d2 + sc[:, ln - 1] * d3 + v2 * bm[3]
    )
    for i in range(ln - 4, 0, -1):
        sc[:, i - 1] = (
            x[:, i] * m1 + x[:, i + 1] * m2 + x[:, i + 2] * m3 + x[:, i + 3] * m4
        )
        sc[:, i - 1] -= (
            sc[:, i] * d1 + sc[:, i + 1] * d2 + sc[:, i + 2] * d3 + sc[:, i + 3] * d4
        )
    return out + sc


def _recursive_gradient(img: np.ndarray, spacing: float):
    """GradientRecursiveGaussianImageFilter (sigma = spacing): (d/dy, d/dx)."""
    c0 = _recursive_coefficients(spacing, spacing, 0)
    c1 = _recursive_coefficients(spacing, spacing, 1)
    img = np.asarray(img, dtype=np.float64)
    gx = _recursive_filter(_recursive_filter(img.T, c0).T, c1)
    gy = _recursive_filter(_recursive_filter(img, c0).T, c1).T
    return gy / spacing, gx / spacing
