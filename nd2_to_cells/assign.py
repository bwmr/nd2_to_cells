"""Frame-to-frame assignment: port of SuperSegger's multiAssignmentSparse.m.

Assigns each region of frame c to zero, one or two regions of frame f, using
shape-aligned overlap, raw overlap, centroid distance, area change and an
outward-motion term. Pairs in c (two c regions -> one f region) and pairs in f
(one c region -> two f regions) are scored too. Selection is a greedy global
minimum followed by the fixProblems and exchangeAssignment repair passes.

The port is bug-for-bug: MATLAB quirks are kept and marked "MATLAB quirk".

Conventions (mirroring MATLAB):
- Region labels are 1-based; per-region lists and arrays have an unused
  entry at index 0, so label k is at index k.
- Frames are objects with ``labels`` (2-D int array), ``num_regs``, ``area``
  (n+1,), ``centroid`` (n+1, 2) as [x, y] 1-based, and ``bbox`` (n+1, 4) as
  MATLAB BoundingBox [x0, y0, w, h].
- Candidate lists ("cell arrays") are Python lists of label lists.
"""

import math

import numpy as np
from scipy import ndimage

NO_OVERLAP = 0.0001
CENTROID_WEIGHT = 5
AREA_FACTOR = 20
AREA_CHANGE_FACTOR = 100
OUTWARD_MOT_FACTOR = 1 / 10
BOX_PAD = 20

_SQUARE5 = np.ones((5, 5), dtype=bool)
_CONN8 = np.ones((3, 3), dtype=int)


# ---------------------------------------------------------------------------
# MATLAB helpers
# ---------------------------------------------------------------------------


def mround(x: float) -> float:
    """MATLAB round: halves away from zero."""
    t = math.trunc(x)
    if abs(x - t) >= 0.5:
        return float(t + math.copysign(1, x))
    return float(t)


def matlab_argmin(a: np.ndarray) -> int:
    """Index of MATLAB's min: first minimum ignoring NaN; 0 if all NaN."""
    if a.size == 0 or np.all(np.isnan(a)):
        return 0
    return int(np.nanargmin(a))


def label_colmajor(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected bwlabel: components numbered in column-major order."""
    lab, n = ndimage.label(mask.T, structure=_CONN8)
    return lab.T, n


def add_bb(bb1, bb2):
    """addBB.m: union of two MATLAB BoundingBoxes."""
    if bb1 is None:
        return bb2
    if bb2 is None:
        return bb1
    ymin = min(bb1[1], bb2[1])
    xmin = min(bb1[0], bb2[0])
    ymax = max(bb1[1] + bb1[3], bb2[1] + bb2[3])
    xmax = max(bb1[0] + bb1[2], bb2[0] + bb2[2])
    return (xmin, ymin, xmax - xmin, ymax - ymin)


def bb_pad(bb, shape: tuple[int, int], pad: int) -> tuple[slice, slice]:
    """getBBpad.m as 0-based (row, col) slices."""
    ymin = math.ceil(bb[1]) - pad
    xmin = math.ceil(bb[0]) - pad
    ymax = ymin + math.floor(bb[3]) + 2 * pad - 1
    xmax = xmin + math.floor(bb[2]) + 2 * pad - 1
    return (
        slice(max(1, ymin) - 1, max(min(ymax, shape[0]), max(1, ymin) - 1)),
        slice(max(1, xmin) - 1, max(min(xmax, shape[1]), max(1, xmin) - 1)),
    )


def imshift(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """imshift.m: out(y, x) = mask(y - dy, x - dx), zero filled."""
    h, w = mask.shape
    out = np.zeros_like(mask)
    if abs(dy) >= h or abs(dx) >= w:
        return out
    out[max(dy, 0) : h + min(dy, 0), max(dx, 0) : w + min(dx, 0)] = mask[
        max(-dy, 0) : h - max(dy, 0), max(-dx, 0) : w - max(dx, 0)
    ]
    return out


def box_limits(frame, regs) -> tuple[slice, slice]:
    """getBoxLimits: padded bounding box of the regions in ``regs``."""
    bb = None
    for reg in regs:
        if reg is None or (isinstance(reg, float) and math.isnan(reg)):
            continue
        bb = add_bb(bb, tuple(frame.bbox[int(reg)]))
    return bb_pad(bb, frame.labels.shape, BOX_PAD)


def reg_properties(frame, regs, ys: slice, xs: slice):
    """regProperties: union mask in the box, summed area, mean centroid."""
    regs = [int(r) for r in regs if not (isinstance(r, float) and math.isnan(r))]
    box = frame.labels[ys, xs]
    mask = np.zeros(box.shape, dtype=bool)
    cx = cy = 0.0
    area = 0.0
    for reg in regs:
        cx = cx + frame.centroid[reg, 0]
        cy = cy + frame.centroid[reg, 1]
        mask |= box == reg
        area = area + frame.area[reg]
    return mask, area, (cx / len(regs), cy / len(regs))


def find_neighbor_pairs(frame_c, regs_in_c: list[int], frame_f):
    """findNeighborPairs: pairs of touching c regions, and f labels near each."""
    regs_set = set(regs_in_c)
    pairs: list[tuple[int, int]] = []
    neigh_f: list[np.ndarray] = [np.zeros(0, dtype=int)] * (frame_c.num_regs + 1)
    for jj in regs_in_c:
        ys, xs = box_limits(frame_c, [jj])
        box_c = frame_c.labels[ys, xs]
        tmp_mask = ndimage.binary_dilation(box_c == jj, structure=_SQUARE5)
        ind = np.unique(box_c[tmp_mask])
        for uu in ind[ind > jj]:
            if int(uu) in regs_set:
                pairs.append((jj, int(uu)))
        if frame_f is not None:
            fn = frame_f.labels[ys, xs][tmp_mask]
            neigh_f[jj] = np.unique(fn[fn != 0]).astype(int)
    return pairs, neigh_f


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------


class AssignResult:
    """Outputs of multiAssignmentSparse (MATLAB names in comments)."""

    def __init__(self, assignments, error, cost, ids_c, ids_f, d_area, rev):
        self.assignments = assignments  # assignments / map.{r,f}
        self.error = error  # errorR / error.{r,f}
        self.cost = cost  # totCost / cost.{r,f}
        self.ids_c = ids_c  # indexC / idsC.{r,f}
        self.ids_f = ids_f  # indexF / idsR.r or idsF.f
        self.d_area = d_area  # dA / dA.{r,f}
        self.rev = rev  # revAssign / revmap.{r,f}


def assign(frame_c, frame_f, forward: bool, da_min: float, da_max: float, fixes=()):
    """Port of multiAssignmentSparse(data_c, data_f, CONST, forward).

    ``fixes``: opt-in departures from MATLAB (track.FIXES).
    """
    sign = 1 if forward else -1
    max_da = max(sign * da_min, sign * da_max)
    min_da = min(sign * da_min, sign * da_max)

    def set_error(da):
        if np.isnan(da):
            return 0
        if da > max_da:
            return 3
        if da < min_da:
            return 2
        return 0

    n_c = frame_c.num_regs
    if frame_f is None:
        return AssignResult(
            [[] for _ in range(n_c + 1)],
            np.zeros(n_c + 1),
            np.zeros(0),
            np.zeros((2, 0)),
            np.zeros((2, 0)),
            np.full(n_c + 1, np.nan),
            [],
        )
    n_f = frame_f.num_regs
    regs_in_c = list(range(1, n_c + 1))  # no manual links
    regs_in_f = list(range(1, n_f + 1))

    # Colony calculation (imfill holes is 4-connected background).
    filled = ndimage.binary_fill_holes(frame_c.labels != 0)
    colony_labels, n_col = label_colmajor(filled)
    col_area = np.bincount(colony_labels.ravel(), minlength=n_col + 1)
    rr, cc = np.indices(colony_labels.shape)
    col_cx = np.bincount(
        colony_labels.ravel(), weights=(cc + 1).ravel(), minlength=n_col + 1
    )
    col_cy = np.bincount(
        colony_labels.ravel(), weights=(rr + 1).ravel(), minlength=n_col + 1
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        col_cent = np.stack([col_cx / col_area, col_cy / col_area], axis=1)

    pairs_f, _ = find_neighbor_pairs(frame_f, regs_in_f, None)
    pairs_f_set = {frozenset(p) for p in pairs_f}
    pairs_c, neigh_f = find_neighbor_pairs(frame_c, regs_in_c, frame_f)
    all_c = [(r, None) for r in regs_in_c] + list(pairs_c)

    n_cols = len(all_c)
    prealloc = n_cols * 10
    index_c = np.full((2, prealloc), np.nan)
    index_f = np.full((2, prealloc), np.nan)
    tot_cost = np.full(prealloc, np.nan)
    overlap = np.full(prealloc, np.nan)  # areaOverlapCost
    area_change = np.full(prealloc, np.nan)
    centroid_cost = np.full(prealloc, np.nan)
    overlap_trans = np.full(prealloc, np.nan)  # areaOverlapTransCost
    outward_mot = np.full(prealloc, np.nan)
    dist_colony_mat = np.full(prealloc, np.nan)
    good_one_to_one = np.zeros(n_cols + 1)  # MATLAB quirk: indexed by column
    dist_from_coln = np.zeros(n_cols + 1)
    counter = 0  # 0-based column of the next candidate

    def grow():
        # MATLAB grows arrays with zeros when assigning past the end.
        nonlocal index_c, index_f, tot_cost, overlap, area_change
        nonlocal centroid_cost, overlap_trans, outward_mot, dist_colony_mat
        if counter < tot_cost.size:
            return
        index_c = np.concatenate([index_c, np.zeros((2, 1))], axis=1)
        index_f = np.concatenate([index_f, np.zeros((2, 1))], axis=1)
        tot_cost = np.append(tot_cost, 0.0)
        overlap = np.append(overlap, 0.0)
        area_change = np.append(area_change, 0.0)
        centroid_cost = np.append(centroid_cost, 0.0)
        overlap_trans = np.append(overlap_trans, 0.0)
        outward_mot = np.append(outward_mot, 0.0)
        dist_colony_mat = np.append(dist_colony_mat, 0.0)

    def score(col, mask_c, area_c, cent_c, f_regs, ys, xs, dist_colony, pair_f):
        mask_f, area_f, cent_f = reg_properties(frame_f, f_regs, ys, xs)
        area_ov = np.count_nonzero(mask_f[mask_c])
        overlap[counter] = area_ov / area_c
        if overlap[counter] == 0:
            overlap[counter] = NO_OVERLAP
        dx = cent_c[0] - cent_f[0]
        dy = cent_c[1] - cent_f[1]
        centroid_cost[counter] = math.sqrt(dx**2 + dy**2)
        if pair_f:
            # MATLAB quirk: `if centroidCost == 0` tests the whole vector.
            zero = bool(np.all(centroid_cost == 0))
        else:
            zero = centroid_cost[counter] == 0
        if zero:
            outward_mot[counter] = 0
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                outward_mot[counter] = np.float64(
                    dist_colony[0] * dx + dist_colony[1] * dy
                ) / np.float64(centroid_cost[counter])
        dist_colony_mat[counter] = math.exp(-dist_from_coln[col] / OUTWARD_MOT_FACTOR)
        if not pair_f:
            area_change[counter] = (area_f - area_c) / area_c
        shifted = imshift(mask_f, int(mround(dx)), int(mround(dy)))
        overlap_trans[counter] = np.count_nonzero(shifted[mask_c]) / area_c
        if overlap_trans[counter] == 0:
            overlap_trans[counter] = NO_OVERLAP
        if pair_f:
            area_change[counter] = (area_f - area_c) / area_c

    for ii, c_regs in enumerate(all_c, start=1):
        is_single = c_regs[1] is None
        if not is_single and (good_one_to_one[c_regs[0]] or good_one_to_one[c_regs[1]]):
            continue
        regs = [c_regs[0]] if is_single else list(c_regs)
        ys, xs = box_limits(frame_c, regs)
        mask_c, area_c, cent_c = reg_properties(frame_c, regs, ys, xs)

        col_overlap = colony_labels[ys, xs][mask_c]
        if col_overlap.sum() == 0:
            dist_colony = (0.0, 0.0)
        else:
            colony_id = int(col_overlap.max())
            dist_colony = (
                cent_c[0] - col_cent[colony_id, 0],
                cent_c[1] - col_cent[colony_id, 1],
            )
        dist_from_coln[ii] = math.sqrt(dist_colony[0] ** 2 + dist_colony[1] ** 2)

        cand = np.unique(np.concatenate([neigh_f[r] for r in regs]))
        cand = [int(x) for x in cand if 1 <= x <= n_f]

        start = counter
        for id_f in cand:
            grow()
            index_c[0, counter] = c_regs[0]
            index_c[1, counter] = np.nan if is_single else c_regs[1]
            index_f[0, counter] = id_f
            score(ii, mask_c, area_c, cent_c, [id_f], ys, xs, dist_colony, False)
            counter += 1

        if is_single and start != counter:
            s = slice(start, counter)
            prov = (
                AREA_CHANGE_FACTOR * 1 / overlap_trans[s]
                + AREA_FACTOR * 1 / overlap[s]
                + centroid_cost[s]
                + AREA_CHANGE_FACTOR * np.abs(area_change[s])
            )
            k = start + matlab_argmin(prov)
            good_one_to_one[ii] = (
                abs(area_change[k]) < 0.15
                and overlap[k] > 0.7
                and overlap_trans[k] > 0.8
            )
        else:
            good_one_to_one[ii] = 1

        if not good_one_to_one[ii]:
            for a in range(len(cand)):
                for b in range(a + 1, len(cand)):
                    sis = (cand[a], cand[b])
                    if frozenset(sis) not in pairs_f_set:
                        continue
                    grow()
                    index_c[0, counter] = c_regs[0]
                    index_c[1, counter] = np.nan if is_single else c_regs[1]
                    index_f[0, counter] = sis[0]
                    index_f[1, counter] = sis[1]
                    score(
                        ii, mask_c, area_c, cent_c, list(sis), ys, xs, dist_colony, True
                    )
                    counter += 1

    penalty = np.zeros(area_change.shape)
    if forward:
        penalty[area_change < -0.1] = 100
    else:
        outward_mot = -outward_mot
        penalty[area_change > 0.1] = 100
    # MATLAB quirk: the > 0.3 line overwrites the > 0.6 penalty.
    if "area_penalty" in fixes:
        penalty[np.abs(area_change) > 0.3] = 50
        penalty[np.abs(area_change) > 0.6] = 1000
    else:
        penalty[np.abs(area_change) > 0.6] = 1000
        penalty[np.abs(area_change) > 0.3] = 50
    overlap_cost = 1 - overlap
    tot_cost = (
        penalty
        + AREA_CHANGE_FACTOR * 1 / overlap_trans
        + CENTROID_WEIGHT * centroid_cost
        + AREA_CHANGE_FACTOR * np.abs(area_change)
        + dist_colony_mat * AREA_FACTOR * 1 / overlap
        + OUTWARD_MOT_FACTOR * outward_mot
    )
    keep = np.any(~np.isnan(index_c), axis=0)
    tot_cost = tot_cost[keep]
    index_c = index_c[:, keep]
    index_f = index_f[:, keep]

    assignments: list[list[int]] = [[] for _ in range(n_c + 1)]
    cost_mat = tot_cost.copy()
    flagger = ~np.isnan(cost_mat)
    while flagger.any():
        ind = int(np.nanargmin(cost_mat))
        assign_tmp = [int(x) for x in index_f[:, ind] if not np.isnan(x)]
        regions_c = index_c[:, ind]
        assignments[int(regions_c[0])] = list(assign_tmp)
        if not np.isnan(regions_c[1]):
            assignments[int(regions_c[1])] = list(assign_tmp)
        c_nonnan = [x for x in regions_c if not np.isnan(x)]
        col_del_f = np.any(np.isin(index_f, assign_tmp), axis=0)
        col_del_c = np.any(np.isin(index_c, c_nonnan), axis=0)
        cost_mat[col_del_c] = np.nan
        cost_mat[col_del_f] = np.nan
        flagger[col_del_c] = False
        flagger[col_del_f] = False

    rev: list[list[int]] = [[] for _ in range(n_f + 1)]
    for ll in range(1, n_c + 1):
        for uu in assignments[ll]:
            rev[uu] = rev[uu] + [ll]

    c_area = frame_c.area
    f_area = frame_f.area

    def best_single(value, cost, idx_f, idx_c):
        """findBestSingleAssign(value_f, totCost, indexF, indexC)."""
        n = idx_f.shape[1]
        sel = (idx_f[0, :] == value) & np.isnan(idx_f[1, :])
        tmp = cost.copy()
        tmp[:n][~sel] = np.nan
        if np.count_nonzero(~np.isnan(tmp)):
            return idx_c[0, matlab_argmin(tmp)]
        return np.nan

    def fix_problems(assg, rev_a, cost, idx_f, idx_c, a_c, a_f):
        left_in_f = [j for j in range(1, len(rev_a)) if not rev_a[j]]
        for jj in left_in_f:
            best_c = best_single(jj, cost, idx_f, idx_c)
            if np.isnan(best_c) or best_c > len(assg) - 1:
                continue
            best_c = int(best_c)
            f_already = assg[best_c]
            if not f_already:
                assg[best_c] = [jj]  # MATLAB quirk: rev_a not updated
                if "assign_rev" in fixes:
                    rev_a[jj] = [best_c]
                continue
            # MATLAB quirk: revAssign{FAlready} keeps only the first element.
            if "assign_all_regions" in fixes:
                rev_to_already = sorted({x for f in f_already for x in rev_a[f]})
            else:
                rev_to_already = list(rev_a[f_already[0]])
            area_c = float(np.sum(a_c[rev_to_already]))
            area_f_before = float(np.sum(a_f[f_already]))
            da_before = _div(area_f_before - area_c, max(area_f_before, area_c))
            if len(rev_to_already) == 2 and set_error(da_before) > 0:
                area_fjj = a_f[jj]
                new_rev = [r for r in rev_to_already if r != best_c]
                area_c = a_c[best_c]
                new_da_jj = _div(area_fjj - area_c, max(area_fjj, area_c))
                if set_error(new_da_jj):
                    continue
                if len(new_rev) != 1:
                    raise RuntimeError("fixProblems: MATLAB errors here (&& on vector)")
                # MATLAB quirk: normalised by the stealer's area.
                norm = a_c[new_rev[0]] if "assign_robbed_area" in fixes else area_c
                new_da_already = _div(
                    area_f_before - a_c[new_rev[0]], max(area_f_before, norm)
                )
                if not set_error(new_da_already):
                    if len(f_already) != 1:
                        raise RuntimeError(
                            "fixProblems: MATLAB errors here (c{[a b]} = x)"
                        )
                    assg[best_c] = [jj]
                    rev_a[jj] = [best_c]
                    rev_a[f_already[0]] = new_rev
            else:
                area_f = area_f_before + a_f[jj]
                da_tmp = _div(area_f - area_c, max(area_f, area_c))
                if set_error(da_before) > 0 and not set_error(da_tmp):
                    assg[best_c] = list(f_already) + [jj]
                    rev_a[jj] = [best_c]

    def exchange(assg, rev_a, cost, idx_f, idx_c):
        left_in_c = [k for k in range(1, len(assg)) if not assg[k]]
        left_in_f = [j for j in range(1, len(rev_a)) if not rev_a[j]]
        for cur in left_in_c:
            best_f = best_single(cur, cost, idx_c, idx_f)
            if np.isnan(best_f):
                continue
            best_f = int(best_f)
            bad_c = 0
            for bad_c in range(1, len(assg)):
                if assg[bad_c] and best_f in assg[bad_c]:
                    break
            else:
                # MATLAB quirk: without a match bad_c is the last region.
                if "assign_exchange" in fixes:
                    continue
            if cost.size == 0:
                continue
            flag_c = (idx_c[0, :] == bad_c) & np.isnan(idx_c[1, :])
            flag_f = (idx_f[0, :] == best_f) & np.isnan(idx_f[1, :])
            tmp = cost.copy()
            tmp[~flag_c] = np.nan
            tmp[flag_f] = np.nan
            if "assign_nan" in fixes and np.isnan(tmp).all():
                continue
            ind = matlab_argmin(tmp)  # MATLAB quirk: all-NaN gives column 1
            second = idx_f[:, ind]
            if np.isnan(second[1]) and int(second[0]) in left_in_f:
                s = int(second[0])
                assg[bad_c] = [s]
                rev_a[s] = [bad_c]
                rev_a[best_f] = [cur]
                assg[cur] = [best_f]

    fix_problems(assignments, rev, overlap_cost, index_f, index_c, c_area, f_area)
    fix_problems(rev, assignments, overlap_cost, index_c, index_f, f_area, c_area)
    exchange(assignments, rev, tot_cost, index_f, index_c)
    exchange(rev, assignments, tot_cost, index_c, index_f)

    d_area = np.full(n_c + 1, np.nan)
    for ll in range(1, n_c + 1):
        carea = c_area[ll]
        farea = float(np.sum(f_area[assignments[ll]])) if assignments[ll] else 0.0
        d_area[ll] = _div(farea - carea, max(carea, farea))
    error = np.zeros(n_c + 1)
    for ll in range(1, n_c + 1):
        error[ll] = set_error(d_area[ll])

    return AssignResult(assignments, error, tot_cost, index_c, index_f, d_area, rev)


def _div(a: float, b: float) -> float:
    """MATLAB double division (x/0 -> ±Inf, 0/0 -> NaN)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.float64(a) / np.float64(b))
