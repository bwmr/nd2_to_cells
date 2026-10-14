"""Cell linking, derived from SuperSegger's trackOptiLinkCellMulti, errorRez
and trackOptiCellMarker (see docs/linking.md).

The structure and the case analysis follow SuperSegger; what differs is
marked "Not SuperSegger", and SuperSegger behaviour that is kept although it
looks wrong is marked "MATLAB quirk".

Region labels are 1-based: per-region arrays and lists have an unused entry at
index 0, so label k is at index k (as in MATLAB's regs.* fields). Times are
1-based file indices, as in MATLAB.
"""

import io
import math
from collections.abc import Callable

import numpy as np
from scipy import ndimage

from nd2_to_cells.assign import assign, matlab_argmin
from nd2_to_cells.mask_edit import merge_regions, missing_seg_2to1

MAX_ITER_PER_FRAME = 3  # trackOptiLinkCellMulti; GUI runs (no CONST.ignoreerror)
SAFETY_MAX_PASSES = 20  # not in MATLAB: then one last pass without mask edits


class Frame:
    """Region table and linking state of one frame (MATLAB data.regs)."""

    def __init__(self, labels: np.ndarray):
        self.labels = np.ascontiguousarray(labels, dtype=np.int32)
        n, self.area, self.centroid, self.bbox = region_props(self.labels)
        self.num_regs = n
        self.edited = False  # labels differ from the Omnipose mask
        self.packed: bytes | None = None  # compressed edited labels
        self.colony = None  # (labels, centroids) cache of assign.colonies
        self.pairs = None  # touching-region pairs cache of assign.neighbor_pairs
        self.reset_lineage()
        empty = [[] for _ in range(n + 1)]
        self.map_r, self.map_f = [list(x) for x in empty], [list(x) for x in empty]
        self.revmap_r: list[list[int]] = []
        self.error_r = np.zeros(n + 1)
        self.cost_r = np.zeros(0)
        self.ids_c_r = self.ids_r_r = np.zeros((2, 0))
        self.d_area_r = np.full(n + 1, np.nan)

    def reset_lineage(self) -> None:
        """Lineage fields as updateRegionFields initialises them."""
        n = self.num_regs
        for name in (
            "ID", "birth", "divide", "ehist", "stat0", "sisterID", "motherID",
        ):  # fmt: skip
            setattr(self, name, np.zeros(n + 1, dtype=np.int64))
        self.daughterID: list[list[int]] = [[] for _ in range(n + 1)]

    def release_labels(self) -> None:
        """Drop the label image; keep a compressed copy if it was edited."""
        if self.labels is None:
            return
        if self.edited:
            buf = io.BytesIO()
            np.savez_compressed(buf, labels=self.labels)
            self.packed = buf.getvalue()
        self.labels = None
        self.invalidate()

    def invalidate(self) -> None:
        """Drop the caches derived from the labels (after an in-place edit)."""
        self.colony = None
        self.pairs = None

    def edited_labels(self) -> np.ndarray | None:
        """The edited label image, or None if the Omnipose mask is unchanged."""
        if self.packed is None:
            return self.labels if self.edited else None
        return np.load(io.BytesIO(self.packed))["labels"]

    def set_assignment(self, side: str, res) -> None:
        """Keep what errorRez reads: the backward fields, the forward map."""
        if side == "r":
            self.map_r, self.error_r, self.cost_r = res.assignments, res.error, res.cost
            self.ids_c_r, self.ids_r_r, self.d_area_r = res.ids_c, res.ids_f, res.d_area
            self.revmap_r = res.rev
        else:
            self.map_f = res.assignments


def region_props(labels: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """regionprops(labels, 'Area', 'Centroid', 'BoundingBox') for 1..max(label).

    Labels below the maximum with no pixels ("ghosts") get Area 0, a NaN
    centroid and BoundingBox [0.5 0.5 0 0], like MATLAB. Centroids are exact
    integer coordinate sums divided by the area, so they match MATLAB bit for
    bit.
    """
    n = int(labels.max()) if labels.size else 0
    flat = labels.ravel()
    rows, cols = np.nonzero(labels)  # foreground only; sums stay exact
    fg = labels[rows, cols]
    area = np.bincount(flat, minlength=n + 1).astype(float)[: n + 1]
    sx = np.bincount(fg, weights=cols + 1, minlength=n + 1)[: n + 1]
    sy = np.bincount(fg, weights=rows + 1, minlength=n + 1)[: n + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        centroid = np.stack([sx / area, sy / area], axis=1)
    area[0] = 0  # background; unused
    centroid[area == 0] = np.nan
    bbox = np.zeros((n + 1, 4))
    bbox[:, :2] = 0.5
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        ys, xs = sl
        bbox[k] = (
            xs.start + 0.5,
            ys.start + 0.5,
            xs.stop - xs.start,
            ys.stop - ys.start,
        )
    return n, area, centroid, bbox


# ---------------------------------------------------------------------------
# Driver: trackOptiLinkCellMulti
# ---------------------------------------------------------------------------


def link_frames(
    load_mask: Callable[[int], np.ndarray],
    n_frames: int,
    params,
    warn: Callable[[str], None] = print,
    progress: Callable[[int], None] | None = None,
) -> list[Frame]:
    """Link all frames of a position; return one Frame per file (1-based t-1).

    ``load_mask(i)`` returns the Omnipose label image of file index i (0-based).
    Each returned Frame keeps its region table and lineage state; the label
    image is released, with a compressed copy kept for frames whose mask was
    edited (see Frame.edited_labels).
    """
    states: list[Frame] = []
    cell_count = 0
    data_r: Frame | None = None
    data_f: Frame | None = None
    raw: dict[int, np.ndarray] = {}  # look-ahead masks by file index
    for t in range(1, n_frames + 1):
        data_c = data_f if data_f is not None else Frame(load_mask(t - 1))
        data_f = Frame(load_mask(t)) if t < n_frames else None
        # Not SuperSegger: masks of files t+1 ... t+k-1 (frames t+2 ... t+k)
        ahead = []
        for i in range(t + 1, min(t + params.division_lookahead, n_frames)):
            if i not in raw:
                raw[i] = load_mask(i).astype(np.int32)
            ahead.append(raw[i])
        raw = {i: m for i, m in raw.items() if i > t}
        data_c, cell_count = process_frame(
            t, data_c, data_r, data_f, params, cell_count, warn, ahead
        )
        states.append(data_c)
        if data_r is not None:
            data_r.release_labels()
        data_r = data_c
        if progress is not None:
            progress(t)
    if data_r is not None:
        data_r.release_labels()
    return states


def process_frame(
    t: int,
    data_c: Frame,
    data_r: Frame | None,
    data_f: Frame | None,
    params,
    cell_count: int,
    warn: Callable[[str], None] = print,
    ahead: list[np.ndarray] | None = None,
) -> tuple[Frame, int]:
    """One frame of trackOptiLinkCellMulti: link, re-running after mask edits.

    ``data_r`` is mutated and keeps changes made by aborted passes, as in
    MATLAB. ``ahead`` holds the masks after data_f, for the division
    look-ahead. Returns (final data_c, cell_count).
    """

    def assign_(a, b, forward):
        return assign(a, b, forward, params.da_min, params.da_max)

    cur_iter = 1
    while True:
        last_count = cell_count
        if data_r is not None:
            data_r.set_assignment("f", assign_(data_r, data_c, True))
        data_c.set_assignment("r", assign_(data_c, data_r, False))
        data_c.set_assignment("f", assign_(data_c, data_f, True))
        final_iteration = cur_iter >= MAX_ITER_PER_FRAME
        no_edits = cur_iter >= SAFETY_MAX_PASSES
        if no_edits:
            warn(f"frame {t}: {cur_iter - 1} passes, finishing without mask edits")
        cell_count, reset = error_rez(
            t,
            data_c,
            data_r,
            data_f,
            params,
            cell_count,
            final_iteration,
            no_edits,
            ahead or [],
        )
        if not reset:
            return data_c, cell_count
        cur_iter += 1
        cell_count = last_count
        data_c = Frame(data_c.labels)
        data_c.edited = True


# ---------------------------------------------------------------------------
# errorRez and helpers
# ---------------------------------------------------------------------------


def create_new_cell(c: Frame, k: int, time: int, cell_count: int) -> int:
    cell_count += 1
    c.birth[k] = time
    c.divide[k] = 0
    c.ehist[k] = 0
    c.stat0[k] = 0
    c.sisterID[k] = 0
    c.motherID[k] = 0
    c.daughterID[k] = []
    c.ID[k] = cell_count
    return cell_count


def continue_cell_line(c: Frame, k: int, r: Frame, reg_r: int, time: int, err: bool):
    if not np.any(c.ID[1:] == r.ID[reg_r]):
        c.birth[k] = r.birth[reg_r]
        c.divide[k] = 0
        c.ehist[k] = int(bool(r.ehist[reg_r]) or bool(err))
        c.stat0[k] = r.stat0[reg_r]
        c.sisterID[k] = r.sisterID[reg_r]
        c.motherID[k] = r.motherID[reg_r]
        c.daughterID[k] = []
        c.ID[k] = r.ID[reg_r]
        r.divide[reg_r] = 0
    # else: "ID PROBLEM" (another region already took the ID); nothing changes


def mark_division_event(
    c: Frame,
    s1: int,
    r: Frame,
    reg_r: int,
    time: int,
    err: int,
    s2: int,
    cell_count: int,
) -> int:
    if c.ID[s1]:  # MATLAB quirk: only sister 1 is checked
        return cell_count
    cell_count += 1
    _new_daughter(c, s1, r, reg_r, time, err, sister_id=cell_count + 1, cid=cell_count)
    cell_count += 1
    _new_daughter(c, s2, r, reg_r, time, err, sister_id=cell_count - 1, cid=cell_count)
    r.divide[reg_r] = int(not err)
    r.daughterID[reg_r] = [cell_count - 1, cell_count]
    return cell_count


def _new_daughter(c, s, r, reg_r, time, err, sister_id, cid) -> None:
    c.birth[s] = time
    c.divide[s] = 0
    c.ehist[s] = err
    c.stat0[s] = int(not err)
    c.sisterID[s] = sister_id
    c.motherID[s] = r.ID[reg_r]
    c.daughterID[s] = []
    c.ID[s] = cid


def create_division(c, r, mother, s1, s2, cell_count, time) -> int:
    r.error_r[mother] = 0
    c.error_r[s1] = 0
    c.error_r[s2] = 0
    return mark_division_event(c, s1, r, mother, time, 0, s2, cell_count)


def map_to_best_of_two(c: Frame, k: int, r: Frame, map_cr: list[int], time: int):
    flag_c = (c.ids_c_r[0] == k) & np.isnan(c.ids_c_r[1])
    flag_r1 = (c.ids_r_r[0] == map_cr[0]) & np.isnan(c.ids_r_r[1])
    flag_r2 = (c.ids_r_r[0] == map_cr[1]) & np.isnan(c.ids_r_r[1])
    cost1 = c.cost_r[flag_c & flag_r1]
    cost2 = c.cost_r[flag_c & flag_r2]
    if cost2.size == 0:
        first = True
    else:
        if cost1.size != 1 or cost2.size != 1:
            raise RuntimeError("mapToBestOfTwo: MATLAB errors here (|| on non-scalar)")
        first = bool(cost1[0] < cost2[0] or np.isnan(cost2[0]))
    c.error_r[k] = 2
    continue_cell_line(c, k, r, map_cr[0] if first else map_cr[1], time, True)


def map_best_of_two(
    c, map_rc, r, map_cr, time, cell_count, data_f, drop_strays, to_delete
):
    """mapBestOfTwo: continue the best of map_rc, make the others new cells."""
    reset = False
    keeper = map_rc[matlab_argmin(c.d_area_r[map_rc])]
    remove = [m for m in map_rc if m != keeper]
    continue_cell_line(c, keeper, r, map_cr[0], time, c.error_r[keeper] > 0)
    c.revmap_r[map_cr[0]] = [keeper]
    for m in remove:
        c.error_r[m] = 1
    mod_ids = remove + [keeper]
    if drop_strays and data_f is not None and _no_fw_mapping(c, remove):
        to_delete.extend(remove)
        reset = True
    else:
        for m in remove:
            cell_count = create_new_cell(c, m, time, cell_count)
    return cell_count, reset, mod_ids


def _no_fw_mapping(c: Frame, regs) -> bool:
    """hasNoFwMapping: isempty(map.f{regs}) (MATLAB errors for != 1 region)."""
    if isinstance(regs, list):
        if len(regs) != 1:
            raise RuntimeError("hasNoFwMapping: MATLAB errors here (comma list)")
        regs = regs[0]
    return not c.map_f[regs]


def _eq_any(a: list[int], b: list[int]) -> bool:
    """any(a == b) with MATLAB implicit expansion (errors on incompatible sizes)."""
    if not a:
        if len(b) > 1:
            raise RuntimeError("errorRez: MATLAB errors here ([] == vector)")
        return False
    if len(a) == len(b) or len(a) == 1 or len(b) == 1:
        if len(a) == len(b):
            return any(x == y for x, y in zip(a, b))
        return any(x == y for x in a for y in b)
    raise RuntimeError("errorRez: MATLAB errors here (incompatible sizes)")


def _orphans(c: Frame, r: Frame, min_area: float) -> dict[int, list[int]]:
    """Previous cells with no forward link, by the c region holding them.

    Not SuperSegger. A cell of r at least min_area large whose forward map is
    empty is assigned to the c region that covers at least half of its
    pixels, if any.
    """
    out: dict[int, list[int]] = {}
    for j in range(1, r.num_regs + 1):
        if r.map_f[j] or r.area[j] < min_area:
            continue
        x0, y0, w, h = r.bbox[j]
        ys = slice(int(y0), int(y0) + int(h))
        xs = slice(int(x0), int(x0) + int(w))
        hit = c.labels[ys, xs][r.labels[ys, xs] == j]
        hit = hit[hit > 0]
        if hit.size:
            k = int(np.bincount(hit).argmax())
            if np.count_nonzero(hit == k) >= 0.5 * r.area[j]:
                out.setdefault(k, []).append(j)
    return out


AHEAD_MARGIN = 20  # px around the two pieces searched in look-ahead masks


def _fuse_ahead(c: Frame, s1: int, s2: int, f: Frame, ahead) -> bool:
    """Do pieces s1 and s2 of c run into the same label in a look-ahead mask?

    Not SuperSegger. Each piece is followed from its regions in f (c.map_f)
    through the masks in ``ahead`` by largest overlap.
    """
    if not ahead or not c.map_f[s1] or not c.map_f[s2]:
        return False
    x0, y0 = np.minimum(c.bbox[s1, :2], c.bbox[s2, :2])
    x1, y1 = np.maximum(
        c.bbox[s1, :2] + c.bbox[s1, 2:], c.bbox[s2, :2] + c.bbox[s2, 2:]
    )
    h, w = f.labels.shape
    ys = slice(max(int(y0) - AHEAD_MARGIN, 0), min(int(y1) + AHEAD_MARGIN, h))
    xs = slice(max(int(x0) - AHEAD_MARGIN, 0), min(int(x1) + AHEAD_MARGIN, w))
    box = f.labels[ys, xs]
    m1, m2 = np.isin(box, c.map_f[s1]), np.isin(box, c.map_f[s2])
    for labels in ahead:
        box = labels[ys, xs]
        nxt = []
        for m in (m1, m2):
            hit = box[m]
            hit = hit[hit > 0]
            if not hit.size:
                return False
            nxt.append(np.bincount(hit).argmax())
        if nxt[0] == nxt[1]:
            return True
        m1, m2 = box == nxt[0], box == nxt[1]
    return False


def delete_regions(c: Frame, regs: list[int]) -> None:
    """deleteRegions: clear each label inside its (pass-start) bounding box."""
    for mm in regs:
        x0, y0, w, h = c.bbox[mm]
        ymin, xmin = math.ceil(y0), math.ceil(x0)
        ys = slice(ymin - 1, ymin + math.floor(h) - 1)
        xs = slice(xmin - 1, xmin + math.floor(w) - 1)
        box = c.labels[ys, xs]
        box[box == mm] = 0
    c.invalidate()


def error_rez(
    time: int,
    c: Frame,
    r: Frame | None,
    f: Frame | None,
    params,
    cell_count: int,
    ignore_error: bool,
    no_edits: bool = False,
    ahead: list[np.ndarray] = (),
) -> tuple[int, bool]:
    """errorRez: resolve the maps into IDs; may edit c.labels.

    Returns (cell_count, reset). ``no_edits`` (safety cap, not in MATLAB)
    also blocks the edits MATLAB doesn't gate by ignoreError. ``ahead``: masks
    after f; a 1 -> 2 split whose pieces fuse again in one of them is merged
    like one that fuses in f (division look-ahead, not SuperSegger).
    """
    ignore_error = ignore_error or no_edits
    drop_strays = params.drop_single_frame_strays and not no_edits
    min_area_merge = params.fragment_merge_area
    da_min, da_max = params.da_min, params.da_max
    to_delete: list[int] = []
    reset = False
    c_area = c.area
    c.ID[:] = 0
    mod_regions: set[int] = set()

    def merge(regs) -> bool:
        nonlocal reset
        if no_edits:
            return False
        done = merge_regions(c, regs)
        if done:
            c.edited = True
        reset = reset or done
        return done

    def merge_or_divide(s1, s2, mother) -> None:
        """Merge a 1 -> 2 split; if the closed union is not connected, divide.

        Not SuperSegger, where both pieces become new cells without a mother.
        """
        nonlocal cell_count
        if not merge([s1, s2]):
            cell_count = create_division(c, r, mother, s1, s2, cell_count, time)

    # Not SuperSegger: previous cells hidden in a continued region (2 -> 1).
    orphans = (
        _orphans(c, r, min_area_merge) if r is not None and not ignore_error else {}
    )

    for k in range(1, c.num_regs + 1):
        if c.ID[k] != 0 or k in mod_regions:
            pass
        else:
            regs_r = c.map_r[k]  # rCellsFromC
            if len(regs_r) == 1 and orphans.get(k):
                # Not SuperSegger: a previous cell without a forward link lies
                # in k, so k holds two cells; handled as 2 -> 1 (E) below.
                regs_r = regs_r + orphans[k]
            if regs_r:
                # MATLAB quirk: x = c{[a b]} keeps the first element.
                c_from_r = r.map_f[regs_r[0]]  # cCellsFromR
                c_transp = c.revmap_r[regs_r[0]]  # cCellsTransp
            else:
                c_from_r, c_transp = [], []

            if len(regs_r) == 0:  # A: stray
                if time != 1 and (not c.map_f[k] and drop_strays):
                    to_delete.append(k)
                    reset = True
                else:
                    cell_count = create_new_cell(c, k, time, cell_count)

            elif len(regs_r) == 1 and len(c_transp) == 1:  # B: continue
                continue_cell_line(c, k, r, regs_r[0], time, c.error_r[k] > 0)

            elif len(regs_r) == 1 and len(c_from_r) == 1 and len(c_transp) == 2:  # C
                s1 = k
                s2s = [m for m in c_transp if m != k]
                if len(s2s) != 1:
                    raise RuntimeError("errorRez C: MATLAB errors here (sister2)")
                s2 = s2s[0]
                mother = regs_r[0]
                tot_c = c.area[s1] + c.area[s2]
                change = (tot_c - r.area[mother]) / tot_c
                good = da_min < change < da_max
                no_match = not c.map_f[s1] or not c.map_f[s2]
                same = not no_match and all(x in c.map_f[s2] for x in c.map_f[s1])
                small = c_area[s1] < min_area_merge or c_area[s2] < min_area_merge
                merge_f = no_match or same or small  # SuperSegger's merge test
                if (
                    good
                    and not ignore_error
                    and f is not None
                    and (merge_f or _fuse_ahead(c, s1, s2, f, ahead))
                ):
                    merge_or_divide(s1, s2, mother)
                    mod_regions.update(c_transp)
                elif good:
                    cell_count = create_division(c, r, mother, s1, s2, cell_count, time)
                    mod_regions.update(c_transp)
                else:
                    cell_count, rst, mod = map_best_of_two(
                        c,
                        c_transp,
                        r,
                        regs_r,
                        time,
                        cell_count,
                        f,
                        drop_strays,
                        to_delete,
                    )
                    reset = reset or rst
                    mod_regions.update(mod)

            elif len(regs_r) == 1 and len(c_from_r) == 2:  # D: 1 -> 2
                mother = regs_r[0]
                s1 = k
                map_rc = list(r.map_f[mother])
                s2s = [m for m in map_rc if m != k]
                s2_map = c.map_r[s2s[0]]  # MATLAB quirk: first of a comma list
                k_in = k in map_rc
                if (
                    len(s2s) == 1
                    and k_in
                    and s2_map
                    and all(x == mother for x in s2_map)
                ):
                    s2 = s2s[0]
                    tot_c = c.area[s1] + c.area[s2]
                    change = (tot_c - r.area[mother]) / tot_c
                    good = da_min < change < da_max
                    no_match = f is not None and (not c.map_f[s1] or not c.map_f[s2])
                    same = (
                        f is not None
                        and not no_match
                        and all(x in c.map_f[s2] for x in c.map_f[s1])
                    )
                    small = c_area[s1] < min_area_merge or c_area[s2] < min_area_merge
                    merge_f = no_match or same or small  # SuperSegger's merge test
                    fuse_later = f is not None and _fuse_ahead(c, s1, s2, f, ahead)
                    if good and not ignore_error and (merge_f or fuse_later):
                        merge_or_divide(s1, s2, mother)
                        mod_regions.update(map_rc)
                    else:
                        cell_count = create_division(
                            c, r, mother, s1, s2, cell_count, time
                        )
                        mod_regions.update(map_rc)
                elif (
                    len(s2s) == 1 and k_in and any(x != mother for x in c.map_r[s2s[0]])
                ):
                    continue_cell_line(c, k, r, regs_r[0], time, c.error_r[k] > 0)
                elif not k_in:  # incorrect mapping 1 to 2: a new cell
                    cell_count = create_new_cell(c, k, time, cell_count)
                    c.error_r[k] = 1
                # else: not resolved; k becomes a new cell below

            elif len(regs_r) > 1:  # E: 2 -> 1, try to split
                if not ignore_error and missing_seg_2to1(c, k, r, regs_r):
                    c.edited = True
                    c.error_r[k] = 0
                    reset = True
                else:
                    map_to_best_of_two(c, k, r, regs_r, time)

            elif len(regs_r) == 1 and len(c_from_r) > 2:  # F: 1 -> 3+
                forw = [x for m in c_from_r for x in c.map_f[m]]
                forward_map = sorted(set(forw))
                occur = [forw.count(v) for v in forward_map]
                # MATLAB quirk: the "no forward match" test (isempty of a cell
                # array) is never true.
                same = len(forward_map) == 1
                some_same = any(o > 1 for o in occur)
                if f is not None and same:
                    if not ignore_error:
                        merge(list(c_from_r))
                        mod_regions.update(c_from_r)
                    else:
                        cell_count, rst, mod = map_best_of_two(
                            c,
                            c_transp,
                            r,
                            regs_r,
                            time,
                            cell_count,
                            f,
                            drop_strays,
                            to_delete,
                        )
                        reset = reset or rst
                        mod_regions.update(mod)
                elif f is not None and some_same:
                    value_fw = [v for v, o in zip(forward_map, occur) if o > 1]
                    to_merge = [m for m in c_from_r if _eq_any(c.map_f[m], value_fw)]
                    merge(to_merge)  # MATLAB quirk: not gated by ignoreError
                    mod_regions.update(to_merge)

            # G: no case applies; k becomes a new cell below

        if c.ID[k] == 0:
            cell_count = create_new_cell(c, k, time, cell_count)

    if to_delete:
        delete_regions(c, to_delete)
        c.edited = True
    return cell_count, reset


# ---------------------------------------------------------------------------
# trackOptiCellMarker
# ---------------------------------------------------------------------------


def cell_marker(states: list[Frame], min_cycle_frames: int) -> None:
    """Mark complete cell cycles (stat0 = 2) by walking back from divisions.

    A cycle is complete if it spans at least min_cycle_frames frames, birth
    and division frame included (SuperSegger's MIN_CELL_AGE + 1). Not
    SuperSegger: a lineage whose backward map breaks is dropped, instead of
    continuing from a stale region.
    """
    n_im = len(states)
    clist: list[list[int]] = []  # [ID, current region, birth]
    for i in range(n_im - 1, 1, -1):
        c = states[i - 1]
        for ii in range(1, c.num_regs + 1):
            if c.divide[ii] and c.stat0[ii] and not c.ehist[ii]:
                if i - c.birth[ii] + 1 >= min_cycle_frames:
                    clist.append([int(c.ID[ii]), ii, int(c.birth[ii])])
        for row in clist:
            jj = row[1]
            if i >= row[2]:
                if jj > c.num_regs or not c.map_r[jj]:
                    row[2] = n_im + 1  # never matches i >= row[2] again
                    continue
                row[1] = c.map_r[jj][0]
                c.stat0[jj] = 2
