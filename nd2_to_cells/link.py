"""Cell linking: port of SuperSegger's trackOptiLinkCellMulti, errorRez and
trackOptiCellMarker (bug-for-bug; see docs/supersegger_port.md).

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
        self.reset_lineage()
        empty = [[] for _ in range(n + 1)]
        self.map_r, self.map_f = [list(x) for x in empty], [list(x) for x in empty]
        self.revmap_r: list[list[int]] = []
        self.revmap_f: list[list[int]] = []
        self.error_r = np.zeros(n + 1)
        self.error_f = np.zeros(n + 1)
        self.cost_r = self.cost_f = np.zeros(0)
        self.ids_c_r = self.ids_r_r = self.ids_c_f = self.ids_f_f = np.zeros((2, 0))
        self.d_area_r = np.full(n + 1, np.nan)
        self.d_area_f = np.full(n + 1, np.nan)

    def reset_lineage(self) -> None:
        """Lineage fields as updateRegionFields initialises them."""
        n = self.num_regs
        for name in (
            "ID", "birth", "birthF", "death", "deathF", "age", "divide",
            "ehist", "stat0", "sisterID", "motherID",
        ):  # fmt: skip
            setattr(self, name, np.zeros(n + 1, dtype=np.int64))
        self.daughterID: list[list[int]] = [[] for _ in range(n + 1)]
        self.label: list[str] = [""] * (n + 1)  # error.label

    def release_labels(self) -> None:
        """Drop the label image; keep a compressed copy if it was edited."""
        if self.labels is None:
            return
        if self.edited:
            buf = io.BytesIO()
            np.savez_compressed(buf, labels=self.labels)
            self.packed = buf.getvalue()
        self.labels = None

    def edited_labels(self) -> np.ndarray | None:
        """The edited label image, or None if the Omnipose mask is unchanged."""
        if self.packed is None:
            return self.labels if self.edited else None
        return np.load(io.BytesIO(self.packed))["labels"]

    def set_assignment(self, side: str, res) -> None:
        if side == "r":
            self.map_r, self.error_r, self.cost_r = res.assignments, res.error, res.cost
            self.ids_c_r, self.ids_r_r, self.d_area_r = res.ids_c, res.ids_f, res.d_area
            self.revmap_r = res.rev
        else:
            self.map_f, self.error_f, self.cost_f = res.assignments, res.error, res.cost
            self.ids_c_f, self.ids_f_f, self.d_area_f = res.ids_c, res.ids_f, res.d_area
            self.revmap_f = res.rev


def region_props(labels: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """regionprops(labels, 'Area', 'Centroid', 'BoundingBox') for 1..max(label).

    Labels below the maximum with no pixels ("ghosts") get Area 0, a NaN
    centroid and BoundingBox [0.5 0.5 0 0], like MATLAB. Centroids are exact
    integer coordinate sums divided by the area, so they match MATLAB bit for
    bit.
    """
    n = int(labels.max()) if labels.size else 0
    flat = labels.ravel()
    rows, cols = np.indices(labels.shape)
    area = np.bincount(flat, minlength=n + 1).astype(float)[: n + 1]
    sx = np.bincount(flat, weights=(cols + 1).ravel(), minlength=n + 1)[: n + 1]
    sy = np.bincount(flat, weights=(rows + 1).ravel(), minlength=n + 1)[: n + 1]
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
    for t in range(1, n_frames + 1):
        data_c = data_f if data_f is not None else Frame(load_mask(t - 1))
        data_f = Frame(load_mask(t)) if t < n_frames else None
        data_c, cell_count, _ = process_frame(
            t, data_c, data_r, data_f, params, cell_count, warn
        )
        states.append(data_c)
        if data_r is not None:
            data_r.release_labels()
        data_r = data_c
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
) -> tuple[Frame, int, int]:
    """One frame of trackOptiLinkCellMulti: link, re-running after mask edits.

    ``data_r`` is mutated and keeps changes made by aborted passes, as in
    MATLAB. Returns (final data_c, cell_count, number of passes).
    """
    cur_iter = 1
    while True:
        last_count = cell_count
        if data_r is not None:
            data_r.set_assignment(
                "f", assign(data_r, data_c, True, params.da_min, params.da_max)
            )
        data_c.set_assignment(
            "r", assign(data_c, data_r, False, params.da_min, params.da_max)
        )
        data_c.set_assignment(
            "f", assign(data_c, data_f, True, params.da_min, params.da_max)
        )
        final_iteration = cur_iter >= MAX_ITER_PER_FRAME
        no_edits = cur_iter >= SAFETY_MAX_PASSES
        if no_edits:
            warn(f"frame {t}: {cur_iter - 1} passes, finishing without mask edits")
        cell_count, reset = error_rez(
            t, data_c, data_r, data_f, params, cell_count, final_iteration, no_edits
        )
        if not reset:
            return data_c, cell_count, cur_iter
        cur_iter += 1
        cell_count = last_count
        data_c = Frame(data_c.labels)
        data_c.edited = True


# ---------------------------------------------------------------------------
# errorRez and helpers
# ---------------------------------------------------------------------------


def _mf(x) -> str:
    """num2str for a scalar or a label list."""
    if isinstance(x, list):
        return "  ".join(str(int(v)) for v in x)
    return str(int(x))


def create_new_cell(c: Frame, k: int, time: int, cell_count: int) -> int:
    cell_count += 1
    c.death[k] = time
    c.deathF[k] = 1
    c.birth[k] = time
    c.birthF[k] = 1
    c.age[k] = 1
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
        c.death[k] = time
        c.deathF[k] = 1
        c.birth[k] = r.birth[reg_r]
        c.birthF[k] = 0
        c.age[k] = r.age[reg_r] + 1
        c.divide[k] = 0
        c.ehist[k] = int(bool(r.ehist[reg_r]) or bool(err))
        c.stat0[k] = r.stat0[reg_r]
        c.sisterID[k] = r.sisterID[reg_r]
        c.motherID[k] = r.motherID[reg_r]
        c.daughterID[k] = []
        c.ID[k] = r.ID[reg_r]
        r.death[reg_r] = time
        r.deathF[reg_r] = 0
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
    c.death[s] = time
    c.deathF[s] = 1
    c.birth[s] = time
    c.birthF[s] = 1
    c.age[s] = 1
    c.divide[s] = 0
    c.ehist[s] = err
    c.stat0[s] = int(not err)
    c.sisterID[s] = sister_id
    c.motherID[s] = r.ID[reg_r]
    c.daughterID[s] = []
    c.ID[s] = cid


def create_division(c, r, mother, s1, s2, cell_count, time) -> int:
    # The [L1,L2,Sc] suffix of MATLAB's label is not computed (cellprops3).
    c.label[s1] = (
        f"Frame: {time}, reg: {s1} and {s2} . cell division from mother reg{mother}"
    )
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
    c.label[k] = f"Frame: {time}, reg: {k}. Map to minimum cost"
    continue_cell_line(c, k, r, map_cr[0] if first else map_cr[1], time, True)


def map_best_of_two(
    c, map_rc, r, map_cr, time, cell_count, data_f, remove_stray, to_delete
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
    if remove_stray and data_f is not None and _no_fw_mapping(c, remove):
        c.label[remove[0]] = (
            f"Frame: {time}, reg: {remove[0]} was not the best match for "
            f"{_mf(map_cr)} and was deleted.{keeper} was."
        )
        to_delete.extend(remove)
        reset = True
    else:
        for m in remove:
            cell_count = create_new_cell(c, m, time, cell_count)
            c.label[m] = (
                f"Frame: {time}, reg: {m} was not the best match for "
                f"{_mf(map_cr)}. Converted into a new cell."
            )
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


def delete_regions(c: Frame, regs: list[int]) -> None:
    """deleteRegions: clear each label inside its (pass-start) bounding box."""
    for mm in regs:
        x0, y0, w, h = c.bbox[mm]
        ymin, xmin = math.ceil(y0), math.ceil(x0)
        ys = slice(ymin - 1, ymin + math.floor(h) - 1)
        xs = slice(xmin - 1, xmin + math.floor(w) - 1)
        box = c.labels[ys, xs]
        box[box == mm] = 0


def error_rez(
    time: int,
    c: Frame,
    r: Frame | None,
    f: Frame | None,
    params,
    cell_count: int,
    ignore_error: bool,
    no_edits: bool = False,
) -> tuple[int, bool]:
    """Port of errorRez: resolve the maps into IDs; may edit c.labels.

    Returns (cell_count, reset). ``no_edits`` (safety cap, not in MATLAB)
    also blocks the edits MATLAB doesn't gate by ignoreError.
    """
    ignore_error = ignore_error or no_edits
    remove_stray = params.remove_stray and not no_edits
    min_area_merge = params.small_area_merge
    da_min, da_max = params.da_min, params.da_max
    to_delete: list[int] = []
    reset = False
    c_area = c.area
    c.ID[:] = 0
    mod_regions: set[int] = set()

    def merge(regs):
        nonlocal reset
        if no_edits:
            return
        done = merge_regions(c, regs)
        if done:
            c.edited = True
        reset = reset or done

    for k in range(1, c.num_regs + 1):
        if c.ID[k] != 0 or k in mod_regions:
            pass
        else:
            regs_r = c.map_r[k]  # rCellsFromC
            regs_f = c.map_f[k]  # fCellsFromC
            if regs_r:
                # MATLAB quirk: x = c{[a b]} keeps the first element.
                c_from_r = r.map_f[regs_r[0]]  # cCellsFromR
                c_transp = c.revmap_r[regs_r[0]]  # cCellsTransp
            else:
                c_from_r, c_transp = [], []

            if len(regs_r) == 0:  # A: stray
                if time != 1 and (not c.map_f[k] and remove_stray):
                    c.label[k] = (
                        f"Frame: {time}, reg: {k}. is a stray region - Deleted."
                    )
                    to_delete.append(k)
                    reset = True
                else:
                    if time != 1:
                        c.label[k] = f"Frame: {time}, reg: {k}. is a stray region."
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
                if (
                    good
                    and not ignore_error
                    and f is not None
                    and (no_match or same or small)
                ):
                    merge([s1, s2])
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
                        remove_stray,
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
                    if good and not ignore_error and (no_match or same or small):
                        merge([s1, s2])
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
                elif not k_in:
                    cell_count = create_new_cell(c, k, time, cell_count)
                    c.error_r[k] = 1
                    c.label[k] = (
                        f"Frame: {time}, reg: {k}. Incorrect Mapping 1 to 2 - "
                        "making a new cell"
                    )
                else:
                    c.label[k] = (
                        f"Frame: {time}, reg: {k}. Error not fixed - two to 1 but "
                        "don't know what to do."
                    )

            elif len(regs_r) > 1:  # E: 2 -> 1, try to split
                success = False
                if not ignore_error:
                    success = missing_seg_2to1(c, k, r, regs_r, f, regs_f)
                if success:
                    c.edited = True
                    c.error_r[k] = 0
                    c.label[k] = (
                        f"Frame: {time}, reg: {k}. Segment added to fix 2 to 1 error"
                    )
                    reset = True
                else:
                    map_to_best_of_two(c, k, r, regs_r, time)

            elif len(regs_r) == 1 and len(c_from_r) > 2:  # F: 1 -> 3+
                no_match = False  # MATLAB quirk: isempty of a cell array
                forw = [x for m in c_from_r for x in c.map_f[m]]
                forward_map = sorted(set(forw))
                occur = [forw.count(v) for v in forward_map]
                same = not no_match and len(forward_map) == 1
                some_same = not no_match and any(o > 1 for o in occur)
                if f is not None and (no_match or same):
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
                            remove_stray,
                            to_delete,
                        )
                        reset = reset or rst
                        mod_regions.update(mod)
                elif f is not None and some_same:
                    value_fw = [v for v, o in zip(forward_map, occur) if o > 1]
                    to_merge = [m for m in c_from_r if _eq_any(c.map_f[m], value_fw)]
                    merge(to_merge)  # MATLAB quirk: not gated by ignoreError
                    mod_regions.update(to_merge)

            else:  # G
                c.label[k] = f"Frame: {time}, reg: {k}. Error not fixed"

        if c.ID[k] == 0:
            cell_count = create_new_cell(c, k, time, cell_count)

    if to_delete:
        delete_regions(c, to_delete)
        c.edited = True
    return cell_count, reset


# ---------------------------------------------------------------------------
# trackOptiCellMarker
# ---------------------------------------------------------------------------


def cell_marker(states: list[Frame], min_cell_age: int) -> None:
    """Mark complete cell cycles (stat0 = 2) by walking back from divisions."""
    n_im = len(states)
    clist: list[list[int]] = []  # [ID, current region, birth]
    for i in range(n_im - 1, 1, -1):
        c = states[i - 1]
        for ii in range(1, c.num_regs + 1):
            if c.divide[ii] and c.stat0[ii] and not c.ehist[ii]:
                if i - c.birth[ii] >= min_cell_age:
                    clist.append([int(c.ID[ii]), ii, int(c.birth[ii])])
                else:
                    c.label[ii] += (
                        f"Frame: {i}, reg: {ii}. Division Failed: Cell age "
                        f"{i - c.birth[ii]}< min age {min_cell_age}."
                    )
        for row in clist:
            jj = row[1]
            if i >= row[2]:
                # MATLAB quirk: on an error (no backward map) the try/catch
                # skips the update, so jj goes stale for the next frame.
                if jj > c.num_regs or not c.map_r[jj]:
                    continue
                row[1] = c.map_r[jj][0]
                c.stat0[jj] = 2
