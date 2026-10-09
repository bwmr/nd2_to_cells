"""Parity check 3: teacher-forced errorRez / driver vs SuperSegger err files.

For each frame t, data_r is MATLAB's err(t-1) (fields that frame t mutates are
reset to their pre-t values: death = t-1, deathF = 1, divide = 0,
daughterID = []; stat0 is reset from 2 to 1). All passes then run from the
Omnipose mask of t, and the result is compared with err(t) / masksOS(t):
  - final mask (with edits) == MATLAB's final mask
  - per region: ID, birth, birthF, age, ehist, stat0 (pre-CellMarker), motherID,
    sisterID, error.r (except mothers that divide at t+1), error.label prefix
  - data_r side effects: death, deathF, divide, daughterID of err(t-1)

Usage: python dev/parity_link.py <xy dir> [--start T] [--stop T]
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import imageio.v3 as iio
import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ssref  # noqa: E402

from nd2_to_cells.link import Frame, process_frame  # noqa: E402

# Effective MATLAB settings of the reference run (GUI, 100XEc).
PARAMS = SimpleNamespace(
    da_min=-0.2, da_max=0.3, small_area_merge=50, remove_stray=False, min_cell_age=3
)
C_FIELDS = ("ID", "birth", "birthF", "age", "ehist", "motherID", "sisterID")
R_FIELDS = ("death", "deathF", "divide")


def int_vec(regs, name, n):
    v = ssref.vec(getattr(regs, name))
    return np.rint(v).astype(np.int64) if v.size == n else np.zeros(n, np.int64)


def frame_from_err(regs, t_prev: int) -> Frame:
    fr = Frame(np.asarray(regs.regs_label))
    n = fr.num_regs
    for name in C_FIELDS:
        getattr(fr, name)[1:] = int_vec(regs, name, n)
    fr.stat0[1:] = int_vec(regs, "stat0", n) > 0
    fr.death[1:] = t_prev
    fr.deathF[1:] = 1
    fr.divide[1:] = 0
    fr.daughterID = [[] for _ in range(n + 1)]
    fr.error_r[1:] = ssref.vec(regs.error.r)
    return fr


def cell_counts(xy: Path):
    """t -> cell_count before frame t (largest ID born before t, from clist)."""
    clist = sio.loadmat(xy / "clist.mat", squeeze_me=True, struct_as_record=False)
    cdef = [str(x) for x in np.atleast_1d(clist["def"])]
    cdata = np.asarray(clist["data"])
    c_id = cdata[:, cdef.index("Cell ID")]
    c_birth = cdata[:, cdef.index("Cell birth time")]
    return lambda t: int(c_id[c_birth <= t - 1].max()) if t > 1 else 0


def labels_of(regs, n):
    lab = regs.error.label
    lab = np.atleast_1d(lab) if n != 1 else np.array([lab], dtype=object)
    return [""] + [x if isinstance(x, str) else "" for x in lab]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xy_dir", type=Path)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--stop", type=int, default=None)
    args = ap.parse_args()
    params = PARAMS

    xy = args.xy_dir
    errs = ssref.seg_files(xy, "err")
    masks = sorted((xy / "masks").glob("*cp_masks.png"))
    n_files = len(errs)
    stop = args.stop or n_files

    cell_count_of = cell_counts(xy)

    fails: Counter = Counter()
    examples: list[str] = []
    passes: Counter = Counter()
    t0 = time.time()
    prev_regs = ssref.load_regs(errs[args.start - 2]) if args.start > 1 else None
    for t in range(args.start, stop + 1):
        cur = ssref.load_regs(errs[t - 1])
        n = int(cur.num_regs)
        data_r = frame_from_err(prev_regs, t - 1) if prev_regs is not None else None
        data_c = Frame(iio.imread(masks[t - 1]))
        data_f = Frame(iio.imread(masks[t])) if t < n_files else None
        cell_count = cell_count_of(t)
        out, _, n_pass = process_frame(
            t, data_c, data_r, data_f, params, cell_count, warn=lambda m: None
        )
        passes[n_pass] += 1
        ml_mask = np.asarray(cur.regs_label).astype(np.int32)
        if out.num_regs != n or not np.array_equal(out.labels, ml_mask):
            fails["mask"] += 1
            same_shape = out.labels.shape == ml_mask.shape
            n_px = int(np.sum(out.labels != ml_mask)) if same_shape else "shape"
            examples.append(
                f"t={t} mask: py num_regs {out.num_regs} edited {out.edited}, "
                f"ml num_regs {n}, differing px {n_px}"
            )
        else:
            bad = []
            for name in C_FIELDS:
                if not np.array_equal(getattr(out, name)[1:], int_vec(cur, name, n)):
                    bad.append(name)
            if not np.array_equal(
                out.stat0[1:], (int_vec(cur, "stat0", n) > 0).astype(int)
            ):
                bad.append("stat0")
            keep = int_vec(cur, "divide", n) == 0
            if not np.array_equal(
                (out.error_r[1:] > 0)[keep], (ssref.vec(cur.error.r) > 0)[keep]
            ):
                bad.append("error.r")
            ml_lab = labels_of(cur, n)
            if any(not ml_lab[k].startswith(out.label[k]) for k in range(1, n + 1)):
                bad.append("label")
            for b in bad:
                fails[b] += 1
            if bad:
                k = (
                    next(
                        k
                        for k in range(1, n + 1)
                        if out.ID[k] != int_vec(cur, "ID", n)[k - 1]
                        or not ml_lab[k].startswith(out.label[k])
                    )
                    if any(b in ("ID", "label") for b in bad)
                    else None
                )
                ex = f"t={t} fields {bad}"
                if k:
                    ml_id = int_vec(cur, "ID", n)[k - 1]
                    ex += f" reg {k}: py ID {out.ID[k]} '{out.label[k]}'"
                    ex += f" | ml ID {ml_id} '{ml_lab[k]}'"
                examples.append(ex)
        if data_r is not None:
            nr = data_r.num_regs
            bad = [
                name
                for name in R_FIELDS
                if not np.array_equal(
                    getattr(data_r, name)[1:], int_vec(prev_regs, name, nr)
                )
            ]
            if data_r.daughterID != ssref.cell_list(prev_regs.daughterID, nr):
                bad.append("daughterID")
            for b in bad:
                fails["r." + b] += 1
            if bad:
                examples.append(f"t={t} data_r fields {bad}")
        prev_regs = cur
        if t % 20 == 0:
            print(f"  t={t} ({time.time() - t0:.0f} s) {dict(fails)}", flush=True)

    print(f"{xy.name}: frames {args.start}-{stop}, {time.time() - t0:.0f} s")
    print("passes per frame:", dict(sorted(passes.items())))
    print("failures per check:", dict(fails) or "none")
    for e in examples[:40]:
        print("  ", e)


if __name__ == "__main__":
    main()
