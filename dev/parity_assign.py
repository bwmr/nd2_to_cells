"""Parity checks 1 and 2: region table and assignment vs SuperSegger err files.

For every frame t of a position:
  1. region_props(err(t) mask) must equal err(t).regs.props.
  2. assign(err(t) -> err(t-1), reverse) must reproduce map.r / cost.r /
     idsC.r / idsR.r / dA.r, and assign(err(t) -> err(t+1), forward) must
     reproduce map.f / cost.f / idsC.f / idsF.f / dA.f. (Both maps on disk are
     computed against the final, edited masks; error.r and revmap.r are left
     out because errorRez rewrites them.)

Usage: python dev/parity_assign.py <xy dir> [--start T] [--stop T]
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ssref  # noqa: E402

from nd2_to_cells.assign import assign  # noqa: E402
from nd2_to_cells.link import Frame  # noqa: E402

DA_MIN, DA_MAX = -0.2, 0.3


def compare(tag, t, res, regs, sub, ids_f_name, n, fails, examples, cost_diff):
    m = getattr(regs, "map")
    ref_map = ssref.cell_list(getattr(m, sub), n)
    ref_cost = ssref.vec(getattr(regs.cost, sub))
    ref_ids_c = ssref.ids(getattr(regs.idsC, sub))
    ref_ids_f = ssref.ids(getattr(getattr(regs, ids_f_name), sub))
    ref_da = ssref.vec(getattr(regs.dA, sub))
    if res.assignments != ref_map:
        fails[f"{tag}.map"] += 1
        bad = [k for k in range(1, n + 1) if res.assignments[k] != ref_map[k]]
        examples.append(
            f"t={t} {tag}.map regs {bad[:5]}: "
            f"py {[res.assignments[k] for k in bad[:3]]} "
            f"ml {[ref_map[k] for k in bad[:3]]}"
        )
    if not ssref.nan_equal(res.cost, ref_cost, rtol=1e-9):
        fails[f"{tag}.cost"] += 1
        examples.append(f"t={t} {tag}.cost len py {res.cost.size} ml {ref_cost.size}")
    elif res.cost.size:
        with np.errstate(invalid="ignore"):
            d = np.nanmax(np.abs(res.cost - ref_cost))
        cost_diff.append(0.0 if np.isnan(d) else d)
    if not ssref.nan_equal(res.ids_c, ref_ids_c):
        fails[f"{tag}.idsC"] += 1
    if not ssref.nan_equal(res.ids_f, ref_ids_f):
        fails[f"{tag}.idsF"] += 1
    if not ssref.nan_equal(res.d_area[1:], ref_da, rtol=1e-12):
        fails[f"{tag}.dA"] += 1
        examples.append(f"t={t} {tag}.dA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xy_dir", type=Path)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--stop", type=int, default=None)
    args = ap.parse_args()

    files = ssref.seg_files(args.xy_dir, "err")
    n_files = len(files)
    stop = args.stop or n_files
    cache: dict[int, tuple] = {}

    def get(i):
        if i not in cache:
            regs = ssref.load_regs(files[i - 1])
            cache[i] = (regs, Frame(np.asarray(regs.regs_label)))
        return cache[i]

    fails: Counter = Counter()
    examples: list[str] = []
    cost_diff: list[float] = []
    t0 = time.time()
    for t in range(args.start, stop + 1):
        regs, fr = get(t)
        n = int(regs.num_regs)
        area, cent, bbox = ssref.props_arrays(regs)
        if not (
            fr.num_regs == n
            and ssref.nan_equal(fr.area, area)
            and ssref.nan_equal(fr.centroid[1:], cent[1:])
            and ssref.nan_equal(fr.bbox[1:], bbox[1:])
        ):
            fails["props"] += 1
            examples.append(f"t={t} props")
        prev = get(t - 1)[1] if t > 1 else None
        res = assign(fr, prev, False, DA_MIN, DA_MAX)
        compare("r", t, res, regs, "r", "idsR", n, fails, examples, cost_diff)
        nxt = get(t + 1)[1] if t < n_files else None
        res = assign(fr, nxt, True, DA_MIN, DA_MAX)
        compare("f", t, res, regs, "f", "idsF", n, fails, examples, cost_diff)
        cache.pop(t - 1, None)
        if t % 20 == 0:
            print(f"  t={t} ({time.time() - t0:.0f} s) fails so far {dict(fails)}")

    print(f"{args.xy_dir.name}: frames {args.start}-{stop}, {time.time() - t0:.0f} s")
    print("failures per check:", dict(fails) or "none")
    if cost_diff:
        print(f"max |cost diff| on matching frames: {max(cost_diff):.3g}")
    for e in examples[:25]:
        print("  ", e)


if __name__ == "__main__":
    main()
