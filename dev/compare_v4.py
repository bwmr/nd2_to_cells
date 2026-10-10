"""Compare the MATLAB-faithful registration (ITK v3 path) with ITK v4 (SimpleITK).

For every dumped split call (dev/dump_splits.py), runs missingSeg2to1 with
nd2_to_cells.imreg (as SuperSegger does it) and with a SimpleITK v4
registration (regular-step gradient descent, mean squares, affine, 3-level
pyramid, MATLAB's optimizer settings without MATLAB's per-level step
schedule), and reports time, success and a quality proxy: the mean IoU of the
two pieces with the previous frame's two cells and, when the next frame also
has two, with those (best matching, unregistered).

Needs SimpleITK (pip install SimpleITK). If *_ml1.mat oracle results exist,
MATLAB (1 thread) is included in the table.

Usage: python dev/compare_v4.py <dump dir>
"""

import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nd2_to_cells.mask_edit as mask_edit  # noqa: E402
from nd2_to_cells import imreg  # noqa: E402

register_v4 = imreg.register_masks_v4


def run(d, register):
    mask_edit.register_masks = register
    frame_c = SimpleNamespace(
        labels=d["rl"].astype(np.int32), num_regs=int(d["num_regs"])
    )
    frame_r = SimpleNamespace(labels=d["rlr"].astype(np.int32))
    rlf = d["rlf"]
    frame_f = SimpleNamespace(labels=rlf.astype(np.int32)) if rlf.size else None
    regs_r = [int(x) for x in np.atleast_1d(d["regR"])]
    regs_f = [int(x) for x in np.atleast_1d(d["regF"])] if np.size(d["regF"]) else []
    t0 = time.time()
    ok = mask_edit.missing_seg_2to1(
        frame_c, int(d["regC"]), frame_r, regs_r, frame_f, regs_f
    )
    return ok, frame_c.labels, time.time() - t0


def iou(a, b):
    u = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / u if u else 0.0


def quality(d, rl):
    """Mean IoU of the two pieces with the r cells (and f cells if two)."""
    n = int(d["num_regs"])
    pieces = [rl == n + 1, rl == n + 2]
    if not (pieces[0].any() and pieces[1].any()):
        return None, None
    out = []
    for key, regs in (("rlr", d["regR"]), ("rlf", d["regF"])):
        regs = [int(x) for x in np.atleast_1d(regs)]
        if len(regs) < 2 or not d[key].size:
            out.append(None)
            continue
        cells = [d[key] == regs[0], d[key] == regs[1]]
        straight = (iou(pieces[0], cells[0]) + iou(pieces[1], cells[1])) / 2
        crossed = (iou(pieces[0], cells[1]) + iou(pieces[1], cells[0])) / 2
        out.append(max(straight, crossed))
    return tuple(out)


def main():
    dump = Path(sys.argv[1])
    cases = sorted(
        p
        for p in dump.glob("split_*.mat")
        if re.fullmatch(r"split_t\d{3}_\d{3}\.mat", p.name)
    )
    v3 = mask_edit.register_masks
    stats = {
        k: {"ok": 0, "t": 0.0, "qr": [], "qf": []}
        for k in ("v3 port", "v4 sitk", "MATLAB 1t")
    }
    agree = 0
    for case in cases:
        d = sio.loadmat(case, squeeze_me=True)
        results = {"v3 port": run(d, v3), "v4 sitk": run(d, register_v4)}
        ml = case.with_name(case.stem + "_ml1.mat")
        if ml.exists():
            m = sio.loadmat(ml, squeeze_me=True)
            results["MATLAB 1t"] = (
                bool(m["ml_success"]),
                m["ml_rl"].astype(np.int32),
                float(m["ml_time"]),
            )
        agree += results["v3 port"][0] == results["v4 sitk"][0]
        line = case.stem
        for k, (ok, rl, dt) in results.items():
            s = stats[k]
            s["ok"] += ok
            s["t"] += dt
            qr, qf = quality(d, rl) if ok else (None, None)
            if qr is not None:
                s["qr"].append(qr)
            if qf is not None:
                s["qf"].append(qf)
            line += f" | {k}: {int(ok)} {dt:4.1f}s"
        print(line, flush=True)
    print(f"\n{len(cases)} split calls; v3 and v4 agree on success in {agree}")
    for k, s in stats.items():
        n = (
            len(cases)
            if k != "MATLAB 1t"
            else sum(1 for c in cases if c.with_name(c.stem + "_ml1.mat").exists())
        )
        if not n:
            continue
        mean = lambda v: f"{np.mean(v):.3f} (n={len(v)})" if v else "-"  # noqa: E731
        print(
            f"{k:10s}: success {s['ok']}/{n}, time {s['t'] / n:.1f} s/call, "
            f"IoU with previous cells {mean(s['qr'])}, with next cells {mean(s['qf'])}"
        )


if __name__ == "__main__":
    main()
