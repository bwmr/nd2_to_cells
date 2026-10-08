"""Compare Python missingSeg2to1 with MATLAB on dumped split calls.

For every split_*.mat in a dump directory (dev/dump_splits.py), re-runs the
Python split and compares success and the resulting mask with MATLAB results
written by dev/matlab/oracle_splits.m: *_ml1.mat (1 thread, as on Windows)
and *_ml.mat (default threads), when present.

Usage: python dev/compare_splits.py <dump dir>
"""

import re
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nd2_to_cells.mask_edit import missing_seg_2to1  # noqa: E402


def python_split(d):
    frame_c = SimpleNamespace(labels=d["rl"].astype(np.int32), num_regs=int(d["num_regs"]))
    frame_r = SimpleNamespace(labels=d["rlr"].astype(np.int32))
    rlf = d["rlf"]
    frame_f = SimpleNamespace(labels=rlf.astype(np.int32)) if rlf.size else None
    regs_r = [int(x) for x in np.atleast_1d(d["regR"])]
    regs_f = [int(x) for x in np.atleast_1d(d["regF"])] if np.size(d["regF"]) else []
    ok = missing_seg_2to1(frame_c, int(d["regC"]), frame_r, regs_r, frame_f, regs_f)
    return ok, frame_c.labels


def main():
    dump = Path(sys.argv[1])
    cases = sorted(p for p in dump.glob("split_*.mat") if re.fullmatch(r"split_t\d{3}_\d{3}\.mat", p.name))
    tally: dict[str, Counter] = {"py vs ml1": Counter(), "ml10 vs ml1": Counter()}
    rows = []
    for case in cases:
        d = sio.loadmat(case, squeeze_me=True)
        t0 = time.time()
        ok, rl = python_split(d)
        dt = time.time() - t0
        res = {"py": (ok, rl)}
        for tag, suffix in (("ml1", "_ml1.mat"), ("ml10", "_ml.mat")):
            f = case.with_name(case.stem + suffix)
            if f.exists():
                m = sio.loadmat(f, squeeze_me=True)
                res[tag] = (bool(m["ml_success"]), m["ml_rl"].astype(np.int32))
        line = f"{case.stem}: py {int(ok)} ({dt:.1f} s)"
        if "ml1" in res:
            ref_ok, ref_rl = res["ml1"]
            for name, key in (("py vs ml1", "py"), ("ml10 vs ml1", "ml10")):
                if key not in res:
                    continue
                k_ok, k_rl = res[key]
                same_mask = np.array_equal(k_rl, ref_rl)
                outcome = "same mask" if same_mask else (
                    "same success, mask differs" if k_ok == ref_ok else "success differs"
                )
                tally[name][outcome] += 1
            line += f" | ml1 {int(ref_ok)}" + (f" ml10 {int(res['ml10'][0])}" if "ml10" in res else "")
            if ok and ref_ok:
                new = rl != d["rl"].astype(np.int32)
                ref_new = ref_rl != d["rl"].astype(np.int32)
                line += f" | changed px py {int(new.sum())} ml1 {int(ref_new.sum())} both {int((new & ref_new & (rl == ref_rl)).sum())}"
        rows.append(line)
    for r in rows:
        print(r)
    for name, c in tally.items():
        if c:
            print(f"{name}: {dict(c)} of {sum(c.values())}")


if __name__ == "__main__":
    main()
