"""Dump the inputs and Python outputs of every missingSeg2to1 call.

Runs teacher-forced frames (as in parity_link.py) and writes one .mat file per
split call to <out dir>, for dev/matlab/oracle_splits.m to replay in MATLAB.

Usage: python dev/dump_splits.py <xy dir> <out dir> [--start T] [--stop T]
"""

import argparse
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parity_link  # noqa: E402
import ssref  # noqa: E402

import nd2_to_cells.link as link  # noqa: E402
from nd2_to_cells.mask_edit import missing_seg_2to1  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xy_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--stop", type=int, default=None)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    xy = args.xy_dir
    errs = ssref.seg_files(xy, "err")
    masks = sorted((xy / "masks").glob("*cp_masks.png"))
    stop = args.stop or len(errs)
    state = {"t": 0, "n": 0}

    def wrapped(frame_c, reg_c, frame_r, regs_r, frame_f, regs_f):
        before = frame_c.labels.copy()
        ok = missing_seg_2to1(frame_c, reg_c, frame_r, regs_r, frame_f, regs_f)
        state["n"] += 1
        sio.savemat(
            args.out_dir / f"split_t{state['t']:03d}_{state['n']:03d}.mat",
            {
                "rl": before.astype(np.uint16),
                "rlr": frame_r.labels.astype(np.uint16),
                "rlf": (
                    frame_f.labels.astype(np.uint16)
                    if frame_f is not None
                    else np.zeros((0, 0), np.uint16)
                ),
                "regC": float(reg_c),
                "regR": np.array(regs_r, dtype=float),
                "regF": np.array(regs_f, dtype=float),
                "num_regs": float(frame_c.num_regs),
                "py_rl": frame_c.labels.astype(np.uint16),
                "py_success": float(ok),
            },
            do_compression=True,
        )
        return ok

    link.missing_seg_2to1 = wrapped
    cell_count_of = parity_link.cell_counts(xy)
    prev = ssref.load_regs(errs[args.start - 2]) if args.start > 1 else None
    for t in range(args.start, stop + 1):
        state["t"], state["n"] = t, 0
        cur = ssref.load_regs(errs[t - 1])
        data_r = parity_link.frame_from_err(prev, t - 1) if prev is not None else None
        data_c = link.Frame(iio.imread(masks[t - 1]))
        data_f = link.Frame(iio.imread(masks[t])) if t < len(errs) else None
        link.process_frame(
            t, data_c, data_r, data_f, parity_link.PARAMS, cell_count_of(t),
            warn=lambda m: None,
        )  # fmt: skip
        if state["n"]:
            print(f"t={t}: {state['n']} split call(s)", flush=True)
        prev = cur


if __name__ == "__main__":
    main()
