"""Compare nd2_to_cells track output with SuperSegger output for one position.

Reads our cell*.h5 / Cell*.h5 files and SuperSegger's cell/Cell*.mat files
(lineage fields only) and, with --frames, the err files (regs_label + ID):
  1. totals: cells, complete cycles (Cell), divisions, cells born from one
  2. per-ID agreement of birth, death, divide, motherID, sisterID, daughterID,
     number of frames and Cell/cell naming; first birth frame of a mismatch
  3. (--frames) per frame: identical segmentation partition, and identical
     cell IDs on it
  4. (--frames) lineage structure independent of ID numbering: each
     SuperSegger cell is matched to the cell of ours born in the same frame
     with the largest mask overlap there; death, divide, number of frames,
     mother and daughters are then compared through that matching

Usage: python dev/compare_cells.py <our cell dir> <SuperSegger xy dir>
           [--mat-dir cell_old] [--frames]
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ssref  # noqa: E402

FIELDS = (
    "birth",
    "death",
    "divide",
    "motherID",
    "sisterID",
    "daughterID",
    "n_frames",
    "complete",
)


def read_ours(cell_dir: Path):
    cells = {}
    for f in sorted(cell_dir.glob("[cC]ell*.h5")):
        with h5py.File(f, "r") as h5:
            cid = int(f.stem[4:])
            cells[cid] = {
                "birth": int(h5["birth"][()]),
                "death": int(h5["death"][()]),
                "divide": int(h5["divide"][()]),
                "motherID": int(h5["motherID"][()]),
                "sisterID": int(h5["sisterID"][()]),
                "daughterID": tuple(int(x) for x in h5["daughterID"][:]),
                "n_frames": len(h5["frames"][:]),
                "complete": f.name.startswith("Cell"),
                "frames": h5["frames"][:],
                "r_offset": h5["r_offset"][:],
                "mask": h5["mask"][:],
            }
    return cells


def read_matlab(mat_dir: Path):
    cells = {}
    names = ["birth", "death", "divide", "motherID", "sisterID", "daughterID", "CellA"]
    for f in sorted(mat_dir.glob("[cC]ell*.mat")):
        m = sio.loadmat(f, squeeze_me=True, variable_names=names[:-1])
        cid = int(f.stem[4:])
        n = sio.whosmat(f)
        cella = next((s for s in n if s[0] == "CellA"), None)
        n_cella = int(np.prod(cella[1])) if cella else 0
        cells[cid] = {
            "birth": int(m["birth"]),
            "death": int(m["death"]),
            "divide": int(m["divide"]),
            "motherID": int(m["motherID"]),
            "sisterID": int(m["sisterID"]),
            "daughterID": tuple(int(x) for x in np.atleast_1d(m["daughterID"])),
            # MATLAB duplicates CellA for one-frame cells (birthF and deathF).
            "n_frames": 1 if m["birth"] == m["death"] and n_cella == 2 else n_cella,
            "complete": f.name.startswith("Cell"),
        }
    return cells


def paint(ours, frame: int, shape):
    """Label image of frame (0-based) from our HDF5 crops, labelled by cell ID."""
    img = np.zeros(shape, dtype=np.int64)
    for cid, c in ours.items():
        idx = np.nonzero(c["frames"] == frame)[0]
        if not idx.size:
            continue
        k = idx[0]
        x0, y0 = c["r_offset"][k]
        m = c["mask"][:, :, k]
        img[y0 : y0 + m.shape[0], x0 : x0 + m.shape[1]][m] = cid
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cell_dir", type=Path)
    ap.add_argument("xy_dir", type=Path)
    ap.add_argument("--mat-dir", default="cell_old")
    ap.add_argument("--frames", action="store_true")
    args = ap.parse_args()
    ours = read_ours(args.cell_dir)
    ml = read_matlab(args.xy_dir / args.mat_dir)

    def totals(c):
        return {
            "cells": len(c),
            "Cell (complete)": sum(v["complete"] for v in c.values()),
            "divisions": sum(v["divide"] for v in c.values()),
            "born from division": sum(v["motherID"] != 0 for v in c.values()),
        }

    print("totals   ours:", totals(ours))
    print("totals MATLAB:", totals(ml))

    common = sorted(set(ours) & set(ml))
    diffs: Counter = Counter()
    first_bad = None
    for cid in common:
        bad = [f for f in FIELDS if ours[cid][f] != ml[cid][f]]
        for b in bad:
            diffs[b] += 1
        if bad and (first_bad is None or ml[cid]["birth"] < ml[first_bad]["birth"]):
            first_bad = cid
    same = sum(all(ours[c][f] == ml[c][f] for f in FIELDS) for c in common)
    print(
        f"IDs in both: {len(common)} (ours only {len(set(ours) - set(ml))}, "
        f"MATLAB only {len(set(ml) - set(ours))}); identical in all fields: {same}"
    )
    print("fields that differ (count of IDs):", dict(diffs))
    if first_bad is not None:
        print(
            f"earliest-born differing ID {first_bad}: ours "
            f"{ {f: ours[first_bad][f] for f in FIELDS} }\n    MATLAB {ml[first_bad]}"
        )

    if args.frames:
        errs = ssref.seg_files(args.xy_dir, "err")
        part_same = id_same = 0
        first_part = first_id = None
        match: dict[int, int] = {}  # SuperSegger ID -> our ID
        real = 0  # SuperSegger cells with pixels in their birth frame
        for t, path in enumerate(errs):
            regs = ssref.load_regs(path)
            lab = np.asarray(regs.regs_label).astype(np.int64)
            ids = np.concatenate([[0], np.rint(ssref.vec(regs.ID)).astype(np.int64)])
            ours_img = paint(ours, t, lab.shape)
            for k in range(1, len(ids)):
                mid = int(ids[k])
                if mid in ml and ml[mid]["birth"] == t + 1:
                    hit = ours_img[lab == k]
                    real += hit.size > 0
                    hit = hit[hit > 0]
                    if hit.size:
                        cand = int(np.bincount(hit).argmax())
                        if ours[cand]["birth"] == t + 1:
                            match[mid] = cand
            fg = (lab > 0) | (ours_img > 0)
            pairs = set(zip(ours_img[fg].tolist(), lab[fg].tolist()))
            a = Counter(p[0] for p in pairs)
            b = Counter(p[1] for p in pairs)
            ok = all(a[p[0]] == 1 and b[p[1]] == 1 and p[0] and p[1] for p in pairs)
            if ok:
                part_same += 1
                ml_ids = ids[np.clip(lab, 0, len(ids) - 1)]
                if np.array_equal(ours_img, np.where(lab > 0, ml_ids, 0)):
                    id_same += 1
                elif first_id is None:
                    first_id = t + 1
            elif first_part is None:
                first_part = t + 1
        print(
            f"frames with identical partition: {part_same}/{len(errs)} "
            f"(first different: t={first_part}); identical IDs too: {id_same} "
            f"(first ID difference on an identical partition: t={first_id})"
        )
        lineage_report(ml, ours, match, real)


def lineage_report(ml, ours, match, real):
    """Compare lineage fields of matched cells through the ID matching."""
    tally: Counter = Counter()
    for mid, oid in match.items():
        m, o = ml[mid], ours[oid]
        checks = {
            "death": m["death"] == o["death"],
            "divide": m["divide"] == o["divide"],
            "n_frames": m["n_frames"] == o["n_frames"],
            "complete": m["complete"] == o["complete"],
            "mother": (m["motherID"] == 0 and o["motherID"] == 0)
            or match.get(m["motherID"]) == o["motherID"],
            "daughters": sorted(match.get(d, -1) for d in m["daughterID"] if d)
            == sorted(d for d in o["daughterID"] if d),
        }
        for k, v in checks.items():
            tally[k] += v
        tally["all"] += all(checks.values())
    n = len(match)
    print(
        f"cells matched by birth frame and overlap: {n} of {real} SuperSegger "
        f"cells with pixels at birth ({len(ml) - real} empty 'ghost' cells)"
    )
    print("  agreeing:", {k: f"{v / n:.1%}" for k, v in tally.items()} if n else "-")


if __name__ == "__main__":
    main()
