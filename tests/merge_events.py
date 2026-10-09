"""Merge events in track output: issues A and B of the 260429 linking report.

A merge event is a frame t at which track T absorbs another track X: T's
mask at t covers at least HALF of T's own mask at t-1 and at least HALF of
X's mask at t-1. T's area need not jump: when the larger cell keeps its ID,
the merged object can be less than JUMP (the report's 1.3x) times its area.

Kind "A" (re-merge after division): X is T's sister. Kind "B" (neighbour
absorbed): any other X within NEAR px of T at t-1. Kind "far": X is farther
away, so the merged object spans distant cells; in the SuperSegger port this
comes from MATLAB's label-collision quirk (two regions summed into one label
in mask edits). Only the track output is read, so this works for any
linker: per-cell cell*.h5 files (`nd2_to_cells track`) or one cells.h5
(`--consolidated`, or the IoU linker on main). Tracks whose mask is empty in
every frame (SuperSegger's ghost labels) are skipped.

Usage, for whole positions:
    python tests/merge_events.py <cell dir> [<cell dir> ...] [--csv events.csv]
"""

import argparse
import csv
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import h5py
import numpy as np

JUMP = 1.3  # area ratio between frames that the report counts as a jump
HALF = 0.5  # fraction of the absorbed track's mask that must be covered
NEAR = 50  # px between T and X at t-1 up to which X counts as a neighbour


@dataclass
class Track:
    id: int
    key: str  # file stem or cells.h5 group name
    frames: np.ndarray  # 0-based absolute frame per mask plane
    mask: np.ndarray  # (H, W, T) bool, fixed crop box
    x0: int  # crop box top-left in the full image
    y0: int
    mother: int
    sister: int
    area: np.ndarray = field(init=False)  # pixels per plane
    box: np.ndarray = field(init=False)  # (T, 4) y0, y1, x0, x1 in the full image

    def __post_init__(self):
        self.area = self.mask.reshape(-1, self.mask.shape[2]).sum(0)
        rows, cols = self.mask.any(axis=1), self.mask.any(axis=0)  # (H, T), (W, T)
        self.box = np.stack(
            [
                self.y0 + rows.argmax(0),
                self.y0 + len(rows) - rows[::-1].argmax(0),
                self.x0 + cols.argmax(0),
                self.x0 + len(cols) - cols[::-1].argmax(0),
            ],
            axis=1,
        )
        self.box[self.area == 0] = 0


@dataclass
class Event:
    kind: str  # "A", "B" or "far"
    frame: int  # 0-based frame at which T holds the merged object
    cell: str  # T
    absorbed: str  # X
    area_before: int  # T at t-1
    area_after: int  # T at t
    absorbed_area: int  # X at t-1
    absorbed_ends: bool  # X's last frame is t-1
    frames_since_birth: int  # index of t within T
    gap: int  # px between the bounding boxes of T and X at t-1


def load_tracks(cell_dir: Path) -> dict[int, Track]:
    """Tracks with at least one non-empty mask plane, keyed by cell ID."""
    tracks: dict[int, Track] = {}

    def add(key: str, g) -> None:
        mask = g["mask"][:]
        if not mask.any():
            return
        tracks[int(key[4:])] = Track(
            id=int(key[4:]),
            key=key,
            frames=g["frames"][:].astype(int),
            mask=mask,
            x0=int(g["r_offset"][0, 0]),
            y0=int(g["r_offset"][0, 1]),
            mother=int(g["motherID"][()]),
            sister=int(g["sisterID"][()]),
        )

    cell_dir = Path(cell_dir)
    if (cell_dir / "cells.h5").exists():
        with h5py.File(cell_dir / "cells.h5", "r") as f:
            for key, g in f.items():
                add(key, g)
    else:
        for path in sorted(cell_dir.glob("[cC]ell*.h5")):
            with h5py.File(path, "r") as g:
                add(path.stem, g)
    return tracks


def _overlap(a: Track, i: int, b: Track, j: int) -> int:
    """Pixels shared by a's mask plane i and b's mask plane j."""
    ha, wa = a.mask.shape[:2]
    hb, wb = b.mask.shape[:2]
    y0, y1 = max(a.y0, b.y0), min(a.y0 + ha, b.y0 + hb)
    x0, x1 = max(a.x0, b.x0), min(a.x0 + wa, b.x0 + wb)
    if y0 >= y1 or x0 >= x1:
        return 0
    ma = a.mask[y0 - a.y0 : y1 - a.y0, x0 - a.x0 : x1 - a.x0, i]
    mb = b.mask[y0 - b.y0 : y1 - b.y0, x0 - b.x0 : x1 - b.x0, j]
    return int(np.count_nonzero(ma & mb))


def _gap(a: np.ndarray, b: np.ndarray) -> int:
    """Chebyshev distance between two (y0, y1, x0, x1) boxes; 0 if they touch."""
    dy = max(0, b[0] - a[1], a[0] - b[1])
    dx = max(0, b[2] - a[3], a[2] - b[3])
    return int(max(dy, dx))


def find_events(tracks: dict[int, Track]) -> list[Event]:
    present: dict[int, list[tuple[Track, int]]] = defaultdict(list)
    for tr in tracks.values():
        for j, t in enumerate(tr.frames):
            if tr.area[j]:
                present[int(t)].append((tr, j))
    boxes = {t: np.array([o.box[j] for o, j in lst]) for t, lst in present.items()}
    events = []
    for tr in tracks.values():
        for i in range(1, len(tr.frames)):
            t = int(tr.frames[i])
            if tr.frames[i - 1] != t - 1 or not tr.area[i - 1] or not tr.area[i]:
                continue
            if _overlap(tr, i, tr, i - 1) < HALF * tr.area[i - 1]:
                continue  # T moved onto another object: not a merge
            y0, y1, x0, x1 = tr.box[i]
            b = boxes[t - 1]
            hit = (b[:, 0] < y1) & (b[:, 1] > y0) & (b[:, 2] < x1) & (b[:, 3] > x0)
            for k in np.flatnonzero(hit):
                other, j = present[t - 1][k]
                if other is tr:
                    continue
                if _overlap(tr, i, other, j) >= HALF * other.area[j]:
                    gap = _gap(tr.box[i - 1], other.box[j])
                    if other.id == tr.sister:
                        kind = "A"
                    else:
                        kind = "B" if gap <= NEAR else "far"
                    events.append(
                        Event(
                            kind=kind,
                            frame=t,
                            cell=tr.key,
                            absorbed=other.key,
                            area_before=int(tr.area[i - 1]),
                            area_after=int(tr.area[i]),
                            absorbed_area=int(other.area[j]),
                            absorbed_ends=int(other.frames[-1]) == t - 1,
                            frames_since_birth=i,
                            gap=gap,
                        )
                    )
    return sorted(events, key=lambda e: (e.frame, e.cell, e.absorbed))


def summary(tracks: dict[int, Track], events: list[Event]) -> dict[str, int]:
    pairs = jumps = 0
    for tr in tracks.values():
        area = tr.area
        ok = (np.diff(tr.frames) == 1) & (area[:-1] > 0)
        pairs += int(ok.sum())
        jumps += int((ok & (area[1:] > JUMP * area[:-1])).sum())
    return {
        "tracks": len(tracks),
        "frame pairs": pairs,
        "jumps": jumps,
        "A events": sum(e.kind == "A" for e in events),
        "B events": sum(e.kind == "B" for e in events),
        "far events": sum(e.kind == "far" for e in events),
        "absorbed tracks that end": sum(e.absorbed_ends for e in events),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cell_dirs", nargs="+", type=Path)
    ap.add_argument("--csv", type=Path, help="write all events to this file")
    args = ap.parse_args()
    rows = []
    for cell_dir in args.cell_dirs:
        tracks = load_tracks(cell_dir)
        events = find_events(tracks)
        s = summary(tracks, events)
        s["jumps"] = f"{s['jumps']} ({s['jumps'] / max(s['frame pairs'], 1):.1%})"
        print(f"{cell_dir}: {s}")
        rows += [{"cell_dir": str(cell_dir), **asdict(e)} for e in events]
    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
