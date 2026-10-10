"""Cell tracker: links Omnipose masks across frames, writes cell*.h5.

The linking is a bug-for-bug port of SuperSegger/OmniSegger's trackOpti
steps 3-10, restricted to what determines lineages (see
docs/supersegger_port.md):
    1. trackOptiLinkCellMulti: frame-to-frame assignment (assign.py), error
       resolution with mask merges and splits, re-running a frame after each
       mask edit (link.py, mask_edit.py, imreg.py).
    2. trackOptiCellMarker: complete cell cycles (stat0 = 2).
    3. trackOptiCellFiles: one HDF5 file per cell ID; lineage fields from the
       cell's last frame; Cell{ID} when that frame has stat0 = 2.
Fluorescence, cell geometry, foci and clist are not computed.

Memory design: linking holds 3 full labeled frames (previous, current, next)
plus scalar region tables for all frames; edited masks are kept compressed.
Writing holds one frame plus crop buffers for all tracks until the end, so
memory grows with movie length x cell count.
"""

import re
import tomllib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
from tqdm import tqdm

from nd2_to_cells.link import Frame, cell_marker, link_frames

# ---------------------------------------------------------------------------
# Preset loading
# ---------------------------------------------------------------------------

_PRESET_DIR = Path(__file__).parent.parent / "presets"

# Opt-in departures from SuperSegger, named in a preset's `fixes` list. Each is
# tested on its own against the bug-for-bug baseline (docs/linking_report_tests.md).
FIXES = {
    "label_collision": "labels written by mask edits never reuse a label in use",
    "area_penalty": "area changes above 60 % get the 1000 penalty, not 50",
    "split_needs_two": "a split succeeds only if it yields two pieces",
    "death_last_frame": "death is the cell's last frame (MATLAB's can be one later)",
    "drop_ghosts": "labels without pixels are not written as cells",
    "itk_v4": "split registration with ITK v4 (SimpleITK), not MATLAB's v3 path",
    "seeded_split": "split a fused region with the previous cells as seeds",
    "end_on_merge": "a fused region that is not split becomes a new cell",
    "assign_rev": "fixProblems also updates the reverse map",
    "assign_all_regions": "fixProblems uses all regions an f region maps to",
    "assign_robbed_area": "fixProblems normalises by the robbed cell's area",
    "assign_exchange": "exchangeAssignment skips a region nobody holds",
    "assign_nan": "exchangeAssignment skips an all-NaN cost row",
    "division_both_sisters": "no division if either sister already has an ID",
    "f_no_match": "1 -> 3+: pieces without a forward map count as no match",
    "f_gate": "1 -> 3+: the subset merge is off in the final pass",
    "marker_chain": "cell marker drops a lineage whose backward map breaks",
    "merge_fallback": "a 1 -> 2 merge that fails becomes a division, not orphans",
    "orphan_split": "a cell without forward link inside a continued region: 2 -> 1",
}


@dataclass
class TrackingParams:
    """SuperSegger CONST.trackOpti values as used by a superSeggerGui run."""

    da_max: float = 0.3  # DA_MAX
    da_min: float = -0.2  # DA_MIN
    small_area_merge: int = 50  # SMALL_AREA_MERGE
    remove_stray: bool = False  # REMOVE_STRAY
    min_cell_age: int = 3  # MIN_CELL_AGE (GUI default)
    fixes: tuple[str, ...] = ()  # not in SuperSegger: names from FIXES
    # Not in SuperSegger (1 there): a split whose pieces fuse again within
    # this many frames is merged instead of called a division.
    division_lookahead: int = 1


def load_preset(preset: str) -> TrackingParams:
    """Load tracking parameters from a preset name or .toml file path."""
    path = Path(preset)
    if not path.exists():
        path = _PRESET_DIR / f"{preset}.toml"
    if not path.exists():
        raise FileNotFoundError(
            f"Preset not found: {preset!r}. "
            f"Available presets: {[p.stem for p in _PRESET_DIR.glob('*.toml')]}"
        )
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    p = data.get("tracking", {})
    params = TrackingParams(
        **{k: v for k, v in p.items() if k in TrackingParams.__dataclass_fields__}
    )
    params.fixes = tuple(params.fixes)
    unknown = sorted(set(params.fixes) - set(FIXES))
    if unknown:
        raise ValueError(f"Unknown fixes {unknown} in {path}; known: {sorted(FIXES)}")
    return params


# ---------------------------------------------------------------------------
# Mask file enumeration (no loading)
# ---------------------------------------------------------------------------


def _parse_frame_number(fname: str) -> int:
    m = re.search(r"_t(\d+)xy", fname)
    if m is None:
        raise ValueError(f"Cannot parse frame number from: {fname!r}")
    return int(m.group(1))


def _enumerate_mask_paths(masks_dir: Path) -> list[tuple[int, Path]]:
    """Return sorted list of (0-based-frame-index, path) for all mask PNGs."""
    pngs = sorted(
        [
            f
            for f in masks_dir.iterdir()
            if f.suffix.lower() == ".png" and "cp_masks" in f.name
        ],
        key=lambda f: _parse_frame_number(f.name),
    )
    if not pngs:
        raise FileNotFoundError(f"No *cp_masks.png files in {masks_dir}")
    return [((_parse_frame_number(f.name) - 1), f) for f in pngs]


def _load_mask(path: Path) -> np.ndarray:
    """Load one mask PNG as an int32 labeled array."""
    return iio.imread(path).astype(np.int32)


# ---------------------------------------------------------------------------
# Tracks from the linked frames (trackOptiCellFiles semantics)
# ---------------------------------------------------------------------------


@dataclass
class Track:
    track_id: int
    frames: list[int] = field(default_factory=list)  # 0-based absolute frames
    labels: list[int] = field(default_factory=list)  # region label per frame
    bboxes: list[tuple] = field(default_factory=list)  # (r0, c0, r1, c1)
    birth: int = 0  # 1-based frame number
    death: int = 0  # 1-based frame number
    mother_id: int = 0
    sister_id: int = 0
    daughter_ids: list[int] = field(default_factory=list)
    divide: bool = False
    stat0: int = 0  # 2 = complete cell cycle


def _build_tracks(states: list[Frame], frames: list[int], fixes=()) -> dict[int, Track]:
    """One track per cell ID; lineage fields come from the ID's last frame.

    ``frames[i]`` is the 0-based absolute frame of file i. SuperSegger's birth
    and death are file indices; they are converted to 1-based frame numbers.
    Fixes: "drop_ghosts" skips labels without pixels; "death_last_frame" sets
    death to the last frame (MATLAB's field can be stale, one frame later).
    """
    tracks: dict[int, Track] = {}
    for i, fr in enumerate(states):
        for k in range(1, fr.num_regs + 1):
            if "drop_ghosts" in fixes and not fr.area[k]:
                continue
            cid = int(fr.ID[k])
            track = tracks.setdefault(cid, Track(track_id=cid))
            if track.frames and track.frames[-1] == frames[i]:
                raise RuntimeError(f"cell {cid} appears twice in frame {frames[i] + 1}")
            x0, y0, w, h = fr.bbox[k]
            r0, c0 = int(round(y0 - 0.5)), int(round(x0 - 0.5))
            track.frames.append(frames[i])
            track.labels.append(k)
            track.bboxes.append((r0, c0, r0 + int(h), c0 + int(w)))
            # intUpCell: every frame overwrites the lineage fields.
            track.birth = frames[int(fr.birth[k]) - 1] + 1
            track.death = frames[min(int(fr.death[k]), len(frames)) - 1] + 1
            track.divide = bool(fr.divide[k])
            track.mother_id = int(fr.motherID[k])
            track.sister_id = int(fr.sisterID[k])
            track.daughter_ids = [int(x) for x in fr.daughterID[k]]
            track.stat0 = int(fr.stat0[k])
    if "death_last_frame" in fixes:
        for track in tracks.values():
            track.death = track.frames[-1] + 1
    return tracks


# ---------------------------------------------------------------------------
# Consensus bounding box and HDF5 writing
# ---------------------------------------------------------------------------


def _union_bbox(bboxes: list[tuple]) -> tuple[int, int, int, int]:
    return (
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    )


def _squarify_and_pad(
    bbox: tuple[int, int, int, int],
    pad: int,
    img_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Pad and squarify a bounding box, clamped to image dimensions."""
    min_r, min_c, max_r, max_c = bbox
    h = max_r - min_r
    w = max_c - min_c
    side = max(h, w) + 2 * pad
    r0 = min_r - (side - h) // 2
    c0 = min_c - (side - w) // 2
    r1 = r0 + side
    c1 = c0 + side
    H, W = img_shape
    r0 = max(0, r0)
    c0 = max(0, c0)
    r1 = min(H, r1)
    c1 = min(W, c1)
    return (r0, c0, r1, c1)


def _touches_edge(crop_box: tuple, img_shape: tuple) -> bool:
    r0, c0, r1, c1 = crop_box
    H, W = img_shape
    return r0 == 0 or c0 == 0 or r1 == H or c1 == W


def _alloc_track_buffers(
    tracks: dict[int, "Track"],
    img_shape: tuple[int, int],
    pad: int,
) -> dict[int, dict]:
    """Pre-allocate crop buffers for every track (no mask reads).

    Returns a dict mapping track_id -> buffer dict with:
        crop:         (r0, c0, r1, c1) fixed crop box for the whole track
        mask_stack:   bool array (H_crop, W_crop, n_frames), zeroed
        bb_arr:       int32 array (n_frames, 4)
        r_offset_arr: int32 array (n_frames, 2)
        edge_arr:     bool array (n_frames,)
        frame_index:  dict frame -> position in track.frames
    """
    buffers: dict[int, dict] = {}
    for tid, track in tracks.items():
        n = len(track.frames)
        if n == 0:
            continue
        crop = _squarify_and_pad(_union_bbox(track.bboxes), pad, img_shape)
        r0, c0, r1, c1 = crop
        H_crop, W_crop = r1 - r0, c1 - c0

        # Pre-fill scalar arrays (don't need mask data)
        bb_arr = np.zeros((n, 4), dtype=np.int32)
        r_offset_arr = np.zeros((n, 2), dtype=np.int32)
        edge_arr = np.zeros(n, dtype=bool)
        edge = _touches_edge(crop, img_shape)
        for fi, per_frame_bbox in enumerate(track.bboxes):
            fr0, fc0, fr1, fc1 = per_frame_bbox
            bb_arr[fi] = [fc0 - c0, fr0 - r0, fc1 - fc0, fr1 - fr0]
            r_offset_arr[fi] = [c0, r0]
            edge_arr[fi] = edge

        buffers[tid] = {
            "crop": crop,
            "mask_stack": np.zeros((H_crop, W_crop, n), dtype=bool),
            "bb_arr": bb_arr,
            "r_offset_arr": r_offset_arr,
            "edge_arr": edge_arr,
            "frame_index": {f: i for i, f in enumerate(track.frames)},
        }
    return buffers


def _fill_buffers(
    mask_paths: list[tuple[int, Path]],
    tracks: dict[int, "Track"],
    buffers: dict[int, dict],
    states: list[Frame],
) -> None:
    """Single pass over the masks: load each frame once, fill all active crops.

    Builds the frame -> [track_id] inverted index internally so the caller
    doesn't need to manage it. Frames whose mask was edited during linking
    (merges, splits) use the edited mask; the others are read from masks/.
    """
    from collections import defaultdict

    frame_to_tids: dict[int, list[int]] = defaultdict(list)
    for tid, track in tracks.items():
        if tid not in buffers:
            continue
        for f in track.frames:
            frame_to_tids[f].append(tid)

    for (abs_frame, path), state in zip(mask_paths, states):
        tids = frame_to_tids.get(abs_frame)
        if not tids:
            continue
        labeled = state.edited_labels()
        if labeled is None:
            labeled = _load_mask(path)
        for tid in tids:
            track = tracks[tid]
            buf = buffers[tid]
            fi = buf["frame_index"][abs_frame]
            r0, c0, r1, c1 = buf["crop"]
            lbl = track.labels[fi]
            buf["mask_stack"][:, :, fi] = labeled[r0:r1, c0:c1] == lbl
        del labeled


def _flush_all_tracks_to_h5(
    cell_dir: Path,
    tracks: dict[int, "Track"],
    buffers: dict[int, dict],
) -> None:
    """Write all tracks as groups into a single cells.h5 file."""
    with h5py.File(cell_dir / "cells.h5", "w") as h5:
        for track in tracks.values():
            if track.track_id not in buffers:
                continue
            buf = buffers[track.track_id]
            prefix = "Cell" if track.stat0 == 2 else "cell"
            grp = h5.require_group(f"{prefix}{track.track_id:07d}")
            grp.create_dataset("birth", data=np.int64(track.birth))
            grp.create_dataset("death", data=np.int64(track.death))
            grp.create_dataset("divide", data=np.int8(1 if track.divide else 0))
            grp.create_dataset("motherID", data=np.int64(track.mother_id))
            grp.create_dataset("sisterID", data=np.int64(track.sister_id))
            grp.create_dataset(
                "daughterID", data=np.array(track.daughter_ids, dtype=np.int64)
            )
            grp.create_dataset("frames", data=np.array(track.frames, dtype=np.int64))
            grp.create_dataset("BB", data=buf["bb_arr"])
            grp.create_dataset("r_offset", data=buf["r_offset_arr"])
            grp.create_dataset("edgeFlag", data=buf["edge_arr"])
            grp.create_dataset(
                "mask", data=buf["mask_stack"], compression="gzip", compression_opts=4
            )


def _flush_track_to_h5(
    cell_dir: Path,
    track: "Track",
    buf: dict,
) -> None:
    """Write one HDF5 file from a pre-filled buffer dict."""
    prefix = "Cell" if track.stat0 == 2 else "cell"
    fname = f"{prefix}{track.track_id:07d}.h5"

    with h5py.File(cell_dir / fname, "w") as h5:
        h5.create_dataset("birth", data=np.int64(track.birth))
        h5.create_dataset("death", data=np.int64(track.death))
        h5.create_dataset("divide", data=np.int8(1 if track.divide else 0))
        h5.create_dataset("motherID", data=np.int64(track.mother_id))
        h5.create_dataset("sisterID", data=np.int64(track.sister_id))
        h5.create_dataset(
            "daughterID", data=np.array(track.daughter_ids, dtype=np.int64)
        )
        h5.create_dataset("frames", data=np.array(track.frames, dtype=np.int64))
        h5.create_dataset("BB", data=buf["bb_arr"])
        h5.create_dataset("r_offset", data=buf["r_offset_arr"])
        h5.create_dataset("edgeFlag", data=buf["edge_arr"])
        h5.create_dataset(
            "mask", data=buf["mask_stack"], compression="gzip", compression_opts=4
        )


# ---------------------------------------------------------------------------
# Per-position entry point
# ---------------------------------------------------------------------------


def _track_position(
    xy_dir: Path, params: TrackingParams, pad: int, consolidated: bool = False
) -> None:
    """Run the full tracking pipeline for one xy position."""
    masks_dir = xy_dir / "masks"
    cell_dir = xy_dir / "cell"
    cell_dir.mkdir(exist_ok=True)

    if not masks_dir.exists():
        print(f"  [skip] {xy_dir.name}: no masks/ directory")
        return

    # Remove the cp_output directory Omnipose creates as a CLI artefact
    cp_output_dir = xy_dir / "cp_output"
    if cp_output_dir.exists():
        import shutil

        shutil.rmtree(cp_output_dir)

    try:
        mask_paths = _enumerate_mask_paths(masks_dir)
    except FileNotFoundError as e:
        print(f"  [skip] {xy_dir.name}: {e}")
        return

    if not mask_paths:
        print(f"  [skip] {xy_dir.name}: empty masks/ directory")
        return

    # Missing frames are bridged by the linker; warn since links are less reliable
    frames = {f for f, _ in mask_paths}
    missing = [f + 1 for f in range(min(frames), max(frames)) if f not in frames]
    if missing:
        shown = ", ".join(map(str, missing[:10])) + (" …" if len(missing) > 10 else "")
        print(
            f"  [warn] {xy_dir.name}: {len(missing)} frame(s) missing from masks/ "
            f"(t = {shown}); tracks are linked across the gap(s)"
        )

    # Masks are kept across re-alignment; warn if images were rewritten since
    phase_tifs = list((xy_dir / "phase").glob("*.tif"))
    if phase_tifs and max(f.stat().st_mtime for f in phase_tifs) > min(
        p.stat().st_mtime for _, p in mask_paths
    ):
        print(
            f"  [warn] {xy_dir.name}: phase/ images are newer than masks/ — "
            "masks may be out of date (re-run Omnipose?)"
        )

    # Image shape from first frame only
    img_shape = _load_mask(mask_paths[0][1]).shape[:2]

    def progress(t: int) -> None:
        if t % 50 == 0:
            print(f"  {xy_dir.name}: linked {t}/{len(mask_paths)} frames", flush=True)

    print(f"  {xy_dir.name}: linking {len(mask_paths)} frames...")
    states = link_frames(
        lambda i: _load_mask(mask_paths[i][1]),
        len(mask_paths),
        params,
        warn=lambda msg: print(f"  [warn] {xy_dir.name}: {msg}"),
        progress=progress,
    )
    cell_marker(states, params.min_cell_age, params.fixes)
    tracks = _build_tracks(states, [f for f, _ in mask_paths], params.fixes)
    print(f"  {xy_dir.name}: {len(tracks)} tracks, writing HDF5 file(s)...")

    buffers = _alloc_track_buffers(tracks, img_shape, pad)
    _fill_buffers(mask_paths, tracks, buffers, states)

    # Remove previous track output (both modes) so stale cells don't survive
    stale = [
        f
        for f in cell_dir.iterdir()
        if re.fullmatch(r"[cC]ell\d{7}\.h5|cells\.h5", f.name)
    ]
    for f in stale:
        f.unlink()
    if stale:
        print(f"  {xy_dir.name}: removed {len(stale)} previous cell file(s)")

    if consolidated:
        _flush_all_tracks_to_h5(cell_dir, tracks, buffers)
    else:
        for track in tracks.values():
            if track.track_id in buffers:
                _flush_track_to_h5(cell_dir, track, buffers[track.track_id])


def _track_position_wrapper(args) -> str | None:
    """Run one position; return an error message instead of raising."""
    xy_dir, preset_str, pad, consolidated = args
    try:
        params = load_preset(preset_str)
        _track_position(Path(xy_dir), params, pad, consolidated)
    except Exception as exc:
        import traceback

        print(f"ERROR tracking {xy_dir}: {exc}")
        traceback.print_exc()
        return f"{Path(xy_dir).name}: {exc}"
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_track(
    data_dir: str,
    preset: str = "100XEc",
    workers: int = 1,
    pad: int = 5,
    consolidated: bool = False,
) -> None:
    """Link cells and write HDF5 output for all xy positions.

    Args:
        data_dir:     Directory containing xy*/ subdirectories.
        preset:       Preset name or path to .toml file.
        workers:      Number of parallel worker processes.
        pad:          Padding (px) added around each cell's bounding box.
        consolidated: If True, write one cells.h5 per position (one group per
                      cell) instead of one file per cell.
    """
    data_dir = Path(data_dir)
    xy_dirs = sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and re.match(r"xy\d+$", d.name)],
        key=lambda d: int(d.name[2:]),
    )

    if not xy_dirs:
        raise FileNotFoundError(f"No xy*/ directories found in {data_dir}")

    params = load_preset(preset)
    print(
        f"Tracking {len(xy_dirs)} position(s) with preset '{preset}' "
        f"(da_min={params.da_min}, da_max={params.da_max}, "
        f"small_area_merge={params.small_area_merge}, "
        f"remove_stray={params.remove_stray}, min_cell_age={params.min_cell_age}"
        + (f", fixes={list(params.fixes)})" if params.fixes else ")")
    )

    args = [(str(d), preset, pad, consolidated) for d in xy_dirs]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            errors = list(
                tqdm(
                    pool.map(_track_position_wrapper, args),
                    total=len(args),
                    desc="Tracking positions",
                    unit="pos",
                )
            )
    else:
        errors = [_track_position_wrapper(a) for a in args]

    failed = [e for e in errors if e]
    if failed:
        raise RuntimeError(
            f"Tracking failed for {len(failed)}/{len(args)} position(s):\n  "
            + "\n  ".join(failed)
        )
    print("Tracking complete.")
