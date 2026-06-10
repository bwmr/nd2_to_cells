"""IoU-based cell tracker: links Omnipose masks across frames, writes cell*.h5.

Algorithm (per xy position):
    1. Enumerate mask file paths sorted by frame number (never load all at once).
    2. Stream frame pairs: load frame t and t+1, extract region properties,
       apply area filters and small-region merging, link, then discard t.
    3. Write per-cell HDF5 files by re-reading only the frames each cell
       was alive in (one frame at a time, never the full stack in memory).

Memory design: at most 2 full labeled frames are held in RAM simultaneously.
Region objects store only scalars (label, area, centroid, bbox) — no arrays.
IoU is computed on the fly from label equality within the bbox overlap region.

Derived from ObtrackerPy (https://github.com/alexSysBio/ObtrackerPy) with extensions
for division detection, area filtering, and HDF5 output.
"""

import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
from skimage.measure import regionprops
from tqdm import tqdm

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # fallback for 3.10


# ---------------------------------------------------------------------------
# Preset loading
# ---------------------------------------------------------------------------

_PRESET_DIR = Path(__file__).parent.parent / "presets"


@dataclass
class TrackingParams:
    overlap_limit_min: float = 0.08
    da_max: float = 0.3
    da_min: float = -0.2
    min_area: int = 8
    min_area_no_neigh: int = 30
    small_area_merge: int = 50
    remove_stray: bool = True
    min_cell_age: int = 5
    search_radius: float = 15.0


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
    return TrackingParams(
        **{k: v for k, v in p.items() if k in TrackingParams.__dataclass_fields__}
    )


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
# Region properties — scalars only, no full-frame arrays
# ---------------------------------------------------------------------------


@dataclass
class Region:
    label: int
    area: int
    centroid: tuple[float, float]  # (row, col)
    bbox: tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    # No mask field — computed on demand from the labeled array


def _extract_regions(labeled: np.ndarray, params: TrackingParams) -> list[Region]:
    """Extract region scalars from a labeled mask, applying min_area filter."""
    regions = []
    for prop in regionprops(labeled):
        if prop.area < params.min_area:
            continue
        regions.append(
            Region(
                label=prop.label,
                area=prop.area,
                centroid=(prop.centroid[0], prop.centroid[1]),
                bbox=prop.bbox,
            )
        )
    return regions


def _bboxes_overlap(b1: tuple, b2: tuple) -> bool:
    """True if two (min_r, min_c, max_r, max_c) bboxes have any overlap."""
    return b1[0] < b2[2] and b2[0] < b1[2] and b1[1] < b2[3] and b2[1] < b1[3]


def _bbox_intersect(b1: tuple, b2: tuple) -> tuple[int, int, int, int] | None:
    """Return the intersection bbox, or None if they don't overlap."""
    r0 = max(b1[0], b2[0])
    c0 = max(b1[1], b2[1])
    r1 = min(b1[2], b2[2])
    c1 = min(b1[3], b2[3])
    if r1 <= r0 or c1 <= c0:
        return None
    return (r0, c0, r1, c1)


def _iou_from_labeled(
    labeled_a: np.ndarray,
    label_a: int,
    bbox_a: tuple,
    labeled_b: np.ndarray,
    label_b: int,
    bbox_b: tuple,
) -> float:
    """Compute IoU using only the bbox overlap region — no full-frame arrays."""
    inter_bbox = _bbox_intersect(bbox_a, bbox_b)
    if inter_bbox is None:
        return 0.0
    r0, c0, r1, c1 = inter_bbox
    patch_a = labeled_a[r0:r1, c0:c1] == label_a
    patch_b = labeled_b[r0:r1, c0:c1] == label_b
    inter = int(np.count_nonzero(patch_a & patch_b))
    if inter == 0:
        return 0.0
    # Union = area_a + area_b - inter  (faster than materialising full union)
    area_a = int(
        np.count_nonzero(
            labeled_a[bbox_a[0] : bbox_a[2], bbox_a[1] : bbox_a[3]] == label_a
        )
    )
    area_b = int(
        np.count_nonzero(
            labeled_b[bbox_b[0] : bbox_b[2], bbox_b[1] : bbox_b[3]] == label_b
        )
    )
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-frame filtering and merging (operates on one labeled frame at a time)
# ---------------------------------------------------------------------------


def _has_neighbour(region: Region, all_regions: list[Region]) -> bool:
    """Return True if region has any spatially adjacent neighbour.

    Uses only bbox proximity — no full-frame arrays.
    Two regions are considered adjacent if their bboxes are within 1px.
    """
    r0, c0, r1, c1 = region.bbox
    for other in all_regions:
        if other.label == region.label:
            continue
        or0, oc0, or1, oc1 = other.bbox
        if r1 < or0 - 1 or or1 < r0 - 1 or c1 < oc0 - 1 or oc1 < c0 - 1:
            continue
        return True
    return False


def _apply_area_filters(
    regions: list[Region],
    params: TrackingParams,
) -> list[Region]:
    """Discard isolated sub-threshold regions (MIN_AREA_NO_NEIGH)."""
    return [
        r
        for r in regions
        if r.area >= params.min_area_no_neigh or _has_neighbour(r, regions)
    ]


def _merge_small_regions(
    labeled: np.ndarray,
    regions: list[Region],
    params: TrackingParams,
) -> tuple[np.ndarray, list[Region]]:
    """Merge pairs of adjacent sub-threshold regions into one (in place).

    Operates on the labeled array directly; no full-frame bool copies.
    """
    threshold = params.small_area_merge
    small = [r for r in regions if r.area < threshold]
    if not small:
        return labeled, regions

    merged: set[int] = set()
    for r in small:
        if r.label in merged:
            continue
        r0, c0, r1, c1 = r.bbox
        for other in small:
            if other.label == r.label or other.label in merged:
                continue
            or0, oc0, or1, oc1 = other.bbox
            # Adjacent = bboxes within 1px of each other
            if r1 < or0 - 1 or or1 < r0 - 1 or c1 < oc0 - 1 or oc1 < c0 - 1:
                continue
            labeled[labeled == other.label] = r.label
            merged.add(other.label)
            break

    if not merged:
        return labeled, regions

    # Rebuild region list from updated labeled image
    return labeled, _extract_regions(labeled, params)


# ---------------------------------------------------------------------------
# Track data structure
# ---------------------------------------------------------------------------


@dataclass
class Track:
    track_id: int
    frames: list[int] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    bboxes: list[tuple] = field(default_factory=list)
    areas: list[int] = field(default_factory=list)
    mother_id: int = 0
    sister_id: int = 0
    daughter_ids: list[int] = field(default_factory=list)
    divide: bool = False
    has_predecessor: bool = False


# ---------------------------------------------------------------------------
# Streaming linker — loads one frame pair at a time
# ---------------------------------------------------------------------------


def _normalised_area_change(area_t: int, area_t1: int) -> float:
    if area_t1 == 0:
        return 0.0
    return (area_t1 - area_t) / area_t1


def link_frames_streaming(
    mask_paths: list[tuple[int, Path]],
    params: TrackingParams,
) -> dict[int, Track]:
    """Link regions across all frames, loading only two frames at a time.

    Args:
        mask_paths: List of (0-based-frame-index, path) sorted by frame.
        params:     Tracking parameters.

    Returns:
        Dict mapping track_id -> Track (scalars only, no arrays).
    """
    tracks: dict[int, Track] = {}
    next_id = 1
    active: dict[int, int] = {}  # label_in_current_frame -> track_id

    n_frames = len(mask_paths)

    # Load first frame
    frame_0, path_0 = mask_paths[0]
    labeled_prev = _load_mask(path_0)
    regions_prev = _extract_regions(labeled_prev, params)
    regions_prev = _apply_area_filters(regions_prev, params)
    labeled_prev, regions_prev = _merge_small_regions(
        labeled_prev, regions_prev, params
    )

    # Initialise tracks for first frame
    for r in regions_prev:
        tid = next_id
        next_id += 1
        tracks[tid] = Track(
            track_id=tid,
            frames=[frame_0],
            labels=[r.label],
            bboxes=[r.bbox],
            areas=[r.area],
            has_predecessor=False,
        )
        active[r.label] = tid

    for fi in range(1, n_frames):
        frame_cur, path_cur = mask_paths[fi]
        pass

        labeled_cur = _load_mask(path_cur)
        regions_cur = _extract_regions(labeled_cur, params)
        regions_cur = _apply_area_filters(regions_cur, params)
        labeled_cur, regions_cur = _merge_small_regions(
            labeled_cur, regions_cur, params
        )

        claimed: set[int] = set()
        new_active: dict[int, int] = {}

        for r0 in regions_prev:
            tid = active.get(r0.label)
            if tid is None:
                continue

            candidates_iou = []
            candidates_fallback = []

            for r1 in regions_cur:
                if r1.label in claimed:
                    continue

                if _bboxes_overlap(r0.bbox, r1.bbox):
                    iou = _iou_from_labeled(
                        labeled_prev,
                        r0.label,
                        r0.bbox,
                        labeled_cur,
                        r1.label,
                        r1.bbox,
                    )
                    if iou >= params.overlap_limit_min:
                        candidates_iou.append((iou, r1))
                        continue

                # Centroid-distance fallback
                dist = np.hypot(
                    r1.centroid[0] - r0.centroid[0],
                    r1.centroid[1] - r0.centroid[1],
                )
                if dist <= params.search_radius:
                    candidates_fallback.append((dist, r1))

            # Prefer IoU candidates (sort best-first); fall back to centroid
            if candidates_iou:
                candidates = sorted(
                    candidates_iou, key=lambda x: -x[0]
                )  # highest IoU first
                candidates = [(1.0 / iou, r1) for iou, r1 in candidates]
            elif candidates_fallback:
                candidates = sorted(candidates_fallback, key=lambda x: x[0])
            else:
                candidates = []

            if len(candidates) == 0:
                pass  # track ends naturally

            elif len(candidates) == 1:
                _, r1 = candidates[0]
                da = _normalised_area_change(r0.area, r1.area)
                if params.da_min <= da <= params.da_max:
                    tracks[tid].frames.append(frame_cur)
                    tracks[tid].labels.append(r1.label)
                    tracks[tid].bboxes.append(r1.bbox)
                    tracks[tid].areas.append(r1.area)
                    new_active[r1.label] = tid
                    claimed.add(r1.label)
                # else: track ends (area change too extreme, not a continuation)

            else:
                # Check top 2 for division first (before per-candidate da filter)
                _, r1a = candidates[0]
                _, r1b = candidates[1]
                combined = r1a.area + r1b.area
                da_div = _normalised_area_change(r0.area, combined)

                if (
                    params.da_min <= da_div <= params.da_max
                    and r1b.label not in claimed
                ):
                    tracks[tid].divide = True
                    tid_a = next_id
                    next_id += 1
                    tid_b = next_id
                    next_id += 1
                    tracks[tid].daughter_ids = [tid_a, tid_b]

                    for tid_d, r1_d in [(tid_a, r1a), (tid_b, r1b)]:
                        tracks[tid_d] = Track(
                            track_id=tid_d,
                            frames=[frame_cur],
                            labels=[r1_d.label],
                            bboxes=[r1_d.bbox],
                            areas=[r1_d.area],
                            mother_id=tid,
                            sister_id=tid_b if tid_d == tid_a else tid_a,
                            has_predecessor=True,
                        )
                        new_active[r1_d.label] = tid_d
                        claimed.add(r1_d.label)
                else:
                    # Not a division — link to best candidate that passes da filter
                    for _, r1 in candidates:
                        da = _normalised_area_change(r0.area, r1.area)
                        if params.da_min <= da <= params.da_max:
                            tracks[tid].frames.append(frame_cur)
                            tracks[tid].labels.append(r1.label)
                            tracks[tid].bboxes.append(r1.bbox)
                            tracks[tid].areas.append(r1.area)
                            new_active[r1.label] = tid
                            claimed.add(r1.label)
                            break

        # New regions with no predecessor
        for r1 in regions_cur:
            if r1.label in claimed:
                continue
            tid = next_id
            next_id += 1
            tracks[tid] = Track(
                track_id=tid,
                frames=[frame_cur],
                labels=[r1.label],
                bboxes=[r1.bbox],
                areas=[r1.area],
                has_predecessor=False,
            )
            new_active[r1.label] = tid

        # Advance: current becomes previous, discard previous labeled array
        labeled_prev = labeled_cur
        regions_prev = regions_cur
        active = new_active
        # labeled_prev now holds the only frame in memory

    # Remove stray single-frame tracks with no predecessor and no daughters
    if params.remove_stray:
        tracks = {
            tid: t
            for tid, t in tracks.items()
            if not (not t.has_predecessor and len(t.frames) == 1 and not t.divide)
        }

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
) -> None:
    """Single pass over mask PNGs: load each frame once, fill all active crops.

    Builds the frame -> [track_id] inverted index internally so the caller
    doesn't need to manage it.
    """
    from collections import defaultdict

    frame_to_tids: dict[int, list[int]] = defaultdict(list)
    for tid, track in tracks.items():
        if tid not in buffers:
            continue
        for f in track.frames:
            frame_to_tids[f].append(tid)

    for abs_frame, path in mask_paths:
        tids = frame_to_tids.get(abs_frame)
        if not tids:
            continue
        labeled = _load_mask(path)
        for tid in tids:
            track = tracks[tid]
            buf = buffers[tid]
            fi = buf["frame_index"][abs_frame]
            r0, c0, r1, c1 = buf["crop"]
            lbl = track.labels[fi]
            buf["mask_stack"][:, :, fi] = labeled[r0:r1, c0:c1] == lbl
        del labeled


def _flush_track_to_h5(
    cell_dir: Path,
    track: "Track",
    buf: dict,
    params: TrackingParams,
) -> None:
    """Write one HDF5 file from a pre-filled buffer dict."""
    n_frames = len(track.frames)
    is_complete = track.divide and n_frames >= params.min_cell_age
    prefix = "Cell" if is_complete else "cell"
    fname = f"{prefix}{track.track_id:07d}.h5"

    with h5py.File(cell_dir / fname, "w") as h5:
        h5.create_dataset("birth", data=np.int64(track.frames[0] + 1))
        h5.create_dataset("death", data=np.int64(track.frames[-1] + 1))
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


def _track_position(xy_dir: Path, params: TrackingParams, pad: int) -> None:
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

    # Image shape from first frame only
    img_shape = _load_mask(mask_paths[0][1]).shape[:2]

    print(f"  {xy_dir.name}: linking {len(mask_paths)} frames...")
    tracks = link_frames_streaming(mask_paths, params)
    print(f"  {xy_dir.name}: {len(tracks)} tracks, writing HDF5 files...")

    buffers = _alloc_track_buffers(tracks, img_shape, pad)
    _fill_buffers(mask_paths, tracks, buffers)
    for track in tracks.values():
        if track.track_id in buffers:
            _flush_track_to_h5(cell_dir, track, buffers[track.track_id], params)


def _track_position_wrapper(args):
    xy_dir, preset_str, pad = args
    try:
        params = load_preset(preset_str)
        _track_position(Path(xy_dir), params, pad)
    except Exception as exc:
        import traceback

        print(f"ERROR tracking {xy_dir}: {exc}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_track(
    data_dir: str,
    preset: str = "100XEc",
    workers: int = 1,
    pad: int = 5,
) -> None:
    """Link cells and write per-cell HDF5 files for all xy positions.

    Args:
        data_dir: Directory containing xy*/ subdirectories.
        preset:   Preset name or path to .toml file.
        workers:  Number of parallel worker processes.
        pad:      Padding (px) added around each cell's bounding box.
    """
    data_dir = Path(data_dir)
    xy_dirs = sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and re.match(r"xy\d+$", d.name)],
        key=lambda d: int(d.name[2:]),
    )

    if not xy_dirs:
        print(f"No xy*/ directories found in {data_dir}")
        return

    params = load_preset(preset)
    print(
        f"Tracking {len(xy_dirs)} position(s) with preset '{preset}' "
        f"(overlap_limit_min={params.overlap_limit_min}, "
        f"da_min={params.da_min}, da_max={params.da_max}, "
        f"remove_stray={params.remove_stray})"
    )

    if workers > 1:
        args = [(str(d), preset, pad) for d in xy_dirs]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(
                tqdm(
                    pool.map(_track_position_wrapper, args),
                    total=len(args),
                    desc="Tracking positions",
                    unit="pos",
                )
            )
    else:
        for xy_dir in xy_dirs:
            _track_position(xy_dir, params, pad)

    print("Tracking complete.")
