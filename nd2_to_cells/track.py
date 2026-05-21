"""IoU-based cell tracker: links Omnipose masks across frames, writes cell*.h5.

Algorithm (per xy position):
    1. Load all Omnipose PNG masks; extract per-frame region properties.
    2. Pre-filter: discard tiny / isolated regions per preset thresholds.
    3. Merge: merge adjacent sub-threshold regions into a single region.
    4. Link frame-by-frame:
         - Primary: IoU >= overlap_limit_min AND area change in [da_min, da_max]
         - Fallback: centroid distance < search_radius (for daughters with no
           mask overlap with mother after division)
         - Division: one region in t maps to two in t+1 whose combined area
           is within the normalised area-change bounds.
    5. Apply stray-region policy (remove_stray preset flag).
    6. Write one HDF5 file per track to xy{N}/cell/.

Derived from ObtrackerPy (Papagiannakis & Wimmer, 2024) with extensions
for division detection, area filtering, and HDF5 output.

SuperSegger calibration parameters are loaded from a preset TOML file;
see presets/100XEc.toml for field descriptions.
"""

import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import imageio.v3 as iio
import numpy as np
from skimage.measure import regionprops, label as sk_label
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
        # Try named preset
        path = _PRESET_DIR / f"{preset}.toml"
    if not path.exists():
        raise FileNotFoundError(
            f"Preset not found: {preset!r}. "
            f"Available presets: {[p.stem for p in _PRESET_DIR.glob('*.toml')]}"
        )
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    p = data.get("tracking", {})
    return TrackingParams(**{k: v for k, v in p.items()
                             if k in TrackingParams.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Mask loading
# ---------------------------------------------------------------------------

def _parse_frame_number(fname: str) -> int:
    m = re.search(r"_t(\d+)xy", fname)
    if m is None:
        raise ValueError(f"Cannot parse frame number from: {fname!r}")
    return int(m.group(1))


def load_masks(masks_dir: Path) -> dict[int, np.ndarray]:
    """Load all Omnipose PNG masks from a directory.

    Returns:
        Dict mapping 0-based frame index to labeled integer array.
        Frame index = parsed frame number - 1 (converted to 0-based).
    """
    pngs = sorted(
        [f for f in masks_dir.iterdir()
         if f.suffix.lower() == ".png" and "cp_masks" in f.name],
        key=lambda f: _parse_frame_number(f.name),
    )
    if not pngs:
        raise FileNotFoundError(f"No *cp_masks.png files in {masks_dir}")

    result = {}
    for f in pngs:
        frame_1based = _parse_frame_number(f.name)
        frame_0based = frame_1based - 1
        result[frame_0based] = iio.imread(f).astype(np.int32)
    return result


# ---------------------------------------------------------------------------
# Region properties
# ---------------------------------------------------------------------------

@dataclass
class Region:
    label: int
    area: int
    centroid: tuple[float, float]   # (row, col)
    bbox: tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    mask: np.ndarray                 # bool array, full-frame size


def _extract_regions(labeled: np.ndarray, params: TrackingParams) -> list[Region]:
    """Extract regions from a labeled mask, applying area pre-filters."""
    regions = []
    props = regionprops(labeled)
    for prop in props:
        if prop.area < params.min_area:
            continue
        mask = labeled == prop.label
        regions.append(Region(
            label=prop.label,
            area=prop.area,
            centroid=(prop.centroid[0], prop.centroid[1]),
            bbox=prop.bbox,  # (min_row, min_col, max_row, max_col)
            mask=mask,
        ))
    return regions


def _has_neighbour(region: Region, all_regions: list[Region]) -> bool:
    """Return True if region has any pixel-adjacent neighbour."""
    # Dilate the region mask by 1px (4-connectivity) and check overlap
    r = region
    r0, c0, r1, c1 = r.bbox
    for other in all_regions:
        if other.label == r.label:
            continue
        # Quick bounding-box proximity check first
        or0, oc0, or1, oc1 = other.bbox
        if r1 < or0 - 1 or or1 < r0 - 1 or c1 < oc0 - 1 or oc1 < c0 - 1:
            continue
        # Pixel-level adjacency: expand one region by 1 and test overlap
        expanded = np.zeros_like(r.mask)
        H, W = r.mask.shape
        expanded[
            max(0, r0 - 1):min(H, r1 + 1),
            max(0, c0 - 1):min(W, c1 + 1),
        ] = True
        expanded &= r.mask.__class__(np.ones_like(r.mask, dtype=bool))
        # Simple: check if other.mask overlaps expanded area of r
        if np.any(expanded & other.mask):
            return True
    return False


def _apply_area_filters(
    regions: list[Region],
    params: TrackingParams,
) -> list[Region]:
    """Discard isolated sub-threshold regions (MIN_AREA_NO_NEIGH)."""
    return [
        r for r in regions
        if r.area >= params.min_area_no_neigh or _has_neighbour(r, regions)
    ]


def _merge_small_regions(
    labeled: np.ndarray,
    regions: list[Region],
    params: TrackingParams,
) -> tuple[np.ndarray, list[Region]]:
    """Merge pairs of adjacent sub-threshold regions into one.

    Two regions both below small_area_merge that are adjacent are merged
    by re-labelling both with the smaller label value.  Only one merge
    pass is performed.
    """
    threshold = params.small_area_merge
    small = [r for r in regions if r.area < threshold]
    if not small:
        return labeled, regions

    merged_pairs: set[int] = set()
    new_labeled = labeled.copy()

    for r in small:
        if r.label in merged_pairs:
            continue
        r0, c0, r1, c1 = r.bbox
        H, W = labeled.shape
        expanded = np.zeros((H, W), dtype=bool)
        expanded[
            max(0, r0 - 1):min(H, r1 + 1),
            max(0, c0 - 1):min(W, c1 + 1),
        ] = r.mask[
            max(0, r0 - 1) - r0 + max(0, -(r0 - 1)):,
            max(0, c0 - 1) - c0 + max(0, -(c0 - 1)):,
        ][:min(H, r1 + 1) - max(0, r0 - 1),
          :min(W, c1 + 1) - max(0, c0 - 1)]
        # Simpler: just dilate r.mask by 1 in-place
        expanded = np.zeros((H, W), dtype=bool)
        expanded[max(0, r0-1):min(H, r1+1), max(0, c0-1):min(W, c1+1)] = True
        expanded &= ~r.mask  # border only (excluding self)

        for other in small:
            if other.label == r.label or other.label in merged_pairs:
                continue
            if np.any(expanded & other.mask):
                # Merge other into r (keep r's label)
                new_labeled[new_labeled == other.label] = r.label
                merged_pairs.add(other.label)
                break

    # Rebuild region list from updated labeled image
    new_regions = _extract_regions(new_labeled, params)
    return new_labeled, new_regions


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------

def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute Intersection-over-Union of two boolean masks."""
    inter = np.count_nonzero(mask_a & mask_b)
    if inter == 0:
        return 0.0
    union = np.count_nonzero(mask_a | mask_b)
    return inter / union


def _bboxes_overlap(b1: tuple, b2: tuple) -> bool:
    """Return True if two (min_row, min_col, max_row, max_col) bboxes overlap."""
    return (b1[0] < b2[2] and b2[0] < b1[2] and
            b1[1] < b2[3] and b2[1] < b1[3])


# ---------------------------------------------------------------------------
# Track data structure
# ---------------------------------------------------------------------------

@dataclass
class Track:
    track_id: int
    frames: list[int] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)   # Omnipose label per frame
    bboxes: list[tuple] = field(default_factory=list)  # (min_r, min_c, max_r, max_c)
    centroids: list[tuple] = field(default_factory=list)
    areas: list[int] = field(default_factory=list)
    mother_id: int = 0
    sister_id: int = 0
    daughter_ids: list[int] = field(default_factory=list)
    divide: bool = False
    has_predecessor: bool = False


# ---------------------------------------------------------------------------
# Linker
# ---------------------------------------------------------------------------

def _normalised_area_change(area_t: int, area_t1: int) -> float:
    """(area_t1 - area_t) / area_t1, matching SuperSegger's DA convention."""
    if area_t1 == 0:
        return 0.0
    return (area_t1 - area_t) / area_t1


def link_frames(
    frames_regions: dict[int, list[Region]],
    params: TrackingParams,
) -> dict[int, Track]:
    """Link regions across all frames into tracks.

    Returns:
        Dict mapping track_id -> Track.
    """
    sorted_frames = sorted(frames_regions.keys())
    tracks: dict[int, Track] = {}
    next_id = 1

    # active_links[frame][region_label] = track_id
    active: dict[int, int] = {}  # region_label (in current frame) -> track_id

    for fi, frame in enumerate(sorted_frames):
        regions_t = frames_regions[frame]
        is_last = (fi == len(sorted_frames) - 1)

        if fi == 0:
            # Initialise all regions in first frame as new tracks
            new_active = {}
            for r in regions_t:
                tid = next_id; next_id += 1
                tracks[tid] = Track(
                    track_id=tid,
                    frames=[frame],
                    labels=[r.label],
                    bboxes=[r.bbox],
                    centroids=[r.centroid],
                    areas=[r.area],
                    has_predecessor=False,
                )
                new_active[r.label] = tid
            active = new_active
            continue

        regions_t1 = regions_t  # "t+1" in the description; current frame
        regions_t0 = frames_regions[sorted_frames[fi - 1]]  # previous frame

        # Build lookup: label -> Region for current frame
        t1_by_label: dict[int, Region] = {r.label: r for r in regions_t1}
        # Track which t1 regions have been claimed by a t0 predecessor
        claimed_t1: set[int] = set()
        # Regions in t0 that found a successor
        linked_t0: set[int] = set()

        new_active: dict[int, int] = {}

        # --- For each region in t0, find candidates in t1 ---
        for r0 in regions_t0:
            tid = active.get(r0.label)
            if tid is None:
                continue  # orphan region from t0 (shouldn't happen)

            # Candidate regions in t1 with overlapping bboxes
            candidates_iou = []
            candidates_fallback = []

            for r1 in regions_t1:
                if r1.label in claimed_t1:
                    continue
                da = _normalised_area_change(r0.area, r1.area)
                if da < params.da_min or da > params.da_max:
                    continue
                iou = _iou(r0.mask, r1.mask) if _bboxes_overlap(r0.bbox, r1.bbox) else 0.0
                if iou >= params.overlap_limit_min:
                    candidates_iou.append((iou, r1))
                else:
                    # Centroid-distance fallback
                    dist = np.hypot(
                        r1.centroid[0] - r0.centroid[0],
                        r1.centroid[1] - r0.centroid[1],
                    )
                    if dist <= params.search_radius:
                        candidates_fallback.append((dist, r1))

            # Prefer IoU candidates; fall back to centroid if none
            if candidates_iou:
                candidates = [(1.0 / iou, r1) for iou, r1 in candidates_iou]
            elif candidates_fallback:
                candidates = candidates_fallback
            else:
                candidates = []

            candidates.sort(key=lambda x: x[0])

            if len(candidates) == 0:
                # Track ends — no action needed; track stays in tracks dict
                linked_t0.add(r0.label)  # mark as "handled" (ended)

            elif len(candidates) == 1:
                # Standard 1-to-1 link
                _, r1 = candidates[0]
                tracks[tid].frames.append(frame)
                tracks[tid].labels.append(r1.label)
                tracks[tid].bboxes.append(r1.bbox)
                tracks[tid].centroids.append(r1.centroid)
                tracks[tid].areas.append(r1.area)
                new_active[r1.label] = tid
                claimed_t1.add(r1.label)
                linked_t0.add(r0.label)

            else:
                # Multiple candidates — check for division (top 2)
                _, r1a = candidates[0]
                _, r1b = candidates[1]
                combined_area = r1a.area + r1b.area
                da_div = _normalised_area_change(r0.area, combined_area)

                if (params.da_min <= da_div <= params.da_max
                        and r1b.label not in claimed_t1):
                    # Division event
                    tracks[tid].divide = True

                    tid_a = next_id; next_id += 1
                    tid_b = next_id; next_id += 1

                    tracks[tid].daughter_ids = [tid_a, tid_b]

                    for (tid_d, r1_d) in [(tid_a, r1a), (tid_b, r1b)]:
                        tracks[tid_d] = Track(
                            track_id=tid_d,
                            frames=[frame],
                            labels=[r1_d.label],
                            bboxes=[r1_d.bbox],
                            centroids=[r1_d.centroid],
                            areas=[r1_d.area],
                            mother_id=tid,
                            sister_id=tid_b if tid_d == tid_a else tid_a,
                            has_predecessor=True,
                        )
                        new_active[r1_d.label] = tid_d
                        claimed_t1.add(r1_d.label)

                    linked_t0.add(r0.label)
                else:
                    # Not a clean division; link to best candidate only
                    _, r1 = candidates[0]
                    tracks[tid].frames.append(frame)
                    tracks[tid].labels.append(r1.label)
                    tracks[tid].bboxes.append(r1.bbox)
                    tracks[tid].centroids.append(r1.centroid)
                    tracks[tid].areas.append(r1.area)
                    new_active[r1.label] = tid
                    claimed_t1.add(r1.label)
                    linked_t0.add(r0.label)

        # --- New regions in t1 with no predecessor ---
        for r1 in regions_t1:
            if r1.label in claimed_t1:
                continue
            # This is a new/stray region
            if params.remove_stray and not is_last:
                # We can't know yet if it will have a successor;
                # mark with has_predecessor=False and cull at the end.
                pass
            tid = next_id; next_id += 1
            tracks[tid] = Track(
                track_id=tid,
                frames=[frame],
                labels=[r1.label],
                bboxes=[r1.bbox],
                centroids=[r1.centroid],
                areas=[r1.area],
                has_predecessor=False,
            )
            new_active[r1.label] = tid

        active = new_active

    # --- Post-processing: remove_stray ---
    if params.remove_stray:
        # A stray region: no predecessor AND no successor (single-frame track)
        tracks = {
            tid: t for tid, t in tracks.items()
            if not (not t.has_predecessor and len(t.frames) == 1)
        }

    return tracks


# ---------------------------------------------------------------------------
# Consensus bounding box and mask extraction
# ---------------------------------------------------------------------------

def _union_bbox(bboxes: list[tuple]) -> tuple[int, int, int, int]:
    """Return the union (min_row, min_col, max_row, max_col) of a list of bboxes."""
    min_r = min(b[0] for b in bboxes)
    min_c = min(b[1] for b in bboxes)
    max_r = max(b[2] for b in bboxes)
    max_c = max(b[3] for b in bboxes)
    return (min_r, min_c, max_r, max_c)


def _squarify_and_pad(
    bbox: tuple[int, int, int, int],
    pad: int,
    img_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Pad and squarify a bounding box, clamped to image dimensions.

    Returns:
        (r0, c0, r1, c1) in 0-based half-open convention.
    """
    min_r, min_c, max_r, max_c = bbox
    h = max_r - min_r
    w = max_c - min_c
    side = max(h, w) + 2 * pad
    # Centre the original bbox in the square
    extra_h = side - h
    extra_w = side - w
    r0 = min_r - extra_h // 2
    c0 = min_c - extra_w // 2
    r1 = r0 + side
    c1 = c0 + side
    # Clamp to image
    H, W = img_shape
    r0 = max(0, r0); c0 = max(0, c0)
    r1 = min(H, r1); c1 = min(W, c1)
    return (r0, c0, r1, c1)


def _touches_edge(
    crop_box: tuple[int, int, int, int],
    img_shape: tuple[int, int],
) -> bool:
    r0, c0, r1, c1 = crop_box
    H, W = img_shape
    return r0 == 0 or c0 == 0 or r1 == H or c1 == W


# ---------------------------------------------------------------------------
# HDF5 writing
# ---------------------------------------------------------------------------

def _write_cell_h5(
    cell_dir: Path,
    track: Track,
    masks_by_frame: dict[int, np.ndarray],
    img_shape: tuple[int, int],
    pad: int,
    params: TrackingParams,
) -> None:
    """Write one HDF5 file for a single tracked cell."""
    n_frames = len(track.frames)
    if n_frames == 0:
        return

    # Consensus bounding box (union, then squarify + pad)
    union = _union_bbox(track.bboxes)
    r0, c0, r1, c1 = _squarify_and_pad(union, pad, img_shape)
    H_crop = r1 - r0
    W_crop = c1 - c0

    mask_stack = np.zeros((H_crop, W_crop, n_frames), dtype=bool)
    bb_arr = np.zeros((n_frames, 4), dtype=np.int32)
    r_offset_arr = np.zeros((n_frames, 2), dtype=np.int32)
    edge_arr = np.zeros(n_frames, dtype=bool)

    for fi, (abs_frame, lbl, per_frame_bbox) in enumerate(
        zip(track.frames, track.labels, track.bboxes)
    ):
        labeled = masks_by_frame[abs_frame]
        cell_mask_full = labeled == lbl

        # Crop to consensus box
        cell_mask_crop = cell_mask_full[r0:r1, c0:c1]
        mask_stack[:, :, fi] = cell_mask_crop

        # Per-frame bbox in consensus crop coordinates
        fr0, fc0, fr1, fc1 = per_frame_bbox
        bb_arr[fi] = [
            fc0 - c0,          # x1 (col offset in crop)
            fr0 - r0,          # y1 (row offset in crop)
            fc1 - fc0,         # width
            fr1 - fr0,         # height
        ]
        r_offset_arr[fi] = [c0, r0]  # [x, y] top-left in global image
        edge_arr[fi] = _touches_edge((r0, c0, r1, c1), img_shape)

    # Determine filename capitalisation:
    # Capital C = observed division AND track long enough
    birth_1based = track.frames[0] + 1
    death_1based = track.frames[-1] + 1
    is_complete = (
        track.divide
        and n_frames >= params.min_cell_age
    )
    prefix = "Cell" if is_complete else "cell"
    fname = f"{prefix}{track.track_id:07d}.h5"

    with h5py.File(cell_dir / fname, "w") as h5:
        h5.create_dataset("birth",      data=np.int64(birth_1based))
        h5.create_dataset("death",      data=np.int64(death_1based))
        h5.create_dataset("divide",     data=np.int8(1 if track.divide else 0))
        h5.create_dataset("motherID",   data=np.int64(track.mother_id))
        h5.create_dataset("sisterID",   data=np.int64(track.sister_id))
        h5.create_dataset("daughterID", data=np.array(track.daughter_ids, dtype=np.int64))
        h5.create_dataset("frames",     data=np.array(track.frames, dtype=np.int64))
        h5.create_dataset("BB",         data=bb_arr)
        h5.create_dataset("r_offset",   data=r_offset_arr)
        h5.create_dataset("edgeFlag",   data=edge_arr)
        h5.create_dataset("mask",       data=mask_stack, compression="gzip",
                          compression_opts=4)


# ---------------------------------------------------------------------------
# Per-position entry point
# ---------------------------------------------------------------------------

def _track_position(
    xy_dir: Path,
    params: TrackingParams,
    pad: int,
) -> None:
    """Run the full tracking pipeline for one xy position."""
    masks_dir = xy_dir / "masks"
    cell_dir = xy_dir / "cell"
    cell_dir.mkdir(exist_ok=True)

    if not masks_dir.exists():
        print(f"  [skip] {xy_dir.name}: no masks/ directory")
        return

    # Load masks
    try:
        masks_by_frame = load_masks(masks_dir)
    except FileNotFoundError as e:
        print(f"  [skip] {xy_dir.name}: {e}")
        return

    if not masks_by_frame:
        print(f"  [skip] {xy_dir.name}: empty masks/ directory")
        return

    # Determine image shape from first mask
    img_shape = next(iter(masks_by_frame.values())).shape[:2]

    # Extract regions per frame, apply filters and merging
    frames_regions: dict[int, list[Region]] = {}
    sorted_frame_keys = sorted(masks_by_frame.keys())

    for frame in sorted_frame_keys:
        labeled = masks_by_frame[frame]
        regions = _extract_regions(labeled, params)
        regions = _apply_area_filters(regions, params)
        labeled_merged, regions = _merge_small_regions(labeled, regions, params)
        masks_by_frame[frame] = labeled_merged  # use merged mask for crop extraction
        frames_regions[frame] = regions

    # Link
    tracks = link_frames(frames_regions, params)

    # Write HDF5 files
    for track in tracks.values():
        _write_cell_h5(cell_dir, track, masks_by_frame, img_shape, pad, params)


def _track_position_wrapper(args):
    """Top-level wrapper for ProcessPoolExecutor."""
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
        [d for d in data_dir.iterdir()
         if d.is_dir() and re.match(r"xy\d+$", d.name)],
        key=lambda d: int(d.name[2:]),
    )

    if not xy_dirs:
        print(f"No xy*/ directories found in {data_dir}")
        return

    # Validate preset exists before spawning workers
    params = load_preset(preset)
    print(
        f"Tracking {len(xy_dirs)} position(s) with preset '{preset}' "
        f"(overlap_limit_min={params.overlap_limit_min}, "
        f"da_min={params.da_min}, da_max={params.da_max}, "
        f"remove_stray={params.remove_stray})"
    )

    args = [(str(d), preset, pad) for d in xy_dirs]

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(tqdm(
                pool.map(_track_position_wrapper, args),
                total=len(args),
                desc="Tracking positions",
                unit="pos",
            ))
    else:
        for xy_dir in tqdm(xy_dirs, desc="Tracking positions", unit="pos"):
            _track_position(xy_dir, params, pad)

    print("Tracking complete.")
