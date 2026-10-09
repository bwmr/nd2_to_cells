# nd2_to_cells

Python CLI pipeline converting ND2 microscopy files to per-cell HDF5 files
for downstream timelapse analysis. Replaces the MATLAB SuperSegger steps
(alignment, linking, cell file generation) while keeping Omnipose segmentation
unchanged.

## Installation

```bash
pip install -e .
```

## Workflow

```bash
# 1. Export ND2 → per-position TIFFs
nd2_to_cells export \
  --input experiment.nd2 \
  --output /data/experiment/ \
  --basename 260430 \
  --phase-channel 0

# 2. Drift correction
nd2_to_cells align \
  --data /data/experiment/ \
  --workers 4

# 3. Run Omnipose (external, unchanged)
conda activate omnipose
python -m omnipose --dir /data/experiment/xy01/phase/ \
  --save_png --dir_above --no_npy --in_folders --omni \
  --pretrained_model bact_phase_omni --cluster \
  --mask_threshold 1 --flow_threshold 0 --diameter 30 --exclude_on_edges
# repeat for each xy position, or loop

# 4. Track cells and write HDF5 output
nd2_to_cells track \
  --data /data/experiment/ \
  --preset 100XEc \
  --workers 4
# add --consolidated to write one cells.h5 per position instead of one file per cell
# previous HDF5 output in xy*/cell/ is replaced; masks/ is never modified
```

### Z-stack workflow (alternative to align)

Use this workflow when your ND2 file contains a Z-stack and you want to
preserve individual slices rather than project them:

```bash
# 1. Export all Z slices as individual TIFFs
nd2_to_cells export \
  --input experiment.nd2 \
  --output /data/experiment/ \
  --basename 260430 \
  --export-z-slices

# 2. Assemble slices into per-timepoint ZYX TIFFs
nd2_to_cells assemble \
  --data /data/experiment/ \
  --basename 260430 \
  --workers 4
```

Caveats for the Z-stack workflow:

- `assemble` does **no drift correction**.
- Segmentation and tracking of Z-stack (3D+T) data are not supported; the
  workflow ends with the ZYX stacks in `xy{P}/phase/`, `xy{P}/fluor1/`, …

## Output layout

```
/data/experiment/
  raw_im/                    exported TIFFs (input to align/assemble)
                             (Z-stack files named …_t{T}xy{P}z{Z}c{C}.tif
                              when --export-z-slices is used)
  xy01/
    phase/                   aligned phase TIFFs
    fluor1/                  aligned fluorescence channel 1
    fluor2/                  aligned fluorescence channel 2 (if present)
    masks/                   Omnipose PNG masks (populated by step 3)
    cell/                    HDF5 output (populated by step 4)
      cell0000001.h5         one file per tracked cell (default)
      Cell0000002.h5         capital C = complete cell cycle (birth + division)
      ...                    or, with --consolidated:
      cells.h5               single file; one group per cell
  xy02/ ...
```

## Export options

| Flag | Default | Description |
|------|---------|-------------|
| `--phase-channel N` | `0` | 0-based index of the phase-contrast channel. All other channels become `fluor1`, `fluor2`, … |
| `--z-project mean\|max` | `mean` | Z-projection method applied when the ND2 contains a Z-stack. Ignored when `--export-z-slices` is set. |
| `--export-z-slices` | off | Write each Z slice as a separate TIFF instead of projecting. Output files are named `{basename}_t{T}xy{P}z{Z}c{C}.tif`; positions with a single plane are written as `z1`. |

## Align options

`align` reads `{basename}_t{T}xy{P}c{C}.tif` files from `raw_im/` and writes
drift-corrected TIFFs into `xy{P}/phase/`, `xy{P}/fluor1/`, … Existing
`phase/` and `fluor*/` folders are removed before writing. Other files in
`raw_im/` are ignored.

| Flag | Default | Description |
|------|---------|-------------|
| `--data DIR` | required | Experiment directory containing `raw_im/` and `xy*/`. |
| `--basename STR` | auto | Filename prefix used during export. Required only if `raw_im/` contains more than one basename. |
| `--align-channel NAME` | `phase` | Channel used to compute shifts. |
| `--align-to-first` | off | Register every frame against frame 1 instead of the previous frame. |
| `--max-shift-px N` | `50` | A frame whose shift jumps by more than this from the previous accepted frame is an outlier and keeps that frame's shift, unless the next frame confirms the jump (real stage movement). |
| `--workers N` | `1` | Number of parallel worker processes (one per xy position). |

## Assemble options

`assemble` is used in place of `align` for Z-stack data. It reads the
per-slice TIFFs produced by `export --export-z-slices` and writes one
**ZYX** TIFF per timepoint × channel into `xy{P}/{channel}/`. Existing
`phase/` and `fluor*/` folders are removed before writing.

| Flag | Default | Description |
|------|---------|-------------|
| `--data DIR` | required | Experiment directory containing `raw_im/` and `xy*/`. |
| `--basename STR` | auto | Filename prefix used during export (e.g. `260430`). Required only if `raw_im/` contains Z-slice TIFFs of more than one basename. |
| `--workers N` | `1` | Number of parallel worker processes (one per xy position). |

## Track options

`track` reads Omnipose masks from `xy{P}/masks/` and writes HDF5 output to
`xy{P}/cell/`. Previous `cell*.h5`, `Cell*.h5` and `cells.h5` files there are
replaced; `masks/` is never modified. A warning is printed if `phase/` images
are newer than the masks.

| Flag | Default | Description |
|------|---------|-------------|
| `--data DIR` | required | Experiment directory containing `xy*/`. |
| `--preset NAME\|PATH` | `100XEc` | Preset name from `presets/` or path to a custom `.toml` file. |
| `--pad N` | `5` | Padding in pixels added around each cell's bounding box. |
| `--consolidated` | off | Write one `cells.h5` per position (one group per cell) instead of one file per cell. |
| `--workers N` | `1` | Number of parallel worker processes (one per xy position). |

Division and linking checks are set in the preset (not SuperSegger parameters;
both bundled presets use the defaults below):

| Preset key | Default | Description |
|------------|---------|-------------|
| `min_division_age` | `8` | Minimum track age (frames) before a split counts as a division. Earlier splits are treated as segmentation flicker and the pieces are re-joined under the mother's identity. Tracks present in the first frame are exempt. Depends on the frame interval. |
| `min_sister_ratio` | `0.5` | Minimum area ratio (smaller / larger) of the two pieces for a split to count as a division; unequal splits are re-joined. |
| `confident_iou` | `0.3` | IoU at or above which a link is accepted regardless of `da_min` / `da_max`; the area-change limits then gate only low-overlap or centroid-fallback links. |

## HDF5 cell file layout

In the default mode each `cell{ID:07d}.h5` contains the datasets below at the
root level. With `--consolidated`, all cells are stored in a single `cells.h5`
per position; each cell occupies a group (e.g. `cells.h5/Cell0000002/`) and the
same datasets live inside that group.

Each cell contains:

| Dataset    | Type        | Shape    | Description                                |
|------------|-------------|----------|--------------------------------------------|
| `birth`    | int64       | scalar   | 1-based frame of first appearance          |
| `death`    | int64       | scalar   | 1-based frame of last appearance           |
| `divide`   | int8        | scalar   | 1 if division observed, 0 otherwise        |
| `motherID` | int64       | scalar   | Track ID of mother cell (0 = none)         |
| `sisterID` | int64       | scalar   | Track ID of sister cell (0 = none)         |
| `daughterID` | int64     | (0,) or (2,) | Track IDs of daughter cells           |
| `frames`   | int64       | (T,)     | 0-based absolute frame indices             |
| `BB`       | int32       | (T, 4)   | Per-frame cell bbox (unpadded), relative to crop top-left [x, y, w, h] |
| `r_offset` | int32       | (T, 2)   | Top-left of padded crop in image [x, y] (same for all frames) |
| `edgeFlag` | bool        | (T,)     | True if padded crop touches the image edge (same for all frames)¹ |
| `mask`     | bool        | (H, W, T)| Binary Omnipose mask in consensus crop     |

¹ After `align`, the image edge is the padded canvas edge, not where real
pixels end.

## Presets

Tracking parameters are stored in `presets/` as TOML files. Available presets:

- `100XEc` — *E. coli*, 100x objective, 60 nm/px
- `100XPa` — *P. aeruginosa*, 100x objective, 60 nm/px

Some values are taken from the corresponding SuperSegger `.mat` preset files,
but no key behaves exactly like its SuperSegger counterpart; the comments in
`presets/100XEc.toml` say what each key does. `search_radius`,
`min_division_age`, `min_sister_ratio` and `confident_iou` have no SuperSegger
counterpart.

`min_cycle_frames` (default `6`) is the number of observed frames a cycle
needs to count as complete (`Cell*` files). A track born during the movie
can't divide before `min_division_age` frames, so `min_cycle_frames` only has
an effect if it is larger than `min_division_age`.

Presets that use the earlier key names still load, with a warning naming the
new key: `overlap_limit_min` → `min_link_iou`, `small_area_merge` →
`fragment_merge_area`, `remove_stray` → `drop_single_frame_strays`,
`min_cell_age` → `min_cycle_frames`. Values carry over unchanged. Unknown keys
are ignored with a warning.
Pass a preset name (`--preset 100XEc`, looked up in `presets/`) or a path to
your own `.toml` (`--preset /path/to/custom.toml`).
