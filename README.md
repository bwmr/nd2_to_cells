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
```

## Output layout

```
/data/experiment/
  raw_im/                    pre-alignment TIFF copies
  xy01/
    phase/                   aligned phase TIFFs
    fluor1/                  aligned fluorescence channel 1
    fluor2/                  aligned fluorescence channel 2 (if present)
    masks/                   Omnipose PNG masks (populated by step 3)
    cell/                    HDF5 output (populated by step 4)
      cell0000001.h5         one file per tracked cell (default)
      Cell0000002.h5         capital C = complete cell cycle observed
      ...                    or, with --consolidated:
      cells.h5               single file; one group per cell
  xy02/ ...
```

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
| `BB`       | int32       | (T, 4)   | Padded bbox per frame [x1, y1, w, h]       |
| `r_offset` | int32       | (T, 2)   | Top-left of padded crop [x, y]             |
| `edgeFlag` | bool        | (T,)     | True if crop touches image boundary        |
| `mask`     | bool        | (H, W, T)| Binary Omnipose mask in consensus crop     |

## Presets

Tracking parameters are stored in `presets/` as TOML files. Available presets:

- `100XEc` — *E. coli*, 100x objective, 60 nm/px
- `100XPa` — *P. aeruginosa*, 100x objective, 60 nm/px

Parameters are derived from the corresponding SuperSegger `.mat` preset files.
Custom presets can be added by placing a `.toml` file in the `presets/` directory
and passing `--preset /path/to/custom.toml`.
