# SuperSegger linking port

`nd2_to_cells track` is a bug-for-bug Python port of the lineage-relevant part
of OmniSegger/SuperSegger's `trackOpti` (steps 3–10 of
`BatchSuperSeggerOpti`). Source: `omnisegger/SuperSegger-master` at
`254d081`. The MATLAB path is described in
[`supersegger_linking_reference.md`](supersegger_linking_reference.md), with
corrections in §5 of [`linking_comparison.md`](linking_comparison.md). The
latter compares the IoU linker this port replaced.

The defaults reproduce a `superSeggerGui` run: preset 100XEc, Minimum Cell Age
3, Remove Stray off, `CONST.ignoreerror` unset (3-pass cap).

## What is ported

| SuperSegger | Python |
|---|---|
| `trackOpti` step 3 (`trackOptiStripSmall`) | nothing: commented out in MATLAB |
| `trackOptiLinkCellMulti` (driver, frame re-runs) | `link.link_frames`, `link.process_frame` |
| `updateRegionFields` (props, lineage fields) | `link.Frame`, `link.region_props` |
| `multiAssignmentSparse` (+ `fixProblems`, `exchangeAssignment`, `findNeighborPairs`, `regProperties`, `getBoxLimits`, `imshift`, `getBBpad`, `addBB`) | `assign.py` |
| `errorRez` (+ `createDivision`, `mapBestOfTwo`, `mapToBestOfTwo`, `hasNoFwMapping`) | `link.error_rez` and helpers |
| `createNewCell`, `continueCellLine`, `markDivisionEvent`, `deleteRegions` | `link.py` |
| `merge2Regions` | `mask_edit.merge_regions` |
| `missingSeg2to1` (+ `createMask`, `watershedpaw`) | `mask_edit.missing_seg_2to1` |
| `registerMasks_monointensity` (`imregtform` affine, monomodal) | `imreg.register_masks` |
| `trackOptiCellMarker` | `link.cell_marker` |
| `trackOptiCellFiles` (lineage fields, Cell/cell naming) | `track._build_tracks` + HDF5 writer |

Not ported, because nothing downstream reads it:
- fluorescence background (step 6);
- `CellA` geometry, pole ages and fluorescence statistics (step 7);
- foci (step 8);
- `clist` (step 9);
- region scores (`cellprops3`, `scoreNeuralNet`; only used in log strings);
- skeleton medoids.

Not ported because they are inactive in a GUI run: manual links, the
neighbour flag, `skip > 1`, and the `CONST.ignoreerror` option.

## How the registration was matched

MATLAB's `imregtform` runs ITK's v3 registration framework inside a closed
builtin. `imreg.py` re-implements it. Each piece was pinned down with MATLAB
probes on SuperSegger split calls from 260429:

| Piece | Finding | Probe |
|---|---|---|
| Pyramid | ITK `MultiResolutionPyramidImageFilter`: discrete Gaussian of variance (f/2)² with maximum error 0.1, applied on every level including f = 1, then linear resampling at index (f−1)/2 + f·i | starting MSE at factors 1, 2, 4, 8 agree to the printed precision |
| Metric gradient | Recursive (Deriche) Gaussian of the moving level, σ = pixel spacing, normalised across scale, looked up at the nearest pixel | gradient magnitudes on all three levels agree to every printed digit |
| Transform | ITK affine about the fixed-image centre; MATLAB's `tform` is its inverse | first-step parameter vector has norm exactly 0.0625 |
| Optimizer | Regular-step gradient descent with the scales applied once. Per-level schedule: maximum step × f², minimum step × 10 for each level above the finest | stop messages report each level's current and minimum step |
| Whole path | Single-level, 100 iterations | MATLAB's trace reproduced within print precision; final transform within ~1e-6 |

On the coarse levels, MATLAB's maximum step is 1.0 in matrix units, and the
optimizer becomes chaotic: differences of 1e-7 in the smoothed images grow
into different transforms within a few iterations. MATLAB is not
reproducible across machines either. With 1 thread (what SuperSegger uses on
Windows) and with 10 threads it ends in different transforms, and sometimes in
different split outcomes. Splits are therefore compared by outcome
agreement, measured against MATLAB-vs-MATLAB agreement (below).

## Deliberate differences from MATLAB

- **Safety cap.** After 20 passes on one frame, a final pass runs with all
  mask edits disabled, and a warning is printed. MATLAB can loop forever
  here, because F's subset merge and the `REMOVE_STRAY` deletions are not
  gated by `ignoreError`.
- **Edited masks.** They stay in memory, compressed, and are used for the HDF5
  crops. No `masksOS/` directory is written, and `masks/` is never modified.
- **Output format.** The existing HDF5 layout is kept. MATLAB's duplicated
  `CellA` entry for one-frame cells has no equivalent, because the HDF5 file
  lists `frames` explicitly. `birth` and `death` are converted from file
  indices to frame numbers; they are identical when no frame is missing.
- **Missing mask frames.** They are bridged with a warning. MATLAB stops at
  its mask-count check instead.
- **Large watershed plateaus.** scikit-image's ridge-line watershed is very
  slow on large flat areas. Those arise when the region being split is one
  label shared by distant blobs, an effect of MATLAB's label-collision quirk.
  Above 100 000 plateau pixels, only the cells and a 2-pixel ring are flooded,
  and the rest of the plateau takes its nearest label. This can move ridge
  pixels; on the 260429 cases it cut such calls from minutes to seconds.
- **Where MATLAB would error.** The port raises `RuntimeError`, so that
  position fails, as it would in MATLAB. The error states which MATLAB line
  it mirrors.
- **Two points where the port follows MATLAB rather than the library.** These
  are fidelity fixes, not deviations. A constant image is treated as one
  regional minimum, as in MATLAB (scikit-image finds none). `bwdist`,
  `imhmin` and `imimposemin` run in single precision, as in the toolbox.

## Parity with the 260429 reference run

Run the checks with the scripts in `dev/` (see each file's docstring):

| Check | Script | Result |
|---|---|---|
| Region table vs `regs.props` | `parity_assign.py` | identical, 4 positions × 361 frames |
| Assignment, both directions, every frame pair | `parity_assign.py` | identical maps, ids, `dA`; cost within 2e-10 |
| `errorRez` + driver, teacher-forced, splits disabled | `parity_link.py` | every frame without a split identical (xy01: 191 of 361) |
| Split calls vs MATLAB (1 thread) | `dump_splits.py`, `matlab/oracle_splits.m`, `compare_splits.py` | see below |
| Full run vs SuperSegger cell files | `compare_cells.py` | see below |
