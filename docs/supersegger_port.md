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
| Same, with splits | `parity_link.py` | xy01: 256 of 361 frames identical; the rest differ only through split outcomes |
| Split calls vs MATLAB (1 thread) | `dump_splits.py`, `matlab/oracle_splits.m`, `compare_splits.py` | as close as MATLAB with 10 threads (below) |
| Full run vs SuperSegger cell files | `compare_cells.py --frames` | below |

### Split calls

These are all 93 `missingSeg2to1` calls of an xy01 run. Every call starts
from the same inputs, and is compared with MATLAB on 1 thread, which matches
the Windows reference:

| | same mask | same success, other mask | success differs |
|---|---|---|---|
| Python port | 70 | 15 | 8 |
| MATLAB, 10 threads | 70 | 12 | 11 |

"Same mask" includes calls that fail on both sides and leave the mask
unchanged. Per call, the port takes a median of 0.8 s (maximum 5 s). MATLAB on
1 thread takes 37 s (maximum 87 s).

### ITK v4 (SimpleITK) instead of MATLAB's v3 path

`dev/compare_v4.py` runs the same 93 calls with a SimpleITK v4 registration.
It uses the same optimizer settings, without MATLAB's per-level step schedule.

The quality proxy is the mean IoU of the two new pieces with the two cells of
the previous frame, unregistered. It is a proxy, not ground truth. Pieces are
counted inside the fused region only. In 15 of the 93 calls, an earlier edit
of the same pass had already written label `num_regs + 1` or `+ 2` elsewhere
in the frame (the label-collision quirk). A frame-wide count mistook that for
a piece in two calls, and lowered the IoU in others.

| | success | real two-piece splits | IoU with previous cells | median time per call |
|---|---|---|---|---|
| Port (v3 path) | 52 | 21 | 0.68 | 0.8 s |
| MATLAB, 1 thread | 54 | 22 | 0.68 | 37 s |
| SimpleITK v4 | 60 | 60 | 0.85 | 0.6 s |

The MATLAB path counts a call as successful whenever the result has at most
two labels. More than half of those successes only relabel the region and do
not cut it: 31 of 52 for the port, 32 of 54 for MATLAB. With the v4
registration, every success is a real split, 60 of 93 calls.

v4 agrees with the v3 path on success in only 57 of 93 calls, so adopting it
would give up parity with SuperSegger. The port keeps the v3 path as the
baseline; v4 will be tested later as a separate divergence.

### Full runs from the raw masks

The comparison is against `cell_old/*.mat` and the err files of 260429.

| | xy01 | xy02 | xy03 | xy04 |
|---|---|---|---|---|
| Cells, port / SuperSegger | 3469 / 3492 | 4495 / 4459 | 8229 / 8333 | 4092 / 4090 |
| `Cell` (complete cycles) | 47 / 50 | 63 / 64 | 76 / 77 | 38 / 38 |
| Divisions | 391 / 395 | 538 / 528 | 828 / 839 | 395 / 400 |
| Frames with identical segmentation (of 361) | 244 | 215 | 231 | 295 |
| First frame that differs | 6 | 21 | 23 | 50 |
| Real cells matched (born in the same frame, largest overlap) | 1446 / 1516 | 2300 / 2427 | 3528 / 3636 | 2141 / 2184 |
| Matched cells with identical death, divide, frames, mother, daughters and naming | 80.4 % | 78.5 % | 85.0 % | 89.3 % |

About half of all IDs are ghosts: labels with no pixels. SuperSegger writes
them as well (xy01: 1976). Cell IDs diverge after the first split that
differs, so comparing by ID says little.

Matching runs on lineage structure instead. A SuperSegger cell is matched to
the port's cell that is born in the same frame and has the largest overlap
with it there. Two cells count as agreeing only when their mother and
daughters are also matched to each other.

The teacher-forced checks show that the remaining differences start at split
outcomes, and per-call split agreement is the same as MATLAB's own agreement
across thread counts. A full MATLAB run with 10 threads was not made, so
whole-run variation of MATLAB against itself is not measured.

On xy01, `timelapse_analysis.check_cell_ok` keeps 940 of the port's cells and
951 of SuperSegger's. The HDF5 schema is unchanged.

Runtime: 14 min (852 s) for xy01 run alone, with nothing else running
(2026-10-12; an earlier measurement gave 32 min). Run three at a time, while
MATLAB was also running, positions took 46–65 min.
