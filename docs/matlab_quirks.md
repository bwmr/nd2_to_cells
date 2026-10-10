# MATLAB quirks: inventory and fixes

The port reproduces SuperSegger bug-for-bug: that is the baseline. Each fix
is opt-in, named in a preset's `fixes` list (`track.FIXES`), and is tested on
its own against the baseline. The tests and the event counts (A, B, far) are
described in [`linking_report_tests.md`](linking_report_tests.md).

Status:
- **tested**: the fix exists and has results below;
- **open**: not implemented yet;
- **no effect**: the quirk cannot trigger in our settings.

## Mask edits (`mask_edit.py`)

| # | Quirk | Likely effect | Fix | Status |
|---|---|---|---|---|
| 1 | Merges and splits label their result `num_regs + 1` (and `+ 2`), and `num_regs` is not refreshed within a pass. A merge also adds its label onto any third cell inside the closed union (label sums). Two edits in one pass can therefore give distant regions the same label. | "far" events: one track picks up a distant cell and then jumps there. 1242 baseline events on 260429. | `label_collision` | tested |
| 2 | A split counts as a success when the result's largest label is 2, even if label 1 is missing. The region is then only relabelled, and the frame is re-run. | Wasted passes (29 of 52 split "successes" on xy01), ending in best-of-two anyway. Runtime, and possibly label order. | `split_needs_two` | tested |
| 3 | `watershedpaw` mixes up the x and y bounds when clipping the 3×3 neighbourhood. | Small changes to split geometry near the window edge. | — | negligible: only for ridge pixels on the image border (the watershed window keeps a 16 px margin) |

## Assignment (`assign.py`)

| # | Quirk | Likely effect | Fix | Status |
|---|---|---|---|---|
| 4 | The penalty for an area change above 0.3 (50) is written after the one above 0.6 (1000), so 1000 never applies. | Links onto fused or broken objects cost far less than intended. Could affect A and B. | `area_penalty` | tested |
| 5 | `fixProblems` assigns a leftover f region without updating the reverse map. | The two directions disagree, so errorRez sees false 1→2 or 2→1 cases. | `assign_rev` | tested in `assign_bookkeeping` |
| 6 | `revAssign{FAlready}` uses only the first of several regions. | Wrong area comparison when a cell already maps to two regions. | `assign_all_regions` | tested in `assign_bookkeeping` |
| 7 | The new area change of the robbed cell is normalised by the stealer's area. | Wrong accept/reject when a region is reassigned. | `assign_robbed_area` | tested in `assign_bookkeeping` |
| 8 | `exchangeAssignment`: when no region holds the target, `bad_c` falls through to the last region. | Rare spurious swaps. | `assign_exchange` | tested in `assign_bookkeeping` |
| 9 | `min` over an all-NaN cost row returns column 1. | Rare arbitrary assignment. | `assign_nan` | tested in `assign_bookkeeping` |
| 10 | `if centroidCost == 0` tests the whole vector. | Needs an exact zero centroid shift. | — | no effect in practice |
| 11 | `goodOneToOne` is indexed by column but read by region number. | None: without manual links, columns 1…n are regions 1…n. | — | no effect |

## errorRez and the driver (`link.py`)

| # | Quirk | Likely effect | Fix | Status |
|---|---|---|---|---|
| 12 | 2→1 (two cells map to one region): if the split fails, the cheaper cell continues on the fused region and the other ends without a flag (`mapToBestOfTwo`). | This is the A and B mechanism itself. Options: a better split (ITK v4 registration, #18), a seeded split, or end both tracks with a flag. | `seeded_split`, `end_on_merge` (with `itk_v4`, #18) | tested |
| 13 | 1→2 is called a division or an over-segmentation from one frame of look-ahead only (do both pieces map to the same region at t+1?). | A split that lasts two or more frames becomes a division, and its later re-fusion becomes an A event. A few frames of look-ahead could tell real divisions from temporary splits. | `division_lookahead = k` (preset key; 1 in SuperSegger) | tested with k = 3, 5 |
| 14 | `markDivisionEvent` checks only whether sister 1 already has an ID. | Sister 2's ID can be overwritten. | `division_both_sisters` | tested in `errorrez_bookkeeping` |
| 15 | `x = c{[a b]}` keeps the first element (`cCellsFromR`, `cCellsTransp`, and the 1→2 sister map). | Only the first of two previous cells is examined. | — | no effect: the first element is only read where the list has one element |
| 16 | 1→3+: `isempty` of a cell array is always false, so "no forward match" never triggers the merge. | That branch is dead. | `f_no_match` | tested in `errorrez_bookkeeping` |
| 17 | 1→3+: the subset merge is not gated by `ignoreError`. | A frame can loop forever; the port's safety cap stops it. | `f_gate` | tested in `errorrez_bookkeeping` |

## Cell marker and output (`link.py`, `track.py`)

| # | Quirk | Likely effect | Fix | Status |
|---|---|---|---|---|
| 18 | Registration for splits follows MATLAB's ITK v3 path (not a quirk, an upgrade path). | ITK v4 cut 60 of 93 split calls against 23 (`supersegger_port.md`). Expected to reduce A and B. | `itk_v4` | tested |
| 19 | CellMarker: a failed backward lookup leaves `jj` stale for the next frame. | Wrong `Cell`/`cell` naming on some lineages. | `marker_chain` | implemented; output only |
| 20 | `death` comes from the cell's last frame, where it can be stale (last frame + 1). | Wrong `death` in the HDF5 file. Downstream reads `death`. | `death_last_frame` | implemented; output only |
| 21 | Labels without pixels (ghosts) become cells with empty masks. | About half of all cell files on 260429 are ghosts, each with its own ID. | `drop_ghosts` | implemented; output only |

## Results

Event counts summed over xy01–xy04 (361 frames each), and outcomes of the 10
real-data cases of `test_report_cases.py`:

| Configuration | A | B | far | Cases resolved | Synthetic controls |
|---|---|---|---|---|---|
| baseline | 903 | 1755 | 1242 | 0 / 10 | pass |
| `main`'s IoU linker (19fc751) | 1077 | 1027 | 3 | 1 / 10 | fusions fail |
| `label_collision` | 1259 | 1801 | 3 | 1 / 10 | pass |
| `area_penalty` | 941 | 1765 | 1250 | 0 / 10 | pass |
| `split_needs_two` | 1071 | 1814 | 958 | 0 / 10 | pass |
