# Cell linking

`nd2_to_cells track` links Omnipose masks with SuperSegger/OmniSegger's
`trackOpti` (steps 3–10 of `BatchSuperSeggerOpti`), restricted to what
determines lineages, with seven changes that reduce merged cells. This page
covers:
- what is ported, and what changes and why;
- how it was checked against SuperSegger;
- the results on 260429, and the tests.

Background:
- [`supersegger_linking_reference.md`](supersegger_linking_reference.md)
  describes SuperSegger's linking (source: `omnisegger/SuperSegger-master` at
  `254d081`).
- [`linking_comparison.md`](linking_comparison.md) compares it with the IoU
  linker that this code replaced.

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
| `missingSeg2to1` | `mask_edit.missing_seg_2to1`: a different split (below) |
| `trackOptiCellMarker` | `link.cell_marker` |
| `trackOptiCellFiles` (lineage fields, Cell/cell naming) | `track._build_tracks` + HDF5 writer |

Not ported, because nothing downstream reads it:
- fluorescence background (step 6);
- `CellA` geometry, pole ages and fluorescence statistics (step 7);
- foci (step 8);
- `clist` (step 9);
- region scores (`cellprops3`, `scoreNeuralNet`).

Not ported because they are inactive in a GUI run: manual links, the
neighbour flag, `skip > 1`, and the `CONST.ignoreerror` option.

The presets hold the values of a `superSeggerGui` run: preset 100XEc,
Minimum Cell Age 3, Remove Stray off, `CONST.ignoreerror` unset (3 passes per
frame). `division_lookahead` is added.

## Changes from SuperSegger

In the code, each change is marked "Not SuperSegger". SuperSegger behaviour
that is kept although it looks wrong is marked "MATLAB quirk".

| # | SuperSegger | Here | Effect on 260429 |
|---|---|---|---|
| 1 | Merges and splits label their result `num_regs + 1` (and `+ 2`), not refreshed within a pass. A merge also adds its label onto a third cell inside the closed union. Two edits in one pass can thus give distant regions one label. | New labels lie above every label in use. A merge writes onto background only. | "Far" merges, where a track jumps to a cell about 1000 px away, drop from 1242 to 3. Peak memory drops from 7–10 GB to 1–2 GB per position: the writer kept full-frame crops for those tracks. |
| 2 | A fused region (2 → 1) is split by registering the two previous cells onto it (affine, `imregtform`) and flooding a watershed from them. | All previous cells are moved onto the region together: centroid onto centroid, then the rotation in ±10° (2° steps) that overlaps the region most. Their parts inside the region seed a watershed of its distance transform. | SuperSegger's registration cuts 21 of 93 fused regions on xy01; ITK v4 registration (SimpleITK) cut 60. The seeded split cuts A + B merges by 60 %. The rotation resolves one more report case. |
| 3 | A 1 → 2 split is undone only if its pieces map to one region in the next frame. | Also if they fuse again within `division_lookahead` frames (3). | A merges: 1259 → 632 on top of #1, 301 → 192 on top of #2. |
| 4 | Undoing a split fails if the closed union is not connected (gap ≥ 3 px). Both pieces then become cells without a mother, and the mother ends without daughters. | The split is called a division. | Fixes lineages; the merge count does not change. |
| 5 | When a small cell fuses with a large one, the assignment links the fused region to the large cell only. errorRez sees no 2 → 1 and tries no split. | A previous cell without forward link, at least `fragment_merge_area` px large, with at least half of its pixels inside a continued region, makes that region a 2 → 1. | B merges drop by a third. |
| 6 | `death` comes from the cell's last frame, where it can be one frame later; labels without pixels ("ghosts", about half of all IDs) become cells with empty masks. | `death` is the last frame; ghosts get no cell file. | Output only. |
| 7 | The cell marker keeps a stale region when a lineage's backward map breaks. | The lineage is dropped. | `Cell`/`cell` naming only. |

Kept as in SuperSegger:
- **The assignment** (`multiAssignmentSparse`) is unchanged, quirks included,
  and matches SuperSegger exactly (below). Fixing its quirks (wrong penalty
  order, bookkeeping in `fixProblems` and `exchangeAssignment`) had no effect
  on merges.
- **errorRez**: its bookkeeping quirks (only sister 1 checked in
  `markDivisionEvent`; the dead "no forward match" test in 1 → 3+; the
  ungated subset merge) also had no measurable effect.

Tested and not adopted, all against the SuperSegger-equivalent port:

| Change | Result |
|---|---|
| ITK v4 registration for the split (SimpleITK) | A + B down by a third on top of #1, against 60 % for the seeded split; it needs an extra dependency |
| End both tracks when a fused region is not split | Removes merges by ending tracks (18 % more tracks); after #2 it changes little |
| A split counts as successful only with two pieces | No effect |
| Penalty order fixed, assignment and errorRez bookkeeping fixed | No effect |

Behaviour that MATLAB doesn't have:
- **Safety cap.** After 20 passes on one frame, one last pass runs without
  mask edits, and a warning is printed. MATLAB can loop forever there.
- **Edited masks.** They stay compressed in memory and are used for the HDF5
  crops. No `masksOS/` is written; `masks/` is never modified.
- **Output.** The HDF5 layout is unchanged. `birth`/`death` are frame
  numbers, not file indices: the two differ only when a mask frame is
  missing, which is bridged with a warning (MATLAB stops).
- **Errors.** Where MATLAB would error, a `RuntimeError` names the MATLAB
  line it mirrors.

## How the port was checked

Before the changes, the port was SuperSegger-equivalent. It was checked
against a SuperSegger run of 260429 (all four positions: err files,
`masksOS`, `cell_old/*.mat`). The checks were:
- **Region tables and the assignment** were identical on 4 positions × 361
  frames: maps, IDs and area changes, with costs within 2e-10.
- **errorRez and the driver**, fed MATLAB's previous frame, were identical
  on every frame without a split.
- **Splits** were not: MATLAB's registration is chaotic on its coarse
  pyramid levels. MATLAB with 10 threads does not reproduce MATLAB with 1
  thread either. Over the 93 split calls of xy01, the port agreed with
  1-thread MATLAB on 70 masks, as often as 10-thread MATLAB did.
- **Full runs:** 78–89 % of matched cells agreed on every lineage field.
  Cells were matched by birth frame and overlap.

That port, a preset switch for each change, and the parity scripts (`dev/`)
are in commit a40b03e (branch `claude/pensive-kirch-096b54`).

## Results on 260429

Summed over xy01–xy04, 361 frames each:
- **Merges A, B, far:** the events defined in [Tests](#tests).
- **Absorbed that end:** absorbed cells whose track ends there.
- **Area jumps:** the share of frame pairs where a track's area grows by
  more than 1.3×, where downstream curation cuts.
- **Clean ≥ 60:** tracks without such a jump for at least 60 frames.
- **Time:** xy01 tracked alone.

| Linker | A | B | far | Absorbed that end | Area jumps | Clean ≥ 60 | Report cases | Time | Peak memory |
|---|---|---|---|---|---|---|---|---|---|
| SuperSegger-equivalent | 903 | 1755 | 1242 | 3526 | 5.63 % | 209 | 0 / 10 | 852 s | 3.1–7.5 GB |
| IoU linker (main at 19fc751) | 1077 | 1027 | 3 | 2016 | 5.02 % | 188 | 1 / 10 | 16 s | 0.4 GB |
| **This linker** | **155** | **697** | **4** | **581** | **3.13 %** | **342** | **10 / 10** | 718 s | 0.9–2.5 GB |

What each change contributes, added one at a time:

| Changes | A | B | far | Clean ≥ 60 | Report cases |
|---|---|---|---|---|---|
| none | 903 | 1755 | 1242 | 209 | 0 |
| #1 | 1259 | 1801 | 3 | 215 | 1 |
| #1, #2 without rotation | 301 | 1024 | 3 | 377 | 9 |
| + #3, #4, #6, #7 | 192 | 1078 | 2 | 356 | 9 |
| + #5 | 157 | 683 | 4 | 346 | 9 |
| + rotation in #2 (this linker) | 155 | 697 | 4 | 342 | 10 |

Downstream curation (`amp_testing/01_preprocessing_curation.py`) was run on
the variant without rotation, which equals this linker in aggregate. Sister
lineages share their ancestors' rows and become copies when curation cuts
them before the division, so lineages are counted as distinct sets of rows.

| Linker | Curated lineages | … beyond 150 min | Curated cell-frames | … beyond 150 min |
|---|---|---|---|---|
| SuperSegger-equivalent | 111 | 49 | 8,148 | 2,464 |
| IoU linker | 108 | 24 | 5,566 | 776 |
| This linker, without rotation | **168** | **137** | **22,940** | **12,263** |

It keeps 2.8× the curated cell-frames of the SuperSegger-equivalent port and
4.1× those of the IoU linker; beyond 150 min, 5.0× and 16×. What curation
still cuts appears to come mostly from other area jumps, such as labels
growing into background, rather than from merges: #5 removed a third of the
B merges without changing long clean tracks.

## Tests

```bash
pip install -e ".[dev]"
pytest                                   # about 2.5 min; no data needed
```

The tests follow the downstream linking report of 2026-10-08
(`amp_testing/linking_report/LINKING_ISSUES.md`), which names two linking
problems:
- **A, re-merge after division.** The daughters' labels fuse again. One
  daughter's track continues on the fused object, and the sister's track
  ends.
- **B, neighbour absorbed.** Omnipose fuses two neighbouring cells, and one
  track continues on the fused object.

Its other two types, C (a label grows into background) and D (a track spans
two labels), are segmentation errors and are not tested.

`tests/merge_events.py` finds merges in any track output: per-cell
`cell*.h5` files or one `cells.h5`. At frame t, track T absorbs track X if
T's mask at t covers at least half of T's own mask at t−1 and at least half
of X's mask at t−1. No area jump is required, unlike the report's
`linking_diagnostics.py`: when the larger cell keeps its ID, the merged
object can grow by less than 1.3×.

| Kind | X is | Typical cause |
|---|---|---|
| A | T's sister | daughters fuse again after a division |
| B | another cell, within 50 px of T at t−1 | Omnipose fuses neighbours |
| far | another cell, more than 50 px away | SuperSegger's label collision (#1) |

For whole positions: `python tests/merge_events.py <cell dir> ... [--csv events.csv]`.

| File | What it checks |
|---|---|
| `test_merge_events.py` | the event check itself, on hand-made tracks |
| `test_changes.py` | changes #1, #2 and #6 on hand-made frames |
| `test_scenarios.py` | synthetic movies: growth and division (controls), daughters that fuse (A), neighbours that fuse (B), a small neighbour that fuses (B, change #5), a temporary split that must not become a division (#3), and one too wide to merge back (#4) |
| `test_report_cases.py` | 10 real-data cases: one A and one B per position, plus the locations of the report's figures A and B. Each is a merge event of the SuperSegger-equivalent port, linked as a 13–18-frame window. Neither cell may absorb a track or be absorbed at t. |

The real-data cases are stored in `tests/data/` as crops of the 260429
masks (350 KB in all) and are put back into empty frames of the original
size. In every crop, the SuperSegger-equivalent port still produces the
case's merge.
