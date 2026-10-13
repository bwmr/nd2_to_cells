# MATLAB quirks: inventory, fixes and results

The port reproduces SuperSegger bug-for-bug: that is the baseline. Each fix
is opt-in. It is named in a preset's `fixes` list (`track.FIXES`), or set by
a preset key such as `division_lookahead`, and it was tested against the
baseline on 260429, all four positions. The tests and the event kinds (A, B,
far) are described in [`linking_report_tests.md`](linking_report_tests.md).

## Overview (2026-10-10)

The headline numbers are summed over xy01–xy04, 361 frames each.

| Configuration | Merges (A + B + far) | Absorbed cells that end | Area jumps > 1.3× | Tracks clean ≥ 60 frames | Cases resolved | Peak memory |
|---|---|---|---|---|---|---|
| baseline (SuperSegger) | 3900 | 3526 | 5.63 % | 209 | 0 / 10 | ~7 GB¹ |
| `main`'s IoU linker (19fc751) | 2107 | 2016 | 5.02 % | 188 | 1 / 10 | — |
| `label_collision` | 3063 | 2963 | 4.99 % | 215 | 1 / 10 | — |
| `lc_seeded_split` | 1328 | 1185 | 3.00 % | **377** | 9 / 10 | 1.3 GB |
| `combined` | 1272 | 1121 | 3.02 % | 356 | 9 / 10 | 2.1 GB |
| `combined_end` | 1192 | 1052 | **2.98 %** | 359 | 9 / 10 | 2.1 GB |
| `lc_seeded_orphan` | 892 | 616 | 3.13 % | 373 | 9 / 10 | 1.7 GB |
| `combined_orphan` | **844** | **561** | 3.15 % | 346 | 9 / 10 | 2.6 GB |
| `combined_orphan_plus` | 845 | 580 | 3.16 % | 346 | 8 / 10 | 1.2 GB |
| `combined_orphan_rigid` | 856 | 581 | 3.13 % | 342 | **10 / 10** | 2.5 GB |

¹ Measured on the assignment and errorRez group runs, which give the
baseline's counts. Runs with more mask edits and the label-collision quirk
reach 10 GB.

- `lc_` means on top of `label_collision`.
- `combined` is `label_collision`, `seeded_split`, `merge_fallback` and `division_lookahead = 3`, plus the output fixes `death_last_frame`, `drop_ghosts` and `marker_chain`.
- `combined_end` adds `end_on_merge`, and `combined_orphan` adds
  `orphan_split`.
- `combined_orphan_plus` adds the fixes for #4, #7, #8, #9 and #16 to
  `combined_orphan`. `combined_orphan_rigid` replaces `seeded_split` with
  `rigid_split`. Both were tested in the final run (below).

Findings:

1. **`label_collision` (#1) is the first fix to make.** It removes the "far"
   events (1242 → 3), where one track picks up a cell about 1000 px away and
   jumps there. Peak memory drops from 7–10 GB per position to 1–2 GB: the
   writer kept full-frame-sized crops for those tracks. It also
   stops every other mask-editing fix from adding new far events, so the
   other fixes are best judged on top of it.
2. **The split decides A and B (#12, #18).** MATLAB's affine registration
   splits few fused regions. On top of `label_collision`:
   - ITK v4 (`lc_itk_v4`) cuts A + B by a third;
   - `seeded_split` cuts them by about 60 %.

   `seeded_split` needs no registration. It seeds a watershed with the
   previous cells, aligned by their centroid. It also resolves 9 of the 10
   real-data cases, and in 8 of them both cells carry on.
3. **Look-ahead (#13) halves A.** In the `lc_` runs A goes from 1259 to 632,
   and on top of `seeded_split` from 301 to 192. Two caveats:
   - It needs the division fallback (#22): a merge request for a split with
     a real gap otherwise orphans both pieces.
   - It costs a little in clean long tracks (356 against 377 without it).
     Some temporary splits it undoes may have been real divisions.
4. **`end_on_merge` (#12) removes A and B by ending tracks rather than
   continuing them.** Alone it gives 18 % more tracks. With a good split
   (`combined_end`), only the few merges left over change.
5. **Most remaining merges are small neighbours absorbed within the area
   tolerance (#23).** In `combined`, 91 % of the remaining B events change
   the absorbing cell's area by less than `da_max` = 0.3. The absorbed cell
   is small, with a median of 18 % of the absorber. The assignment links the
   fused region to the large cell only, so errorRez sees no 2→1 and tries no
   split. `orphan_split` handles a previous cell without a forward link,
   lying inside a continued region, as a 2→1:
   - B drops by a third (1078 → 683 in `combined_orphan`);
   - the absorbed cells that end halve (1121 → 561);
   - all merges are 78 % below the baseline and 60 % below `main`.

   Area jumps and long clean tracks do not improve (346 against 356). What
   downstream curation still cuts must therefore come mostly from other area
   jumps, such as labels growing into background (the report's type C).
6. **No effect on merges:**
   - `area_penalty` (#4) and `split_needs_two` (#2);
   - both bookkeeping groups (assignment #5–#9, errorRez #14, #16, #17);
   - `merge_fallback` (#22) alone. It keeps the number of merges
     identical, but turns orphaned pieces back into sisters (B → A), so it
     fixes lineages.
7. **Speed.** Timed alone on xy01 (final run), `combined_orphan` takes
   677 s against 852 s for the baseline: fewer frame re-runs, and no
   registration. `main`'s IoU linker takes 16 s. The earlier CPU figures
   (671 s for `combined` against 1804 s) came from overlapping runs and
   overstated the gap.

Questions for discussion:
- Which fixes become the default, and does a SuperSegger-faithful preset stay
  available?
- Look-ahead: keep it (fewer A events), or not (slightly more long clean
  tracks)? Which k?
- Merges that can't be split: continue the cheaper cell (MATLAB) or end both
  (`end_on_merge`)?
- `orphan_split`: adopt? It cuts merges by a third, but the downstream proxy
  stays flat.
- Since merges are no longer the main source of curation cuts, should the
  next step target the other area jumps (segmentation flicker), for example
  by bridging transient jumps?
- The writer's memory (crop buffers for all tracks until the end) is only a
  problem with far tracks. Is it worth changing anyway?
- `seeded_split` or `rigid_split`? They are equal in aggregate; `rigid_split`
  also resolves the last report case and takes 6 % longer.

## Final test run (2026-10-12)

Code at 331042d, all four positions of 260429. The tracking runs, the timing
runs and the case outcomes ran first, then the parity scripts and the
downstream runs.

Checks:
- **Baseline regression.** On all four positions, the baseline's cell
  files are identical to those of the 2026-10-08 run, every file and every
  dataset.
- **Parity scripts.**
  - `parity_assign.py`: no failures on any of the four positions, and the
    cost agrees within 2.3e-10.
  - `parity_link.py`, xy01: every line is identical to the earlier log,
    timings aside. 256 of 361 frames are identical (`supersegger_port.md`).
- **Determinism.** `combined_orphan` is identical to its 2026-10-10 run.
  Each timing run is identical to the full run of its configuration.

Timing: xy01, one configuration at a time, nothing else running.

| Configuration | Wall | CPU (user + sys) | Peak memory |
|---|---|---|---|
| baseline | 852 s | 851 s | 3.3 GB |
| `combined_orphan` | 677 s | 676 s | 0.9 GB |
| `combined_orphan_rigid` | 718 s | 717 s | 0.9 GB |
| `main`'s IoU linker | 16 s | 16 s | 0.4 GB |

Peak memory is the process's maximum resident set, and it varies between
runs with identical output. On xy01, `combined_orphan` peaked at 0.9 GB here,
and at 2.5 GB in the full run, which ran beside seven others. In the full
runs, the baseline peaked at 3.1–7.5 GB per position and `combined_orphan` at
0.8–2.5 GB.

The two extra configurations (counts in the tables above):
- **`combined_orphan_plus`** (fixes for #4, #7, #8, #9, #16). In aggregate,
  nothing changes, as with the separate runs (finding 6). It loses report
  case xy01_B: one track already holds both cells at t−1, and it has
  absorbed another track by frame t.
- **`combined_orphan_rigid`** (`rigid_split`). Equal in aggregate. It is the
  first configuration to resolve all ten report cases: xy03_B was merged in
  every earlier one.

Downstream: `amp_testing/01_preprocessing_curation.py`, up to the export of
the curated table. It ran on a scratch copy, with the images linked
read-only and the cell files of each configuration.
- **Counting.** A lineage is a root-to-leaf path, so sister lineages share
  their ancestors' rows. When curation cuts them before the division, they
  become copies of each other, so the table counts distinct row sets.
- **Workaround.** Since `timelapse_analysis` 441e5d4, the script's overview
  plot fails on several positions: positions are concatenated with duplicate
  index labels, and seaborn cannot plot those once growth rates hold NaN.
  The copy resets the index after import, which does not change the
  curation.

| Configuration | Curated lineages | … beyond 150 min | Curated cells | Curated cell-frames | … beyond 150 min |
|---|---|---|---|---|---|
| baseline | 111 | 49 | 181 | 8,148 | 2,464 |
| `main`'s IoU linker | 108 | 24 | 152 | 5,566 | 776 |
| `combined_orphan` | **168** | **137** | **312** | **22,940** | **12,263** |

Against the baseline, `combined_orphan` keeps 2.8× the curated cell-frames,
and against `main`, 4.1×. Beyond 150 min the factors are 5.0× and 16×, with
137 lineages against 49 and 24. The merge fixes thus do pay off downstream.
Long clean tracks (346 against 209 for the baseline and 188 for `main`) had
already shown this, even though they barely differ between the
`combined_*` variants.

## Mask edits (`mask_edit.py`)

| # | Quirk | Likely effect | Fix | Status |
|---|---|---|---|---|
| 1 | Merges and splits label their result `num_regs + 1` (and `+ 2`), and `num_regs` is not refreshed within a pass. A merge also adds its label onto any third cell inside the closed union (label sums). Two edits in one pass can therefore give distant regions the same label. | "far" events: one track picks up a distant cell and then jumps there. 1242 baseline events on 260429. | `label_collision` | tested |
| 2 | A split counts as a success when the result's largest label is 2, even if label 1 is missing. The region is then only relabelled, and the frame is re-run. | Wasted passes (31 of 52 split "successes" on xy01), ending in best-of-two anyway. Runtime, and possibly label order. | `split_needs_two` | tested |
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
| 12 | 2→1 (two cells map to one region): if the split fails, the cheaper cell continues on the fused region and the other ends without a flag (`mapToBestOfTwo`). | This is the A and B mechanism itself. Options: a better split (ITK v4 registration, #18), a seeded split, or end both tracks with a flag. | `seeded_split`, `rigid_split` (seeded split after a rotation search), `end_on_merge` (with `itk_v4`, #18) | tested |
| 13 | 1→2 is called a division or an over-segmentation from one frame of look-ahead only (do both pieces map to the same region at t+1?). | A split that lasts two or more frames becomes a division, and its later re-fusion becomes an A event. A few frames of look-ahead could tell real divisions from temporary splits. | `division_lookahead = k` (preset key; 1 in SuperSegger). If the merge fails, the division is called (see #22). | tested with k = 3, 5 |
| 14 | `markDivisionEvent` checks only whether sister 1 already has an ID. | Sister 2's ID can be overwritten. | `division_both_sisters` | tested in `errorrez_bookkeeping` |
| 15 | `x = c{[a b]}` keeps the first element (`cCellsFromR`, `cCellsTransp`, and the 1→2 sister map). | Only the first of two previous cells is examined. | — | no effect: the first element is only read where the list has one element |
| 16 | 1→3+: `isempty` of a cell array is always false, so "no forward match" never triggers the merge. | That branch is dead. | `f_no_match` | tested in `errorrez_bookkeeping` |
| 17 | 1→3+: the subset merge is not gated by `ignoreError`. | A frame can loop forever; the port's safety cap stops it. | `f_gate` | tested in `errorrez_bookkeeping` |
| 22 | 1→2 merge of a temporary split fails when the closed union is not connected: the 3×3 closing bridges gaps of up to 2 px, so pieces 3 px or more apart cannot be merged. Both pieces then become new cells without a mother, and the mother ends without daughters. | Broken lineages after splits; when the pieces later fuse, the merge is a B event. Look-ahead makes many more merge requests and hits this often. | `merge_fallback` | tested |
| 23 | 2→1 hidden by the assignment: when a small cell fuses with a large one, the fused region is linked back to the large cell only. The small cell gets no forward link, the area change stays within `da_max`, and errorRez continues the large cell as if nothing happened. | Most of the B events left after the split fixes (finding 5). | `orphan_split` | tested |

## Cell marker and output (`link.py`, `track.py`)

| # | Quirk | Likely effect | Fix | Status |
|---|---|---|---|---|
| 18 | Registration for splits follows MATLAB's ITK v3 path (not a quirk, an upgrade path). | ITK v4 cut 60 of 93 split calls against 21 (`supersegger_port.md`). Expected to reduce A and B. | `itk_v4` | tested |
| 19 | CellMarker: a failed backward lookup leaves `jj` stale for the next frame. | Wrong `Cell`/`cell` naming on some lineages. | `marker_chain` | implemented; output only |
| 20 | `death` comes from the cell's last frame, where it can be stale (last frame + 1). | Wrong `death` in the HDF5 file. Downstream reads `death`. | `death_last_frame` | implemented; output only |
| 21 | Labels without pixels (ghosts) become cells with empty masks. | About half of all cell files on 260429 are ghosts, each with its own ID. | `drop_ghosts` | implemented; output only |

## All results

All counts are summed over xy01–xy04. Case outcomes use the strict rule in
`test_report_cases.py`: neither cell absorbs, or is absorbed by, any track
at t. "Clean frames" are the frames of each track before its first area jump
above 1.3×, which is where downstream curation cuts. "Clean ≥ 60" counts the
tracks that stay clean for at least an hour.

| Configuration | A | B | far | Absorbed cells that end | Jumps | Tracks | Clean frames | Clean ≥ 60 | Cases | Peak GB |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 903 | 1755 | 1242 | 3526 | 5.63 % | 9769 | 66,505 | 209 | 0 | — |
| `main` IoU linker | 1077 | 1027 | 3 | 2016 | 5.02 % | 5286 | 51,002 | 188 | 1 | — |
| `label_collision` | 1259 | 1801 | 3 | 2963 | 4.99 % | 8744 | 66,202 | 215 | 1 | — |
| `area_penalty` | 941 | 1765 | 1250 | 3582 | 5.67 % | 9765 | 66,166 | 204 | 0 | — |
| `split_needs_two` | 1071 | 1814 | 958 | 3490 | 5.81 % | 9623 | 65,995 | 206 | 0 | — |
| `itk_v4` | 482 | 1314 | 1671 | 2954 | 5.43 % | 9690 | 66,425 | 211 | 1 | — |
| `seeded_split` | 139 | 883 | 2093 | 2552 | 5.02 % | 9927 | 66,394 | 201 | 1 | 8.9 |
| `division_lookahead = 3` | 407 | 2210 | 1655 | 3834 | 5.83 % | 9503 | 64,715 | 207 | 0 | 10.3 |
| `end_on_merge` | 168 | 853 | 927 | 1631 | 3.90 % | 11,513 | 74,503 | 226 | 7 | 6.4 |
| assignment group (#5–#9) | 903 | 1755 | 1243 | 3533 | 5.62 % | 9778 | 66,558 | 209 | 0 | 6.8 |
| errorRez group (#14, #16, #17) | 902 | 1745 | 1255 | 3518 | 5.63 % | 9744 | 66,483 | 209 | 0 | 7.3 |
| `lc_itk_v4` | 709 | 1330 | 3 | 1950 | 3.87 % | 7080 | 72,785 | 279 | 3 | 1.4 |
| `lc_seeded_split` | 301 | 1024 | 3 | 1185 | 3.00 % | 5893 | 76,878 | 377 | 9 | 1.3 |
| `lc_lookahead3` | 632 | 1791 | 3 | 2329 | 4.37 % | 7287 | 64,575 | 236 | 1 | 2.4 |
| `lc_merge_fallback` | 1295 | 1765 | 3 | 2963 | 4.99 % | 8744 | 66,202 | 215 | 1 | 1.6 |
| `combined` | 192 | 1078 | 2 | 1121 | 3.02 % | 5542 | 73,092 | 356 | 9 | 2.1 |
| `combined_end` | 180 | 1010 | 2 | 1052 | 2.98 % | 5618 | 73,742 | 359 | 9 | 2.1 |
| `lc_seeded_orphan` | 203 | 685 | 4 | 616 | 3.13 % | 5446 | 76,236 | 373 | 9 | 1.7 |
| `combined_orphan` | 157 | 683 | 4 | 561 | 3.15 % | 5091 | 71,501 | 346 | 9 | 2.6 |
| `combined_orphan_plus` | 146 | 697 | 2 | 580 | 3.16 % | 5123 | 71,748 | 346 | 8 | 1.2 |
| `combined_orphan_rigid` | 155 | 697 | 4 | 581 | 3.13 % | 5171 | 72,416 | 342 | 10 | 2.5 |

Notes:
- **Peak memory** could only be measured from Python, so it is missing for
  the first runs. Configurations with the label-collision quirk need 6–10 GB
  per position, against 1–2 GB without it.
- **Wall-clock times** are not comparable: the runs overlapped, and the
  machine slept during some of them. Use the final run's timing.
- **Superseded runs.** The first `seeded_split` and look-ahead runs used
  earlier versions: two seeds only, and look-ahead without the division
  fallback. They were re-run, and only the re-runs are listed.
- **Why the counts can rise.** Every extra mask edit adds label collisions
  unless `label_collision` is on. That is why `itk_v4`, `seeded_split` and
  `division_lookahead` alone raise "far".
