# Lineage linking: `nd2_to_cells track` vs OmniSegger

This compares how `nd2_to_cells/track.py` turns Omnipose masks into cell
lineages with what OmniSegger does on the same input. The OmniSegger side is
described in [`supersegger_linking_reference.md`](supersegger_linking_reference.md)
(the "reference doc" below).

| | Version |
|---|---|
| `nd2_to_cells` | `nd2_to_cells/track.py` at `1a7eb27` (`main`) |
| OmniSegger | `SuperSegger-master/` at `254d081` |
| Baseline presets | `presets/100XEc.toml` vs `settings/100XEc.mat` |

> **Changed since `1a7eb27` (2026-10-09).** Four preset keys were renamed:
> `overlap_limit_min` → `min_link_iou`, `small_area_merge` →
> `fragment_merge_area`, `remove_stray` → `drop_single_frame_strays`,
> `min_cell_age` → `min_cycle_frames`. Fragment merging, stray removal and
> the default cycle length now follow OmniSegger more closely (D4, D13, D14).
> The rest of this document, including the §6 runs, describes `1a7eb27`.
> Notes marked *Changed since `1a7eb27`* say what is different now.

**Scope.** From reading the masks up to deciding which cells count as a
complete cycle (`Cell*` files). That covers pre-linking mask clean-up,
frame-to-frame assignment, divisions, error handling, ID bookkeeping and
complete-cycle marking. Fluorescence, cell geometry, `clist` and file contents
are left out, except where they affect lineages.

**Method.** §1–§5 come from reading code. Every OmniSegger claim used here
was checked against the MATLAB source. Errors found in the reference doc have
been fixed there and are listed in §5. Behaviour that depends on runtime
details I couldn't check is marked *inferred*. §6 reports runs of both tools
on the 260429 test data. Line numbers are `track.py:N` for this repo and
`file.m:N` (relative to `SuperSegger-master/`) for OmniSegger.

---

## Summary

1. **`track.py` is a different algorithm, not a port.** It is an IoU linker
   (derived from ObtrackerPy): one forward pass, previous-frame cells handled
   in label order, two frames in memory. OmniSegger runs three cost-based
   assignment passes per frame, decides identity from the *backward* map,
   edits masks (merges and splits) and re-runs the frame. The two share
   parameter names, but most of the shared parameters mean something
   different or are dead in MATLAB (§1.2).
2. **Area limits gate links in `track.py`; in OmniSegger they only flag
   errors.** OmniSegger never breaks a lineage because of an area change. It
   continues the cell and sets `ehist`, which keeps that cycle out of the
   `Cell` set. `track.py` ends the lineage when IoU < `confident_iou` and the
   area change is out of range. It accepts IoU ≥ `confident_iou` silently,
   and it records no error history at all (D11, D13).
3. **Under-segmentation is handled very differently.** When two cells
   collapse into one mask, OmniSegger tries to split the mask again
   (`missingSeg2to1`). `track.py` has no split. One lineage ends, and the
   other carries the two-cell blob. When the blob separates again, the
   re-join rule keeps it together until the track is `min_division_age`
   frames old, and only then records a division. This gives false cycles of
   about `min_division_age` frames. On 260429, 40 % of `track.py`'s observed
   cycles are 8–9 frames long, and most of those show this pattern (D10,
   §6).
4. **The complete-cycle rule differs.** OmniSegger requires ≥ 6 observed
   frames and no flagged error since birth (≥ 4 frames when run from the GUI).
   `track.py` checks `len(frames) ≥ min_cell_age` (5), but that test can never
   fail: a track can't divide before `min_division_age` (8) frames, so every
   `Cell` already has ≥ 8. There is no error check (D13).
5. **`track.py` changes the masks before linking, which OmniSegger doesn't.**
   It drops small regions (`min_area`, `min_area_no_neigh`), but those filters
   are commented out on the OmniSegger path. It also merges small neighbours
   in every frame (`small_area_merge`), whereas OmniSegger merges only two
   regions that map back to the same cell. `remove_stray = true` in
   `100XEc.toml`, but it is effectively off in OmniSegger (D3, D4, D14).
6. **Candidate search.** OmniSegger considers only regions within 2 px of a
   cell's previous footprint, with no fallback. `track.py` requires
   IoU ≥ 0.08, or a centroid within 15 px when nothing overlaps. So it can
   link, and even divide, cells with no overlap, and it drops weak overlaps
   that OmniSegger would consider (D5).
7. **How `ignoreerror` is set changes the reference result** (§1.1).
   `processExp` sets it to 0, which removes the iteration cap. The GUI leaves
   it unset (cap applies), and 1 turns off merges and splits. For
   over-segmentation, a run with `ignoreerror = 1` is furthest from `track.py`.
   For under-segmentation, it is the closest. On 260429, `processExp`'s
   setting never finishes (§6.2).
8. **The checks on 260429 confirm D10–D13** (§6). They also show that
   OmniSegger's own output is noisy: half the cell IDs in the capped run are
   zero-area ghosts, and it still records hundreds of cycles shorter than 8
   frames. Agreement with OmniSegger is therefore not a target in itself.

---

## 1. Baseline settings

### 1.1 Which CONST, and what `ignoreerror` does

The OmniSegger README documents two entry points, and they build CONST
differently:

| Entry point | `CONST.ignoreerror` | `MIN_CELL_AGE` | `REMOVE_STRAY` |
|---|---|---|---|
| `processExp('dir')` | **0** (`processExp.m:114`) | 5 (`loadConstants`) | 0 |
| `superSeggerGui` | absent | **3** (GUI box, `superSeggerGui.m:199`; default in the `.fig`) | GUI checkbox, default off |
| `BatchSuperSeggerOpti` with a `loadConstants` CONST | absent | 5 | 0 |

`ignoreerror` is read only at `trackOptiLinkCellMulti.m:161-162`. There it
overwrites `finalIteration`, which errorRez uses as `ignoreError`:

- **Field = 0 (`processExp`).** `ignoreError` is false on every pass, so the
  3-pass cap never applies. Merges (C, D, F) and splits (E) are allowed on
  every pass, and a frame can repeat without limit. One way that happens: a
  split that "succeeds" without splitting (§5, `missingSeg2to1`) makes the
  frame repeat each time.
- **Field absent (GUI).** Passes 1–2 behave as above. On pass 3, the C/D
  merges and the E split are switched off: E falls back to
  `mapToBestOfTwo`, and D divides. F's subset merge and the `REMOVE_STRAY`
  deletions are not gated by `ignoreError`, so a frame can still reset after
  pass 3.
- **Field = 1.** `ignoreError` is true on every pass. D always records a
  division (no area test). C records a division if the `ok` area test passes,
  else `mapBestOfTwo`. E always uses `mapToBestOfTwo` (one lineage kept and
  flagged, the other ends). Masks are edited only by F's subset merge (and by
  `REMOVE_STRAY` deletions, if that is on).

What this means for comparing with `track.py`:

- **Over-segmentation flicker** (one cell → two pieces for a frame).
  OmniSegger with field = 1 records a division at every flicker, with no age
  or size-ratio gate, so it should show many more short cycles than
  `track.py`. With field = 0 or absent, it merges the pieces when one of the
  merge triggers holds (§3, D9) and otherwise divides.
- **Under-segmentation** (two cells → one mask). With field = 1, OmniSegger
  behaves like `track.py`: one lineage survives and the other ends. The
  differences are that OmniSegger flags it and chooses by cost, while
  `track.py` chooses by label order and the IoU gate. With field = 0 or
  absent, OmniSegger splits the mask and both lineages survive.
- **Recommendation.** Use the capped configuration (field unset) as the
  reference. `processExp`'s field = 0 would be the natural choice, since it
  is the documented command-line path, but on 260429 it never finishes
  (§6.2). Run field = 1 as a second reference to separate assignment
  differences from mask-editing differences. Write down `MIN_CELL_AGE` for
  any GUI run.

### 1.2 Parameters

`loadConstants.m` copies only some fields from the `.mat` preset (reference
doc §8; confirmed). Comparison with `presets/100XEc.toml`
(`100XPa.toml` in brackets where it differs):

| TOML key | TOML value | OmniSegger effective value | Same meaning? |
|---|---|---|---|
| `overlap_limit_min` | 0.08 | not read anywhere | **No.** Only `track.py` has an overlap threshold (an IoU threshold, `track.py:397`). The TOML comment "Equivalent to SuperSegger OVERLAP_LIMIT_MIN" is wrong. |
| `da_max`, `da_min` | 0.3, −0.2 | 0.3, −0.2 (defaults; the `.mat` values are equal but not copied) | **Partly.** OmniSegger: a 1:1 error flag over area ratio [0.80, 1.43] (`/max` normaliser), plus the division `ok` test. `track.py`: a link gate over ratio [0.83, 1.43] (normalised by the new area), plus the division area test, which uses the same normaliser as `ok` (D11). |
| `min_area` | 8 | 8, but unused (`trackOptiStripSmall` is commented out, `trackOpti.m:75-85`) | **No.** Only `track.py` filters (D3). |
| `min_area_no_neigh` | 30 | 30, unused | **No** (D3). |
| `small_area_merge` | 50 [30] | 50 [30] | **No.** Same value, different rule (D4). |
| `remove_stray` | true [false] | 0 for both (the Ec `.mat` has 1, but it isn't copied) | **No.** The value differs for Ec, and so does the mechanism (D14). |
| `min_cell_age` | 5 | 5 (3 from the GUI) | **No.** OmniSegger means ≥ 6 frames; `track.py` means ≥ 5, and in practice ≥ 8 (D13). |
| `search_radius`, `min_division_age`, `min_sister_ratio`, `confident_iou` | 15, 8, 0.5, 0.3 | — | `track.py` only (documented as such) |

*Changed since `1a7eb27`:* the four mismatched keys are renamed (old names still load,
with a warning), and the preset comments no longer claim SuperSegger
equivalence. `min_cycle_frames` is 6. The code default for
`drop_single_frame_strays` is false; `100XEc.toml` keeps it true.

---

## 2. Workflow side by side

| Stage | OmniSegger | `track.py` |
|---|---|---|
| Frame order, time axis | Position in the sorted `*seg.mat` listing. A missing mask blocks the run. | `t` parsed from the filename. Missing frames are bridged with a warning. |
| Mask clean-up | None. Omnipose labels are used as they are. | Size filters, then a per-frame small-region merge |
| Passes per frame | 3: r→c (recomputed), c→r, c→f (look-ahead) | 1: previous → current |
| Candidates | Labels within a 2 px dilation of the region | IoU ≥ 0.08, else centroid ≤ 15 px |
| Score | Cost: shape-aligned overlap, distance, area change, penalties | IoU (distance in the fallback) |
| Assignment | Greedy global minimum over all columns, then 4 repair passes | Each previous cell, in label order, takes its best unclaimed candidate |
| Identity | Backward map (c→r) first, then errorRez branches A–G | Forward choice of the previous cell |
| Division | 1→2 seen in both directions (no area test), or 2→1 seen only backward with `ok` area | Top two candidates: area fits, mother ≥ 8 frames old, sister ratio ≥ 0.5 |
| Over-segmentation | Merge the pieces if a merge trigger holds (closing; one component required) | Re-join the top two candidates when the area fits but age or ratio fails |
| Under-segmentation | Split the mask (registration + watershed), else keep one lineage and flag it | Not handled: one lineage ends |
| Area change outside limits | Link kept, error flagged | Link rejected unless IoU ≥ 0.3 |
| Strays | Kept (`REMOVE_STRAY` effectively 0) | Removed after linking (`remove_stray = true`) |
| Error history | `ehist` per cycle, reset at division | None |
| Complete cycle | Born from division, divides, `ehist = 0`, ≥ 6 frames | Born from division, divides, ≥ 5 frames (≥ 8 in practice) |
| Edited masks | Saved to err files and `masksOS/` | Merges replayed into the HDF5 crops; `masks/` untouched |

---

## 3. Divergences

Each divergence gets one of three classes:

- **Deliberate.** The repo records a choice: the module docstring,
  `AGENTS.md`, a preset or README comment, a commit message, or a decision
  you made earlier.
- **MATLAB quirk.** A bug or oddity in OmniSegger that `track.py` doesn't
  reproduce. Listed so that output differences at these points aren't blamed
  on `track.py`.
- **Unintended / unclear.** No recorded decision.

### D1. Overall structure — *Deliberate*

- **OmniSegger.** Keeps all per-frame state in `seg`/`err` files on disk.
  Runs three assignment passes per frame, rewrites the masks when it merges or
  splits, and re-runs the frame (`trackOptiLinkCellMulti.m`, reference doc §4).
- **`track.py`.** One forward pass over consecutive mask files. It holds two
  labelled frames and stores scalars only (`track.py:308-523`). Design notes
  are in the module docstring and `AGENTS.md`.
- **Effect.** This is the root of D7–D10. With two frames in memory there is
  no look-ahead to t+1, so the next frame can't inform a split or merge
  decision.

### D2. Time axis and missing frames — *Deliberate*

- **OmniSegger.** `time` is the file's position in `dir('*seg.mat')`
  (`trackOptiLinkCellMulti.m:58`), and `birth`/`death` use that index. A
  phase image without a mask fails the count check and then `imread`, which
  aborts the position. A timepoint missing from both phase and masks is
  bridged silently, and later frames are renumbered.
- **`track.py`.** Uses frame = `t − 1` from the filename (`track.py:97`), and
  writes `birth`/`death` as the real `t`. Missing mask files are bridged with
  a warning (`track.py:754-762`). Ages are counted in observed frames
  (`len(frames)`), not elapsed time.
- **Effect.** No difference when `t` starts at 1 with no gaps. Otherwise
  `birth`/`death` differ, and a track that spans a gap is "younger" in
  `track.py` than its elapsed time. That matters for `min_division_age` and
  `min_cell_age`.

### D3. Size filters before linking — *Unintended / unclear*

- **OmniSegger.** `trackOptiStripSmall` is commented out
  (`trackOpti.m:75-85`), so every Omnipose label gets linked, including
  slivers. (OmniSegger's own Omnipose command passes `--exclude_on_edges`, so
  edge cells are removed upstream.)
- **`track.py`.** Drops regions < `min_area` (8 px, `track.py:123`) and
  isolated regions < `min_area_no_neigh` (30 px). "Isolated" means no other
  region's bbox is within 1 px (`track.py:182-208`).
- **Effect.** `track.py` has fewer short tracks made from debris. Real cells
  are much larger than 30 px (≈ 0.1 µm² at 60 nm/px), so lineages are hardly
  affected. The presets present these filters as SuperSegger equivalents,
  which they aren't.

*Changed since `1a7eb27`:* the preset comments no longer call these filters
SuperSegger equivalents.

### D4. Small-region merge — *Unintended / unclear*

- **OmniSegger.** `SMALL_AREA_MERGE` is only used in branches C and D of
  errorRez. When two current regions both map back to the same previous
  cell, they are merged if the `ok` area test passes and at least one of
  these holds: one has no forward map, their forward targets overlap, or
  **either** one is < `SMALL_AREA_MERGE`. The merge is a closing, and it
  happens only if the result is one connected component
  (`merge2Regions.m:39-43`).
- **`track.py`.** In every frame, before linking, merges pairs of regions
  that are **both** < `small_area_merge` and whose bboxes are within 1 px
  (`track.py:211-251`). It uses no information from other frames, does no
  connectivity check, and merges greedily in label order.
- **Effect.** `track.py` can join two unrelated fragments, which OmniSegger
  never does. OmniSegger can merge a small piece into a large sister; this
  pre-merge can't, but the re-join (D9) covers that case. At 50 px the rule
  rarely touches real cells.

*Changed since `1a7eb27`:* the per-frame merge is gone. `fragment_merge_area` now
follows OmniSegger's small-piece trigger: when a cell splits into two pieces
and one is smaller than `fragment_merge_area`, the pieces are merged back if
their combined area fits and they form one region after a 3×3 closing.
OmniSegger's two next-frame triggers are not implemented. On 260429 neither
the old rule nor the new one fires.

### D5. Candidate search — *Deliberate* (`search_radius`) / *Unintended* (`overlap_limit_min`)

- **OmniSegger.** Candidates are the labels under a 5×5 dilation (2 px) of
  the region, searched inside its bbox plus 20 px. There is no threshold and
  no distance fallback (`multiAssignmentSparse.m:47, :542-557`). A cell more
  than about 2 px from its own previous footprint starts a new lineage.
- **`track.py`.** Candidates need IoU ≥ `overlap_limit_min` (0.08). If none
  qualify, regions with a centroid within `search_radius` (15 px) are used
  instead (`track.py:382-421`). The fallback applies only when there are no
  IoU candidates at all.
- **Effect.**
  - `track.py` links cells that jumped up to 15 px. OmniSegger would start new
    lineages for them.
  - Two non-overlapping regions within 15 px can be called a division
    (`track.py:432-473` also runs on fallback candidates).
  - Conversely, `track.py` ignores weak overlaps (IoU < 0.08) that OmniSegger
    would score. An example is a small daughter next to a larger sister that
    took most of the mother's footprint.

*Changed since `1a7eb27`:* `overlap_limit_min` is now `min_link_iou`.

### D6. Similarity score — *Deliberate*

- **OmniSegger.** The cost is dominated by `100/ovT`, the overlap after
  shifting the candidate onto the cell's centroid. It adds 5 × the centroid
  distance, 100 × |dA| and direction and size penalties. Raw overlap counts
  (`wcol`) mostly for isolated cells (reference doc §5.2; confirmed).
- **`track.py`.** Uses IoU, `track.py:152-174`.
- **Effect.** IoU falls with displacement *and* with size change: a cell that
  doubled while staying in place has IoU 0.5. `ovT` largely ignores
  displacement within the candidate window. `track.py` is therefore more
  sensitive to leftover drift between aligned frames, and to growth over long
  frame intervals.

### D7. Assignment order — *Unintended / unclear*

- **OmniSegger.** Picks the cheapest remaining candidate over all c regions,
  single and paired, then runs two `fixProblems` and two
  `exchangeAssignment` passes. Both directions are computed and
  cross-checked in errorRez (`multiAssignmentSparse.m:321-504`).
- **`track.py`.** Visits previous regions in label order (`track.py:374`).
  Each takes its best candidate that no earlier region has claimed, and
  candidates already claimed are invisible (`track.py:383`).
- **Effect.** In dense clusters, a low-label cell can take a region that fits
  a higher-label cell better, and the higher-label cell's lineage ends. A
  mother can also lose one daughter to a neighbour processed earlier. It then
  continues into the other daughter, and the division is missed (D12).
  Results depend on Omnipose's label numbering. OmniSegger's errorRez is also
  label-ordered, but its assignment is global.

### D8. Division rule — *Deliberate* (fix A, `23a99f1`)

- **OmniSegger.**
  - **Branch D:** the forward pass maps R to two regions, and both map back
    only to R. This is a division **without any area test**, unless the
    pieces are merged (`errorRez.m:166-201`).
  - **Branch C:** two regions map back to R while the forward pass says 1→1.
    This is a division if the `ok` area test passes
    (`errorRez.m:118-156`).
  - There is no age or size-ratio gate.
  - A 1→2 pair is only scored when R's best 1:1 candidate isn't a "good"
    match (|dA| < 0.15, ov > 0.7, ovT > 0.8).
- **`track.py`.** Takes the top two candidates. It requires all of:
  - combined area change in [`da_min`, `da_max`], normalised by the summed
    daughter area (the same normaliser as `ok`);
  - mother ≥ `min_division_age` frames old (tracks present in the first
    frame are exempt);
  - smaller / larger daughter ≥ `min_sister_ratio`
    (`track.py:432-473`).

  This test runs whenever there are ≥ 2 candidates, even when one of them is
  an excellent 1:1 match.
- **Effect.** Many fewer divisions (4552 → 1451 on 260429). No track born
  during the movie can divide within its first 8 frames, and divisions
  without a fitting area change are never recorded. OmniSegger accepts both.

### D9. Over-segmentation: re-join vs merge — *Deliberate* (fix A)

- **OmniSegger.** Merges two pieces only when both map back to the same cell
  and a merge trigger holds (D4). The result must be one connected component
  after a 3×3 closing. If it isn't, both pieces become orphan new cells and
  the mother's lineage ends (`merge2Regions.m`, `errorRez.m:330`).
- **`track.py`.** When the area fits but the age or ratio gate fails, it
  relabels candidate 2 as candidate 1 and continues the mother
  (`track.py:474-484`). There is no adjacency, connectivity or look-ahead
  check.
- **Effect.** Fewer false divisions and no orphaned lineages. Two side
  effects:
  - The output mask can hold two disconnected parts under one label.
  - *Inferred:* the mother can swallow a small neighbour. The conditions
    are: the neighbour's region is still unclaimed (its own predecessor has
    a higher label than the mother, or didn't take it), it is the
    second-best candidate with IoU ≥ 0.08, and it is ≤ 43 % of the mother's
    area. Then the area fits and the ratio fails, so it is re-joined. The
    neighbour's own lineage ends.

### D10. Under-segmentation: no split — *Unintended / unclear*

- **OmniSegger.** Two previous cells map to one current region (branch E).
  With mask editing allowed, `missingSeg2to1` registers the two previous
  masks onto the region, splits it by watershed, and re-runs the frame, so
  both lineages continue. If the split isn't allowed or fails,
  `mapToBestOfTwo` continues the lower-cost cell with `error.r = 2`, which
  sets `ehist`. The other lineage ends.
- **`track.py`.** The first previous cell (label order) whose link passes
  `_accept_link` takes the merged region. For two cells of similar size,
  each has IoU ≈ 0.5 with the blob, so this passes via `confident_iou`.
  The other lineage ends, and no flag is set. When the blob separates again (`track.py:432-490`):
  - **Track at least `min_division_age` frames long:** a division is
    recorded at once. The two original cells come out as "daughters".
  - **Younger track, or unequal pieces:** the pieces are re-joined, and this
    repeats every frame until the track reaches `min_division_age`. The
    division is recorded then.
- **Effect.**
  - One lineage is lost per merge event.
  - Each merge event produces a false division.
  - The HDF5 masks show two cells as one in the frames in between.
  - If two sisters re-merge soon after a real division, or cells in a cluster
    flicker repeatedly, the false cycles come out ≈ `min_division_age` frames
    long. The cause is segmentation, as the earlier review concluded, but the
    cycle length is set by this linker rule, not by biology. On 260429 this
    accounts for more than half of the 8–9-frame cycles (§6.4).

### D11. Area change on a 1:1 link — *Unintended / unclear* (partly *Deliberate*: `confident_iou`, fix C)

- **OmniSegger.** Never rejects a link because of area. The error flag uses
  `dA = (A_t − A_{t−1}) / max(A_t, A_{t−1})`, which accepts the area ratio
  A_t/A_{t−1} ∈ [0.80, 1.43] in both directions
  (`multiAssignmentSparse.m:434-452`). Outside that range it sets
  `error = 2/3`, which feeds `ehist` when branch B continues the cell. The
  cost also penalises area change, but every candidate stays assignable.
- **`track.py`.** Accepts a link with IoU ≥ `confident_iou` regardless of
  area. Otherwise it requires `(A_t − A_{t−1}) / A_t` ∈ [−0.2, 0.3]
  (`track.py:287-290, 366-372`), which is ratio [0.83, 1.43]. A failed link
  ends the track, and the region starts a new track with no mother.
- **Effect.** At low IoU, `track.py` breaks lineages that OmniSegger keeps
  and flags. At high IoU, `track.py` keeps them without a flag. Both
  directions change who counts as a complete cycle (D13). The shrink limit is
  slightly stricter in `track.py` (0.83 vs 0.80).

### D12. Missed divisions — *Unintended / unclear*

- **OmniSegger.** If only one daughter maps back to R, branch B continues R
  into it. In the c→r pass the 2→1 area flag is computed per single region,
  so that daughter gets `error = 3` (dA ≈ +0.5 against the reverse limit
  0.2). Branch B passes that into `ehist`, so the mother's cycle can't become
  `Cell` (`errorRez.m:112-115`, `multiAssignmentSparse.m:450-452`).
- **`track.py`.** If the second daughter has IoU < 0.08 or was claimed
  earlier (D7), there is only one candidate. IoU ≈ 0.5 ≥ `confident_iou`, so
  the mother continues into the first daughter without a flag. The second
  daughter becomes a new track with no mother.
- **Effect.** In `track.py`, a mother track can span two generations and
  still be named `Cell`, with a cycle about twice as long. OmniSegger keeps
  such cycles out of `Cell`.

### D13. Complete-cycle rule — *Unintended / unclear*

- **OmniSegger.** `trackOptiCellMarker.m:99-102` requires all of:
  - `divide` (set on the mother's last frame);
  - `stat0 ≥ 1` (born from a division);
  - `ehist == 0`;
  - `last − birth ≥ MIN_CELL_AGE`, i.e. ≥ 6 observed frames (≥ 4 with the
    GUI default of 3).

  It then marks `stat0 = 2` back to birth, and the file is `Cell` when the
  last frame has `stat0 == 2`.
- **`track.py`.** Requires `mother_id ≠ 0`, `divide` and
  `len(frames) ≥ min_cell_age` (`track.py:652-658`).
  - The length test can never fail. A track born during the movie can't
    divide until it is `min_division_age` (8) frames long
    (`track.py:443-447`), so every `Cell` has ≥ 8 frames. `min_cell_age` has
    no effect while `min_division_age ≥ min_cell_age`.
  - To mean the same thing as OmniSegger's 5, `min_cell_age` would have to
    be 6.
  - There is no error condition.
- **Effect.** Compared with OmniSegger, `track.py`'s `Cell` set leaves out
  6–7-frame cycles. It includes cycles OmniSegger would flag: confident links
  with large area jumps, missed divisions (D12), and the blob cycles from D10.

*Changed since `1a7eb27`:* the key is now `min_cycle_frames`, with a default of 6
(OmniSegger's 5). It still has no effect while `min_division_age` is larger,
and there is still no error condition.

### D14. Stray removal — *Unintended / unclear*

- **OmniSegger.** `REMOVE_STRAY` is effectively 0: it isn't copied from the
  `.mat`, and the GUI checkbox defaults to off. When it is on, errorRez
  deletes the region from the mask (t > 1, no forward map; on the last frame
  every stray) and re-runs the frame (`errorRez.m:90-111`).
- **`track.py`.** `remove_stray = true` in `100XEc.toml` (false in
  `100XPa.toml`). After linking, it drops tracks that are one frame long,
  have no predecessor and don't divide (`track.py:514-521`). That includes
  first-frame cells that vanish after frame 1, regions that appear in the
  last frame, and one-frame tracks created by a rejected link (D11). The
  regions stay in the masks but get no cell file.
- **Effect.** The 100XEc output has fewer one-frame `cell` files than
  OmniSegger. Lineages aren't affected, because these tracks have no
  relatives. The preset comment copies the nominal `.mat` value, not the
  value MATLAB actually uses.

*Changed since `1a7eb27`:* the key is now `drop_single_frame_strays`, and cells in
the first frame are kept, as in OmniSegger. The code default is false;
`100XEc.toml` keeps it true, which still differs from OmniSegger's effective
setting. On 260429 this keeps 11 more one-frame tracks.

### D15. ID numbering — *Unintended / unclear* (cosmetic)

- **OmniSegger.** IDs are handed out while visiting current regions in label
  order. Daughters get n+1 and n+2 in label order, and IDs are rolled back
  when a frame re-runs.
- **`track.py`.** First-frame tracks are numbered in label order. In later
  frames, daughters are numbered while visiting previous cells, with the
  higher-IoU daughter first, and new tracks after them. Stray removal leaves
  gaps in the numbering.
- **Effect.** IDs can't be compared between the two outputs. Compare by
  lineage structure and position instead.

### MATLAB quirks not reproduced

These are behaviours of OmniSegger that `track.py` doesn't have. Where outputs
differ at one of these points, the difference comes from OmniSegger.

| Quirk | Where | Effect on OmniSegger output |
|---|---|---|
| Branch B continues a lineage even when the forward pass saw a division | `errorRez.m:112` | Missed divisions, flagged via `ehist` (D12) |
| D divides without an area test; C needs `ok` | `errorRez.m:156, :199-201` | Division counts are asymmetric |
| Area penalty: the `>0.3 → 50` line overwrites `>0.6 → 1000` | `multiAssignmentSparse.m:302-310` | Large growth (forward) or shrinkage (reverse) is under-penalised. The total cost for shrinkage still rises (via `ovT`). |
| `fixProblems`' second call accepts the ratio range [0.70, 1.25] instead of [0.80, 1.43] | `multiAssignmentSparse.m:357-358` | Repair steals and splits use different area limits |
| `exchangeAssignment` can swap to an unrelated region (loop variable or `min` over all-NaN) | `multiAssignmentSparse.m:481-493` | Rare wrong links in repair |
| A failed merge orphans both pieces | `merge2Regions.m:43`, `errorRez.m:330` | The mother's lineage ends with no division |
| `missingSeg2to1` accepts `max(label) == 2`, even if label 1 is missing | `missingSeg2to1.m:92-105` | The region is relabelled but not split, and the frame repeats (without limit when `ignoreerror = 0`) |
| `num_regs` isn't refreshed between edits in one pass *(inferred)* | `merge2Regions.m:45-50`, `missingSeg2to1.m:93-95` | Two edits in one pass can write the same new label, so two blobs share one ID |
| Label gaps after edits give zero-area regions *(inferred)* | `updateRegionFields.m:33-39` | One-frame "ghost" cells |
| `markDivisionEvent` checks only sister 1's ID *(inferred)* | `markDivisionEvent.m:40` | C can overwrite an ID handed out earlier in the pass |
| An aborted pass leaves `death`, `divide` and `daughterID` in `data_r` | `trackOptiLinkCellMulti.m:182-185` | `daughterID` can point at IDs re-issued after the rollback |
| `ehist` resets at each division | `markDivisionEvent.m:50-51` | Daughters of an erroneous division can still be `Cell` |

---

## 4. Checks on a shared dataset

Run OmniSegger with `ignoreerror` unset (3-pass cap), and again with
`ignoreerror = 1`. (`processExp`'s `ignoreerror = 0` doesn't finish on
260429, §6.2.) Run `track.py` with `100XEc.toml`. Then compare the items
below. The results for 260429 are in §6.

1. **Cycle-length histograms** of cells born from a division, per tool. In
   `track.py`, check whether cycles of `min_division_age`–`min_division_age + 1`
   frames follow a merge (D10). The sign is that the mother's sister ended
   1–2 frames after its own birth, at the frame where the mother's area
   roughly doubled.
2. **`Cell` sets.** Match cells by position and birth frame. Expect
   `track.py` to lack the 6–7-frame cycles, and to include cycles that
   OmniSegger marks with `ehist = 1`.
3. **Lineage breaks from rejected links** (D11). Count `track.py` tracks with
   no mother that start in the same frame, and overlap the region, where
   another track ended.
4. **Missed divisions** (D12). Find `track.py` mothers whose area drops by
   about half without a division. In OmniSegger, find branch B continuations
   whose `error.label` mentions a division.
5. **OmniSegger artefacts.** One-frame cells with zero area (ghosts), frames
   that took many passes (the `output_log.txt` from `processExp`), and labels
   shared by two blobs. Exclude these before counting differences.

---

## 5. Corrections made to the reference doc

Found by checking the reference doc against the source at `254d081`. All of
them have been applied to `supersegger_linking_reference.md`; this list
records what changed. Section numbers refer to that file.

- **§1.** The disabled `trackOptiStripSmall` block is at `trackOpti.m:75-85`,
  not 69-78.
- **§2.** The alignment-skip logic is at `BatchSuperSeggerOpti.m:118-133`.
  The mask-count check is at `:359-375`, with a second copy at `:328-344`.
  It counts directory entries minus 2, so hidden files such as `.DS_Store`
  cause a false mismatch. It blocks on `input()` only without `autoomni`, and
  it doesn't re-check after Enter. `cp_masks/` passes the check, but
  `ssoSegFunPerReg` reads only `masks/`.
- **§3.** Only `data_c` and `data_f` go through `updateRegionFields`.
  `data_r` is loaded from the err file as it is.
- **§4.**
  - The c→f map isn't only a merge look-ahead. It also feeds
    `missingSeg2to1` (the 2:1:2 case, `errorRez.m:257`) and the
    `REMOVE_STRAY` test.
  - The 3-pass cap doesn't switch off every edit: F's subset merge
    (`errorRez.m:298-312`) and the `REMOVE_STRAY` deletions ignore
    `ignoreError`.
- **§5.2.**
  - The penalty-overwrite bug inverts the penalty P, but not the total cost
    for shrinkage, because `ovT ≤ A_F/A_C` keeps `100/ovT` large. The doc's
    example ("a 25 % shrink costs more than a 70 % shrink") holds for P
    alone. The bug does bite for forward growth: +100 % gets 50 instead of
    1000.
  - `wcol ≈ 1` also holds for a pair column that makes up a whole two-cell
    colony of near-equal areas.
  - `out` is a unit-vector projection scaled by the distance to the colony
    centroid. It can reach ±10 at 100 px from the colony centre, which is
    more than a tie-breaker in large colonies.
- **§5.3.**
  - `fixProblems`: the second call accepts the ratio range [0.70, 1.25]. The
    "assign" branch updates only one of `map` and `revmap`.
  - `exchangeAssignment` has a second bug: with no other column, it falls
    back to column 1.
  - The 2→1 error flag uses the single region's area, so in the c→r pass
    both daughters get `error = 3`. `createDivision` clears this, but branch
    B and `mapBestOfTwo` read it.
- **§6.**
  - C also divides when `manual_link` is set.
  - D has no overall next-frame test; the `oneIsSmall` trigger works on the
    last frame too.
  - D′: if the sister's backward map is empty, `k` falls to "Error not fixed"
    and becomes a new cell with no mother.
  - E: "allowed" checks only `ignoreError`, and ties in `mapToBestOfTwo` go
    to R(2).
  - F, three corrections:
    - under `ignoreError`, merge-all becomes `mapBestOfTwo`;
    - the subset merge checks neither `ignoreError` nor `manual_link`;
    - F never checks `k ∈ fwd`.
  - "Merging allowed when `~ignoreError` and no manual link" is true only
    for C and D.
  - `mapBestOfTwo` deletes the removed sister only when a next frame also
    exists.
- **§6, `missingSeg2to1`.**
  - The seeds come from the interior distance
    `−(bwdist(~r1) + bwdist(~r2))`.
  - `watershedpaw` gives each ridge pixel to the labelled neighbour with the
    lowest *image value*, not the lowest label.
  - Acceptance is `max(label) == 2` (`missingSeg2to1.m:92`), not "exactly
    {1, 2}" (see §3, quirks).
- **§7.1.** Only frames 2…N−1 are re-saved.
- **§7.6.** One-frame cells get two copies of the same frame in `CellA`
  (`intInitCell` then `intUpCell`, `trackOptiCellFiles.m:100-108, :166,
  :192`), so "`CellA{k}` = frame `birth + k − 1`" doesn't hold for them. Old
  cell files are deleted in `trackOpti.m:226-227`.
- **§8.** `MIN_CELL_AGE` is overridden by the GUI (`superSeggerGui.m:199`,
  default 3 in the `.fig`).
- **§9.6.** This is understated. If the re-run never relinks R, `divide = 1`
  stays as well, and `daughterID` points at IDs that are re-issued after the
  `cell_count` rollback.
- **§10.** Out of date. It has been replaced by this document.

---

## 6. Results on 260429

The §4 checks were run on 2026-10-08/09, on a copy of
`amp_testing/01_SW5-20-1_2xMIC/260429`: 361 frames at 1 min/frame, AMP added
at frame 22. Results cover **xy01, xy02 and xy04**. xy03 was dropped because
the capped OmniSegger run had slowed to about 50 minutes per frame and was
stopped at frame 346 of 361.

The run scripts, exports and logs were temporary and haven't been kept.

### 6.1 Setup

- **Input.** Phase images and Omnipose masks only; fluorescence was left out.
- **`track.py`.** At `1a7eb27` with `100XEc.toml`, run through
  `link_frames_streaming`, which is the linking step of `nd2_to_cells track`.
- **OmniSegger.** At `254d081`, in MATLAB R2026b, through
  `BatchSuperSeggerOpti` stages 3–5 (mask import, linking,
  `trackOptiCellMarker`). CONST is built as in `processExp`, but with the
  `100XEc` preset, no alignment and no foci.
- **Two OmniSegger configurations.**
  - **Capped:** `ignoreerror` unset, so the 3-pass cap applies. This is the
    GUI / `loadConstants` behaviour, except that `MIN_CELL_AGE` is 5.
  - **`ignoreerror = 1`:** no merges or splits except F's subset merge.

  Both configurations link the same imported masks.
- **One change to OmniSegger, for speed.** `find_medoid` skeletonised a
  full-frame mask for every cell, about 0.57 s per cell per call, and linking
  calls it on two frames per pass. It was replaced by a version that works on
  the cell's bounding box, with 2 px of padding. On four test
  frames with 21–178 cells, the resulting seg files were identical to the
  original's in every field (`isequaln`). The medoid isn't read by linking.
- **Matching cells between tools.** By birth frame and the Omnipose label
  the cell started from. This is approximate, because OmniSegger relabels
  regions it merges or splits.
- **Runtime.**
  - `track.py`: 15–22 s per position.
  - OmniSegger mask import: 6 min for all positions on 10 workers.
  - OmniSegger linking with `ignoreerror = 1`: 65 min for all positions.
  - OmniSegger linking, capped: 8.5–11.3 h per position (wall clock, part of
    it with the computer asleep).

### 6.2 `processExp`'s setting doesn't finish

With `ignoreerror = 0`, OmniSegger never gets past xy01 frame 6. A verbose
run of frames 1–7 shows why:

1. Region 6 is two previous cells in one mask (branch E), so
   `missingSeg2to1` is called.
2. The watershed puts only label 2 inside the region. The
   `max(label) == 2` test accepts it anyway: the region is relabelled
   `num_regs + 2`, and labels 6 and `num_regs + 1` are left empty.
3. The frame re-runs. The relabelled region is again two cells in one, so
   the same thing happens with the next two labels.

Each pass adds two zero-area labels, and without the cap the loop never ends.
This is the `missingSeg2to1` quirk and the ghost regions from §3, together
with reference doc §9.1.

### 6.3 Summary

| | `track.py` | OmniSegger, capped | OmniSegger, `ignoreerror = 1` |
|---|---|---|---|
| Cell IDs | 3,298 | 12,051 | 8,011 |
| of which zero-area for their whole life | 0 | 5,913 | 131 |
| one-frame IDs | 348 | 7,809 | 3,037 |
| Divisions | 871 | 1,327 | 2,512 |
| Observed cycles (born from a division, divide) | 671 | 882 | 2,282 |
| Median cycle length (frames) | 11 | 7 | 4 |
| `Cell` | 671 | 128 | 226 |
| `Cell` with the GUI's `MIN_CELL_AGE` of 3 | — | 153 | 273 |

Rebuilding OmniSegger's complete-cycle rule from the exported fields
reproduces `trackOptiCellMarker` exactly (128 and 226). This confirms the
reference doc's description of that rule.

### 6.4 Checks

**1. Cycle lengths (D10).** The table shows observed cycles per length. The
second number is the share whose sister was "absorbed": the sister lasted
≤ 2 frames without dividing, and the cell's area grew ≥ 1.5× in the frame
after the sister ended.

| Cycle length (frames) | `track.py` | OmniSegger, capped | OmniSegger, `ignoreerror = 1` |
|---|---|---|---|
| 1–7 | 0 | 484 · 17 % | 1,544 · 39 % |
| 8–9 | **267 · 54 %** | 52 · 8 % | 135 · 21 % |
| 10–14 | 130 · 52 % | 89 · 9 % | 187 · 21 % |
| ≥ 15 | 274 · 22 % | 257 · 3 % | 416 · 13 % |

- 40 % of `track.py`'s cycles fall at `min_division_age` (8–9 frames), and
  more than half of those absorbed their sister. That is the D10 mechanism.
- The share grows during the movie: 8–9-frame cycles make up 16 % of cycles
  born in frames 1–60, and 47 % of those born after frame 240. This fits the
  earlier finding that the remaining short cycles sit in late, clustered
  cells.
- OmniSegger rarely shows the absorbed-sister pattern (8 % in the capped
  run). It records flicker as short cycles instead. Its mask edits cut those
  about threefold compared with `ignoreerror = 1`, but don't remove them.

**2. `Cell` sets.** The sets barely overlap.

| | vs. capped | vs. `ignoreerror = 1` |
|---|---|---|
| `Cell` in both | 41 | 59 |
| `Cell` only in `track.py` | 630 | 612 |
| — no OmniSegger cell born from the same region | 361 | 160 |
| — OmniSegger cell doesn't divide | 138 | 175 |
| — OmniSegger cell has `ehist = 1` | 81 | 243 |
| — OmniSegger cell not born from a division | 35 | 4 |
| — OmniSegger cell has < 6 frames | 15 | 30 |
| `Cell` only in OmniSegger | 87 | 167 |
| — no `track.py` track born from the same region | 79 | 152 |

**3. Lineage breaks from rejected links (D11).** 108 `track.py` tracks start
without a mother in the frame after another track ended at an overlapping
region. In 92 of them the overlap is real (IoU 0.08–0.3) but the area
change is outside `track.py`'s limits, so the link was refused. OmniSegger
continues the lineage in 67 of these (capped) or 66 (`ignoreerror = 1`).

**4. Area drops below 0.65× within a lineage (D12, D13).**
- `track.py` has 1,050 such drops, and 110 `Cell`s contain one. OmniSegger
  records a division at the same place for 114 of the drops (capped) or 182
  (`ignoreerror = 1`).
- OmniSegger has 1,192 (capped) and 861 (`ignoreerror = 1`) drops of its own.
  All are flagged, so none ends up in a `Cell`.

**5. OmniSegger artefacts.**
- **Ghosts.** The capped run has 5,913 cell IDs that are zero-area for their
  whole life, half of all its IDs. They are labels left empty by merges and
  failed splits. The `ignoreerror = 1` run, which barely edits masks, has
  131.
- **Masks in several pieces.** 638 regions (capped) and 133
  (`ignoreerror = 1`) have more than one connected component.
- **Other error labels in the capped run:** 59 "Error not fixed" and 152
  "Converted into a new cell".
- **For comparison:** `track.py` replayed 4,863 region merges (small-region
  merges and re-joins) into its output masks.

### 6.5 What this shows

- **D10 is confirmed, and it is the main source of `track.py`'s short
  cycles.** The 8–9-frame peak comes from the re-join rule acting on
  under-segmentation in late, crowded frames. OmniSegger doesn't produce it.
- **D13 is confirmed.** `min_cell_age` has no effect: every observed
  `track.py` cycle is a `Cell` (671 of 671).
- **D11 and D12 are confirmed.** `track.py` breaks lineages that OmniSegger
  keeps, and continues lineages through halvings that OmniSegger flags. Both
  affect the `Cell` set.
- **OmniSegger is not a ground truth on this data.**
  - Half of the capped run's cell IDs are ghosts.
  - It records 484 cycles shorter than 8 frames even with mask editing.
  - Its two configurations disagree with each other by almost a factor of
    two in `Cell`s (128 vs 226).
  - The default `processExp` setting can't finish at all.

  A low overlap between the `Cell` sets says that both tools struggle with
  this movie's late frames. It doesn't say which one is right.
