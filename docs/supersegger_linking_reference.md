# OmniSegger / SuperSegger linking path — reference

What this covers: the MATLAB code that runs when OmniSegger
(`github.com/bwmr/omnisegger`, `SuperSegger-master/`) gets **already-aligned
images and external Omnipose masks**, links cells through time, and writes the
per-cell `cell*.mat` / `Cell*.mat` files. It's meant as the baseline for
comparing against `nd2_to_cells track`.

All paths below are relative to `omnisegger/SuperSegger-master/`. Line numbers
match commit `254d081`.

> **How this was produced.** I worked this out by reading the code. I did not
> run MATLAB. Wherever a behaviour depends on MATLAB semantics I couldn't
> check (for example `regionprops` on label images with gaps), the text marks
> it as *inferred* and says how to check it on real output.
>
> On 2026-10-08 every claim was checked again against the source at
> `254d081`, and the corrections are folded in below. The list of what changed
> is in [`linking_comparison.md`](linking_comparison.md) §5.

---

## 1. The code path in scope

```
BatchSuperSeggerOpti(dirname, skip=1, clean=0, CONST, startEnd=[2 10])
 ├─ alignment skipped (startEnd(1)>1, or raw_im/ already exists)
 ├─ trackOptiPD                    move root *.tif into xyNN/{phase,fluorN}/
 ├─ intProcessXY  (parfor over xy)
 │   ├─ Omnipose check: xyNN/masks/ must exist and hold as many files as phase/
 │   ├─ doSeg (per frame)           → seg/*_seg.mat
 │   │    └─ CONST.seg.segFun = ssoSegFunPerReg → intMakeRegs (reads the PNG mask)
 │   └─ trackOpti
 │        ├─ trackOptiLinkCellMulti → seg/*_err.mat          (step 4, "linking")
 │        │    ├─ updateRegionFields
 │        │    ├─ transferManualLinks   (no-op unless manual links exist)
 │        │    ├─ multiAssignmentSparse ×3 per frame (r→c, c→r, c→f)
 │        │    └─ errorRez
 │        │         ├─ createNewCell / continueCellLine / markDivisionEvent
 │        │         ├─ merge2Regions        (mask edit → frame re-run)
 │        │         ├─ missingSeg2to1       (mask edit → frame re-run)
 │        │         │    ├─ registerMasks_monointensity
 │        │         │    └─ watershedpaw
 │        │         └─ deleteRegions        (only if REMOVE_STRAY)
 │        ├─ trackOptiCellMarker    marks complete cell cycles (stat0 = 2)
 │        ├─ trackOptiFluor         per-frame background fluorescence
 │        ├─ trackOptiMakeCell      per-region CellA structs (geometry, pole, fluor)
 │        ├─ saveosmasks            xyNN/masksOS/*_os_masks.png
 │        ├─ trackOptiFindFoci      only if sum(CONST.trackLoci.numSpots) > 0
 │        ├─ trackOptiClist         xyNN/clist.mat
 │        └─ trackOptiCellFiles     xyNN/cell/{cell,Cell}NNNNNNN.mat
```

`trackOptiStripSmall` (step 3) is commented out in `frameLink/trackOpti.m:75-85`.
That means `MIN_AREA`, `MIN_AREA_NO_NEIGH` and `OVERLAP_LIMIT_MIN` have **no
effect** on this path (see §9).

`skip > 1` (`trackOptiSkipMerge`) is not covered. These notes assume `skip = 1`.

### Minimal file set (the "extracted" path)

| Role | File |
|---|---|
| Driver | `batch/BatchSuperSeggerOpti.m`, `batch/trackOptiPD.m`, `frameLink/trackOpti.m` |
| Mask import | `segmentation/doSeg.m`, `segmentation/ssoSegFunPerReg.m`, `trainingConstants/intMakeRegs.m`, `segmentation/intImLoader.m`, `viz/find_medoid.m` |
| Region features | `segmentation/cellprops3.m` (`regionScoreFun.props`), `segmentation/scoreNeuralNet.m` (`regionScoreFun.fun`) |
| Linking | `frameLink/trackOptiLinkCellMulti.m`, `frameLink/updateRegionFields.m`, `frameLink/transferManualLinks.m`, `frameLink/multiAssignmentSparse.m`, `Internal/imshift.m`, `Internal/addBB.m`, `Internal/getBB.m`, `Internal/getBBpad.m` |
| Error resolution | `frameLink/errorRez.m`, `frameLink/createNewCell.m`, `frameLink/continueCellLine.m`, `frameLink/markDivisionEvent.m`, `frameLink/merge2Regions.m`, `frameLink/missingSeg2to1.m`, `frameLink/registerMasks_monointensity.m`, `frameLink/watershedpaw.m`, `frameLink/deleteRegions.m` |
| Post-linking | `frameLink/trackOptiCellMarker.m`, `fluorescence/trackOptiFluor.m`, `cell/trackOptiMakeCell.m`, `cell/toMakeCell.m`, `cell/intMakeRod.m`, `cell/rodGeom.m`, `cell/makeColonyDist.m`, `fluorescence/trackOptiCellFluor.m`, `Internal/saveosmasks.m`, `cell/trackOptiClist.m`, `cell/trackOptiCellFiles.m`, `gate/gate.m` → `gateTool('strip')` |
| Constants | `settings/loadConstants.m`, `settings/100XEc.mat`, `settings/100XPa.mat` |

Dead code in `frameLink/` that is **not** on this path: `splitAreaErrors.m`,
`splitAtBestSeg.m`, `missingSeg2to1_*draft*.m`, `bwdistpaw.m`.

---

## 2. Input requirements and how mask import works

### Directory and naming

- Before `trackOptiPD` runs, the images sit in the experiment root and are
  named `{base}t{TTTTT}xy{PPP}c{C}.tif`. `c1` is phase (it's only used as the
  image carrier, but it has to exist), and `c2…` become `fluor1…`.
- Alignment is skipped when `startEnd(1) > 1` or when `raw_im/` already exists
  with TIFFs or a `cropbox.mat` (`BatchSuperSeggerOpti.m:118-133`). Without
  a crop box, `crop_box_array = cell(1,10000)` and every `crop_box` is empty.
- Masks go in `xyNN/masks/{base}t{TTTTT}xy{PPP}c1_cp_masks.png`
  (`ssoSegFunPerReg.m:92`). TIFF masks are found as `…cp_masks.tif`
  (no `c1`).
- If `masks/` is empty, or **the mask count ≠ the phase-file count**,
  `intProcessXY` prints the Omnipose command and blocks on `input()`
  (`BatchSuperSeggerOpti.m:359-375`). A second copy at `:328-344` runs when
  neither `masks/` nor `cp_masks/` exists. With `autoomni` it runs Omnipose
  instead of blocking.
  - The count is the number of directory entries minus 2, so hidden files
    such as `.DS_Store` cause a false mismatch.
  - There is no re-check after Enter. A phase file without a mask then makes
    `imread` fail inside the `parfor`, which aborts the run.
  - `cp_masks/` passes this check, but `ssoSegFunPerReg` reads only `masks/`
    and waits with `pause` until it exists (`ssoSegFunPerReg.m:69-75`).
- The dataset path must not contain another directory called `seg`. The
  mask directory is found with `extractBefore(dataname, '/seg/')`.

### `doSeg` → `ssoSegFunPerReg` → `intMakeRegs`

For each frame `i` (index into the sorted phase list):

1. The phase image is loaded. If there are several z planes, they are averaged.
2. `ssoSegFunPerReg` **skips all SuperSegger segmentation** (the original
   code is commented out) and calls `intMakeRegs(maskpath, data, CONST)`.
3. `intMakeRegs`:
   - `regs_label = double(imread(mask))`. The Omnipose labels are used as
     they are, with no relabelling, no hole filling, no size filter, and no
     border removal.
   - `num_regs = max(label)`.
   - `props = regionprops(…,'BoundingBox','Orientation','Centroid','Area')`.
   - `props(ii).Medoid` = medoid of the region's **skeleton**
     (`bwmorph skel` → the pixel with minimum mean distance to all other
     skeleton pixels). This is an OmniSegger addition. It later becomes
     `coord.rcm`.
   - `info(ii,:) = cellprops3(mask, props)`: 21 shape features. Column 1 is
     `L1` (long-axis extent after rotating to the major axis) and column 2 is
     `L2` (mean width).
   - `scoreRaw = scoreNeuralNet(info, E)` and `score = scoreRaw > 0`. **The
     linking logic never uses the region score.** It only shows up in log
     strings and in the clist.
4. Each fluorescence channel is loaded **as a full frame** into
   `data.fluorN`, and the phase image into `data.phase`.
   `data.imRange = [min;max]` per channel.
5. The result is saved to `seg/{base}t{TTTTT}xy{PPP}_seg.mat`.

Each seg/err file therefore holds full-frame phase, every fluor image, and the
label image. The design keeps all per-frame state in these files on disk.

---

## 3. Per-frame data model (`regs` in seg/err files)

`updateRegionFields` (re)builds these fields from `regs_label` every time
`data_c` or `data_f` is loaded for linking. It overwrites any earlier values.
`data_r` is loaded from its err file as it is.

| Field | Meaning |
|---|---|
| `regs_label`, `num_regs` (= `max(label)`), `props` (+`Medoid`) | geometry |
| `info`, `L1`, `L2`, `scoreRaw`, `score`, `eccentricity` | shape features |
| `map.r{k}`, `map.f{k}` | region IDs in the previous / next frame that region `k` is assigned to |
| `revmap.r{j}`, `revmap.f{j}` | inverse of the above (which current regions point at `j`) |
| `error.r(k)`, `error.f(k)` | 0 = ok, 2 = area change too negative, 3 = too positive; set to 1 by some resolution branches |
| `cost.r/f`, `idsC.r/f`, `idsR.r`, `idsF.f`, `dA.r/f` | the flattened candidate cost vector and its index columns, plus the per-region area change |
| `ID` | cell (track) ID; 0 = unassigned |
| `birth`, `birthF` | first frame index; 1 if born in this frame |
| `death`, `deathF` | last frame seen so far; 1 if this is (so far) the last frame |
| `age` | frames since birth, starting at 1 |
| `divide` | 1 on the mother's **last** frame if it divided |
| `motherID`, `sisterID`, `daughterID{}` | lineage |
| `stat0` | 0 = born by appearance; 1 = born from a division; 2 = full cell cycle (set later) |
| `ehist` | 1 if any linking error happened since birth |
| `error.label{k}` | human-readable log of what happened |
| `manual_link.r/f`, `ignoreError` | GUI-edit hooks (inactive on a fresh run) |

**Time is the file index.** `time` in linking is the position of the file in
the sorted `dir('*seg.mat')` listing (1-based), not the `t` number in the
filename. `birth`, `death` and every frame count downstream use this index.
Nothing checks for gaps: a timepoint missing from both phase and masks drops
out, and its neighbours are linked as adjacent frames. `dir` name order is
the only sorting, so unpadded `t` numbers would be mis-ordered.

---

## 4. Linking driver — `trackOptiLinkCellMulti`

```
delete all *err.mat                       (clean_flag = 1 from trackOpti)
cell_count = 0; time = 1
while time <= numIm:
    data_r = load err(time-1)  (or [] at time 1)          — already linked
    data_f = load seg(time+1)  (or [] at last frame); updateRegionFields
    data_c = load err(time) if it exists, else seg(time);  updateRegionFields
    data_r.map.f  = assign(data_r → data_c, forward=1)     (recomputed!)
    data_c.map.r  = assign(data_c → data_r, forward=0)
    data_c.map.f  = assign(data_c → data_f, forward=1)
    finalIteration = (curIter >= 3)
    if isfield(CONST,'ignoreerror'): finalIteration = CONST.ignoreerror   ← see §9.1
    [data_c, data_r, cell_count, reset] = errorRez(..., ignoreError = finalIteration)
    if reset:  cell_count = lastCellCount; data_c.ID = 0; (repeat same frame)
    else:      time++, curIter = 1
    save data_r → err(time-1), data_c → err(time)          (saved even when reset)
```

Design points:

- **Three assignment passes per frame.** The pass from the previous frame
  (`r→c`) is recomputed against the current frame, so both directions see the
  same, possibly corrected, `c` masks. The forward pass (`c→f`) uses the next
  frame's **uncorrected** seg masks. It's used as a look-ahead for merge
  decisions, by the `REMOVE_STRAY` test, and by `missingSeg2to1` in the 2:1:2
  case (`errorRez.m:257`).
- **Identity comes from the backward map.** `errorRez` reads `c.map.r` first
  and uses `r.map.f` and `c.revmap.r` as consistency checks (§6).
- **Mask edits restart the frame.** A merge or split rewrites
  `c.regs_label`, saves it, and reprocesses the same `time` from the edited
  err file. IDs handed out in the aborted pass are rolled back. Changes the
  aborted pass made to `data_r` (death, divide, daughterID) were already
  saved and are **not** rolled back (§9.6).
- **Iteration cap.** `maxIterPerFrame = 3`. On the third pass `ignoreError`
  is set, which turns off the C/D merges and the E split. F's subset merge
  (`errorRez.m:298-312`) and the `REMOVE_STRAY` deletions don't check
  `ignoreError`, so a frame can still reset after pass 3, and nothing bounds
  the loop. This cap is **switched off** when `CONST.ignoreerror` exists
  (§9.1).

---

## 5. Frame-to-frame assignment — `multiAssignmentSparse(data_c, data_f, CONST, forward)`

This is one directed pass: each region in the "c" frame gets assigned to 0, 1
or 2 regions in the "f" frame. In the reverse pass, "f" is actually the
previous frame and `forward = 0`.

### 5.1 Candidates

- **Colonies.** `colony_labels = bwlabel(imfill(regs_label>0,'holes'))`.
  Cells that touch (no background pixel between them) belong to one colony.
- **Neighbour candidates.** For each region `c`, `findNeighborPairs`
  dilates its mask with a **5×5 square** (2 px), inside its bbox plus 20 px
  of padding. The labels under that dilated mask in the other frame are its
  candidates (`neighF`).
  ⇒ **A cell must overlap, or come within 2 px of, its own previous footprint
  to be linked at all.** There is no search radius and no fallback on
  centroid distance.
- **Pairs.** Two regions in the same frame form a pair if one falls inside
  the 2-px dilation of the other.
  - **Pairs in c** (two c regions → one f region) are evaluated for every
    neighbour pair whose members don't already have a "good one-to-one" match.
  - **Pairs in f** (one c region → two f regions, i.e. a division when
    running forward) are evaluated only when the best one-to-one match for that c
    region is **not** good. Both f regions must be candidates of c and must
    neighbour each other.
- **"Good one-to-one" pre-filter** (`:238`). Among single f candidates, take
  the one with the lowest *provisional* cost
  `100/ovT + 20/ov + d + 100|dA|`. It counts as good when
  `|dA| < 0.15 and ov > 0.7 and ovT > 0.8`. Good regions skip the pair search
  in f, and pairs in c that contain them are skipped too. A c region with no
  candidates at all also counts as good (`:241-244`), so every pair
  containing it is skipped.
- Two-to-two mappings are never considered.

### 5.2 Cost per candidate (c-set C → f-set F)

For combined sets: area = sum, centroid = **unweighted mean** of the member
centroids, mask = union.

```
ov   = |C ∩ F| / |C|                          (floor 1e-4)
ovT  = |C ∩ shift(F, round(cent_C − cent_F))| / |C|   (overlap after centroid alignment)
d    = ||cent_C − cent_F||
dA   = (A_F − A_C) / A_C
wcol = exp(−||cent_C − cent_colony|| / 0.1)
out  = (cent_C − cent_colony) · (cent_C − cent_F) / d

P(dA): forward: 100 if dA < −0.1;   reverse: 100 if dA > +0.1
       then 1000 if |dA| > 0.6,  then 50 if |dA| > 0.3   (later lines overwrite earlier)

cost = P(dA) + 100/ovT + 5·d + 100·|dA| + wcol·20/ov + 0.1·out     (reverse: out → −out)
```

Notes on the terms:

- **`ovT` (overlap after aligning shapes) carries the most weight.** Plain
  overlap `ov` is multiplied by `wcol`, which is ≈ 0 unless the region's
  centroid lies within about 0.3 px of its colony's centroid. In practice,
  `ov` counts mainly for isolated single cells, whose colony is the cell
  itself. It also counts for a pair-in-c column that makes up a whole
  two-cell colony of near-equal areas, and for a cell at the centroid of a
  symmetric cluster. Otherwise, inside a touching cluster the raw-overlap
  term is effectively turned off.
- `maskF` is cropped to C's padded box before the shift (`:194`), so `ovT` is
  underestimated when the displacement exceeds 20 px.
- **Bug in the area penalty:** because the `> 0.3 → 50` line runs last, it
  overwrites the 1000. The penalty that results is: `|dA| > 0.3 → 50`;
  otherwise 100 for the disallowed direction (between 10 % and 30 %);
  otherwise 0. For P alone, a 25 % forward shrink (100) costs more than a
  70 % shrink (50). The **total** cost isn't inverted for shrinkage, because
  `ovT ≤ A_F/A_C` keeps `100/ovT` large (≥ 258 at −25 %, ≥ 453 at −70 %).
  The bug does bite for forward growth (and reverse shrinkage), where `ovT`
  can stay ≈ 1: dA = +1.0 gets 50 instead of 1000.
- `out` rewards moving outward from the colony centroid (forward). It projects
  onto the unit displacement vector, so its size depends on the distance to
  the colony centroid, not on how far the cell moved. At 100 px from the
  colony centre it reaches ±10, comparable to `5·d` at 2 px. Only near the
  colony centre is it a tie-breaker.
- In the pair-in-f branch, `if centroidCost == 0` tests the **whole vector**
  (`:273`), so it is never true. A zero displacement gives `out = NaN`, so the
  cost is NaN and that candidate silently drops out.

### 5.3 Selection

1. **Greedy global minimum.** Repeatedly pick the cheapest remaining
   candidate column, assign it, and remove every column that involves any of
   its c or f regions. If a pair in c wins, both c regions get the same f
   assignment. This is greedy, not Hungarian or optimal.
2. **`fixProblems` (twice, once per direction).** For each f region left
   unassigned, find the c region with the best single-region *overlap*
   (`1 − ov`) for it, then:
   - if that c has no assignment → assign it;
   - if c's current target is shared by two c regions *and* that mapping
     violates the area limits → "steal": reassign c to the orphan when both
     resulting area changes are acceptable;
   - otherwise, if the current mapping violates the area limits and giving
     c both its current target and the orphan fixes that → map c to both.
     This can create a 1→3 map.

   Details:
   - The best-c search may return the first member of a c-pair column.
   - The "assign" branch (`:381`) updates `map` but not `revAssign`, and the
     mirrored call does the opposite, so the two disagree.
   - The steal check at `:396` normalises by
     `max(areaFBefore, areaC_of_stealer)`, not by the new area.
   - The second call swaps the areas but keeps the pass's `minDA`/`maxDA`.
     It therefore accepts `A_f/A_c` ∈ **[0.70, 1.25]**, not [0.80, 1.43].
3. **`exchangeAssignment` (twice).** For each unassigned c region, take its
   best single f. If another c holds that f and that c's second-best single f
   is unassigned, swap them.
   (If no c holds it, the loop variable ends at the last region and that
   region gets used anyway. This is a latent bug.) A second bug: if the
   unassigned c has no other column, `min` over all-NaN returns column 1
   (`:493`), so c can be swapped to an unrelated f. Swaps have no area
   check.
4. **Area-change error flag.** `dA = (ΣA_F − A_C) / max(ΣA_F, A_C)`. Note
   that it uses a **different normaliser** from the cost (`/ max` instead of
   `/ A_C`). `error = 2` if `dA < minDA` and `3` if `dA > maxDA`. With
   `DA_MIN = −0.2` and `DA_MAX = 0.3`:
   - forward: `minDA = −0.2`, `maxDA = 0.3`
   - reverse (sign-flipped): `minDA = −0.3`, `maxDA = 0.2`
   - In terms of the forward area ratio `A_t / A_{t−1}`, both passes accept
     **[0.80, 1.43]** for 1:1 and 1:2 maps.
   - For a c-pair (2→1), `:450` uses the single c's area, not ΣA_C. In c→r,
     both daughters → mother get dA ≈ +0.5, so `error = 3`. `createDivision`
     clears it, but branch B and `mapBestOfTwo` read it.
   - An unassigned c gets dA = −1 (`error = 2`). With `data_f = []` (last
     frame, or t = 1 for `map.r`), the map is empty and the error is 0.

Returns: `assignments{c}`, `errorR`, `totCost` (flattened), `indexC`,
`indexF`, `dA`, `revAssign{f}`.

---

## 6. Error resolution and ID assignment — `errorRez`

Regions are visited in **ascending label order**. A region skips its branch
if it already has an ID (for example, sister 2 after a division) or is in
`modRegions`. It still reaches the final `ID == 0 → createNewCell` check.
Notation, for current region `k`:

- `R = c.map.r{k}` (where `k` points back to)
- `fwd = r.map.f{R}` (where `R` points forward to)
- `back = c.revmap.r{R}` (which current regions point back at `R`)

| # | Condition | Action |
|---|---|---|
| A | `R` empty | **New cell** (labelled "stray" when `t > 1`). If `REMOVE_STRAY` is set, `t > 1`, and `k` also has no forward map → delete the region and re-run the frame. On the last frame no region has a forward map, so every stray there is deleted. |
| B | `|R| = 1`, `|back| = 1` | **Continue** `R`'s ID. `errorStat = error.r(k) > 0` → `ehist`. *This branch wins even if `fwd` says `R` divided.* A daughter continued this way usually has `error.r = 3` (§5.3), so `ehist = 1`. |
| C | `|R| = 1`, `|fwd| = 1`, `|back| = 2` (back says 2→1, forward says 1→1) | `ok = −0.2 < (A_s1 + A_s2 − A_R)/(A_s1 + A_s2) < 0.3`. **Merge** sisters if `ok` ∧ merging allowed ∧ next frame exists ∧ (either sister has no forward map ∨ s1's forward targets ⊆ s2's ∨ either sister < `SMALL_AREA_MERGE` px). Else, if `ok` (or a manual link) → **division**. Else → **mapBestOfTwo**. |
| D | `|R| = 1`, `|fwd| = 2`, `k ∈ fwd`, and the other member of `fwd` maps back only to `R` (`:174`) | Same `ok` test and merge triggers as C → **merge**, but there is no overall next-frame test: "no forward map" and "forward targets ⊆" need a next frame, "either sister small" doesn't. Otherwise → **division, even when `ok` is false**. (The inner `mapBestOfTwo` at `:194-196` is unreachable.) |
| D′ | `|fwd| = 2`, sister maps back elsewhere | **Continue** `R` with `k`. If the sister's backward map is **empty**, `any([])` is false: `k` falls through to "Error not fixed" and becomes a **new cell** with no mother (`:236-243`). |
| D″ | `|fwd| = 2`, `k ∉ fwd` | **New cell**, `error.r = 1`. |
| E | `|R| = 2` (two previous cells → one current region: under-segmentation) | If `~ignoreError` (manual links aren't checked) → **missingSeg2to1** (split `k`, re-run the frame). If that fails or isn't allowed → **mapToBestOfTwo**: continue whichever `R` has the lower single-to-single `cost.r` (ties go to `R(2)`), `error.r = 2` (→ `ehist`). The other `R` ends. |
| F | `|R| = 1`, `|fwd| > 2` | Merge **all** of `fwd` if a sister has no forward map or all share one forward target, else merge the subset that shares a forward target. *`haveNoMatch` is always false here* (`:281`, `isempty` of a non-empty cell array). If nothing merges, `k` falls through to **new cell** with no mother. Under `ignoreError` the merge-all becomes `mapBestOfTwo` (`:294`). The subset merge checks neither `ignoreError` nor manual links. F never checks `k ∈ fwd`, and the shared targets can be several, so unrelated groups can go into one `merge2Regions` call. |
| G | anything else | Label "Error not fixed" → **new cell**. |

After the branch runs, any region whose `ID` is still 0 → `createNewCell`.

In C and D, merging is allowed when `~ignoreError` and the region has no
manual link. E and F's merge-all check only `~ignoreError`, and F's subset
merge checks neither.

### ID bookkeeping

- **`createNewCell`**: `ID = ++cell_count`, `birth = death = t`,
  `birthF = deathF = 1`, `age = 1`, `stat0 = 0`, `ehist = 0`, mother and
  sister IDs = 0, `divide = 0`, `daughterID = []`. `error.r` is left as it is.
- **`continueCellLine(k ← R)`**: copies ID, birth, stat0, mother, sister and
  `ehist |= errorStat` from `R`. Sets `age = age_R + 1`, `death = t`,
  `deathF = 1`, and in r sets `death_R = t`, `deathF_R = 0`, `divide_R = 0`.
  If another current region already took that ID, this logs "ID PROBLEM" and
  does nothing, so `k` later becomes a new cell.
- **`createDivision` → `markDivisionEvent`**: clears `error.r` on mother and
  sisters. Each daughter gets a new ID (s1 = n+1, s2 = n+2), `birth = t`,
  `stat0 = 1`, **`ehist = 0` (error history resets at every division)**,
  `motherID = ID_R` and cross-linked `sisterID`. In the mother:
  `divide = 1`, `daughterID = [n+1, n+2]`. The guard checks only sister 1's
  ID (`markDivisionEvent.m:40`), so sister 2's ID is overwritten even if it
  already has one. *Inferred:* C doesn't check sister 2's own map or ID, so a
  division can overwrite an ID handed out earlier in the same pass.
- **`mapBestOfTwo`**: the keeper is the sister with the **minimum signed**
  `dA.r`, not the minimum `|dA|` (`:445`). With both sisters pointing at
  the same mother, that means the **larger** sister. The other sister gets
  `error.r = 1` and becomes a new cell (or is deleted if `REMOVE_STRAY` is set,
  it has no forward map, and a next frame exists).

### Mask-editing operations

- **`merge2Regions`**: union of the regions, then dilate and erode with a 3×3
  square (a closing). The merge goes ahead only if the closed union is
  **one** connected component. If so, it clears the old labels and writes the
  union as label `num_regs + 1`.
  - If the closing gives more than one component, nothing happens and
    `reset = false`. Neither sister has an ID yet, so **both become orphan
    new cells** and the mother's lineage just ends.
  - Only `list_merge(1)` and `(2)` are cleared (`:47-48`). When F merges ≥ 3
    regions, the third keeps its label and gets the merge label added on top
    (corrupted label sums). `rl + labels` also adds onto pixels of any
    non-merged neighbour that lie inside the closing.
  - *Inferred:* errorRez neither refreshes `num_regs` between edits nor stops
    after one. A second merge or split in the same pass therefore writes
    `num_regs + 1` again (`merge2Regions.m:45-50`, `missingSeg2to1.m:93-95`),
    and after the re-run two disjoint blobs share one label.
- **`missingSeg2to1`** (splits a region that two previous cells collapsed into):
  1. Build a full-frame image with the two previous cells labelled 1 and 2,
     and register it (affine, intensity-based, `imregtform` 'monomodal', 3
     pyramid levels, nearest-neighbour warp) onto the binary mask of `k`.
     The moving image is min-max normalised first (labels become 0.5 and 1),
     and both images are Gaussian-filtered (σ ≈ 1)
     (`registerMasks_monointensity.m:28-34, 61-67`).
  2. If the next frame also has `k` → 2 cells, register those too, and
     pick the label pairing (0.5 + 2 / 1 + 4 vs 0.5 + 4 / 1 + 2) with more
     agreeing pixels.
  3. Seeds = extended minima (h = 2) of the interior distance
     `−(bwdist(~r1) + bwdist(~r2))`, imposed with `imimposemin`. Watershed
     with `watershedpaw`, which gives each ridge pixel the label of its
     labelled 3×3 neighbour with the lowest *image value*, so no dividing line
     is left. Restrict to `k`'s mask.
  4. The acceptance test is `max(label) == 2` (`missingSeg2to1.m:92`), not
     "exactly {1, 2}". Labels are written as `num_regs + 1` and
     `num_regs + 2`. If only label 2 falls inside `k`, the whole of `k` is
     relabelled `num_regs + 2` without a split. Success is still reported
     (the label image changed), so the frame re-runs, without limit when
     the cap is off (§9.1).
- **Label gaps (inferred).** Both edits leave the old label numbers unused,
  and nothing relabels the image afterwards (the `bwlabel` in
  `updateRegionFields` is commented out). `num_regs = max(label)`, and
  `regionprops` returns entries with `Area = 0` for missing labels, so each
  merge or split probably produces **zero-area "ghost" regions**. Ghosts have
  no candidates (and a NaN area change, so they are never flagged), so they
  land in branch A, get new IDs, and end up as one-frame
  `cell*.mat` files with `coord.A = 0`. The `Caution : cell N has a mask of 0`
  log line confirms it. Check on real data with
  `find(cellfun(@(c) c.CellA{1}.coord.A, …) == 0)`.

---

## 7. Post-linking steps

### 7.1 `trackOptiCellMarker` — complete cell cycles

Goes **backwards** over frames `i = N−1 … 2` (the first and last frames are
never examined as division frames). A region is the end of a complete cycle when:

```
divide(ii) == 1  ∧  stat0(ii) ≥ 1 (born from a division)  ∧  ehist(ii) == 0
                 ∧  (i − birth(ii)) ≥ MIN_CELL_AGE (5)      ⇒ at least 6 frames observed
```

It then walks back via `map.r{jj}(1)` and sets `stat0 = 2` in each frame
until it reaches the birth frame. Frames 2…N−1 are re-saved.

### 7.2 `trackOptiFluor` — background

For each frame and channel:
`flNbg = mean(fluor(~imdilate(regs_label > 0, disk(5))))`, restricted to the
crop box if one exists. This is one scalar per frame and is **not**
subtracted from anything.

### 7.3 `trackOptiMakeCell` — `CellA{region}`

For each frame and region (needs `data_r.CellA` from the previous iteration,
so this runs sequentially):

- Crop = bbox + **5 px** padding, clipped to the image, so the crop size
  changes from frame to frame. Fields: `xx`, `yy`, `r_offset`, `BB`,
  `mask`, `phase`, `fluorN` (raw crops), `fluorNmm` (whole-frame
  [min max]).
- `edgeFlag`: the bbox touches the image border.
- `cellLength = [L1, L2]` from `cellprops3`.
- `toMakeCell`:
  - `coord.e1/e2` from `regionprops` Orientation, with the sign of `e1`
    aligned to the previous frame's `e1`.
  - `length` = extents of the mask after rotation.
  - `coord.r_center` = centre of the rotated extents.
  - **`coord.rcm` = skeleton medoid** (the name says centre of mass, but it
    isn't one).
  - `coord.A`, `coord.box`, `coord.xaxis/yaxis`.
  - `Lrod`, `Rrod` from a spherocylinder fit to area and the summed distance
    transform.
- **Pole age.** A daughter's old-pole orientation is
  `sign((centroid − sister_centroid)·e1)`, and its old-pole age is the
  mother's `op_age + 1` or `np_age + 1`. Continuing cells inherit the
  mother's pole. When there's an error or no previous region → NaN.
- `flN = trackOptiCellFluor`: `sum` over the mask (raw, no background
  subtraction), fluorescence-weighted centroid `r`, and second moments
  `Ixx`, `Iyy`, `Ixy`, plus `bg = flNbg` (frame scalar).
- `cell_dist`: minimum distance from the cell to the colony edge (colony =
  close/fill/erode with disk(5)). `gray` = mean phase value in the mask.

### 7.4 `saveosmasks`

Writes each err frame's (possibly edited) `regs_label` to
`xyNN/masksOS/{base}t…xy…_os_masks.png`. The output keeps the label gaps.

### 7.5 `trackOptiClist`

Writes one row per cell ID to `xyNN/clist.mat`: birth and death values,
lengths, areas, fluor sum, mean and background, mother and daughter IDs,
generation, progenitor, growth rate `= (ln L_death − ln L_birth)/age`. It also
builds `data3D`, a per-frame time series. Rows are created when an ID larger
than the largest ID seen so far appears, which assumes IDs only increase over
time (true here).

### 7.6 `trackOptiCellFiles` — the files you parse

`ID_LIST = gate(clist).data(:,1)`. With no gate set, that is every cell. A
gate that removes every row also gives every cell, because an empty
`ID_LIST` means "all" (`trackOptiCellFiles.m:99`). The
step goes forward through the err files and keeps an accumulator per ID:

- `birthF ∧ deathF` (one-frame cell) → init, update, save
- `birthF` → init
- `deathF` → update, save
- else → update

Anything still open at the end is saved too.

**File name:** `Cell%07d.mat` if `stat0` in the cell's **last** frame
`== 2`, else `cell%07d.mat` (`trackOptiCellFiles.m:224`). Old files are
deleted first, by `trackOpti.m:226-227`.

**Contents (top level):** `CellA {1×n}`, `ID`, `birth`, `death`, `divide`,
`motherID`, `sisterID`, `daughterID`, `neighbors`, `stat0`, `ehist`,
`contactHist` (the last three are copied from the last frame).

**`CellA{k}`** is the `trackOptiMakeCell` struct for frame `birth + k − 1`
(file-index time, contiguous by construction, so no explicit time field),
plus `r` (regionprops centroid), `error.label`, `ehist`, `contactHist`,
`stat0`, and `locusN` if foci were fitted. One-frame cells are the
exception: `intInitCell` and then `intUpCell` both add the same frame
(`trackOptiCellFiles.m:100-108, :166, :192`), so `numel(CellA) = 2`.

---

## 8. Effective parameters

`loadConstants.m` sets defaults and then copies **only some** fields from the
preset `.mat`. Several preset values are silently ignored:

| Parameter | `loadConstants` default | in `100XEc.mat` | in `100XPa.mat` | **Effective** | Used by |
|---|---|---|---|---|---|
| `DA_MIN` | −0.2 | −0.2 | −0.2 | **−0.2** (not copied) | area error flag, C/D `ok` test |
| `DA_MAX` | 0.3 | 0.3 | 0.3 | **0.3** (not copied) | same |
| `REMOVE_STRAY` | 0 | **1** | 0 | **0** (not copied; the GUI can set it) | branch A, `mapBestOfTwo` |
| `MIN_CELL_AGE` | 5 | 5 | 5 | **5** (not copied); the GUI sets it from its `cell_age` box, default **3** (`superSeggerGui.m:199`) | CellMarker |
| `SMALL_AREA_MERGE` | 55 | 50 | 30 | **50 / 30** (copied) | C/D merge test |
| `MIN_AREA` | 5 | 8 | 8 | 8 / 8 (copied) | **unused** (StripSmall disabled) |
| `MIN_AREA_NO_NEIGH` | 30 | 30 | 30 | 30 (copied) | **unused** |
| `OVERLAP_LIMIT_MIN` | 0.08 | 0.08 | 0.08 | 0.08 | **unused anywhere** |
| `linkFun` | `@multiAssignmentSparse` | same | same | default (not copied) | linking |
| `seg.segFun` | — | `@ssoSegFunPerReg` | same | from preset | mask import |
| `ignoreerror` | absent | — | — | `processExp` sets it to 0 | §9.1 |

Hard-coded values in `multiAssignmentSparse`:

| Constant | Value |
|---|---|
| dilation for candidates and pairs | 5×5 square |
| bbox padding | 20 px |
| `centroidWeight` | 5 |
| `areaFactor` | 20 |
| `areaChangeFactor` | 100 |
| `outwardMotFactor` | 0.1 |
| `noOverlap` floor | 1e-4 |
| good one-to-one | \|dA\| < 0.15, ov > 0.7, ovT > 0.8 |
| directional penalty | 100 at 10 % |
| \|dA\| penalties | 50 above 30 % (1000 above 60 % never applies) |

Elsewhere: `maxIterPerFrame` = 3, the merge closing uses a 3×3 square, the
CellA pad is 5 px, and the background and colony dilation uses disk(5).

`nd2_to_cells/presets/100XEc.toml` copies `remove_stray = true` and
`overlap_limit_min` from the `.mat`. Neither matches what MATLAB actually
runs: `REMOVE_STRAY` is effectively 0, and no overlap threshold exists at all.

---

## 9. Quirks that affect comparisons

1. **`CONST.ignoreerror` turns off the 3-pass cap**
   (`trackOptiLinkCellMulti.m:161-163`). Whenever the field exists (and
   `processExp` always sets it, to 0), `finalIteration` is overwritten on every
   pass. Merges and splits are never switched off, and a frame can repeat
   without limit. If CONST came from `loadConstants` directly, the field is
   absent and the cap applies. On the third pass the C/D merges and the E
   split are disabled, but F's subset merge and the `REMOVE_STRAY` deletions
   still run (§4). `processExp` sets the field at `processExp.m:114`, and the
   GUI never sets it. **Check which CONST your runs used**
   (`xyNN/../CONST.mat`).
2. **The backward map decides identity.** Branch B continues a lineage even
   when the forward pass detected a division.
3. **The division rules aren't symmetric.** C (2→1 seen only backwards)
   needs `ok` area; D (1→2 seen in both directions) divides **without any
   area test**.
4. **The area normalisation differs at every stage**: `/A_C` in the cost,
   `/max` in the error flag, `/ΣA_daughters` in the C/D `ok` test.
5. **Error history resets at each division.** `ehist` only covers the
   current cell cycle. A cell born from an erroneous division still gets
   `stat0 = 1`.
6. **Aborted passes leave state behind in `data_r`.** If a pass that called
   `markDivisionEvent` is reset, the mother keeps `daughterID`.
   `continueCellLine` later sets `divide = 0` but does not clear `daughterID`.
   If the re-run never relinks R, `divide = 1` stays too, and `daughterID`
   points at IDs that are re-issued to other cells after the `cell_count`
   rollback. Edits from an aborted `continueCellLine` (`death = t`,
   `deathF = 0` on R) also persist.
7. **A failed merge orphans both sisters** (§6), so the lineage breaks
   instead of a division being recorded.
8. **Ghost zero-area regions after merges and splits** (§6, inferred).
9. **No gap closing and no search radius.** A cell that skips a frame or
   moves more than about 2 px past its own footprint starts a new lineage.
10. **The first and last frames** are never division frames in CellMarker.
    On the last frame (no `data_f`), branch C never merges, and branch D
    merges only through the `oneIsSmall` test.
11. **Greedy assignment depends on order.** The global-minimum greedy pick
    plus the repair passes (four calls, each looping in label order and
    editing as it goes) are deterministic but not optimal. `errorRez`
    resolves regions in label order, so the Omnipose label numbering can
    change results.
12. **A split can "succeed" without splitting** (§6, `missingSeg2to1` step
    4). The frame then re-runs on every pass, without limit when the cap is
    off (§9.1).
13. **Label collisions** (*inferred*, §6 `merge2Regions`). Two edits in one
    pass can write the same new label, so two disjoint blobs share one ID.
14. **One-frame cells have two `CellA` entries** (§7.6).

---

## 10. Comparison with `nd2_to_cells/track.py`

Moved to [`linking_comparison.md`](linking_comparison.md). That file compares
the current linker stage by stage, classifies each divergence, and lists
corrections to this document found when checking it against the source.
