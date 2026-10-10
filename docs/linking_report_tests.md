# Tests for the 260429 linking report

The downstream analysis reported that tracked cells change area abruptly
(`amp_testing/linking_report/LINKING_ISSUES.md`, 2026-10-08). The report
names four event types. Two are linking problems:

- **A, re-merge after division.** The daughters' labels fuse again. One
  daughter's track continues on the fused object, and the sister's track
  ends.
- **B, neighbour absorbed.** Omnipose fuses two neighbouring cells, and one
  track continues on the fused object.

The other two types, C (a label grows into background) and D (a track spans
two labels), are segmentation junk. They are not tested.

## Merge events

`tests/merge_events.py` finds merges in any track output: per-cell
`cell*.h5` files, or one `cells.h5` (written with `--consolidated`, or by the
IoU linker on `main`). At frame t, track T absorbs track X if T's mask at t
covers at least half of T's own mask at t−1 and at least half of X's mask at
t−1.

| Kind | X is | Typical cause |
|---|---|---|
| A | T's sister | daughters fuse again after a division |
| B | another cell, within 50 px of T at t−1 | Omnipose fuses neighbours |
| far | another cell, more than 50 px away | in the port, MATLAB's label-collision quirk: one region spans distant blobs |

Differences from the report's `linking_diagnostics.py`:
- No area jump is required. The report's 1.3× jump misses merges where the
  larger cell keeps its ID; the port does this at the report's figure B.
- Kinds come from the tracks, not from the Omnipose labels, so merges made by
  the linker itself are found too.

## Baseline numbers

Events per position, on full runs of 361 frames:

| | A, port | A, IoU | B, port | B, IoU | far, port | far, IoU |
|---|---|---|---|---|---|---|
| xy01 | 158 | 192 | 297 | 172 | 186 | 1 |
| xy02 | 186 | 261 | 434 | 274 | 280 | 1 |
| xy03 | 356 | 440 | 617 | 385 | 562 | 1 |
| xy04 | 203 | 184 | 407 | 195 | 214 | 0 |
| **all** | **903** | **1077** | **1755** | **1026** | **1242** | **3** |

- "Port" is the bug-for-bug SuperSegger port on this branch.
- "IoU" is `main`'s IoU linker: the `cells.h5` of 2026-10-07 that the report
  analysed, which predates PR #4.

Findings:
- **A** is about as common in both linkers.
- **B** is 1.7× as common in the port.
- **far** occurs only in the port. The report saw the same kind of behaviour
  in SuperSegger's own output: tracks that jump 150–660 px onto a different
  cell.
- **Clean cases work.** On clean synthetic rods, the port splits fused labels
  correctly, so the real-data failures come from the split step (registration
  and watershed). Over 93 split calls, the MATLAB-faithful registration cut
  the region in 23, and ITK v4 in 60 (see `supersegger_port.md`).

## Tests

```bash
pip install -e ".[dev]"
pytest                                   # fast tests, a few seconds
ND2C_TEST_DATA=".../amp_testing/01_SW5-20-1_2xMIC/260429" pytest   # + real data
```

Every test that takes the `config` fixture runs once per configuration in
`CONFIGS`. With real data, that costs about 7 min per configuration.

| File | What it checks |
|---|---|
| `test_merge_events.py` | the event check itself, on hand-made tracks |
| `test_fixes.py` | each opt-in fix against the MATLAB behaviour it replaces |
| `test_scenarios.py` | synthetic movies: growth and division (controls), daughters that fuse (A), neighbours that fuse (B), and a temporary split that must not become a division (expected to fail without `division_lookahead` ≥ 3). The baseline passes the first four. |
| `test_report_cases.py` | 10 real-data cases: one A and one B per position, plus the locations of the report's figures A and B. Each is linked as a 13–18-frame window that reproduces the full run's event. |

In `test_report_cases.py`:
- `test_baseline_event` pins the baseline: every case still produces its
  event.
- `test_merge_resolved` states the desired outcome: neither of the two cells
  absorbs a track, or is absorbed, at t. One track holding both cells at t−1
  also counts, as long as that track absorbed no other track up to t; that
  is the outcome when the split was never called a division. Each
  configuration is a strict xfail for every case it is not listed as
  resolving in `RESOLVED`.

The real-data tests read the masks in place and write only to pytest's
temporary directory.

## Evaluating a change

1. **Add the change as a configuration.** Make it switchable from the
   preset: a name in `track.FIXES` for the `fixes` list, or a key in
   `TrackingParams`. Then add an entry to `CONFIGS` in `tests/conftest.py`
   with the preset values to change. Every test that takes `config` now also
   runs on it.
2. **Run `pytest`.** The controls must pass. For each case the change
   resolves, add its name to `RESOLVED[config]`. Strict xfail fails the run
   until you do, and also if a listed case fails again.
3. **Count events on whole positions.** Run `track` with the configuration on
   all four positions, then `python tests/merge_events.py <cell dir> ...`.
   Compare the result with the table above.

The baseline itself does not change. The `dev/` parity scripts still compare
it with SuperSegger.
