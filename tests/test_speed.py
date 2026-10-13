"""Linking speed: 16 dense frames of xy01 (260512, 100XPa), frames 180-195.

The window needs 34 linking passes (error_rez edits masks in 15 of the 16
frames) and took 94 s before the assignment was sped up. The crop is the full
frame, because the cost scales with the image size.

- test_same_output: the lineage and the edited masks equal the golden output
  in tests/data/xy01_speed_golden.npz, made with the code before the
  speed-up (regenerate with ``python -m tests.test_speed``).
- test_colony_once_per_pass: the colony labelling in assign() runs at most
  once per linking pass, not three times.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from nd2_to_cells import assign as assign_mod
from nd2_to_cells import link as link_mod
from nd2_to_cells.link import link_frames
from nd2_to_cells.track import load_preset

pytestmark = pytest.mark.slow

DATA = Path(__file__).parent / "data"
FIELDS = ("ID", "motherID", "sisterID", "divide", "birth", "stat0")


def run() -> list:
    with np.load(DATA / "xy01_speed.npz") as d:
        masks = d["masks"]
    return link_frames(lambda i: masks[i], len(masks), load_preset("100XPa"))


def summarize(states) -> dict[str, np.ndarray]:
    out = {}
    for i, s in enumerate(states):
        for name in FIELDS:
            out[f"{i}_{name}"] = getattr(s, name)
        out[f"{i}_daughterID"] = np.array([",".join(map(str, d)) for d in s.daughterID])
        lab = s.edited_labels()
        out[f"{i}_edited"] = np.array(
            "" if lab is None else hashlib.sha1(np.ascontiguousarray(lab)).hexdigest()
        )
    return out


def test_same_output():
    got = summarize(run())
    with np.load(DATA / "xy01_speed_golden.npz") as golden:
        assert set(got) == set(golden.files)
        for key in golden.files:
            assert np.array_equal(got[key], golden[key]), key


def test_colony_once_per_pass(monkeypatch):
    fills, passes = [], []
    orig_fill = assign_mod.ndimage.binary_fill_holes
    orig_rez = link_mod.error_rez

    def counting_fill(x, *a, **k):
        fills.append(1)
        return orig_fill(x, *a, **k)

    def counting_rez(*a, **k):
        passes.append(1)
        return orig_rez(*a, **k)

    monkeypatch.setattr(assign_mod.ndimage, "binary_fill_holes", counting_fill)
    monkeypatch.setattr(link_mod, "error_rez", counting_rez)
    run()
    # Each pass has three assign() calls; the colonies are of data_c (new in
    # every pass) and data_r (kept from its own last pass).
    assert len(passes) > 30
    assert len(fills) <= len(passes)


if __name__ == "__main__":
    np.savez_compressed(DATA / "xy01_speed_golden.npz", **summarize(run()))
