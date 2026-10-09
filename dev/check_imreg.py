"""Check nd2_to_cells.imreg against MATLAB probe results for one split dump.

Usage: python dev/check_imreg.py <split_*.mat>
"""

import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nd2_to_cells import imreg  # noqa: E402

d = sio.loadmat(sys.argv[1], squeeze_me=True)
rl, rlr = d["rl"].astype(int), d["rlr"].astype(int)
reg_c, reg_r = int(d["regC"]), np.atleast_1d(d["regR"]).astype(int)
fixed = (rl == reg_c).astype(float)
moving = (rlr == reg_r[0]) + 2.0 * (rlr == reg_r[1])
moving = moving / moving.max()

# Gradient magnitude at the identity, per level (GradientMagnitudeTolerance 1e10).
saved = imreg.GRAD_TOL
imreg.GRAD_TOL = 1e10
mags = []
orig = imreg._optimize


def probe(fix, mov, p, centre, scales, *a):
    _, g = imreg._metric(fix, mov, p, centre)
    mags.append(float(np.linalg.norm(g / scales)))
    return p


imreg._optimize = probe
imreg.register(moving, fixed)
imreg._optimize = orig
imreg.GRAD_TOL = saved
print("gradient magnitude per level:", [f"{m:.7g}" for m in mags])

for its in (1, 100):
    trace = []
    t0 = time.time()
    p = imreg.register(moving, fixed, max_iter=its, trace=trace)
    T = imreg.to_matlab_tform(p, fixed.shape)
    t_str = np.array2string(T, precision=6, suppress_small=True)
    print(f"max_iter {its}: {time.time() - t0:.1f} s, T = {t_str}")
    for lev in range(3):
        vals = [v for lv, v in trace if lv == lev]
        first = [f"{v:.8f}" for v in vals[:4]]
        print(f"  level {lev + 1}: {len(vals)} its, first {first}")
