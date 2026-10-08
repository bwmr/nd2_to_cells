"""Load SuperSegger reference output (seg/err .mat files) for parity checks.

Not part of the package. Run the parity scripts with the repo root on
sys.path, e.g. ``python dev/parity_assign.py <xy dir>``.
"""

from pathlib import Path

import numpy as np
import scipy.io as sio


def seg_files(xy_dir: Path, kind: str) -> list[Path]:
    """Sorted *_seg.mat or *_err.mat files of a position (MATLAB dir order)."""
    return sorted((Path(xy_dir) / "seg").glob(f"*_{kind}.mat"))


def load_regs(path: Path, extra: tuple[str, ...] = ()):
    """Return the regs struct (scipy mat_struct) of a seg/err file."""
    m = sio.loadmat(
        path,
        squeeze_me=True,
        struct_as_record=False,
        mat_dtype=True,
        variable_names=["regs", *extra],
    )
    return m["regs"] if not extra else m


def cell_list(x, n: int) -> list[list[int]]:
    """MATLAB cell array of vectors -> 1-based list of int lists (index 0 unused)."""
    if n == 0:
        return [[]]
    if not (isinstance(x, np.ndarray) and x.dtype == object):
        x = np.array([x], dtype=object) if n == 1 else x
    out: list[list[int]] = [[]]
    for v in np.atleast_1d(x):
        out.append([int(e) for e in np.atleast_1d(np.asarray(v, dtype=float)).ravel()])
    return out


def vec(x) -> np.ndarray:
    """MATLAB row vector -> 1-D float array (handles squeezed scalars/empties)."""
    return np.atleast_1d(np.asarray(x, dtype=float)).ravel()


def ids(x) -> np.ndarray:
    """MATLAB 2xN index matrix -> (2, N) float array."""
    a = np.asarray(x, dtype=float)
    return a.reshape(2, -1) if a.size else np.zeros((2, 0))


def props_arrays(regs):
    """regs.props -> (area, centroid, bbox) with an unused row 0."""
    n = int(regs.num_regs)
    props = np.atleast_1d(regs.props) if n else []
    area = np.zeros(n + 1)
    cent = np.full((n + 1, 2), np.nan)
    bbox = np.zeros((n + 1, 4))
    for k, p in enumerate(props, start=1):
        area[k] = float(p.Area)
        cent[k] = vec(p.Centroid)
        bbox[k] = vec(p.BoundingBox)
    return area, cent, bbox


def nan_equal(a: np.ndarray, b: np.ndarray, rtol: float = 0.0) -> bool:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return False
    both_nan = np.isnan(a) & np.isnan(b)
    with np.errstate(invalid="ignore"):
        close = np.isclose(a, b, rtol=rtol, atol=0.0) | (a == b)
    return bool(np.all(both_nan | close))
