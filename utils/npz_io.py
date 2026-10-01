"""
Sparse-COO .npz serialisation used by the CMRx4DFlow challenge for
reconstructed images (fields: coords, data, shape), following the organisers'
reference implementation (save_coo_npz / load_coo_npz).
"""

import os

import numpy as np


def save_coo_npz(path, arr):
    """
    Save a dense complex array as a sparse-COO-encoded, compressed .npz with
    fields: coords, data, shape.

    Parameters
    ----------
    path : str
        Output file path, e.g. ".../P006/img_ktGaussian20.npz".
    arr : np.ndarray
        Dense array to save (cast to complex64). For challenge submissions
        this is expected to already have the segmentation mask applied, so
        most background voxels are exactly zero and the sparse encoding is
        actually small.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    arr = np.asarray(arr).astype("complex64")
    coords = np.argwhere(arr != 0).astype(np.int32)
    data = arr[tuple(coords.T)] if coords.size else arr.reshape(-1)[:0]

    np.savez_compressed(
        path,
        coords=coords,
        data=data,
        shape=np.array(arr.shape, dtype=np.int64),
    )


def load_coo_npz(path, as_dense=True):
    """
    Load a sparse-COO .npz saved by save_coo_npz.

    Parameters
    ----------
    path : str
        File path to load.
    as_dense : bool
        If True (default), reconstructs and returns a dense ndarray.
        If False, returns the raw (coords, data, shape) tuple.
    """
    z = np.load(path)
    coords = z["coords"]
    data = z["data"]
    shape = tuple(z["shape"])

    if not as_dense:
        return coords, data, shape

    out = np.zeros(shape, dtype=data.dtype)
    if coords.size:
        out[tuple(coords.T)] = data
    return out
