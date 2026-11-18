"""Median filter for the near-wall velocity samples (the raw boundary-value-caching values next to the skin are very noisy)."""
import numpy as np
from scipy.spatial import cKDTree


def median_velocity(pts, nz, V, q_pts, q_nz, k=64):
    """Median of the velocity over the k nearest samples on the same side (upper / lower) of the body."""
    out = np.zeros((len(q_pts), 3))
    for sgn in (1, -1):
        m = nz * sgn > 0
        mq = q_nz * sgn > 0
        if not mq.any():
            continue
        tree = cKDTree(pts[m])
        _, nn = tree.query(q_pts[mq], k=k)
        out[mq] = np.median(V[m][nn], axis=1)
    return out
