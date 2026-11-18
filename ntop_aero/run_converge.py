"""The same slice solved with more and more random walks: how the picture and the lift settle.

  python run_converge.py   -> results/converge_V03.npz
"""
from __future__ import annotations

import json
import time

import numpy as np

from study_noise import D_KUTTA, kutta_lsq
from wos_lift import HERE, WosLift, sdf_eval

LEVELS = [(2000, 1), (8000, 1), (30000, 1), (60000, 2), (60000, 10), (250000, 10)]

if __name__ == '__main__':
    S = WosLift('V03', 'geo/V03_h20_q150k.npz', 16)
    xs, zs = np.arange(-1.0, 16.01, 0.11), np.arange(-3.3, 3.31, 0.11)
    X, Z = np.meshgrid(xs, zs, indexing='ij')
    sl = np.c_[X.ravel(), np.full(X.size, 3.4), Z.ravel()]
    sdf, _, _ = sdf_eval('V03', sl, 'conv')
    ok = sdf > 0.03
    kp = np.vstack([S.kutta_points(d) for d in D_KUTTA])
    pts = np.vstack([sl[ok], kp])
    out = dict(pts=sl.astype(np.float32), shape=np.array(X.shape), sdf=sdf, ok=ok)
    info = []
    for i, (cache, walks) in enumerate(LEVELS):
        t = time.perf_counter()
        u, g = S.solve(pts, cache=cache, walks=walks)
        dt = time.perf_counter() - t
        Vb = S.basis_velocity(pts, g)
        gx, gz = kutta_lsq(S, Vb[-len(kp):], len(D_KUTTA))
        full = np.zeros((len(sl), Vb.shape[1], 3), np.float32)
        full[ok] = Vb[:-len(kp)]
        out[f'Vb_{i}'] = full
        out[f'gx_{i}'], out[f'gz_{i}'] = gx * S.R, gz * S.R
        f = [S.forces(gx, gz, a) for a in range(0, 13)]
        info.append(dict(cache=cache, walks=walks, total_walks=cache * walks, seconds=dt, CL=[p['CL'] for p in f], CDi=[p['CDi'] for p in f]))
        print(f'{cache} x {walks}: {dt:.1f} s  CL4 {f[4]["CL"]:.4f}  CL8 {f[8]["CL"]:.4f}', flush=True)
    out['info'] = json.dumps(dict(levels=info, R=S.R))
    np.savez_compressed(HERE / 'results' / 'converge_V03.npz', **out)
