"""The sphere check of check_sphere.py with the wall taken from an implicit (exact distance on grids), no triangles.

  python check_sphere_implicit.py -> results/sphere_check_implicit.json
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

import implicit_wall as IW
from wos_lift import HERE, WosLift

if __name__ == '__main__':
    a, c = 5.0, np.array([6.0, 0.0, 0.15])
    meta = IW.sphere_grid(HERE / 'geo' / 'sphere_M0.00_grid.npz', a, c)
    t = time.perf_counter()
    S = WosLift('V03', channels=16, implicit='geo/sphere_M0.00_grid.npz', print_logs='-v' in sys.argv)
    init = time.perf_counter() - t
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(400, 3)); dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    dist = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]                       # fractions of the radius off the wall
    pts = np.vstack([c + dirs * a * (1 + d) for d in dist])
    rows = []
    for cache, walks in ((15000, 10), (60000, 10)):
        t = time.perf_counter()
        u, g = S.solve(pts, cache=cache, walks=walks)
        dt = time.perf_counter() - t
        x = pts - c
        r = np.linalg.norm(x, axis=1)
        ex = np.zeros_like(x)
        ex[:, 0] = a ** 3 / (2 * r ** 3)
        ex += -1.5 * a ** 3 * (x[:, 0] / r ** 5)[:, None] * x
        err = np.linalg.norm(g[:, 0] - ex, axis=1)                       # as a fraction of flight speed
        uex = a ** 3 * x[:, 0] / (2 * r ** 3) * S.scale                  # exact perturbation potential, normalised units
        n = len(dirs)
        per = []
        for i, d in enumerate(dist):
            e = err[i * n:(i + 1) * n]
            sl = slice(i * n, (i + 1) * n)
            fit = float((u[sl, 0] * uex[sl]).sum() / (uex[sl] ** 2).sum())
            per.append(dict(distance_over_radius=d, rms_error=float(np.sqrt((e ** 2).mean())), median_error=float(np.median(e)),
                            p95_error=float(np.percentile(e, 95)), potential_ratio_to_exact=fit))
        rows.append(dict(cache=cache, walks=walks, seconds=dt, by_distance=per))
        print(cache, walks, f'{dt:.1f} s', [(p['distance_over_radius'], round(p['rms_error'], 4), round(p['potential_ratio_to_exact'], 4)) for p in per], flush=True)
    (HERE / 'results' / 'sphere_check_implicit.json').write_text(json.dumps(dict(radius_m=a, grid=meta, init_seconds=init, points_per_distance=len(dirs), runs=rows,
        note='error = |computed - exact| perturbation velocity as a fraction of flight speed; potential ratio = least-squares ratio of computed to exact potential'), indent=1))
