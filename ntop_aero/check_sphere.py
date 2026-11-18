"""Check of the solver itself on a case with an exact answer: uniform flow past a sphere.

Exact: phi = U x (1 + a^3 / (2 r^3)). The GPU boundary-value-caching solve is compared with it at points at
several distances from the wall. Channel 0 of the lifting PDE is the plain x-onset problem of the stock demo.

  python check_sphere.py -> results/sphere_check.json
"""
from __future__ import annotations

import json
import time

import numpy as np
import trimesh

from wos_lift import HERE, WosLift

if __name__ == '__main__':
    a, c = 5.0, np.array([6.0, 0.0, 0.15])
    m = trimesh.creation.icosphere(subdivisions=6, radius=a)
    v = m.vertices + c
    (HERE / 'geo').mkdir(exist_ok=True)
    np.savez(HERE / 'geo' / 'sphere.npz', vertices=v.astype(np.float32), faces=m.faces.astype(np.int32),
             meta=json.dumps(dict(variant='sphere', faces=int(len(m.faces)), radius_m=a)))
    S = WosLift('V03', 'geo/sphere.npz', 16)
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
        # exact perturbation velocity for unit onset along x
        ex = np.zeros_like(x)
        ex[:, 0] = a ** 3 / (2 * r ** 3)
        ex += -1.5 * a ** 3 * (x[:, 0] / r ** 5)[:, None] * x
        err = np.linalg.norm(g[:, 0] - ex, axis=1)                       # as a fraction of flight speed
        vtot = np.linalg.norm(ex + np.array([1.0, 0, 0]), axis=1)
        n = len(dirs)
        per = []
        for i, d in enumerate(dist):
            e = err[i * n:(i + 1) * n]
            bias = np.linalg.norm((g[i * n:(i + 1) * n, 0] - ex[i * n:(i + 1) * n]).mean(0))
            per.append(dict(distance_over_radius=d, rms_error=float(np.sqrt((e ** 2).mean())), median_error=float(np.median(e)), p95_error=float(np.percentile(e, 95))))
        rows.append(dict(cache=cache, walks=walks, seconds=dt, by_distance=per))
        print(cache, walks, f'{dt:.1f} s', [(p['distance_over_radius'], round(p['rms_error'], 4)) for p in per], flush=True)
    (HERE / 'results' / 'sphere_check.json').write_text(json.dumps(dict(radius_m=a, faces=int(len(m.faces)), points_per_distance=len(dirs), runs=rows,
                                                                        note='error = |computed - exact| perturbation velocity as a fraction of flight speed'), indent=1))
