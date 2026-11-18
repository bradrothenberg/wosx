"""Scatter and run time of the lifting solve against mesh size, cache size and walks per cache point.

  python study_noise.py geo/V03_h20_q150k.npz 30000:5 125000:5 500000:5 --repeat 3 --tag q150k
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from wos_lift import HERE, WosLift

ALPHAS = (0.0, 2.0, 4.0, 6.0, 8.0, 12.0)
D_KUTTA = (0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25)


def kutta_lsq(S, Vb, nd):
    """Least-squares Kutta over nd distances behind the edge. Vb: [nd * ns, 2 + ns, 3]."""
    A = Vb[:, 2:, 2]
    gx = np.linalg.lstsq(A, -Vb[:, 0, 2], rcond=None)[0]
    gz = np.linalg.lstsq(A, -Vb[:, 1, 2], rcond=None)[0]
    return gx, gz


def kutta_edge(Vb, ns, d, chord):
    """Kutta condition taken to the edge itself. kutta_lsq asks for no flow through the wake plane AT the read-out
    points, 0.06 to 0.25 m behind the edge. In the true flow that velocity is zero only at the edge and grows like
    sqrt(d / (d + c)) behind it (flat-plate form, c = local chord), so kutta_lsq acts like a longer chord and gives too
    much circulation (in 2-D: a factor 1 + 2 d / c). Here the velocity behind strip i is allowed to be
    B_i * sqrt(d / (d + c_i)), with B_i unknown; what must vanish is the part that grows toward the edge.
    Vb: [nd * ns, 2 + ns, 3], rows ordered distance by distance; d: [nd] and chord: [ns] in the same length unit."""
    d, chord = np.asarray(d, float), np.asarray(chord, float)
    nd = len(d)
    A = Vb[:, 2:, 2]
    F = np.zeros((nd * ns, ns))
    for j in range(nd):
        F[j * ns + np.arange(ns), np.arange(ns)] = -np.sqrt(d[j] / (d[j] + chord))
    M = np.hstack([A, F])
    gx = np.linalg.lstsq(M, -Vb[:, 0, 2], rcond=None)[0][:ns]
    gz = np.linalg.lstsq(M, -Vb[:, 1, 2], rcond=None)[0][:ns]
    return gx, gz


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mesh')
    ap.add_argument('configs', nargs='+', help='cache:walks')
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--tag', default='noise')
    ap.add_argument('--eps', type=float, default=1e-4)
    ap.add_argument('--channels', type=int, default=16)
    ap.add_argument('--xi', type=float, nargs='+', default=[0.25])
    ap.add_argument('--core', type=float, default=0.05)
    ap.add_argument('--mach', type=float, default=0.0)
    a = ap.parse_args()
    t0 = time.perf_counter()
    S = WosLift('V03', a.mesh, a.channels, xi=a.xi, w=[1.0 / len(a.xi)] * len(a.xi), core_m=a.core, mach=a.mach)
    init = time.perf_counter() - t0
    pts = np.vstack([S.kutta_points(d) for d in D_KUTTA])
    out = dict(mesh=S.mesh_meta, init_seconds=init, configs=[])
    for cfg in a.configs:
        cache, walks = (int(v) for v in cfg.split(':'))
        rows = []
        for r in range(a.repeat):
            S.T.rec.clear()
            S.T.lap('x')
            t = time.perf_counter()
            u, g = S.solve(pts, cache=cache, walks=walks, eps=a.eps)
            dt = time.perf_counter() - t
            Vb = S.basis_velocity(pts, g)
            gx, gz = kutta_lsq(S, Vb, len(D_KUTTA))
            f = [S.forces(gx, gz, al) for al in ALPHAS]
            rows.append(dict(seconds=dt, timing=dict(S.T.rec), CL=[p['CL'] for p in f], CDi=[p['CDi'] for p in f], e=[p['e'] for p in f],
                             gx=(gx * S.R).tolist(), gz=(gz * S.R).tolist()))
        CL = np.array([r['CL'] for r in rows])
        CD = np.array([r['CDi'] for r in rows])
        sec = np.mean([r['seconds'] for r in rows])
        walk = np.mean([r['timing']['gpu_walks'] for r in rows])
        print(f'cache {cache:7d} x {walks:3d} walks: solve {sec:6.1f} s (walks {walk:6.1f} s)  CL4 {CL[:, 2].mean():.4f} +- {CL[:, 2].std(ddof=1) if len(CL) > 1 else 0:.4f}'
              f'  CL8 {CL[:, 4].mean():.4f} +- {CL[:, 4].std(ddof=1) if len(CL) > 1 else 0:.4f}  CDi8 {CD[:, 4].mean():.5f} +- {CD[:, 4].std(ddof=1) if len(CD) > 1 else 0:.5f}'
              f'  CL0 {CL[:, 0].mean():+.4f}', flush=True)
        out['configs'].append(dict(cache=cache, walks=walks, runs=rows))
    (HERE / 'results' / f'noise_{a.tag}.json').write_text(json.dumps(out, indent=1))
    print('init', round(init, 1), 's')
