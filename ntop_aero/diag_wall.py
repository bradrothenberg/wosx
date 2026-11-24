"""Mesh wall against implicit wall with the SAME direct random walks (no boundary value caching).

Walk on Stars is run from the same points in the flow with the two kinds of wall. The potential of every channel
should agree within the scatter of the walks. Also gives the mean walk length and the time of each.

  python diag_wall.py W12T12 0.0 --walks 2000
  python diag_wall.py V03 0.7 --walks 2000   -> results/diag_wall_<key>_M<mach>.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import bodies
from wos_lift import HERE, WosLift


def walk(S, y, walks, eps=1e-4, max_walk=10000, rr=0.99):
    """Direct walks from points y (walk space). Returns potential [n, ch] (walk space), mean walk length, counts, seconds."""
    wosx = S.wosx
    ch = dict(dim=3, channels=S.channels)
    sp = []
    for q in y:
        p = wosx.Solvers.GPUSamplePoint(dim=3)
        p.pt = wosx.GPUFloat3(float(q[0]), float(q[1]), float(q[2]))
        p.normal = wosx.GPUFloat3(0.0, 0.0, 0.0)
        p.type = wosx.Solvers.SampleType.InDomain
        p.estimation_quantity = wosx.Solvers.EstimationQuantity.Solution
        sp.append(p)
    sp = wosx.Solvers.GPUSamplePointList(sp, dim=3)
    ws = wosx.Solvers.GPUWalkSettings()
    ws.epsilon_shell_for_absorbing_boundary = eps
    ws.epsilon_shell_for_reflecting_boundary = eps
    ws.silhouette_precision = eps
    ws.russian_roulette_threshold = rr
    ws.max_walk_length = max_walk
    ws.steps_before_using_maximal_spheres = max_walk
    ws.solve_double_sided = 0
    ws.ignore_absorbing_boundary_contribution = 1
    ws.ignore_reflecting_boundary_contribution = 0
    ws.ignore_source_contribution = 1
    solver = wosx.Solvers.GPUWalkOnStarsSolver(S.task, ws, 131072, True, False, **ch)
    solver.populate_sample_points(sp)
    t = time.perf_counter()
    solver.solve(walks)
    dt = time.perf_counter() - t
    st = wosx.Solvers.GPUSampleStatisticsList(**ch)
    solver.get_sample_statistics(st)
    n = len(y)
    u = np.zeros((n, S.channels)); wl = np.zeros(n); cnt = np.zeros(n)
    for i in range(n):
        u[i] = np.asarray(st[i].get_estimated_solution()).reshape(-1)
        wl[i] = st[i].get_mean_walk_length()
        cnt[i] = st[i].get_solution_estimate_count()
    return u, wl, cnt, dt


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('key')
    ap.add_argument('mach', type=float)
    ap.add_argument('--walks', type=int, default=2000)
    ap.add_argument('--points', type=int, default=240)
    ap.add_argument('--grid', default='')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    key = a.key.upper()
    grid = a.grid or f'geo/{key}_M{a.mach:.2f}_grid.npz'
    mesh, _ = bodies.ensure_mesh(key)
    M = WosLift(key, mesh, 16, mach=a.mach, kernel='exp')
    # points in the flow at set distances from the wall, along the outward normal of wall points of the field
    g = np.load(HERE / grid)
    rng = np.random.default_rng(1)
    dist = [0.005, 0.02, 0.08, 0.3, 1.0]                 # m
    j = rng.choice(len(g['pts']), a.points, replace=False)
    pts = np.concatenate([g['pts'][j[k::len(dist)]] + d * g['nrm'][j[k::len(dist)]] for k, d in enumerate(dist)]).astype(np.float64)
    lab = np.concatenate([np.full(len(j[k::len(dist)]), d) for k, d in enumerate(dist)])
    x = M.to_n(pts); y = (x / (x * x).sum(1)[:, None]).astype(np.float32)
    um, wlm, cm, tm = walk(M, y, a.walks)
    mesh_R = M.R
    I = WosLift(key, channels=16, mach=a.mach, kernel='exp', implicit=grid)
    xi = I.to_n(pts); yi = (xi / (xi * xi).sum(1)[:, None]).astype(np.float32)
    ui, wli, ci, ti = walk(I, yi, a.walks)
    # Kelvin transform back: potential = (walk-space value) * |y|, in the length unit of each run
    pm = um * np.linalg.norm(y, axis=1)[:, None]
    pi = ui * np.linalg.norm(yi, axis=1)[:, None]
    rows = []
    out = dict(key=key, mach=a.mach, walks=a.walks, grid=grid, mesh=mesh, seconds_mesh=tm, seconds_implicit=ti, rows=rows)
    print(f'mesh {tm:.1f} s   implicit {ti:.1f} s   walks {a.walks}')
    for d in dist:
        m = lab == d
        # potentials in metres per unit onset: normalised potential times the body radius of each run
        A = pm[m] * mesh_R; B = pi[m] * I.R
        row = dict(distance_m=d, points=int(m.sum()), walk_length_mesh=float(wlm[m].mean()), walk_length_implicit=float(wli[m].mean()),
                   walks_counted_mesh=float(cm[m].mean()), walks_counted_implicit=float(ci[m].mean()))
        for name, c in (('x_onset', 0), ('z_onset', 1), ('strip_mid', 2 + (A.shape[1] - 3) // 2)):
            row[name] = dict(ratio=float((A[:, c] * B[:, c]).sum() / (A[:, c] ** 2).sum()), rms_mesh=float(np.sqrt((A[:, c] ** 2).mean())),
                             rms_diff=float(np.sqrt(((A[:, c] - B[:, c]) ** 2).mean())))
        rows.append(row)
        print(f"  d {d:5.3f} m  walk length mesh {row['walk_length_mesh']:7.1f} implicit {row['walk_length_implicit']:7.1f}  counted {row['walks_counted_mesh']:.0f}/{row['walks_counted_implicit']:.0f}  "
              + '  '.join(f"{k}: ratio {row[k]['ratio']:.4f} diff {row[k]['rms_diff'] / max(row[k]['rms_mesh'], 1e-30):.4f}" for k in ('x_onset', 'z_onset', 'strip_mid')), flush=True)
    (HERE / 'results' / f'diag_wall_{key}_M{a.mach:.1f}{a.tag}.json').write_text(json.dumps(out, indent=1))
