"""One GPU run that returns the flow everywhere it is asked for (slices, a shell just off the skin, a coarse
3-D box for particles), for every channel. Any angle of attack is a linear combination afterwards.

  python run_fields.py --cache 400000 --walks 3
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from study_noise import D_KUTTA, kutta_lsq
from wos_lift import HERE, WosLift, sdf_eval
from wos_surface import sample_on_mesh

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mesh', default='geo/V03_h20_q150k.npz')
    ap.add_argument('--cache', type=int, default=400000)
    ap.add_argument('--walks', type=int, default=3)
    ap.add_argument('--shell', type=float, default=0.06)
    ap.add_argument('--nshell', type=int, default=60000)
    ap.add_argument('--tag', default='fields')
    a = ap.parse_args()
    t0 = time.perf_counter()
    S = WosLift('V03', a.mesh, 16)
    rng = np.random.default_rng(7)
    sets = {}

    def grid(xs, ys, zs):
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        return np.c_[X.ravel(), Y.ravel(), Z.ravel()], X.shape

    xs, zs = np.arange(-4.0, 18.01, 0.11), np.arange(-4.5, 4.51, 0.11)
    sets['slice_y34'] = grid(xs, [3.4], zs)
    sets['slice_y0'] = grid(xs, [0.0], zs)
    sets['plane_x136'] = grid([13.6], np.arange(-13.0, 13.01, 0.13), np.arange(-5.0, 5.01, 0.13))
    sets['box'] = grid(np.arange(-4.0, 20.01, 0.4), np.arange(0.0, 12.01, 0.4), np.arange(-4.0, 4.01, 0.4))
    surf, nrm = sample_on_mesh(S.V, S.F, np.ones(len(S.F), bool), a.nshell, rng)
    # exact wall normal from the nTop field, then the shell point a fixed distance off the skin
    val, grad, tq = sdf_eval('V03', surf, 'surf')
    n_sdf = grad / np.maximum(np.linalg.norm(grad, axis=1), 1e-9)[:, None]
    shell = surf + a.shell * n_sdf
    sets['shell'] = (shell, (len(shell),))
    kp = np.vstack([S.kutta_points(d) for d in D_KUTTA])
    sets['kutta'] = (kp, (len(kp),))
    names = list(sets)
    allp = np.vstack([sets[k][0] for k in names])
    sdf, _, tq2 = sdf_eval('V03', allp, 'all')
    ok = sdf > 0.03
    print(f'{len(allp)} points, {ok.sum()} outside the body; SDF queries {tq["wall"]:.1f} + {tq2["wall"]:.1f} s', flush=True)
    S.T.lap('points_and_sdf')
    u, g = S.solve(allp[ok], cache=a.cache, walks=a.walks)
    Vb_ok = S.basis_velocity(allp[ok], g)
    S.T.lap('lattice_velocity_cpu')
    Vb = np.zeros((len(allp), Vb_ok.shape[1], 3), np.float32)
    Vb[ok] = Vb_ok
    U = np.zeros((len(allp), u.shape[1]), np.float32)
    U[ok] = u
    out, i0 = {}, 0
    for k in names:
        p, shp = sets[k]
        sl = slice(i0, i0 + len(p)); i0 += len(p)
        out[k + '_pts'] = p.astype(np.float32); out[k + '_shape'] = np.array(shp)
        out[k + '_Vb'] = Vb[sl]; out[k + '_sdf'] = sdf[sl]; out[k + '_u'] = U[sl]
    gx, gz = kutta_lsq(S, Vb[-len(kp):].astype(np.float64), len(D_KUTTA))
    out['gx'], out['gz'] = gx, gz
    out['surf_pts'], out['surf_normals'] = surf.astype(np.float32), n_sdf.astype(np.float32)
    S.T.rec['total_wall'] = time.perf_counter() - t0
    out['meta'] = json.dumps(dict(timing=S.T.rec, cache=a.cache, walks=a.walks, R=S.R, centre=S.c.tolist(), core_m=S.core * S.R, shell_m=a.shell,
                                  lattice_A=S.lat.A.tolist(), lattice_B=S.lat.B.tolist(), ns=S.lat.ns))
    np.savez_compressed(HERE / 'results' / f'{a.tag}_V03.npz', **out)
    print(json.dumps({k: round(v, 1) for k, v in S.T.rec.items()}))
    for al in (0, 4, 8):
        print(al, {k: round(v, 4) for k, v in S.forces(gx, gz, al).items() if k in ('CL', 'CDi', 'e')})
