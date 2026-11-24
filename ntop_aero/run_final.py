"""Alpha sweep of one implicit body with error bars: repeated independent GPU runs at one Mach number.

  python run_final.py --key V05 --mach 0.7 --repeat 8

The body is named by its key (a delta variant of the trade study or a test wing of bodies.py). If its surface
triangles are not in geo/ yet they are made from the field first (bodies.ensure_mesh), so this one command goes
from the implicit body to the lift curve.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import bodies
from study_noise import D_KUTTA, kutta_edge, kutta_lsq
from wos_lift import HERE, WosLift

ALPHAS = [float(a) for a in range(0, 13)]
D_FAR = (0.35, 0.50, 0.70, 1.00)      # extra read-out distances behind the edge, m: saved for later study, not used for the forces
D_ALL = tuple(D_KUTTA) + D_FAR

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', default='V03')
    ap.add_argument('--mesh', default='')
    ap.add_argument('--mach', type=float, default=0.0)
    ap.add_argument('--cache', type=int, default=125000)
    ap.add_argument('--walks', type=int, default=5)
    ap.add_argument('--repeat', type=int, default=8)
    ap.add_argument('--channels', type=int, default=16)
    ap.add_argument('--tag', default='')
    ap.add_argument('--core', type=float, default=0.05, help='vortex core radius, m')
    ap.add_argument('--kernel', default='alg', choices=('alg', 'exp'), help="vortex core model: 'alg' (report so far) or 'exp' (dies out fast)")
    ap.add_argument('--field-device', default=None, help="'cpu' builds the triangles on the processor")
    ap.add_argument('--implicit', default='', help="wall from the implicit, no triangles: a grid file of implicit_grid.py, or 'auto' for geo/KEY_Mx.xx_grid.npz")
    a = ap.parse_args()
    a.key = a.key.upper()
    mesh_seconds = None
    if a.implicit == 'auto':
        a.implicit = f'geo/{a.key}_M{a.mach:.2f}_grid.npz'
    if not a.mesh and not a.implicit:
        a.mesh, mesh_seconds = bodies.ensure_mesh(a.key, device=a.field_device)
    t0 = time.perf_counter()
    S = WosLift(a.key, a.mesh, a.channels, mach=a.mach, core_m=a.core, kernel=a.kernel, implicit=a.implicit or None)
    init = dict(S.T.rec)
    init['wall'] = time.perf_counter() - t0
    init['field_to_triangles'] = mesh_seconds          # None: triangles were already in geo/
    pts = np.vstack([S.kutta_points(d / S.beta) for d in D_ALL])
    n7 = len(D_KUTTA) * S.lat.ns
    runs = []
    chord = (S.lat.xte_mid - S.lat.xle_mid) * S.beta                 # true local chord of each strip, m
    kdir = HERE / 'results' / 'kutta'
    kdir.mkdir(exist_ok=True)
    for r in range(a.repeat):
        S.T.rec.clear(); S.T.lap('x')
        t = time.perf_counter()
        u, g = S.solve(pts, cache=a.cache, walks=a.walks)
        Vb = S.basis_velocity(pts, g)
        gx, gz = kutta_lsq(S, Vb[:n7], len(D_KUTTA))
        dt = time.perf_counter() - t
        f = [S.forces(gx, gz, al) for al in ALPHAS]
        ex, ez = kutta_edge(Vb[:n7], S.lat.ns, D_KUTTA, chord)        # second read-out of the same run: Kutta condition at the edge itself
        fe = [S.forces(ex, ez, al) for al in ALPHAS]
        np.savez_compressed(kdir / f'final_{a.key}_M{a.mach:.1f}{a.tag}_r{r}.npz', Vb=Vb.astype(np.float32), d=np.array(D_ALL), chord=chord, R=S.R, beta=S.beta)
        runs.append(dict(seconds=dt, timing={k: v for k, v in S.T.rec.items() if k != 'x'}, gx=(gx * S.R / S.beta).tolist(), gz=(gz * S.R / S.beta).tolist(),
                         CL=[p['CL'] for p in f], CDi=[p['CDi'] for p in f], e=[p['e'] for p in f], span_ccl=[p['span_ccl'] for p in f],
                         gx_edge=(ex * S.R / S.beta).tolist(), gz_edge=(ez * S.R / S.beta).tolist(), CL_edge=[p['CL'] for p in fe], CDi_edge=[p['CDi'] for p in fe]))
        print(f'run {r}: {dt:.1f} s  CL0 {f[0]["CL"]:+.4f}  CL4 {f[4]["CL"]:.4f}  CL8 {f[8]["CL"]:.4f}  CDi8 {f[8]["CDi"]:.5f}  | edge: CL0 {fe[0]["CL"]:+.4f}  CL8 {fe[8]["CL"]:.4f}', flush=True)
    CL = np.array([r['CL'] for r in runs]); CD = np.array([r['CDi'] for r in runs])
    out = dict(variant=a.key, mach=a.mach, beta=S.beta, mesh=S.mesh_meta, settings=vars(a), init=init, alphas_deg=ALPHAS, strips_per_side=S.lat.ns,
               span_y=S.lat.ym.tolist(), runs=runs,
               CL_mean=CL.mean(0).tolist(), CL_std=CL.std(0, ddof=1).tolist(), CDi_mean=CD.mean(0).tolist(), CDi_std=CD.std(0, ddof=1).tolist(),
               seconds_per_run=float(np.mean([r['seconds'] for r in runs])),
               reference=S.ref, method='WoSX GPU boundary value caching, 16 channels, horseshoe wake + Kutta, forces from the Trefftz plane'
               + ('; wall from the implicit on grids, no triangles' if a.implicit else '')
               + ('' if a.mach == 0 else '; Mach number by the Goethert rule (body stretched in x by 1/beta)'))
    (HERE / 'results' / f'final_{a.key}_M{a.mach:.1f}{a.tag}.json').write_text(json.dumps(out, indent=1))
    print('mean CL', np.round(CL.mean(0), 4).tolist()); print('std  CL', np.round(CL.std(0, ddof=1), 4).tolist())
    print('mean CDi', np.round(CD.mean(0), 5).tolist())
