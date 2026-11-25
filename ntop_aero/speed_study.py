"""Speed against accuracy of the WoSX lift run on one body: the same run with fewer start points, fewer walks,
shorter walks, start points weighted to the trailing edge, or another grid.

  python speed_study.py V03 0.7 --grid geo/V03_M0.70_grid.npz --tag a 240000:10 60000:10 60000:10:imp 60000:10:cap2000

A configuration is cache:walks, then optional words: imp (weighted start points), capN (walks cut at N steps),
rrX (Russian roulette threshold X). Each is run --repeat times. Read-out: 'edge, 0.16 to 1.0 m' of kutta_study.py.
Every row records if another compute job was on the GPU, because that changes the times.
-> results/speed_<key>_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time

import numpy as np

from paths import other_job_running
from run_final import D_ALL
from study_noise import kutta_edge
from wos_lift import HERE, WosLift

DEG = math.pi / 180
SEL = list(range(4, len(D_ALL)))                         # 0.16 to 1.0 m


def gpu_shared():
    """Other compute work seen before a run (the 'busy_process' of paths.py, GPU load): the times are not clean if any."""
    try:
        u = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20).stdout.strip()
        return dict(other_job_running=other_job_running(), gpu_util_and_mem_before=u)
    except Exception as e:                               # noqa: BLE001
        return dict(error=str(e))


def read_out(S, Vb):
    ns = S.lat.ns
    chord = (S.lat.xte_mid - S.lat.xle_mid) * S.beta
    rows = np.concatenate([np.arange(j * ns, (j + 1) * ns) for j in SEL])
    gx, gz = kutta_edge(Vb[rows], ns, np.array(D_ALL)[SEL], chord)
    dy, Sref = S.lat.dy, S.ref['s_ref_m2']
    lin = 4.0 * (gz * dy).sum() * S.R / Sref             # small-angle lift slope per rad
    cl0 = 4.0 * (gx * dy).sum() * S.R / S.beta / Sref
    return float(lin), float(cl0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('key')
    ap.add_argument('mach', type=float)
    ap.add_argument('configs', nargs='+')
    ap.add_argument('--grid', default='')
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--tag', default='a')
    ap.add_argument('--imp-delta', type=float, default=0.10)
    ap.add_argument('--imp-uniform', type=float, default=0.3)
    a = ap.parse_args()
    grid = a.grid or f'geo/{a.key}_M{a.mach:.2f}_grid.npz'
    out = dict(key=a.key, mach=a.mach, grid=grid, read_out='edge, 0.16 to 1.0 m', rows=[])
    solvers = {}
    name = HERE / 'results' / f'speed_{a.key}_{a.tag}.json'
    for cfg in a.configs:
        w = cfg.split(':')
        cache, walks, opts = int(w[0]), int(w[1]), w[2:]
        imp = 'imp' in opts
        cap = next((int(o[3:]) for o in opts if o.startswith('cap')), 10000)
        rr = next((float(o[2:]) for o in opts if o.startswith('rr')), 0.99)
        if imp not in solvers:
            t = time.perf_counter()
            S = WosLift(a.key, channels=16, mach=a.mach, implicit=grid,
                        importance=dict(d=np.array(D_ALL)[SEL], delta=a.imp_delta, uniform=a.imp_uniform) if imp else None)
            solvers[imp] = (S, dict(S.T.rec, wall=time.perf_counter() - t))
        S, init = solvers[imp]
        pts = np.vstack([S.kutta_points(d / S.beta) for d in D_ALL])
        runs = []
        for r in range(a.repeat):
            sh = gpu_shared()
            S.T.rec.clear(); S.T.lap('x')
            t = time.perf_counter()
            u, g = S.solve(pts, cache=cache, walks=walks, max_walk=cap, rr=rr)
            Vb = S.basis_velocity(pts, g)
            lin, cl0 = read_out(S, Vb)
            dt = time.perf_counter() - t
            runs.append(dict(seconds=dt, timing={k: v for k, v in S.T.rec.items() if k != 'x'}, lin=lin, cl0=cl0, gpu=sh))
        L = np.array([q['lin'] for q in runs]); C = np.array([q['cl0'] for q in runs])
        row = dict(config=cfg, cache=cache, walks=walks, importance=imp, max_walk=cap, rr=rr, init=init, runs=runs,
                   lin_per_deg=float(L.mean() * DEG), lin_std_per_deg=float(L.std(ddof=1) * DEG) if len(L) > 1 else None, cl0=float(C.mean()),
                   cl0_std=float(C.std(ddof=1)) if len(C) > 1 else None,
                   seconds=float(np.mean([q['seconds'] for q in runs])), walk_seconds=float(np.mean([q['timing']['gpu_walks'] for q in runs])),
                   setup_seconds=float(np.mean([q['timing']['gpu_solver_setup'] for q in runs])), splat_seconds=float(np.mean([q['timing']['gpu_splat'] for q in runs])),
                   gpu_shared=any(q['gpu'].get('other_job_running') for q in runs))
        out['rows'].append(row)
        name.write_text(json.dumps(out, indent=1))
        sd = row['lin_std_per_deg']
        print(f"{cfg:26s} slope {row['lin_per_deg']:.5f} +- {0 if sd is None else sd:.5f} /deg  CL0 {row['cl0']:+.4f}  run {row['seconds']:6.1f} s "
              f"(set-up {row['setup_seconds']:.1f}, walks {row['walk_seconds']:.1f}, splat {row['splat_seconds']:.1f})  shared GPU: {row['gpu_shared']}", flush=True)
