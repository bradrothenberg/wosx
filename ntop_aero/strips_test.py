"""Does the WoSX lift slope depend on the number of wake strips? The module is built for 16 channels, so at most 13
strips per side fit in one run; fewer strips show the trend.

  python strips_test.py V03 0.7 geo/V03_M0.70_grid_f.npz 6 9 13   -> results/strips_<key>.json
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from run_final import D_ALL
from speed_study import DEG, read_out
from wos_lift import HERE, WosLift

if __name__ == '__main__':
    key, mach, grid = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    out = dict(key=key, mach=mach, grid=grid, settings=dict(cache=60000, walks=10, max_walk=4000, repeats=3), rows=[])
    for ns in [int(q) for q in sys.argv[4:]]:
        for spacing in (('uniform', 'cosine') if ns == 13 else ('uniform',)):
            S = WosLift(key, channels=16, ns=ns, mach=mach, implicit=grid, importance=dict(d=D_ALL[4:]), spacing=spacing)
            pts = np.vstack([S.kutta_points(d / S.beta) for d in D_ALL])
            res = []
            for r in range(3):
                t = time.perf_counter()
                u, g = S.solve(pts, cache=60000, walks=10, max_walk=4000)
                res.append(read_out(S, S.basis_velocity(pts, g)) + (time.perf_counter() - t,))
            res = np.array(res)
            row = dict(strips=ns, spacing=spacing, lin_per_deg=float(res[:, 0].mean() * DEG), lin_std_per_deg=float(res[:, 0].std(ddof=1) * DEG), cl0=float(res[:, 1].mean()),
                       cl0_std=float(res[:, 1].std(ddof=1)), seconds=float(res[:, 2].mean()))
            out['rows'].append(row)
            print(f"{key} strips {ns:2d} {spacing}: slope {row['lin_per_deg']:.5f} +- {row['lin_std_per_deg']:.5f} /deg  CL0 {row['cl0']:+.4f} +- {row['cl0_std']:.4f}  {row['seconds']:.1f} s", flush=True)
            (HERE / 'results' / f'strips_{key}.json').write_text(json.dumps(out, indent=1))
