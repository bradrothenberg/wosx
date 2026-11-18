"""Different ways to read the trailing-edge (Kutta) condition from the SAME GPU runs.

run_final.py saves the velocities at the read-out points behind the edge for every run (results/kutta/*.npz):
11 distances from 0.06 to 1.0 m, 13 strips, every unknown. This script turns them into lift slopes with

  points  no flow through the wake plane AT the read-out points (the condition of the report up to section 09)
  edge    the velocity may grow like sqrt(d / (d + c)) behind the edge; only the part that grows toward the edge must vanish
  single  'points' at one distance only, to show how the answer moves with the distance

  python kutta_study.py W12T12 0.0 _c240
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from study_noise import kutta_edge

HERE = Path(__file__).resolve().parent
R = HERE / 'results'
DEG = math.pi / 180


def load(key, mach, tag):
    w = json.loads((R / f'final_{key}_M{mach:.1f}{tag}.json').read_text())
    runs = []
    for r in range(len(w['runs'])):
        f = R / 'kutta' / f'final_{key}_M{mach:.1f}{tag}_r{r}.npz'
        if f.exists():
            runs.append(np.load(f))
    return w, runs


def slope(w, k, gx, gz):
    """Lift slope per rad between 0 and 8 deg from normalised strip strengths."""
    beta, S = float(k['beta']), w['reference']['s_ref_m2']
    ym = np.array(w['span_y']); dy = 2 * ym[0]
    cl = []
    for al in (0.0, 8.0):
        a = math.atan(beta * math.tan(math.radians(al)))
        gam = (math.cos(a) * gx + math.sin(a) * gz) * float(k['R']) / beta
        cl.append(4.0 * (gam * dy).sum() / S)
    lin = 4.0 * (gz * dy).sum() * float(k['R']) / S       # small-angle slope per rad: d(alpha_transformed)/d(alpha) = beta at zero angle
    return (cl[1] - cl[0]) / 8.0 / DEG, cl[0], lin


def read(k, ns, sel, how):
    d = k['d'][sel]
    rows = np.concatenate([np.arange(j * ns, (j + 1) * ns) for j in sel])
    Vb = k['Vb'][rows].astype(float)
    if how == 'edge':
        return kutta_edge(Vb, ns, d, k['chord'])
    A = Vb[:, 2:, 2]
    return np.linalg.lstsq(A, -Vb[:, 0, 2], rcond=None)[0], np.linalg.lstsq(A, -Vb[:, 1, 2], rcond=None)[0]


def variants(nd):
    v = {'points, 0.06 to 0.25 m (report)': (list(range(7)), 'points'), 'edge, 0.06 to 0.25 m': (list(range(7)), 'edge')}
    if nd > 7:
        v['edge, 0.06 to 1.0 m'] = (list(range(nd)), 'edge')
        v['edge, 0.16 to 1.0 m'] = (list(range(4, nd)), 'edge')
        v['edge, 0.25 to 1.0 m'] = (list(range(6, nd)), 'edge')
        v['points, 0.35 to 1.0 m'] = (list(range(7, nd)), 'points')
    return v


def study(key, mach, tag, quiet=False):
    w, runs = load(key, mach, tag)
    ns = w['strips_per_side']
    out = dict(key=key, mach=mach, tag=tag, runs=len(runs), variants={}, single={})
    if not runs:
        return out
    nd = len(runs[0]['d'])
    for name, (sel, how) in variants(nd).items():
        res = np.array([slope(w, k, *read(k, ns, sel, how)) for k in runs])
        out['variants'][name] = dict(slope=float(res[:, 0].mean()), std=float(res[:, 0].std(ddof=1)) if len(res) > 1 else 0.0, se=float(res[:, 0].std(ddof=1) / math.sqrt(len(res))) if len(res) > 1 else 0.0,
                                     cl0=float(res[:, 1].mean()), cl0_std=float(res[:, 1].std(ddof=1)) if len(res) > 1 else 0.0, slope_runs=res[:, 0].tolist(),
                                     lin=float(res[:, 2].mean()), lin_se=float(res[:, 2].std(ddof=1) / math.sqrt(len(res))) if len(res) > 1 else 0.0, lin_runs=res[:, 2].tolist())
    if nd > 7:                                            # no model: 'points' at each distance from 0.16 m on, parabola in d, value at d = 0
        dd = np.array(runs[0]['d'][4:], float)
        ex = []
        for k in runs:
            sl = np.array([slope(w, k, *read(k, ns, [j], 'points')) for j in range(4, nd)])
            ex.append([np.polyfit(dd, sl[:, 0], 2)[-1], np.polyfit(dd, sl[:, 1], 2)[-1], np.polyfit(dd, sl[:, 2], 2)[-1]])
        ex = np.array(ex)
        out['variants']['points, parabola to the edge'] = dict(slope=float(ex[:, 0].mean()), std=float(ex[:, 0].std(ddof=1)) if len(ex) > 1 else 0.0,
                                                               se=float(ex[:, 0].std(ddof=1) / math.sqrt(len(ex))) if len(ex) > 1 else 0.0, cl0=float(ex[:, 1].mean()),
                                                               cl0_std=float(ex[:, 1].std(ddof=1)) if len(ex) > 1 else 0.0, slope_runs=ex[:, 0].tolist(),
                                                               lin=float(ex[:, 2].mean()), lin_se=float(ex[:, 2].std(ddof=1) / math.sqrt(len(ex))) if len(ex) > 1 else 0.0, lin_runs=ex[:, 2].tolist())
    for j in range(nd):
        res = np.array([slope(w, k, *read(k, ns, [j], 'points')) for k in runs])
        out['single'][f"{float(runs[0]['d'][j]):.2f}"] = dict(lin=float(res[:, 2].mean()), slope=float(res[:, 0].mean()), se=float(res[:, 0].std(ddof=1) / math.sqrt(len(res))) if len(res) > 1 else 0.0, cl0=float(res[:, 1].mean()))
    if not quiet:
        print(f'{key} Mach {mach} {tag}: {len(runs)} runs, slope per rad (per deg), 1 std of one run, CL at zero angle')
        for name, v in out['variants'].items():
            print(f"  {name:34s} {v['slope']:.4f} ({v['slope'] * DEG:.5f})  +- {v['std']:.4f}   CL0 {v['cl0']:+.4f} +- {v['cl0_std']:.4f}   small-angle {v['lin']:.4f} +- {v['lin_se']:.4f} (se)")
        print('  points at one distance:', '  '.join(f"{d} m: {v['slope']:.3f}" for d, v in out['single'].items()))
    return out


if __name__ == '__main__':
    key, mach, tag = sys.argv[1], float(sys.argv[2]), (sys.argv[3] if len(sys.argv) > 3 else '')
    o = study(key, mach, tag)
    (R / f'kutta_study_{key}_M{mach:.1f}{tag}.json').write_text(json.dumps(o, indent=1))
