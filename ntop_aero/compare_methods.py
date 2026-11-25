"""Speed and accuracy of four methods on the ten delta variants at Mach 0.7: the CFD of the trade study (reference),
the flat vortex lattice, the panel code and WoSX with the wall from the field.

  python compare_methods.py --wos _fast --panel-grid 32 24      -> results/compare_methods<tag>.json

Read-only inputs:
  CFD      <study>/aero/results/table.json   lift at 0, 2, 4, 6, 8 degrees, run time per angle
  lattice  results/vlm_V..json
  panel    results/panel_V.._M0.7_cmp.json   (panel_body.run(key, 0.7, grids=..., tag='_cmp'))
  WoSX     results/final_V.._M0.7<tag>.json and results/kutta/..., read-out 'edge, 0.16 to 1.0 m'

One definition for every method: lift slope = (CL at 8 degrees - CL at 0 degrees) / 8, per degree, on the variant's
own reference area. Lift at zero angle is the CL at 0 degrees. K is the wake (Trefftz) drag factor CDi / CL^2.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

import kutta_study as KS
from paths import STUDY

HERE = Path(__file__).resolve().parent
R = HERE / 'results'
KEYS = [f'V{i:02d}' for i in range(1, 11)]
DEG = math.pi / 180


def wake_k(gam, ya, yb, S):
    g2 = np.r_[gam, gam[::-1]]; a = np.r_[ya, -yb[::-1]]; b = np.r_[yb, -ya[::-1]]; yc = 0.5 * (a + b)
    wv = (g2[None] * (-1.0 / (yc[:, None] - a[None]) + 1.0 / (yc[:, None] - b[None]))).sum(axis=1) / (4 * math.pi)
    cl = 2.0 * (g2 * (b - a)).sum() / S
    return float(-(g2 * wv * (b - a)).sum() * 2.0 / S / cl ** 2)


def cfd(tab, k):
    v = tab['variants'][k]
    p = {q['alpha_deg']: q for q in v['points']}
    return dict(slope=(p[8.0]['cl'] - p[0.0]['cl']) / 8.0, cl0=p[0.0]['cl'], fit_slope=v['fits']['cl_alpha_per_deg'],
                k_polar=v['fits']['k_polar_fit'], seconds_per_angle=float(np.mean([q['seconds'] for q in v['points']])),
                seconds_six_angles=float(sum(q['seconds'] for q in v['points'])), cells=v['solver']['cells'])


def lattice(k):
    c = json.loads((R / f'vlm_{k}.json').read_text())['cases']['mach_0.7']
    p = {q['alpha_deg']: q for q in c['points']}
    return dict(slope=(p[8.0]['CL'] - p[0.0]['CL']) / 8.0, cl0=p[0.0]['CL'], k=p[4.0]['CDi_trefftz'] / p[4.0]['CL'] ** 2,
                seconds=float(c['seconds_sweep']), panels=int(c['panels']))


def panel(k, nc, ms, mach=0.7):
    f = R / f'panel_{k}_M{mach:.1f}_cmp.json'
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    beta, S = math.sqrt(1 - mach * mach), d['reference']['s_ref_m2']
    out = {}
    for g in d['grids']:
        dy, gx, gz, y = np.array(g['dy']), np.array(g['gamma_x']) * beta, np.array(g['gamma_z']), np.array(g['span_y'])
        cl = []
        for al in (0.0, 8.0):
            a = math.atan(beta * math.tan(al * DEG))
            cl.append(4.0 * ((math.cos(a) * gx + math.sin(a) * gz) * dy).sum() / (S * beta))
        a4 = math.atan(beta * math.tan(4 * DEG))
        out[(g['nc'], g['ms'])] = dict(slope=(cl[1] - cl[0]) / 8.0, cl0=cl[0], k=wake_k((math.cos(a4) * gx + math.sin(a4) * gz) / beta, y - dy / 2, y + dy / 2, S),
                                       seconds=g['seconds'], panels=g['panels'], small_angle_slope=g['cl_alpha'] * DEG)
    if (nc, ms) not in out:
        return None
    fine = out[max(out, key=lambda q: out[q]['panels'])]
    return dict(out[(nc, ms)], grid=[nc, ms], finest=dict(fine), seconds_sections=d['sections'].get('seconds_sections'), sections_device=d['sections'].get('device'),
                all_grids={f'{a}x{b}': v for (a, b), v in out.items()})


def wos(k, tag, mach=0.7):
    f = R / f'final_{k}_M{mach:.1f}{tag}.json'
    if not f.exists():
        return None
    w, runs = KS.load(k, mach, tag)
    ns, nd = w['strips_per_side'], len(runs[0]['d'])
    sel = list(range(4, nd))
    res, ks = [], []
    ym = np.array(w['span_y']); dy = 2 * ym[0]
    for q in runs:
        gx, gz = KS.read(q, ns, sel, 'edge')
        res.append(KS.slope(w, q, gx, gz))
        a4 = math.atan(float(q['beta']) * math.tan(4 * DEG))
        gam = (math.cos(a4) * gx + math.sin(a4) * gz) * float(q['R']) / float(q['beta'])
        ks.append(wake_k(gam, ym - dy / 2, ym + dy / 2, w['reference']['s_ref_m2']))
    res = np.array(res)
    n = len(res)
    t = [r['timing'] for r in w['runs']]
    return dict(slope=float(res[:, 0].mean() * DEG), slope_std=float(res[:, 0].std(ddof=1) * DEG) if n > 1 else None, cl0=float(res[:, 1].mean()),
                cl0_std=float(res[:, 1].std(ddof=1)) if n > 1 else None, k=float(np.mean(ks)), runs=n,
                seconds_run=float(np.mean([r['seconds'] for r in w['runs']])), seconds_walks=float(np.mean([q['gpu_walks'] for q in t])),
                seconds_setup=float(np.mean([q['gpu_solver_setup'] for q in t])), seconds_init=w['init'].get('wall'),
                seconds_grid=w['mesh'].get('seconds_total'), grid_shape=w['mesh'].get('shape'), settings={q: w['settings'][q] for q in ('cache', 'walks')},
                gpu_shared=w.get('gpu_shared'))


def score(model, ref):
    """Level and scatter of model / reference, and the share of pairs ranked the same way."""
    m, r = np.array(model, float), np.array(ref, float)
    q = m / r
    pairs = list(itertools.combinations(range(len(m)), 2))
    same = sum(1 for i, j in pairs if (m[i] - m[j]) * (r[i] - r[j]) > 0)
    # one level factor fitted to the reference, tested on the variant that was left out of the fit
    loo = np.array([q[i] / np.delete(q, i).mean() - 1.0 for i in range(len(q))]) if len(q) > 2 else np.zeros(0)
    return dict(n=len(m), level=float(q.mean()), level_min=float(q.min()), level_max=float(q.max()), scatter=float(q.std(ddof=1)) if len(m) > 2 else None,
                pairs_right=same, pairs=len(pairs), within_5pct=int((np.abs(q - 1) <= 0.05).sum()), ratios=q.tolist(),
                after_level_fit=dict(rms=float(np.sqrt((loo ** 2).mean())), max_abs=float(np.abs(loo).max()), within_5pct=int((np.abs(loo) <= 0.05).sum()),
                                     residuals=loo.tolist()) if len(loo) else None)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--wos', nargs='*', default=['_fast'])
    ap.add_argument('--panel-grid', type=int, nargs=2, default=[32, 24])
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    tab = json.loads((STUDY / 'aero' / 'results' / 'table.json').read_text())
    out = dict(mach=0.7, definition='slope = (CL(8 deg) - CL(0 deg)) / 8 per degree; own reference area', cfd_lift_uncertainty=tab['cl_grid_uncertainty'], variants={}, scores={})
    for k in KEYS:
        v = dict(cfd=cfd(tab, k), lattice=lattice(k), panel=panel(k, *a.panel_grid))
        for t in a.wos:
            v['wos' + t] = wos(k, t)
        out['variants'][k] = v
    methods = ['lattice', 'panel'] + ['wos' + t for t in a.wos]
    print(f"{'':4s} {'CFD':>8s} " + ' '.join(f'{m:>20s}' for m in methods) + '   slope per degree (ratio to CFD)')
    for k in KEYS:
        v = out['variants'][k]
        print(f"{k:4s} {v['cfd']['slope']:8.5f} " + ' '.join((f"{v[m]['slope']:11.5f} ({v[m]['slope'] / v['cfd']['slope']:5.3f})" if v[m] else f"{'-':>20s}") for m in methods))
    for m in methods:
        ks = [k for k in KEYS if out['variants'][k][m]]
        if len(ks) < 2:
            continue
        V = out['variants']
        s = dict(keys=ks, slope=score([V[k][m]['slope'] for k in ks], [V[k]['cfd']['slope'] for k in ks]),
                 cl0_diff=dict(mean=float(np.mean([V[k][m]['cl0'] - V[k]['cfd']['cl0'] for k in ks])), std=float(np.std([V[k][m]['cl0'] - V[k]['cfd']['cl0'] for k in ks], ddof=1)),
                               max_abs=float(np.max(np.abs([V[k][m]['cl0'] - V[k]['cfd']['cl0'] for k in ks])))),
                 cl4=score([V[k][m]['cl0'] + 4 * V[k][m]['slope'] for k in ks], [V[k]['cfd']['cl0'] + 4 * V[k]['cfd']['slope'] for k in ks]),
                 k_vs_polar=score([V[k][m]['k'] for k in ks], [V[k]['cfd']['k_polar'] for k in ks]))
        # lift at 4 degrees with two fitted numbers (a factor on the slope, a shift of the lift at zero angle), tested on the variant left out
        sm, sc = np.array([V[k][m]['slope'] for k in ks]), np.array([V[k]['cfd']['slope'] for k in ks])
        zm, zc = np.array([V[k][m]['cl0'] for k in ks]), np.array([V[k]['cfd']['cl0'] for k in ks])
        res = np.array([((zm[i] - np.delete(zm - zc, i).mean()) + 4 * sm[i] / np.delete(sm / sc, i).mean()) / (zc[i] + 4 * sc[i]) - 1.0 for i in range(len(ks))])
        s['cl4_after_fit'] = dict(rms=float(np.sqrt((res ** 2).mean())), max_abs=float(np.abs(res).max()), within_5pct=int((np.abs(res) <= 0.05).sum()), residuals=res.tolist())
        raw = (zm + 4 * sm) / (zc + 4 * sc) - 1.0
        s['cl4_as_is'] = dict(rms=float(np.sqrt((raw ** 2).mean())), max_abs=float(np.abs(raw).max()), within_5pct=int((np.abs(raw) <= 0.05).sum()))
        out['scores'][m] = s
        sl = s['slope']
        print(f"{m:10s} n {sl['n']:2d}  slope level {sl['level']:.3f} ({sl['level_min']:.3f} to {sl['level_max']:.3f}) scatter {100 * (sl['scatter'] or 0):.1f} %  pairs {sl['pairs_right']}/{sl['pairs']}"
              f"  | CL0 - CFD: mean {s['cl0_diff']['mean']:+.4f} std {s['cl0_diff']['std']:.4f}  | CL at 4 deg level {s['cl4']['level']:.3f} scatter {100 * (s['cl4']['scatter'] or 0):.1f} % pairs {s['cl4']['pairs_right']}/{s['cl4']['pairs']}")
        f_ = sl['after_level_fit']
        print(f"{'':10s} within 5 % of CFD: {sl['within_5pct']}/{sl['n']} as is; with one fitted level factor (left-out test): {f_['within_5pct']}/{sl['n']}, rms {100 * f_['rms']:.1f} %, worst {100 * f_['max_abs']:.1f} %"
              f"  | K level {s['k_vs_polar']['level']:.3f} scatter {100 * (s['k_vs_polar']['scatter'] or 0):.1f} %")
        print(f"{'':10s} CL at 4 deg within 5 % of CFD: {s['cl4_as_is']['within_5pct']}/{sl['n']} as is (rms {100 * s['cl4_as_is']['rms']:.1f} %, worst {100 * s['cl4_as_is']['max_abs']:.1f} %); with fitted factor and zero-angle shift:"
              f" {s['cl4_after_fit']['within_5pct']}/{sl['n']}, rms {100 * s['cl4_after_fit']['rms']:.1f} %, worst {100 * s['cl4_after_fit']['max_abs']:.1f} %")
    (R / f'compare_methods{a.tag}.json').write_text(json.dumps(out, indent=1))
