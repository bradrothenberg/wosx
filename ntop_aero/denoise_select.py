"""Neural-network de-noising of the WoSX velocity next to the skin, with the size of the network chosen by a second,
independent WoSX run (no clean data is used for training or for the choice).

  run A   results/fields_V03.npz    (python run_fields.py)                 training data
  run B   results/fields2_V03.npz   (python run_fields.py --tag fields2)   same points, new random walks: the judge
  score   results/panel_cp_V03_nc48.npz, the pressure map of the panel code at 4 degrees (no noise); used for the
          final score only

Candidates: local averages over k neighbours, and small networks (position on the skin, side -> velocity).
For each one the error against run B is measured. B has its own noise, so this error cannot reach zero, but the
candidate that is closest to the true flow has the lowest one.

Run in the study Python:  <study python> denoise_select.py   -> results/denoise_select.json
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

import denoise_test as T

HERE = T.HERE


def velocity(D, alpha_deg):
    Vb = D['shell_Vb'].astype(np.float64)
    a = math.radians(alpha_deg)
    gam = (math.cos(a) * T.GX + math.sin(a) * T.GZ) / T.R
    return math.cos(a) * Vb[:, 0] + math.sin(a) * Vb[:, 1] + np.einsum('k,nkj->nj', gam, Vb[:, 2:])


def huber(e, delta=0.05):
    a = np.abs(e)
    return float(np.where(a < delta, 0.5 * a * a, delta * (a - 0.5 * delta)).mean())


if __name__ == '__main__':
    DA, DB = T.D, np.load(HERE / 'results' / 'fields2_V03.npz')
    sp, sn = DA['surf_pts'].astype(np.float64), DA['surf_normals'].astype(np.float64)
    assert np.array_equal(DA['surf_pts'], DB['surf_pts']), 'the two runs must use the same points'
    VA, VB = velocity(DA, T.ALPHA), velocity(DB, T.ALPHA)
    side = np.sign(sn[:, 2]); side[side == 0] = 1.0
    P = np.load(HERE / 'results' / 'denoise_cp.npz')
    Q, cp_ref, use, qside = P['Q'], P['cp_ref'], P['use'], P['qside']
    tree = {s: cKDTree(sp[side == s]) for s in (1.0, -1.0)}
    idx = {s: np.flatnonzero(side == s) for s in (1.0, -1.0)}

    def neighbours(x, sd, k):
        res = np.zeros((len(x), k), int)
        for s in (1.0, -1.0):
            m = sd == s
            _, nn = tree[s].query(x[m], k=k)
            res[m] = idx[s][nn.reshape(m.sum(), k)]
        return res

    def score(name, at_samples, at_panel, seconds, kind):
        e = ((1.0 - (at_panel ** 2).sum(-1)) - cp_ref)[use]
        row = dict(name=name, kind=kind, judge_loss=huber(at_samples - VB), cp_rms=float(np.sqrt((e ** 2).mean())), cp_median_abs=float(np.median(np.abs(e))),
                   cp_p90_abs=float(np.percentile(np.abs(e), 90)), seconds=seconds)
        print(f"{name:28s} loss against the second run {row['judge_loss']:.6f} | Cp error against the panel map: median {row['cp_median_abs']:.4f}  90 % below {row['cp_p90_abs']:.3f}  rms {row['cp_rms']:.3f} | {seconds:.1f} s", flush=True)
        return row

    rows = []
    nq1 = neighbours(Q, qside, 1)[:, 0]
    rows.append(score('raw, one run', VA, VA[nq1], 0.0, 'raw'))
    rows.append(score('raw, mean of the two runs', 0.5 * (VA + VB), 0.5 * (VA + VB)[nq1], 0.0, 'two runs (the judge is not independent)'))
    for k in (8, 16, 32, 64, 128, 256, 512):
        t = time.perf_counter()
        ns, nq = neighbours(sp, side, k), neighbours(Q, qside, k)
        rows.append(score(f'mean of {k} neighbours', VA[ns].mean(1), VA[nq].mean(1), time.perf_counter() - t, 'local mean'))
        rows.append(score(f'median of {k} neighbours', np.median(VA[ns], axis=1), np.median(VA[nq], axis=1), time.perf_counter() - t, 'local median'))
    nets = {}
    for nf in (0, 1, 2, 3, 4, 6):
        for width in (64, 256):
            t = time.perf_counter()
            predict, hist, dev = T.train(sp, side, VA, seed=0, steps=2000, width=width, nf=nf)
            nets[(nf, width)] = predict
            rows.append(score(f'network, {nf} bands, width {width}', predict(sp, side), predict(Q, qside), time.perf_counter() - t, 'network'))
            rows[-1].update(nf=nf, width=width, device=dev)
    best = {}
    for kind in ('local mean', 'local median', 'network'):
        c = [r for r in rows if r['kind'] == kind]
        best[kind] = min(c, key=lambda r: r['judge_loss'])
        oracle = min(c, key=lambda r: r['cp_median_abs'])
        print(f"chosen by the second run, {kind}: {best[kind]['name']}  (Cp median error {best[kind]['cp_median_abs']:.4f}; the best possible choice was {oracle['name']} with {oracle['cp_median_abs']:.4f})")
    # forces from the pressure at every sample point (equal areas), for the chosen candidates
    mesh = np.load(HERE / 'geo' / 'V03_h20_q150k.npz')
    tri = mesh['vertices'][mesh['faces']].astype(np.float64)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
    half = 2.0 if sp[:, 1].min() > -0.05 else 1.0
    a = math.radians(T.ALPHA)
    kb = int(best['local mean']['name'].split()[2])
    cand = {'raw, one run': VA, best['local mean']['name']: VA[neighbours(sp, side, kb)].mean(1), best['network']['name']: nets[(best['network']['nf'], best['network']['width'])](sp, side)}
    forces = {}
    for name, vv in cand.items():
        cl, cd = T.forces(1.0 - (vv ** 2).sum(-1), sn, half * area / len(sp), a)
        forces[name] = dict(cl_pressure=cl, cd_pressure=cd)
        print(f'{name:28s} lift from pressure {cl:.4f}  drag from pressure {cd:+.5f}')
    pp = json.loads((HERE / 'results' / 'panel_pressure_V03.json').read_text())['grids'][2]
    out = dict(alpha_deg=T.ALPHA, samples=int(len(sp)), compared_panel_points=int(use.sum()), rows=rows, chosen={k: v['name'] for k, v in best.items()}, forces=forces,
               forces_reference=dict(panel_cl_pressure=pp['cl_pressure'], panel_cl_wake=pp['cl_wake'], panel_cd_pressure=pp['cd_pressure']),
               note='judge_loss: Huber loss (0.05) of the velocity against the second WoSX run at the 60,000 sample points; Cp errors against the panel code, 5 to 95 % chord')
    (HERE / 'results' / 'denoise_select.json').write_text(json.dumps(out, indent=1))
