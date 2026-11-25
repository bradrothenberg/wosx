"""Can a small neural network take the noise out of the WoSX velocity next to the skin?

Data (already on disk): one WoSX run of V03 at Mach 0 (results/fields_V03.npz), velocity at 60,000 points 0.06 m off
the skin. Reference: the pressure map of the panel code at 4 degrees (results/panel_cp_V03_nc48.npz), which has no noise.

Three ways to read the pressure at the panel points:
  raw      the nearest WoSX sample
  median   median velocity of the 64 nearest samples on the same side (the filter the report used so far)
  network  a small network (position on the skin, side -> velocity), trained on the noisy samples themselves with a
           robust loss; no clean data is used. 15 % of the samples are held out to choose when to stop.

Run in the study Python (it has torch):  <study python> denoise_test.py   -> results/denoise_test.json, results/denoise_cp.npz
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
D = np.load(HERE / 'results' / 'fields_V03.npz')
meta = json.loads(str(D['meta']))
R = meta['R']
w0 = json.loads((HERE / 'results' / 'final_V03_M0.0.json').read_text())
GX = np.mean([r['gx'] for r in w0['runs']], axis=0)
GZ = np.mean([r['gz'] for r in w0['runs']], axis=0)
ALPHA = 4.0
S_REF = w0['reference']['s_ref_m2']


def velocity(name, alpha_deg):
    Vb = D[name + '_Vb'].astype(np.float64)
    a = math.radians(alpha_deg)
    gam = (math.cos(a) * GX + math.sin(a) * GZ) / R
    return math.cos(a) * Vb[:, 0] + math.sin(a) * Vb[:, 1] + np.einsum('k,nkj->nj', gam, Vb[:, 2:])


def features(x, lo, hi, side, nf=6):
    u = (x - lo) / (hi - lo) * 2 - 1
    f = [u, side[:, None]]
    for k in range(nf):
        f += [torch.sin((2 ** k) * math.pi * u), torch.cos((2 ** k) * math.pi * u)]
    f.append(f[0] * side[:, None])
    return torch.cat(f, 1)


def train(x, side, v, seed=0, steps=4000, width=256, delta=0.05, nf=6, weight_decay=0.0):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    X = torch.tensor(x, dtype=torch.float32, device=dev); Sd = torch.tensor(side, dtype=torch.float32, device=dev)
    Y = torch.tensor(v, dtype=torch.float32, device=dev)
    lo, hi = X.min(0).values, X.max(0).values
    F = features(X, lo, hi, Sd, nf)
    perm = torch.randperm(len(X), generator=g).to(dev)
    nv = int(0.15 * len(X)); va, tr = perm[:nv], perm[nv:]
    net = torch.nn.Sequential(torch.nn.Linear(F.shape[1], width), torch.nn.SiLU(), torch.nn.Linear(width, width), torch.nn.SiLU(),
                              torch.nn.Linear(width, width), torch.nn.SiLU(), torch.nn.Linear(width, 3)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    loss_f = torch.nn.HuberLoss(delta=delta)
    best, best_state, hist = 1e9, None, []
    for it in range(steps):
        opt.zero_grad()
        loss = loss_f(net(F[tr]), Y[tr]); loss.backward(); opt.step(); sched.step()
        if it % 100 == 99:
            with torch.no_grad():
                lv = float(loss_f(net(F[va]), Y[va]))
            hist.append((it + 1, float(loss), lv))
            if lv < best:
                best, best_state = lv, {k: q.clone() for k, q in net.state_dict().items()}
    net.load_state_dict(best_state)

    def predict(xq, sq):
        with torch.no_grad():
            return net(features(torch.tensor(xq, dtype=torch.float32, device=dev), lo, hi, torch.tensor(sq, dtype=torch.float32, device=dev), nf)).double().cpu().numpy()
    return predict, hist, dev


def forces(cp, nrm, area_each, a):
    f = -(cp[:, None] * nrm).sum(0) * area_each / S_REF
    return float(f[2] * math.cos(a) - f[0] * math.sin(a)), float(f[0] * math.cos(a) + f[2] * math.sin(a))


if __name__ == '__main__':
    sp, sn = D['surf_pts'].astype(np.float64), D['surf_normals'].astype(np.float64)
    ok = D['shell_sdf'] > 0.03
    V = velocity('shell', ALPHA)
    side = np.sign(sn[:, 2]); side[side == 0] = 1.0
    P = np.load(HERE / 'results' / 'panel_cp_V03_nc48.npz')
    Q, cp_ref = P['Q'].reshape(-1, 3), P['cp'].reshape(-1)
    ms, n2 = P['Q'].shape[:2]
    # chord position of each panel point within its strip, to leave out the leading and trailing 5 %
    xq = P['Q'][:, :, 0]
    frac = ((xq - xq.min(1, keepdims=True)) / (xq.max(1, keepdims=True) - xq.min(1, keepdims=True))).reshape(-1)
    print('samples', len(sp), 'usable', int(ok.sum()), '| y range of the samples', sp[:, 1].min(), sp[:, 1].max(), '| panel points', Q.shape, 'y', Q[:, 1].min(), Q[:, 1].max())
    tree = {s: cKDTree(sp[ok & (side == s)]) for s in (1.0, -1.0)}
    idx = {s: np.flatnonzero(ok & (side == s)) for s in (1.0, -1.0)}
    # side of each panel point: the side of the nearest sample of either kind
    d_up, _ = tree[1.0].query(Q); d_lo, _ = tree[-1.0].query(Q)
    qside = np.where(d_up <= d_lo, 1.0, -1.0)
    near = np.minimum(d_up, d_lo)
    use = (frac > 0.05) & (frac < 0.95) & (near < 0.25) & (np.abs(Q[:, 1]) <= np.abs(sp[:, 1]).max())
    out = dict(alpha_deg=ALPHA, samples=int(len(sp)), usable=int(ok.sum()), panel_points=int(len(Q)), compared=int(use.sum()), wosx_run=dict(cache=meta['cache'], walks=meta['walks'], shell_m=meta['shell_m']),
               note='reference: panel code, Mach 0, 6,144 panels, pressure on the skin; WoSX values are 0.06 m off the skin', methods={})
    cps = {}

    def knn(k):
        res = np.zeros((len(Q), k), int)
        for s in (1.0, -1.0):
            m = qside == s
            _, nn = tree[s].query(Q[m], k=k)
            res[m] = idx[s][nn.reshape(m.sum(), k)]
        return res
    nn64 = knn(64)
    cps['raw'] = 1.0 - (V[nn64[:, 0]] ** 2).sum(-1)
    cps['mean of 64'] = 1.0 - (V[nn64].mean(1) ** 2).sum(-1)
    cps['median of 64'] = 1.0 - (np.median(V[nn64], axis=1) ** 2).sum(-1)
    t = time.perf_counter()
    preds, hists = [], []
    for seed in range(3):
        predict, hist, dev = train(sp[ok], side[ok], V[ok], seed=seed)
        preds.append(predict); hists.append(hist)
    t_train = time.perf_counter() - t
    vq = np.mean([p(Q, qside) for p in preds], axis=0)
    cps['network'] = 1.0 - (vq ** 2).sum(-1)
    one = preds[0](Q, qside)
    cps['network, one of the three'] = 1.0 - (one ** 2).sum(-1)
    for name, c in cps.items():
        e = (c - cp_ref)[use]
        out['methods'][name] = dict(rms=float(np.sqrt((e ** 2).mean())), median_abs=float(np.median(np.abs(e))), p90_abs=float(np.percentile(np.abs(e), 90)), bias=float(np.median(e)))
        print(f"{name:26s} Cp error against the panel map: rms {out['methods'][name]['rms']:.3f}  median |e| {out['methods'][name]['median_abs']:.4f}  90 % below {out['methods'][name]['p90_abs']:.3f}  median e {out['methods'][name]['bias']:+.4f}")
    out['reference_cp'] = dict(rms=float(np.sqrt((cp_ref[use] ** 2).mean())), p10=float(np.percentile(cp_ref[use], 10)), p90=float(np.percentile(cp_ref[use], 90)))
    out['train'] = dict(seconds=t_train, device=dev, networks=3, steps=4000, history_first=hists[0][::5])
    # forces from the pressure at every sample point (equal areas: the points are uniform over the skin)
    mesh = np.load(HERE / 'geo' / 'V03_h20_q150k.npz')
    tri = mesh['vertices'][mesh['faces']].astype(np.float64)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
    half = 2.0 if sp[:, 1].min() > -0.05 else 1.0
    a = math.radians(ALPHA)
    _, nn_s = cKDTree(sp[ok]).query(sp, k=1)
    v_net = np.mean([p(sp, side) for p in preds], axis=0)
    med = np.zeros_like(V)
    for s in (1.0, -1.0):
        m = side == s
        _, nn = tree[s].query(sp[m], k=64)
        med[m] = np.median(V[idx[s][nn]], axis=1)
    fz = {'raw': V[np.flatnonzero(ok)[nn_s]], 'median of 64': med, 'network': v_net}
    for name, vv in fz.items():
        cl, cd = forces(1.0 - (vv ** 2).sum(-1), sn, half * area / len(sp), a)
        out['methods'][name].update(cl_pressure=cl, cd_pressure=cd)
        print(f'{name:26s} lift from pressure {cl:.4f}  drag from pressure {cd:+.5f}')
    cl_wake = float(2.0 * 2.0 * ((math.cos(a) * GX + math.sin(a) * GZ) * (2 * np.array(w0['span_y'])[0])).sum() / S_REF)
    pp = json.loads((HERE / 'results' / 'panel_pressure_V03.json').read_text())['grids'][2]
    out['forces_reference'] = dict(wosx_cl_wake=cl_wake, panel_cl_pressure=pp['cl_pressure'], panel_cl_wake=pp['cl_wake'], panel_cd_pressure=pp['cd_pressure'], area_m2=half * area)
    print('WoSX lift from the wake', round(cl_wake, 4), '| panel code: lift from pressure', round(pp['cl_pressure'], 4), 'from the wake', round(pp['cl_wake'], 4), 'drag from pressure', round(pp['cd_pressure'], 5), '| train', round(t_train, 1), 's on', dev)
    (HERE / 'results' / 'denoise_test.json').write_text(json.dumps(out, indent=1))
    np.savez_compressed(HERE / 'results' / 'denoise_cp.npz', Q=Q, cp_ref=cp_ref, use=use, qside=qside, **{'cp_' + k.replace(' ', '_').replace(',', ''): v for k, v in cps.items()},
                        sp=sp.astype(np.float32), side=side, cp_raw_s=1.0 - (fz['raw'] ** 2).sum(-1), cp_med_s=1.0 - (med ** 2).sum(-1), cp_net_s=1.0 - (v_net ** 2).sum(-1))
