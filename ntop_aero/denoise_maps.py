"""Pressure maps for the de-noising figure: the chosen candidates of denoise_select.py at every sample point.

Run in the study Python after denoise_select.py:  <study python> denoise_maps.py   -> results/denoise_maps.npz
"""
from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree

import denoise_select as S
import denoise_test as T

if __name__ == '__main__':
    sel = json.loads((T.HERE / 'results' / 'denoise_select.json').read_text())
    net = [r for r in sel['rows'] if r['name'] == sel['chosen']['network']][0]
    sp, sn = T.D['surf_pts'].astype(np.float64), T.D['surf_normals'].astype(np.float64)
    VA = S.velocity(T.D, T.ALPHA)
    side = np.sign(sn[:, 2]); side[side == 0] = 1.0
    out = dict(sp=sp.astype(np.float32), side=side, cp_raw=1.0 - (VA ** 2).sum(-1))
    for k in (64, int(sel['chosen']['local mean'].split()[2])):
        v = np.zeros_like(VA)
        for s in (1.0, -1.0):
            m = side == s
            _, nn = cKDTree(sp[m]).query(sp[m], k=k)
            v[m] = VA[m][nn].mean(1)
        out[f'cp_mean{k}'] = 1.0 - (v ** 2).sum(-1)
    predict, hist, dev = T.train(sp, side, VA, seed=0, steps=2000, width=net['width'], nf=net['nf'])
    out['cp_net'] = 1.0 - (predict(sp, side) ** 2).sum(-1)
    P = np.load(T.HERE / 'results' / 'denoise_cp.npz')
    out.update(Q=P['Q'], cp_ref=P['cp_ref'], qside=P['qside'], cp_net_q=1.0 - (predict(P['Q'], P['qside']) ** 2).sum(-1), use=P['use'])
    e = (out['cp_net_q'] - out['cp_ref'])[out['use']]
    print('network', net['name'], 'median error at the panel points', float(np.median(np.abs(e))), 'device', dev)
    np.savez_compressed(T.HERE / 'results' / 'denoise_maps.npz', **out)
