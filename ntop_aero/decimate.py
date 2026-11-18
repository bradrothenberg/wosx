"""Thin the SDF surface mesh for the solver's ray queries (quadric decimation), keep the trailing edge.

  python decimate.py geo/V03_h20.npz 150000   -> geo/V03_h20_q150k.npz
"""
import json
import sys
import time

import numpy as np


def decimate(src, target, fix=True):
    import fast_simplification as fs
    d = np.load(src)
    v, f = d['vertices'], d['faces']
    t = time.perf_counter()
    v2, f2 = fs.simplify(v.astype(np.float32), f.astype(np.int32), target_count=target, agg=5)
    dt = time.perf_counter() - t
    turned = None
    if fix:                                              # every triangle outward (see bodies.fix_winding)
        from bodies import fix_winding
        f2, turned = fix_winding(v2, f2)
    tri = v2[f2].astype(np.float64)
    vol = float(np.einsum('ij,ij->i', tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
    e = np.sort(np.concatenate([f2[:, [0, 1]], f2[:, [1, 2]], f2[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    meta = json.loads(str(d['meta']))
    meta.update(decimated_from_faces=int(len(f)), faces=int(len(f2)), vertices=int(len(v2)), seconds_decimate=round(dt, 2),
                volume_m3=vol, open_edges=int((cnt != 2).sum()), faces_turned_outward=turned, bbox=[v2.min(0).tolist(), v2.max(0).tolist()])
    out = src.replace('.npz', f'_q{target // 1000}k.npz')
    np.savez_compressed(out, vertices=v2.astype(np.float32), faces=f2.astype(np.int32), meta=json.dumps(meta))
    return out, meta


if __name__ == '__main__':
    out, meta = decimate(sys.argv[1], int(sys.argv[2]))
    print(out, json.dumps({k: meta[k] for k in ('faces', 'seconds_decimate', 'volume_m3', 'open_edges', 'bbox')}))
