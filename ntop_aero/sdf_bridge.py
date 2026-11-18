"""Bridge from the nTop implicit model (signed distance field) of a delta variant to WoSX.

Runs in the Python of the study (torch + the study's field library), reads the study read-only.

  python sdf_bridge.py mesh V03 --h 0.04          -> geo/V03_h40.npz  (vertices, faces, normals, timing)
  python sdf_bridge.py eval V03 pts.npy out.npz   -> field value and gradient at the points (exact field)

Why a mesh at all: WoSX's Walk on Stars (needed for the no-flow-through wall condition) asks for ray hits and
silhouette distances, which it gets from a triangle BVH (FCPW). Its GPU SDF-grid path only exists for
absorbing (Dirichlet) walls. So the field is turned into surface triangles once (no volume mesh), and the
field itself is used for everything else: inside/outside masks, wall normals, offset shells, streamline stops.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from paths import STUDY  # noqa: E402
sys.path.insert(0, str(STUDY / 'aero'))
import common as C  # noqa: E402
sys.path.insert(0, str(HERE))
import bodies  # noqa: E402


def field_of(key):
    """Field graph, key, meshing box (half body, y >= 0) and a note on where the field comes from."""
    pad = 0.25
    if bodies.is_local(key):                      # plain test wing, written as a field graph in bodies.py
        sys.path.insert(0, str(STUDY / 'model'))
        import fieldlab as fl
        w = bodies.get(key)
        return w.field(fl), w.key, w.box(pad), 'ntop_aero/bodies.py TaperWing: ' + w.note + ' (fieldlab graph)'
    key = C.use(key)
    import oml
    p = C.planform(key)
    return (oml.flow_body('faired'), key, ([p.x_nose - pad, 0.0, -1.3], [p.x_tail + pad, p.semispan + pad, 1.7]),
            'study model/oml.py flow_body(faired): the same field graph that is written as the nTop recipe')


def cmd_mesh(a):
    t0 = time.time()
    field, key, (lo, hi), source = field_of(a.key)
    import fieldlab as fl
    t1 = time.time()
    v, f, n = fl.mesh_field(field, lo, hi, a.h, pin_y0=True)
    on = np.abs(v[:, 1]) < 1e-9
    idx = np.arange(len(v))
    mir = np.where(on, idx, idx + len(v))
    v = np.vstack([v, v * np.array([1, -1, 1], np.float32)])
    n = np.vstack([n, n * np.array([1, -1, 1], np.float32)])
    f = np.vstack([f, mir[f][:, ::-1]])
    used = np.zeros(len(v), bool)
    used[f.ravel()] = True
    remap = np.cumsum(used) - 1
    v, n, f = v[used], n[used], remap[f]
    t2 = time.time()
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    tri = v[f].astype(np.float64)
    vol = float(np.einsum('ij,ij->i', tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
    area = float(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2)
    out = HERE / 'geo'
    out.mkdir(exist_ok=True)
    name = out / f'{key}_h{int(round(a.h * 1000))}.npz'
    meta = dict(variant=key, h_m=a.h, vertices=int(len(v)), faces=int(len(f)), open_edges=int((cnt != 2).sum()),
                volume_m3=vol, wetted_area_m2=area, seconds_field_build=round(t1 - t0, 2),
                seconds_mesh=round(t2 - t1, 2), device=str(fl.DEV), field_nodes=int(fl.node_count(field)),
                source=source)
    np.savez_compressed(name, vertices=v.astype(np.float32), faces=f.astype(np.int32), normals=n.astype(np.float32),
                        meta=json.dumps(meta))
    print(json.dumps(meta), flush=True)


def cmd_eval(a):
    t0 = time.time()
    field, key, _, _ = field_of(a.key)
    import fieldlab as fl
    pts = np.load(a.pts).astype(np.float32)
    t1 = time.time()
    val, g = fl.gradient(field, pts)
    np.savez(a.out, value=val.astype(np.float32), gradient=g.astype(np.float32),
             seconds_field_build=t1 - t0, seconds_eval=time.time() - t1)
    print(json.dumps(dict(points=len(pts), seconds_eval=round(time.time() - t1, 3))), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('mesh'); m.add_argument('key'); m.add_argument('--h', type=float, default=0.04)
    e = sub.add_parser('eval'); e.add_argument('key'); e.add_argument('pts'); e.add_argument('out')
    a = ap.parse_args()
    dict(mesh=cmd_mesh, eval=cmd_eval)[a.cmd](a)
