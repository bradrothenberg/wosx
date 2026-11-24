"""An implicit body on grids for the WoSX implicit wall. No triangles are made.

Runs in the Python of the study (torch + the study's field library), reads the study read-only.

  python implicit_grid.py V03 --mach 0.7                     -> geo/V03_M0.70_grid.npz
  python implicit_grid.py W12T12 --mach 0 --shape 768 512 160

What is written (lengths in metres; x is the stretched coordinate x / beta of the Goethert rule):
  near    conservative signed distance at the cell centres of a fine grid around the half body (y >= 0).
          Next to the wall it is the field itself times kappa <= 1 (so its zero set is the zero set of the field);
          away from the wall it is the distance to the wall points, less a margin.
  far     the same on a coarse grid around it
  pts, nrm, area   points on the zero set of the field with unit normals and area weights (both halves). They are
          the places where the field changes sign between neighbouring cell centres, moved onto the zero set by
          bisection on the field itself. They give the samples for boundary value caching and the wetted area.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sdf_bridge as SB  # noqa: E402
import bodies  # noqa: E402


def name_of(key, mach, tag=''):
    return HERE / 'geo' / f'{key}_M{mach:.2f}_grid{tag}.npz'


def centres(lo, hi, n):
    h = (hi - lo) / n
    return lo + h * (np.arange(n) + 0.5), h


def main(a):
    import torch
    import torch.nn.functional as TF
    from scipy.spatial import cKDTree
    T = {}
    t0 = time.time()
    field, key, (lo, hi), source = SB.field_of(a.key)
    import fieldlab as fl
    beta = math.sqrt(1.0 - a.mach * a.mach)
    lo, hi = np.array(lo, float), np.array(hi, float)
    T['field_build'] = time.time() - t0

    def val(P):                                          # P in stretched coordinates
        out = np.empty(len(P), np.float32)
        for s in range(0, len(P), 4_000_000):
            Q = np.array(P[s:s + 4_000_000], np.float32)
            Q[:, 0] *= beta
            out[s:s + len(Q)] = np.asarray(fl.probe(field, Q))
        return out

    def grad(P):                                         # gradient in stretched coordinates
        out = np.empty((len(P), 3), np.float32)
        for s in range(0, len(P), 1_000_000):
            Q = np.array(P[s:s + 1_000_000], np.float32)
            Q[:, 0] *= beta
            out[s:s + len(Q)] = np.asarray(fl.gradient(field, Q)[1])
        out[:, 0] *= beta
        return out

    # ---------------------------------------------------------------- near box: the body plus a pad
    t = time.time()
    cx, hx0 = centres(lo[0] / beta, hi[0] / beta, 256); cy, hy0 = centres(0.0, hi[1], 160); cz, hz0 = centres(lo[2], hi[2], 256)
    G = np.stack(np.meshgrid(cx, cy, cz, indexing='ij'), -1).reshape(-1, 3)
    ins = (val(G) < 0).reshape(256, 160, 256)
    zin = cz[ins.any((0, 1))]; xin = cx[ins.any((1, 2))]; yin = cy[ins.any((0, 2))]
    pad = a.pad
    nlo = np.array([xin.min() - hx0 - pad, 0.0, zin.min() - hz0 - pad])
    nhi = np.array([xin.max() + hx0 + pad, yin.max() + hy0 + pad, zin.max() + hz0 + pad])
    nx, ny, nz = a.shape
    X, hx = centres(nlo[0], nhi[0], nx); Y, hy = centres(nlo[1], nhi[1], ny); Z, hz = centres(nlo[2], nhi[2], nz)
    h = np.array([hx, hy, hz])
    T['box'] = time.time() - t

    # ---------------------------------------------------------------- the field at every cell centre
    t = time.time()
    F = np.empty((nx, ny, nz), np.float32)
    YZ = np.stack(np.meshgrid(Y, Z, indexing='ij'), -1).reshape(-1, 2)
    slab = max(1, 4_000_000 // (ny * nz))
    for i in range(0, nx, slab):
        xs = X[i:i + slab]
        P = np.empty((len(xs), ny * nz, 3), np.float32)
        P[:, :, 0] = xs[:, None]; P[:, :, 1:] = YZ[None]
        F[i:i + slab] = val(P.reshape(-1, 3)).reshape(len(xs), ny, nz)
    T['field_on_grid'] = time.time() - t
    evals = nx * ny * nz

    # ---------------------------------------------------------------- wall points: sign changes between neighbours
    t = time.time()
    S = F < 0
    pts, axes = [], []
    for ax in range(3):
        sl0 = [slice(None)] * 3; sl1 = [slice(None)] * 3
        sl0[ax] = slice(0, -1); sl1[ax] = slice(1, None)
        idx = np.argwhere(S[tuple(sl0)] != S[tuple(sl1)])
        A = np.stack([X[idx[:, 0]], Y[idx[:, 1]], Z[idx[:, 2]]], 1)
        B = A.copy(); B[:, ax] += h[ax]
        fa = val(A) < 0
        for _ in range(18):                              # bisection on the field itself
            M = 0.5 * (A + B)
            same = (val(M) < 0) == fa
            A = np.where(same[:, None], M, A); B = np.where(same[:, None], B, M)
        pts.append(0.5 * (A + B)); axes.append(np.full(len(A), ax))
        evals += 19 * len(A)
    pts = np.concatenate(pts); axes = np.concatenate(axes)
    g = grad(pts)
    nrm = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-30)   # out of the body, into the flow
    cell_face = np.array([hy * hz, hx * hz, hx * hy])
    area = np.abs(nrm[np.arange(len(pts)), axes]) * cell_face[axes]
    T['wall_points'] = time.time() - t
    area_half = float(area.sum())
    mir = np.array([1.0, -1.0, 1.0])
    pts_all = np.concatenate([pts, pts * mir]); nrm_all = np.concatenate([nrm, nrm * mir]); area_all = np.concatenate([area, area])

    # ---------------------------------------------------------------- conservative distance
    t = time.time()
    rng = np.random.default_rng(0)
    sub = pts_all if len(pts_all) <= 3_000_000 else pts_all[rng.choice(len(pts_all), 3_000_000, replace=False)]
    tree = cKDTree(sub)
    spacing = float(max(h)) * (len(pts_all) / len(sub)) ** 0.5
    m = 4                                                # medium grid: blocks of m cells
    mx, my, mz = nx // m, ny // m, nz // m
    assert nx % m == 0 and ny % m == 0 and nz % m == 0, 'grid shape must be a multiple of 4'
    Xm, hmx = centres(nlo[0], nhi[0], mx); Ym, hmy = centres(nlo[1], nhi[1], my); Zm, hmz = centres(nlo[2], nhi[2], mz)
    Gm = np.stack(np.meshgrid(Xm, Ym, Zm, indexing='ij'), -1).reshape(-1, 3)
    dm = tree.query(Gm, workers=-1)[0].astype(np.float32).reshape(mx, my, mz)
    margin = float(np.linalg.norm([hmx, hmy, hmz])) + spacing
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    up = TF.interpolate(torch.from_numpy(dm)[None, None].to(dev), size=(nx, ny, nz), mode='trilinear', align_corners=False)[0, 0].cpu().numpy()
    # gradient size of the field next to the wall (stretched space), for kappa
    band = a.band
    near_wall = np.abs(F) < band
    gx = (F[2:, 1:-1, 1:-1] - F[:-2, 1:-1, 1:-1]) / (2 * hx)
    gy = (F[1:-1, 2:, 1:-1] - F[1:-1, :-2, 1:-1]) / (2 * hy)
    gz = (F[1:-1, 1:-1, 2:] - F[1:-1, 1:-1, :-2]) / (2 * hz)
    gn = np.sqrt(gx * gx + gy * gy + gz * gz)[near_wall[1:-1, 1:-1, 1:-1]]
    del gx, gy, gz
    gstat = [float(q) for q in np.percentile(gn, [50, 90, 99, 100])]
    kappa = 1.0 / max(1.0, gstat[1])
    floor = kappa * band
    D = (np.sign(F) * np.maximum(up - margin, floor)).astype(np.float32)
    del up
    # next to the wall: the field times kappa, but never more than the distance to the closest wall point (the field
    # can be far from a distance in places, for example past a clamped leading edge)
    ib = np.argwhere(near_wall)
    dk = tree.query(np.stack([X[ib[:, 0]], Y[ib[:, 1]], Z[ib[:, 2]]], 1), workers=-1)[0].astype(np.float32)
    fb = F[near_wall]
    capped = float((kappa * np.abs(fb) > dk).mean())
    D[near_wall] = np.sign(fb) * np.minimum(kappa * np.abs(fb), dk)
    T['distance_near'] = time.time() - t

    # ---------------------------------------------------------------- far grid and the body radius
    t = time.time()
    c = np.asarray(bodies.centre(key), float) / np.array([beta, 1.0, 1.0])
    assert val(c[None].astype(np.float32))[0] < 0, 'the inversion centre must be inside the body'
    R = float(np.linalg.norm(pts_all - c, axis=1).max()) * 1.0002
    fh = a.far * R
    flo = np.array([c[0] - fh, 0.0, c[2] - fh]); fhi = np.array([c[0] + fh, fh, c[2] + fh])
    fx, fy, fz = a.far_shape
    Xf, hfx = centres(flo[0], fhi[0], fx); Yf, hfy = centres(flo[1], fhi[1], fy); Zf, hfz = centres(flo[2], fhi[2], fz)
    Gf = np.stack(np.meshgrid(Xf, Yf, Zf, indexing='ij'), -1).reshape(-1, 3)
    # the far grid is coarse: a thinned set of wall points is enough, with a margin of three of its spacings
    far_pts = pts_all if len(pts_all) <= 200_000 else pts_all[rng.choice(len(pts_all), 200_000, replace=False)]
    far_spacing = float(max(h)) * (len(pts_all) / len(far_pts)) ** 0.5
    df = cKDTree(far_pts).query(Gf, workers=-1)[0].astype(np.float32).reshape(fx, fy, fz)
    far_margin = float(np.linalg.norm([hfx, hfy, hfz])) + 3.0 * far_spacing
    Dfar = np.maximum(df - far_margin, 0.5 * pad).astype(np.float32)
    T['distance_far'] = time.time() - t

    meta = dict(variant=key, source=source, mach=a.mach, beta=beta, shape=[nx, ny, nz], cell_m=[float(q) for q in h],
                near_lo=nlo.tolist(), near_hi=nhi.tolist(), far_shape=[fx, fy, fz], far_lo=flo.tolist(), far_hi=fhi.tolist(),
                far_cell_m=[float(hfx), float(hfy), float(hfz)], pad_m=pad, band_m=band, kappa=kappa,
                gradient_size_in_band_p50_p90_p99_max=gstat, band_cells=int(len(ib)), band_cells_capped_by_distance=capped, margin_m=margin, far_margin_m=far_margin,
                wall_points=int(len(pts_all)), wetted_area_m2_stretched=2 * area_half, centre=c.tolist(), radius_m=R,
                field_evaluations=int(evals), field_nodes=int(fl.node_count(field)), device=str(fl.DEV),
                seconds={k: round(v, 2) for k, v in T.items()}, seconds_total=round(time.time() - t0, 2))
    out = name_of(key, a.mach, a.tag)
    np.savez(out, near=D, far=Dfar, pts=pts_all.astype(np.float32), nrm=nrm_all.astype(np.float32),
             area=area_all.astype(np.float32), meta=json.dumps(meta))
    print(json.dumps(meta), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('key')
    ap.add_argument('--mach', type=float, default=0.0)
    ap.add_argument('--shape', type=int, nargs=3, default=[1024, 512, 256], help='cells of the fine grid in x, y, z (multiples of 4)')
    ap.add_argument('--far-shape', type=int, nargs=3, default=[192, 96, 192])
    ap.add_argument('--far', type=float, default=3.0, help='half size of the far grid, in body radii')
    ap.add_argument('--pad', type=float, default=0.30, help='clear distance between the body and the faces of the fine grid, m')
    ap.add_argument('--band', type=float, default=0.06, help='the field itself is kept within this distance of the wall, m')
    ap.add_argument('--tag', default='')
    main(ap.parse_args())
