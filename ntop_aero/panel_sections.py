"""Sections of an implicit body for the panel code: upper and lower surface height on a (span station, chord fraction) grid.

Runs in the Python of the study (torch + the study's field library), reads the study read-only.

  python panel_sections.py V03      -> geo/V03_sections.npz
  python panel_sections.py W12T12   -> geo/W12T12_sections.npz   (check of this step against the wing built from its formula)

For each station y and each chord fraction xi the body surface is found on the vertical line through
(xle(y) + xi * chord(y), y) by bisection on the field value, to about 1e-9 m. This needs a body with one upper and
one lower surface over its planform and the plane z = 0 inside it; both are checked here.
Stations: the tip-clustered rows of every span count in MS. Chord fractions: a cosine grid of NC intervals; the
grids used by the panel code (16 to 96 intervals) are subsets of it.
"""
from __future__ import annotations

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

NC = 192
MS = (16, 24, 32, 40, 48, 56)


def main(key):
    t0 = time.time()
    field, key, (lo, hi), source = SB.field_of(key)
    import fieldlab as fl
    pf = bodies.planform(key)
    b = float(pf.semispan)
    t1 = time.time()

    def val(P):
        out = []
        for s in range(0, len(P), 2_000_000):
            out.append(np.asarray(fl.probe(field, P[s:s + 2_000_000])))
        return np.concatenate(out)

    ys = np.unique(np.round(np.concatenate([b * np.sin(0.5 * math.pi * np.arange(m + 1) / m) for m in MS]), 9))
    ys[-1] = b
    ye = ys.copy(); ye[-1] = b - 1e-6                    # the last station is read just inside the tip
    xi = 0.5 * (1.0 - np.cos(math.pi * np.arange(NC + 1) / NC))
    xle = np.asarray(pf.xle(ys), float); xte = np.asarray(pf.xte(ys), float)
    X = xle[:, None] + (xte - xle)[:, None] * xi[None]
    X[:, -1] = xte - 1e-6                                # just inside a blunt trailing edge
    Y = np.repeat(ye[:, None], NC + 1, axis=1)
    n = X.size
    zlo, zhi = float(lo[2]), float(hi[2])

    # checks: z = 0 inside, and one upper and one lower surface on every vertical line
    f0 = val(np.c_[X.ravel(), Y.ravel(), np.zeros(n)]).reshape(X.shape)
    zs = np.arange(zlo, zhi + 1e-9, 0.01)
    crossings = 0
    for j in range(0, len(ys), 8):
        P = np.c_[np.repeat(X[j], len(zs)), np.full(len(zs) * (NC + 1), ye[j]), np.tile(zs, NC + 1)]
        ins = (val(P) < 0).reshape(NC + 1, len(zs))
        crossings = max(crossings, int(np.abs(np.diff(ins.astype(np.int8), axis=1)).sum(1).max()))
    inside0 = f0 < 0

    def root(sign):
        a = np.zeros(n); c = np.full(n, zhi if sign > 0 else zlo)   # a inside, c outside
        for _ in range(44):
            m = 0.5 * (a + c)
            ins = val(np.c_[X.ravel(), Y.ravel(), m]) < 0
            a = np.where(ins, m, a); c = np.where(ins, c, m)
        return (0.5 * (a + c)).reshape(X.shape)

    zu, zl = root(+1), root(-1)
    zu[~inside0] = 0.0; zl[~inside0] = 0.0               # on the leading edge line (and a closed tip) the body has no thickness
    meta = dict(variant=key, source=source, stations=int(len(ys)), chord_intervals=NC, span_counts=list(MS),
                z0_inside_fraction=float(inside0[:, 1:-1].mean()), max_surface_crossings_on_a_line=crossings,
                te_thickness_m=[float((zu - zl)[:, -1].min()), float((zu - zl)[:, -1].max())],
                tip_thickness_max_m=float((zu - zl)[-1].max()), max_thickness_m=float((zu - zl).max()),
                seconds_field_build=round(t1 - t0, 1), seconds_sections=round(time.time() - t1, 1), device=str(fl.DEV))
    np.savez_compressed(HERE / 'geo' / f'{key}_sections.npz', ys=ys, xi=xi, xle=xle, xte=xte, zu=zu, zl=zl, meta=json.dumps(meta))
    print(json.dumps(meta), flush=True)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'V03')
