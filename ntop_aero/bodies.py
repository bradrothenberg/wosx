"""Bodies this folder can run, and the step from an implicit body to the triangles WoSX needs.

Two kinds of body, both implicit field graphs (negative inside, metres, x aft, y starboard, z up):
  * the delta variants of the trade study (V00 ... V10): <study>/model/oml.py, read-only (paths.py says where the study is);
  * plain test wings defined here (TaperWing): a straight-tapered wing with a NACA 00xx section, written with
    the study's field library (fieldlab), so the same graph can be emitted as an nTop recipe.

  ensure_mesh('V05')  -> path of geo/V05_h20_q150k.npz; builds it from the field if it is not there yet.

Pure numpy: this module is imported both by the WoSX Python and by the study Python (sdf_bridge.py).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from paths import STUDY, STUDY_PY  # noqa: E402  (the study folder; a Python with torch and the study's field library)

NACA = (0.2969, -0.1260, -0.3516, 0.2843, -0.1036)      # NACA 4-digit thickness, closed trailing edge


class TaperWing:
    """Straight-tapered wing, symmetric NACA 00xx section in planes y = const, flat tip, apex at the origin."""

    def __init__(self, key, name, cr, ct, semispan, le_sweep_deg, tc, note=''):
        self.key, self.name, self.cr, self.ct, self.semispan, self.tc, self.note = key, name, cr, ct, semispan, tc, note
        self.le_sweep_deg, self.m = le_sweep_deg, math.tan(math.radians(le_sweep_deg))
        self.x_nose, self.x_tail = 0.0, self.m * semispan + max(ct, cr - self.m * semispan)

    def chord(self, y):
        return self.cr + (self.ct - self.cr) * np.abs(np.asarray(y, float)) / self.semispan

    def xle(self, y):
        return self.m * np.abs(np.asarray(y, float))

    def xte(self, y):
        return self.xle(y) + self.chord(y)

    @property
    def area(self):
        return (self.cr + self.ct) * self.semispan

    @property
    def mac(self):
        l = self.ct / self.cr
        return 2 / 3 * self.cr * (1 + l + l * l) / (1 + l)

    @property
    def volume(self):
        """Exact volume of the field's body, m3."""
        sec = 2 * 5 * self.tc * (NACA[0] * 2 / 3 + NACA[1] / 2 + NACA[2] / 3 + NACA[3] / 4 + NACA[4] / 5)   # section area / c^2
        return 2 * sec * self.semispan * (self.cr ** 2 + self.cr * self.ct + self.ct ** 2) / 3

    def reference(self):
        return dict(s_ref_m2=self.area, c_ref_m=self.mac, b_ref_m=2 * self.semispan, moment_ref_m=[0.0, 0.0, 0.0])

    @property
    def centre(self):
        """Point inside the body for the Kelvin inversion: root, 30 % chord (thickest place)."""
        return (0.30 * self.cr, 0.0, 0.0)

    def box(self, pad=0.25):
        """Meshing box of the half body. Its z range is a multiple of 0.04 m, so that z = 0 is a grid plane for the
        usual cell sizes: the thin trailing edge lies on that plane and is then kept to within one cell."""
        t = math.ceil((0.5 * self.tc * self.cr + pad) / 0.04) * 0.04
        return [self.x_nose - pad, 0.0, -t], [self.x_tail + pad, self.semispan + pad, t]

    def half_thickness(self, x, y):
        c = self.chord(y)
        xi = np.clip((np.asarray(x, float) - self.xle(y)) / c, 0.0, 1.0)
        return 5 * self.tc * c * (NACA[0] * np.sqrt(xi) + xi * (NACA[1] + xi * (NACA[2] + xi * (NACA[3] + xi * NACA[4]))))

    def field(self, fl):
        """Field graph (fieldlab nodes). (|z| - half thickness), divided by its slope so that the value is close to a
        distance near the surface, cut by the leading edge, the trailing edge and the tip plane."""
        ay = fl.absf(fl.Y)
        c = self.cr + (self.ct - self.cr) / self.semispan * ay
        xle = self.m * ay
        xi = fl.clamp((fl.X - xle) / c, 1e-6, 1.0)
        r = fl.sqrt(xi)
        k = 5.0 * self.tc
        yt = k * c * (NACA[0] * r + xi * (NACA[1] + xi * (NACA[2] + xi * (NACA[3] + xi * NACA[4]))))
        slope = k * (0.5 * NACA[0] / r + NACA[1] + xi * (2 * NACA[2] + xi * (3 * NACA[3] + xi * (4 * NACA[4]))))
        f = (fl.absf(fl.Z) - yt) / fl.sqrt(1.0 + slope * slope)
        return fl.fmax(f, xle - fl.X, fl.X - (xle + c), ay - self.semispan)


# Warren 12 planform (a standard check case for lifting-surface codes), 5 times its unit size so that it is as large
# as the delta variants: root chord 7.5 m, tip chord 2.5 m, semispan 5 sqrt(2) m, aspect ratio 2 sqrt(2).
_W12 = dict(cr=7.5, ct=2.5, semispan=5 * math.sqrt(2.0), le_sweep_deg=math.degrees(math.atan((0.5 + math.sqrt(2.0)) / math.sqrt(2.0))))
LOCAL = {f'W12T{int(round(100 * t)):02d}': TaperWing(f'W12T{int(round(100 * t)):02d}', f'Warren 12, {int(round(100 * t))} % thick', tc=t, **_W12,
                                                    note='Warren 12 planform times 5, NACA 00xx section')
         for t in (0.06, 0.09, 0.12, 0.18)}


def is_local(key):
    return key.upper() in LOCAL


def get(key):
    return LOCAL[key.upper()]


def planform(key):
    """Object with xle(y), xte(y), semispan."""
    if is_local(key):
        return get(key)
    sys.path.insert(0, str(STUDY / 'study')) if str(STUDY / 'study') not in sys.path else None
    import variants
    return variants.planform(key)


def reference(key):
    if is_local(key):
        return get(key).reference()
    return json.loads((STUDY / 'study' / 'sizing.json').read_text())['variants'][key]['reference']


def centre(key):
    return get(key).centre if is_local(key) else (6.0, 0.0, 0.15)


def fix_winding(v, f):
    """Make all triangles face the same way, outward. Marching cubes followed by a per-face orientation test can
    leave single faces turned over at sharp edges; a turned face has the wrong sign of the wall condition.
    Returns (faces, number of faces turned)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order, connected_components
    nf = len(f)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    fid = np.tile(np.arange(nf), 3)
    fwd = e[:, 0] < e[:, 1]
    key = np.sort(e, axis=1)
    o = np.lexsort((key[:, 1], key[:, 0]))
    key, fid, fwd = key[o], fid[o], fwd[o]
    same = (key[1:] == key[:-1]).all(1)
    first = np.r_[True, ~same]
    cnt = np.diff(np.r_[np.nonzero(first)[0], len(key)])
    i = np.nonzero(first)[0][cnt == 2]                  # edges with exactly two faces
    a, b = fid[i], fid[i + 1]
    flip = (fwd[i] == fwd[i + 1]).astype(np.int8)       # same direction in both faces: one of them is turned
    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(nf, nf)).tocsr()
    g = g + g.T
    par = {}
    for x, y, w in zip(a.tolist(), b.tolist(), flip.tolist()):
        par[(x, y)] = w; par[(y, x)] = w
    ncomp, lab = connected_components(g, directed=False)
    turned = np.zeros(nf, bool)
    seen = np.zeros(nf, bool)
    for c in range(ncomp):
        root = int(np.nonzero(lab == c)[0][0])
        order, pred = breadth_first_order(g, root, directed=False)
        for n in order[1:].tolist():
            turned[n] = turned[pred[n]] ^ bool(par[(pred[n], n)])
        seen[order] = True
    f2 = f.copy()
    f2[turned] = f2[turned][:, ::-1]
    tri = v[f2].astype(np.float64)
    if np.einsum('ij,ij->i', tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() < 0:
        f2 = f2[:, ::-1].copy()
        turned = ~turned
    return f2, int(turned.sum())


def mesh_path(key, h=0.02, faces=150000):
    return f'geo/{key}_h{int(round(h * 1000))}_q{faces // 1000}k.npz'


def ensure_mesh(key, h=0.02, faces=150000, device=None, log=print):
    """Surface triangles of the implicit body, made once and kept in geo/. Returns (relative path, seconds or None).
    device: 'cpu' evaluates the field on the processor (use it when the graphics card is busy)."""
    key = key.upper()
    rel = mesh_path(key, h, faces)
    if (HERE / rel).exists():
        return rel, None
    t = time.perf_counter()
    env = dict(os.environ)
    if device:
        env['FIELDLAB_DEVICE'] = device
    r = subprocess.run([STUDY_PY, str(HERE / 'sdf_bridge.py'), 'mesh', key, '--h', str(h)], capture_output=True, text=True, env=env)
    if r.returncode:
        raise RuntimeError('field to triangles failed:\n' + r.stderr[-2000:])
    log('field -> triangles: ' + r.stdout.strip().splitlines()[-1])
    from decimate import decimate
    out, meta = decimate(str(HERE / f'geo/{key}_h{int(round(h * 1000))}.npz'), faces)
    assert Path(out).resolve() == (HERE / rel).resolve(), (out, rel)
    dt = time.perf_counter() - t
    log(f'thinned to {meta["faces"]} triangles, open edges {meta["open_edges"]}, {dt:.1f} s in all')
    return rel, dt
