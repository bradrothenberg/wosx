"""A small panel code, to check how much section thickness raises the lift of the Warren 12 test wing.

Potential flow on the real thick surface, the classic way (Morino): flat triangles, each with a constant source and a
constant doublet, the potential inside the body set to zero, a flat wake behind the trailing edge. CPU, numpy.
It is independent of WoSX and of the vortex lattice: no random walks, no horseshoe vortices, no read-out points.

  python panel_check.py kernels     the two panel integrals against brute-force sums
  python panel_check.py sphere      flow past a sphere (exact answer)
  python panel_check.py section     a very long wing: lift of the 2-D section (known thick-section values)
  python panel_check.py wing        Warren 12 at several thicknesses and grids -> results/panel_wing.json
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NACA = (0.2969, -0.1260, -0.3516, 0.2843, -0.1036)        # closed trailing edge, as in bodies.py


# ------------------------------------------------------------------------------------------------ panel integrals
def influence(P, A, B, C, chunk=160):
    """P [n,3] field points; triangles A, B, C [m,3], counter-clockwise seen from the side the normal points to.
    Returns Om [n,m] = integral of n.(P-Q)/r^3 dS (the solid angle, positive on the normal side)
    and      S [n,m] = integral of dS/r."""
    nrm = np.cross(B - A, C - A)
    nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
    Om = np.empty((len(P), len(A))); S = np.empty_like(Om)
    edges = []
    for a, b in ((A, B), (B, C), (C, A)):
        e = b - a; L = np.linalg.norm(e, axis=1)
        edges.append((L, np.cross(e / L[:, None], nrm)))  # length, in-plane normal pointing out of the triangle
    for s in range(0, len(P), chunk):
        p = P[s:s + chunk, None, :]
        r = [A[None] - p, B[None] - p, C[None] - p]
        l = [np.linalg.norm(x, axis=-1) for x in r]
        num = (r[0] * np.cross(r[1], r[2])).sum(-1)
        den = l[0] * l[1] * l[2] + (r[0] * r[1]).sum(-1) * l[2] + (r[1] * r[2]).sum(-1) * l[0] + (r[2] * r[0]).sum(-1) * l[1]
        om = -2.0 * np.arctan2(num, den)
        z = -(r[0] * nrm[None]).sum(-1)
        sv = -np.abs(z) * np.abs(om)
        for k, (L, m) in enumerate(edges):
            h = (r[k] * m[None]).sum(-1)                  # distance of the point from the edge, positive on the inner side
            la, lb = l[k], l[(k + 1) % 3]
            sv += h * np.log((la + lb + L[None]) / np.maximum(la + lb - L[None], 1e-300))
        Om[s:s + chunk] = om; S[s:s + chunk] = sv
    return Om, S


def tri_geom(A, B, C):
    n = np.cross(B - A, C - A); a2 = np.linalg.norm(n, axis=1)
    return (A + B + C) / 3.0, n / a2[:, None], 0.5 * a2


# ------------------------------------------------------------------------------------------------ solver
def solve(A, B, C, onsets, mirror=False, wake=None, te_up=None, te_lo=None, kutta_onset=False):
    """Perturbation potential phi on every triangle for each onset flow (unit vectors).
    mirror: the body is the half y >= 0 and the flow is symmetric in y.
    wake: (WA, WB, WC, strip) wake triangles (normal +z) and the strip each belongs to; te_up / te_lo [strips, k]: triangle
    indices next to the trailing edge whose mean potential sets the wake strength of the strip."""
    cen, nrm, area = tri_geom(A, B, C)
    N = len(A)
    Om, S = influence(cen, A, B, C)
    Om[np.arange(N), np.arange(N)] = -2.0 * math.pi       # the point sits just inside its own panel
    if mirror:
        cm = cen * np.array([1.0, -1.0, 1.0])
        o2, s2 = influence(cm, A, B, C)
        Om += o2; S += s2
    K = None
    if wake is not None:
        WA, WB, WC, strip = wake
        ns = te_up.shape[0]
        ow, _ = influence(cen, WA, WB, WC)
        if mirror:
            ow += influence(cm, WA, WB, WC)[0]
        W = np.zeros((N, ns))
        np.add.at(W.T, strip, ow.T)
        K = np.zeros((ns, N))
        for k in range(ns):
            K[k, te_up[k]] += 1.0 / te_up.shape[1]
            K[k, te_lo[k]] -= 1.0 / te_lo.shape[1]
        M = Om + W @ K
    else:
        M = Om
    out = []
    lu = None
    import scipy.linalg as sla
    lu = sla.lu_factor(M)
    for V in onsets:
        V = np.asarray(V, float)
        rhs = -S @ (nrm @ V)
        if K is not None and kutta_onset:                 # variant: the wake carries the jump of the TOTAL potential at the edge panels
            dz = np.array([(cen[te_up[k]] @ V).mean() - (cen[te_lo[k]] @ V).mean() for k in range(K.shape[0])])
            rhs = rhs - W @ dz
        phi = sla.lu_solve(lu, rhs)
        gam = (K @ phi + (dz if (K is not None and kutta_onset) else 0.0)) if K is not None else None
        out.append((phi, gam))
    return out, (cen, nrm, area)


# ------------------------------------------------------------------------------------------------ bodies
def sphere(nt=24, nphi=48):
    th = np.linspace(0, math.pi, nt + 1); ph = np.linspace(0, 2 * math.pi, nphi + 1)
    pt = lambda i, j: np.array([math.cos(th[i]), math.sin(th[i]) * math.cos(ph[j]), math.sin(th[i]) * math.sin(ph[j])])
    T = []
    for i in range(nt):
        for j in range(nphi):
            q = [pt(i, j), pt(i + 1, j), pt(i + 1, j + 1), pt(i, j + 1)]
            for tri in ((q[0], q[1], q[2]), (q[0], q[2], q[3])):
                if np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) > 1e-12:
                    T.append(tri)
    T = np.array(T)
    return orient(T[:, 0], T[:, 1], T[:, 2])


def orient(A, B, C, mirror=False):
    cen, nrm, area = tri_geom(A, B, C)
    vol = ((cen * nrm).sum(1) * area).sum() / 3.0 * (2.0 if mirror else 1.0)
    if vol < 0:
        B, C = C, B
    return A, B, C, abs(vol)


def wing(cr, ct, b, tan_le, tc, nc=32, ms=24, far=400.0, span_spacing='tip'):
    """Half wing y in [0, b], NACA 00xx sections, flat cut tip. Returns triangles, wake and trailing-edge index sets."""
    th = np.linspace(0, math.pi, nc + 1)
    xu = 0.5 * (1 - np.cos(th))                           # 0 -> 1
    xi = np.r_[xu[::-1], xu[1:]]                          # lower: 1 -> 0, upper: 0 -> 1 ; 2 nc + 1 points
    sgn = np.r_[-np.ones(nc + 1), np.ones(nc)]
    yt = 5.0 * tc * (NACA[0] * np.sqrt(xi) + xi * (NACA[1] + xi * (NACA[2] + xi * (NACA[3] + xi * NACA[4]))))
    yt[0] = yt[-1] = 0.0
    k = np.arange(ms + 1) / ms
    ys = b * (np.sin(0.5 * math.pi * k) if span_spacing == 'tip' else k)
    Pt = np.zeros((ms + 1, 2 * nc + 1, 3))
    for j, y in enumerate(ys):
        c = cr + (ct - cr) * y / b
        Pt[j, :, 0] = tan_le * y + c * xi; Pt[j, :, 1] = y; Pt[j, :, 2] = sgn * yt * c
    T, tag = [], []
    for j in range(ms):
        for i in range(2 * nc):
            q = [Pt[j, i], Pt[j, i + 1], Pt[j + 1, i + 1], Pt[j + 1, i]]
            T += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
            tag += [(j, i), (j, i)]
    for i in range(nc):                                   # flat tip cap, normal +y
        q = [Pt[ms, i], Pt[ms, i + 1], Pt[ms, 2 * nc - i - 1], Pt[ms, 2 * nc - i]]
        for tri in ((q[0], q[1], q[2]), (q[0], q[2], q[3])):
            if np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) > 1e-14:
                T.append(tri); tag.append((-1, -1))
    T = np.array(T); tag = np.array(tag)
    A, B, C, vol = orient(T[:, 0], T[:, 1], T[:, 2], mirror=True)
    te_lo = np.array([np.where((tag[:, 0] == j) & (tag[:, 1] == 0))[0] for j in range(ms)])
    te_up = np.array([np.where((tag[:, 0] == j) & (tag[:, 1] == 2 * nc - 1))[0] for j in range(ms)])
    WA, WB, WC, strip = [], [], [], []
    for j in range(ms):
        a, d = Pt[j, 0].copy(), Pt[j + 1, 0].copy()
        bb, cc = a.copy(), d.copy(); bb[0] = cc[0] = far
        WA += [a, a]; WB += [bb, cc]; WC += [cc, d]; strip += [j, j]
    S = (cr + ct) * b
    return dict(A=A, B=B, C=C, wake=(np.array(WA), np.array(WB), np.array(WC), np.array(strip)), te_up=te_up, te_lo=te_lo, ys=ys, S=S, volume=vol, tag=tag, chord=lambda y: cr + (ct - cr) * y / b)


def lift_slope(w, kutta_onset=False):
    """Small-angle lift slope per rad from the wake strength for a unit onset flow along z."""
    (_, (phi, gam)), geo = (lambda r: ((None, r[0][0]), r[1]))(solve(w['A'], w['B'], w['C'], [(0.0, 0.0, 1.0)], mirror=True, wake=w['wake'], te_up=w['te_up'], te_lo=w['te_lo'], kutta_onset=kutta_onset))
    dy = np.diff(w['ys'])
    return 4.0 * float((gam * dy).sum()) / w['S'], gam


# ------------------------------------------------------------------------------------------------ checks
def check_kernels():
    rng = np.random.default_rng(1)
    A, B, C = np.array([[0.0, 0.0, 0.0]]), np.array([[1.0, 0.1, 0.0]]), np.array([[0.2, 0.9, 0.3]])
    cen, nrm, area = tri_geom(A, B, C)
    n = 700                                              # brute force: many small pieces
    u, v = np.meshgrid((np.arange(n) + 0.5) / n, (np.arange(n) + 0.5) / n)
    keep = u + v < 1
    u, v = u[keep], v[keep]
    Q = A + u[:, None] * (B - A) + v[:, None] * (C - A); dS = area[0] / len(Q)
    worst = [0.0, 0.0]
    for P in np.r_[rng.normal(size=(6, 3)) * 0.8 + cen, cen + 0.4 * nrm, cen - 0.05 * nrm]:
        d = P - Q; r = np.linalg.norm(d, axis=1)
        om_ref = ((d @ nrm[0]) / r ** 3).sum() * dS; s_ref = (1 / r).sum() * dS
        om, s = influence(P[None], A, B, C)
        worst = [max(worst[0], abs(om[0, 0] - om_ref)), max(worst[1], abs(s[0, 0] - s_ref) / s_ref)]
        print(f'  point {np.round(P, 2)}: solid angle {om[0, 0]:+.5f} (sum {om_ref:+.5f})   source {s[0, 0]:.5f} (sum {s_ref:.5f})')
    print(f'largest difference: solid angle {worst[0]:.1e}, source {100 * worst[1]:.3f} %')


def check_sphere():
    for nt, nphi in ((12, 24), (24, 48), (36, 72)):
        A, B, C, vol = sphere(nt, nphi)
        res, (cen, nrm, area) = solve(A, B, C, [(1.0, 0.0, 0.0)])
        phi = res[0][0]
        exact = 0.5 * cen[:, 0] / np.linalg.norm(cen, axis=1) ** 3      # perturbation potential of a unit sphere, at the panel centre
        err = math.sqrt(((phi - exact) ** 2 * area).sum() / (exact ** 2 * area).sum())
        print(f'sphere {len(A):5d} panels: volume {vol / (4 * math.pi / 3):.4f} of exact, potential error {100 * err:.2f} % (rms)')


def check_section(tcs=(0.06, 0.12, 0.18), ncs=(24, 48, 96)):
    """Rectangular wing 400 chords long: the middle is a 2-D section. cl_alpha = 2 Gamma / c."""
    out = {}
    for tc in tcs:
        row = []
        for nc in ncs:
            w = wing(1.0, 1.0, 200.0, 0.0, tc, nc=nc, ms=7, span_spacing='uniform', far=40000.0)
            _, gam = lift_slope(w)
            row.append(2.0 * gam[0])
        rich = row[-1] + (row[-1] - row[-2])              # error falls like 1 / panels (seen below)
        out[tc] = dict(nc=list(ncs), cl_alpha=row, extrapolated=rich)
        print(f't/c {100 * tc:4.0f} %: ' + '  '.join(f'{n} panels {v:.4f}' for n, v in zip(ncs, row)) + f'   -> {rich:.4f} = 2 pi x {rich / (2 * math.pi):.4f}')
    return out


W12 = dict(cr=1.5, ct=0.5, b=math.sqrt(2.0), tan_le=1.91421 / math.sqrt(2.0))


def run_wing():
    out = dict(planform='Warren 12', published_per_rad=2.743, grids=[], note='small-angle lift slope per rad from the wake strength; flat wake in the plane z = 0')
    grids = ((24, 16), (32, 24), (48, 32), (64, 40))
    tcs = (0.03, 0.06, 0.09, 0.12, 0.15, 0.18)
    for nc, ms in grids:
        for tc in tcs:
            t0 = time.perf_counter()
            w = wing(tc=tc, nc=nc, ms=ms, **W12)
            cla, gam = lift_slope(w)
            cla2, _ = lift_slope(w, kutta_onset=True) if (nc, ms) == grids[1] else (None, None)
            dt = time.perf_counter() - t0
            out['grids'].append(dict(nc=nc, ms=ms, tc=tc, panels=int(len(w['A'])), cl_alpha=cla, cl_alpha_total_jump=cla2, seconds=dt))
            print(f'nc {nc:3d} ms {ms:3d} t/c {100 * tc:4.0f} %  panels {len(w["A"]):6d}  CL_alpha {cla:.4f}' + (f'  (total-jump variant {cla2:.4f})' if cla2 else '') + f'  {dt:.0f} s', flush=True)
            (HERE / 'results' / 'panel_wing.json').write_text(json.dumps(out, indent=1))
    return out


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'kernels'
    if what == 'kernels':
        check_kernels()
    elif what == 'sphere':
        check_sphere()
    elif what == 'section':
        r = check_section()
        (HERE / 'results' / 'panel_section.json').write_text(json.dumps({str(k): v for k, v in r.items()}, indent=1))
    elif what == 'wing':
        run_wing()
