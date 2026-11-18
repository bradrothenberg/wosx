"""The panel code of panel_check.py on an implicit body (a study variant or a test wing), from the body's field.

  python panel_body.py check            the test wing built from its field against the wing built from its formula
  python panel_body.py run V03 0.7      grid study at Mach 0.7 -> results/panel_V03_M0.7.json
  python panel_body.py pressure V03     Mach 0: lift from surface pressure against lift from the wake

Geometry: geo/<key>_sections.npz (panel_sections.py: the field's surface on a station x chord-fraction grid).
A blunt trailing edge is closed for the panel model: half the base thickness is taken off each side, blended in over
the last `te_blend` of the chord. The wake leaves from the closed edge, straight aft.
Mach number: Goethert rule, the same as in wos_lift.py. The body is stretched in x by 1 / beta and solved as
incompressible; small-angle lift slope = 4 sum(Gamma_z dy) / S, lift at zero angle = 4 sum(Gamma_x dy) / (S beta).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import bodies
import panel_check as PC

HERE = Path(__file__).resolve().parent


def sections(key):
    d = np.load(HERE / 'geo' / f'{key}_sections.npz')
    return {k: d[k] for k in ('ys', 'xi', 'xle', 'xte', 'zu', 'zl')}, json.loads(str(d['meta']))


def points(sec, nc, ms, beta=1.0, te_blend=0.10):
    """Section loops Pt [ms + 1, 2 nc + 1, 3]: lower side from the trailing edge to the leading edge, then the upper
    side back. Returns Pt, station y, and whether the tip is closed (no thickness at the last station)."""
    b = sec['ys'][-1]
    ys = b * np.sin(0.5 * math.pi * np.arange(ms + 1) / ms)
    j = np.searchsorted(sec['ys'], ys - 1e-7)
    assert np.abs(sec['ys'][j] - ys).max() < 1e-6, 'span count not in the sections file'
    NC = len(sec['xi']) - 1
    assert NC % nc == 0, 'chord count not a divisor of the sections grid'
    i = np.arange(0, NC + 1, NC // nc)
    xi = sec['xi'][i]
    zu, zl = sec['zu'][j][:, i].copy(), sec['zl'][j][:, i].copy()
    base = (zu - zl)[:, -1]
    g = np.clip((xi - (1.0 - te_blend)) / te_blend, 0.0, 1.0)
    zu -= 0.5 * base[:, None] * g[None]; zl += 0.5 * base[:, None] * g[None]
    assert (zu - zl).min() > -1e-9
    zu[:, 0] = zl[:, 0] = 0.5 * (zu + zl)[:, 0]
    zu[:, -1] = zl[:, -1] = 0.5 * (zu + zl)[:, -1]
    closed_tip = bool((zu - zl)[-1].max() < 2e-3 * (zu - zl).max())
    if closed_tip:
        zu[-1] = zl[-1] = 0.5 * (zu + zl)[-1]
    x = (sec['xle'][j][:, None] + (sec['xte'] - sec['xle'])[j][:, None] * xi[None]) / beta
    Pt = np.zeros((ms + 1, 2 * nc + 1, 3))
    Pt[:, :nc + 1, 0] = x[:, ::-1]; Pt[:, :nc + 1, 2] = zl[:, ::-1]
    Pt[:, nc:, 0] = x; Pt[:, nc:, 2] = zu
    Pt[:, :, 1] = ys[:, None]
    return Pt, ys, closed_tip


def loft(Pt, ys, closed_tip, far=None, sym=True):
    """Triangles, wake and trailing-edge index sets from section loops (the same layout as panel_check.wing).
    sym: the quads of the upper side are cut along the mirror image of the diagonal used on the lower side, so that a
    body that is symmetric in z gets a symmetric panel set (and exactly zero lift at zero angle)."""
    ms, nc = Pt.shape[0] - 1, (Pt.shape[1] - 1) // 2
    far = far if far is not None else Pt[:, :, 0].max() + 400.0 * (Pt[0, :, 0].max() - Pt[0, :, 0].min())
    T, tag, qP = [], [], []
    for j in range(ms):
        for i in range(2 * nc):
            q = [Pt[j, i], Pt[j, i + 1], Pt[j + 1, i + 1], Pt[j + 1, i]]
            qP.append(0.5 * (q[1] + q[3]) if (sym and i >= nc) else 0.5 * (q[0] + q[2]))     # middle of the cut
            T += [(q[0], q[1], q[3]), (q[1], q[2], q[3])] if (sym and i >= nc) else [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
            tag += [(j, i), (j, i)]
    if not closed_tip:
        for i in range(nc):                               # flat tip cap
            q = [Pt[ms, i], Pt[ms, i + 1], Pt[ms, 2 * nc - i - 1], Pt[ms, 2 * nc - i]]
            ok = [tri for tri in ((q[0], q[1], q[2]), (q[0], q[2], q[3])) if np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) > 1e-14]
            T += ok; tag += [(-1, i)] * len(ok)
            qP.append(np.mean([np.mean(t, axis=0) for t in ok], axis=0))
    T = np.array(T); tag = np.array(tag)
    quad = np.where(tag[:, 0] >= 0, tag[:, 0] * 2 * nc + tag[:, 1], ms * 2 * nc + tag[:, 1])     # the quad each triangle belongs to
    A, B, C, vol = PC.orient(T[:, 0], T[:, 1], T[:, 2], mirror=True)
    area = PC.tri_geom(A, B, C)[2]
    assert area.min() > 1e-12, 'degenerate panel'
    te_lo = np.array([np.where((tag[:, 0] == j) & (tag[:, 1] == 0))[0] for j in range(ms)])
    te_up = np.array([np.where((tag[:, 0] == j) & (tag[:, 1] == 2 * nc - 1))[0] for j in range(ms)])
    WA, WB, WC, strip = [], [], [], []
    for j in range(ms):
        a, d = Pt[j, 0].copy(), Pt[j + 1, 0].copy()
        bb, cc = a.copy(), d.copy(); bb[0] = cc[0] = far
        WA += [a, a]; WB += [bb, cc]; WC += [cc, d]; strip += [j, j]
    return dict(A=A, B=B, C=C, wake=(np.array(WA), np.array(WB), np.array(WC), np.array(strip)), te_up=te_up, te_lo=te_lo, ys=ys, volume=vol, tag=tag, Pt=Pt,
                quad=quad, qP=np.array(qP), q_up=np.arange(ms) * 2 * nc + 2 * nc - 1, q_lo=np.arange(ms) * 2 * nc)


def solve(w, kutta_onset=False):
    """Wake strength per strip for a unit onset flow along x and along z (stretched frame), and the potentials."""
    res, geo = PC.solve(w['A'], w['B'], w['C'], [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)], mirror=True, wake=w['wake'], te_up=w['te_up'], te_lo=w['te_lo'], kutta_onset=kutta_onset)
    return res, geo


def solve_quads(w, onsets=((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))):
    """The classic low-order panel layout: one unknown per quad (its two triangles carry the same doublet), the wall
    condition at the middle of the quad. Returns [(phi per quad, wake strength per strip)] per onset flow."""
    import scipy.linalg as sla
    A, B, C, quad, P = w['A'], w['B'], w['C'], w['quad'], w['qP']
    cen, nrm, area = PC.tri_geom(A, B, C)
    nq = len(P)
    mir = np.array([1.0, -1.0, 1.0])
    Om, S = PC.influence(P, A, B, C)
    own = quad[None, :] == np.arange(nq)[:, None]
    Om[own] = 0.0                                         # own quad: the point lies in its surface
    o2, s2 = PC.influence(P * mir, A, B, C)
    Om += o2; S += s2
    M = np.zeros((nq, nq))
    np.add.at(M.T, quad, Om.T)
    M[np.arange(nq), np.arange(nq)] -= 2.0 * math.pi
    WA, WB, WC, strip = w['wake']
    ns = len(w['q_up'])
    ow = PC.influence(P, WA, WB, WC)[0] + PC.influence(P * mir, WA, WB, WC)[0]
    W = np.zeros((nq, ns))
    np.add.at(W.T, strip, ow.T)
    K = np.zeros((ns, nq))
    K[np.arange(ns), w['q_up']] = 1.0; K[np.arange(ns), w['q_lo']] = -1.0
    lu = sla.lu_factor(M + W @ K)
    out = []
    for V in onsets:
        phi = sla.lu_solve(lu, -S @ (nrm @ np.asarray(V, float)))
        out.append((phi, K @ phi))
    return out, (cen, nrm, area)


def lift(w, res, S, beta):
    dy = np.diff(w['ys'])
    gx, gz = res[0][1], res[1][1]
    return 4.0 * float((gz * dy).sum()) / S, 4.0 * float((gx * dy).sum()) / (S * beta)


# ------------------------------------------------------------------------------------------------ pressure
def surface_velocity(w, geo, phi, V):
    """Velocity on each quad of the lofted grid (mean of its two triangles), from the potential on the surface:
    least-squares slope of the total potential over the neighbouring quads, in the tangent plane."""
    cen, nrm, area = geo
    tag = w['tag']
    ms, nq = int(tag[:, 0].max()) + 1, int(tag[:, 1].max()) + 1
    body = tag[:, 0] >= 0
    Q = np.zeros((ms, nq, 3)); Nq = np.zeros((ms, nq, 3)); Aq = np.zeros((ms, nq)); F = np.zeros((ms, nq))
    tot = phi + cen @ np.asarray(V, float)
    for a_, v_ in ((Q, cen * area[:, None]), (Nq, nrm * area[:, None])):
        np.add.at(a_, (tag[body, 0], tag[body, 1]), v_[body])
    np.add.at(Aq, (tag[body, 0], tag[body, 1]), area[body]); np.add.at(F, (tag[body, 0], tag[body, 1]), (tot * area)[body])
    Q /= Aq[..., None]; F /= Aq
    Avec = Nq.copy()                                      # area vector of the quad (sum of its triangles)
    Nq /= np.linalg.norm(Nq, axis=-1, keepdims=True)
    Vel = np.zeros((ms, nq, 3))
    mir = np.array([1.0, -1.0, 1.0])
    for j in range(ms):
        for i in range(nq):
            nb = []
            for dj, di in ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                jj, ii = j + dj, i + di
                if ii < 0 or ii >= nq or jj >= ms:
                    continue
                if jj < 0:                                # mirror image across y = 0
                    nb.append((Q[0, ii] * mir - Q[j, i], F[0, ii] - F[j, i]))
                else:
                    nb.append((Q[jj, ii] - Q[j, i], F[jj, ii] - F[j, i]))
            D = np.array([d for d, _ in nb]); df = np.array([f for _, f in nb])
            n = Nq[j, i]
            t1 = D[0] - (D[0] @ n) * n; t1 /= np.linalg.norm(t1); t2 = np.cross(n, t1)
            M = np.c_[D @ t1, D @ t2]
            wgt = 1.0 / np.linalg.norm(D, axis=1)
            g = np.linalg.lstsq(M * wgt[:, None], df * wgt, rcond=None)[0]
            Vel[j, i] = g[0] * t1 + g[1] * t2
    return Q, Avec, Vel


def pressure_lift(w, geo, res, S, alpha_deg):
    """Mach 0. Lift coefficient at alpha from the surface pressure, and from the wake strength."""
    a = math.radians(alpha_deg)
    V = np.array([math.cos(a), 0.0, math.sin(a)])
    phi = math.cos(a) * res[0][0] + math.sin(a) * res[1][0]
    Q, Avec, Vel = surface_velocity(w, geo, phi, V)
    cp = 1.0 - (Vel ** 2).sum(-1)
    Fc = -2.0 * (cp[..., None] * Avec).sum((0, 1)) / S    # both halves; the tip cap (if any) has no z area
    cl_p = Fc[2] * math.cos(a) - Fc[0] * math.sin(a)
    cd_p = Fc[0] * math.cos(a) + Fc[2] * math.sin(a)
    dy = np.diff(w['ys'])
    gam = math.cos(a) * res[0][1] + math.sin(a) * res[1][1]
    return float(cl_p), float(cd_p), 4.0 * float((gam * dy).sum()) / S, Q, cp


# ------------------------------------------------------------------------------------------------ runs
def check():
    """The same wing two ways: from its formula (panel_check.wing) and from its field (sections + loft)."""
    sec, meta = sections('W12T12')
    wg = bodies.get('W12T12')
    for nc, ms in ((24, 16), (32, 24)):
        Pt, ys, closed = points(sec, nc, ms)
        w = loft(Pt, ys, closed, sym=False)
        res, geo = solve(w)
        a_field, cl0 = lift(w, res, wg.area, 1.0)
        res_s, _ = solve(loft(Pt, ys, closed))
        a_sym, cl0_sym = lift(w, res_s, wg.area, 1.0)
        print(f'   mirrored diagonals on the upper side: lift slope {a_sym:.5f} ({100 * (a_sym / a_field - 1):+.3f} %), CL0 {cl0_sym:+.2e}')
        w2 = PC.wing(wg.cr, wg.ct, wg.semispan, wg.m, wg.tc, nc=nc, ms=ms)
        a_formula, _ = PC.lift_slope(w2)
        print(f'W12T12 nc {nc} ms {ms}: panels {len(w["A"])} / {len(w2["A"])}, closed tip {closed}, largest point difference {np.abs(np.sort(w["A"], 0) - np.sort(w2["A"], 0)).max():.1e} m, '
              f'lift slope from field {a_field:.5f}, from formula {a_formula:.5f} ({100 * (a_field / a_formula - 1):+.3f} %), CL0 {cl0:+.5f}', flush=True)


def run(key, mach, grids=((24, 16), (32, 24), (48, 32), (64, 40), (96, 48)), tag=''):
    sec, meta = sections(key)
    ref = bodies.reference(key)
    S, beta = ref['s_ref_m2'], math.sqrt(1.0 - mach * mach)
    name = HERE / 'results' / f'panel_{key}_M{mach:.1f}{tag}.json'
    out = dict(variant=key, mach=mach, reference=ref, sections=meta, grids=[],
               note='small-angle lift slope per rad and lift at zero angle from the wake strength; Goethert rule; blunt trailing edge closed over the last te_blend of the chord')
    for g in grids:
        nc, ms = g[0], g[1]
        opt = g[2] if len(g) > 2 else {}
        t0 = time.perf_counter()
        Pt, ys, closed = points(sec, nc, ms, beta=beta, te_blend=opt.get('te_blend', 0.10))
        w = loft(Pt, ys, closed)
        res, geo = solve(w, kutta_onset=opt.get('kutta_onset', False))
        cla, cl0 = lift(w, res, S, beta)
        dt = time.perf_counter() - t0
        dy = np.diff(ys)
        out['grids'].append(dict(nc=nc, ms=ms, panels=int(len(w['A'])), options=opt, closed_tip=closed, volume_m3=float(w['volume'] * beta), cl_alpha=cla, cl0=cl0, seconds=dt,
                                 span_y=(0.5 * (ys[1:] + ys[:-1])).tolist(), dy=dy.tolist(), gamma_z=res[1][1].tolist(), gamma_x=(res[0][1] / beta).tolist()))
        print(f'{key} Mach {mach} nc {nc:3d} ms {ms:3d} {opt} panels {len(w["A"]):6d}  CL_alpha {cla:.4f}  CL0 {cl0:+.5f}  volume {w["volume"] * beta:.3f} m3  {dt:.0f} s', flush=True)
        name.write_text(json.dumps(out, indent=1))
    return out


def wing_points(cr, ct, b, tan_le, tc, nc, ms, camber=None):
    """Section loops of a tapered wing with NACA 00xx sections, from the formula (the points of panel_check.wing)."""
    th = np.linspace(0, math.pi, nc + 1)
    xu = 0.5 * (1 - np.cos(th))
    xi = np.r_[xu[::-1], xu[1:]]
    sgn = np.r_[-np.ones(nc + 1), np.ones(nc)]
    N = PC.NACA
    yt = 5.0 * tc * (N[0] * np.sqrt(xi) + xi * (N[1] + xi * (N[2] + xi * (N[3] + xi * N[4]))))
    yt[0] = yt[-1] = 0.0
    ys = b * np.sin(0.5 * math.pi * np.arange(ms + 1) / ms)
    Pt = np.zeros((ms + 1, 2 * nc + 1, 3))
    for j, y in enumerate(ys):
        c = cr + (ct - cr) * y / b
        Pt[j, :, 0] = tan_le * y + c * xi; Pt[j, :, 1] = y; Pt[j, :, 2] = sgn * yt * c
    return Pt, ys


def run_wing(grids=((24, 16), (32, 24), (48, 32), (64, 40)), tcs=(0.03, 0.06, 0.09, 0.12, 0.15, 0.18), tag=''):
    """Warren 12 again, with both ways to cut the quads: 'same' (panel_check.wing) and 'mirrored' (symmetric in z)."""
    W = PC.W12
    S = (W['cr'] + W['ct']) * W['b']
    name = HERE / 'results' / f'panel_wing_diag{tag}.json'
    out = dict(planform='Warren 12', published_per_rad=2.743, grids=[])
    for nc, ms in grids:
        for tc in tcs:
            Pt, ys = wing_points(W['cr'], W['ct'], W['b'], W['tan_le'], tc, nc, ms)
            row = dict(nc=nc, ms=ms, tc=tc)
            for lab, sym in (('quads', True), ('mirrored', True), ('same', False)):
                t0 = time.perf_counter()
                w = loft(Pt, ys, False, sym=sym)
                res, geo = solve_quads(w) if lab == 'quads' else solve(w)
                cla, cl0 = lift(w, res, S, 1.0)
                row[lab] = dict(cl_alpha=cla, cl0=cl0, seconds=time.perf_counter() - t0)
                row['panels'] = int(len(w['A']))
            out['grids'].append(row)
            print(f"nc {nc:3d} ms {ms:3d} t/c {100 * tc:4.0f} %  panels {row['panels']:6d}  quads {row['quads']['cl_alpha']:.4f} ({row['quads']['seconds']:.0f} s)  mirrored {row['mirrored']['cl_alpha']:.4f} (CL0 {row['mirrored']['cl0']:+.1e})  same {row['same']['cl_alpha']:.4f} (CL0 {row['same']['cl0']:+.4f})  {row['mirrored']['seconds']:.0f} s", flush=True)
            name.write_text(json.dumps(out, indent=1))


def check_camber(ncs=(24, 48, 96), x0=-0.08, y0=0.05, tau_deg=10.0):
    """A cambered section with an exact answer (Karman-Trefftz): lift at zero angle and lift slope of a very long wing."""
    n, a_ = 2.0 - math.radians(tau_deg) / math.pi, 1.0
    z0 = complex(x0, y0); R = abs(a_ - z0); beta = math.asin(y0 / R)

    def section(nc):
        th = -beta - 2.0 * math.pi * np.arange(2 * nc + 1) / (2 * nc)        # from the trailing edge, lower side first
        zeta = z0 + R * np.exp(1j * th)
        wn = ((zeta - a_) / (zeta + a_)) ** n
        z = n * a_ * (1 + wn) / (1 - wn)
        z[0] = z[-1] = n * a_
        return z

    zf = section(4000); c = zf.real.max() - zf.real.min()
    exact0, exact_a = 8 * math.pi * R * math.sin(beta) / c, 8 * math.pi * R * math.cos(beta) / c
    out = dict(section='Karman-Trefftz', x0=x0, y0=y0, trailing_edge_angle_deg=tau_deg, thickness_over_chord=float((zf.imag.max() - zf.imag.min()) / c),
               exact=dict(cl0=exact0, cl_alpha=exact_a), grids=[])
    print(f'section: thickness {100 * out["thickness_over_chord"]:.1f} % of chord, exact cl at zero angle {exact0:.4f}, exact slope {exact_a:.4f} per rad')
    for nc in ncs:
        z = section(nc)
        ys = 200.0 * c * np.arange(8) / 7
        Pt = np.zeros((8, 2 * nc + 1, 3))
        Pt[:, :, 0] = z.real[None]; Pt[:, :, 2] = z.imag[None]; Pt[:, :, 1] = ys[:, None]
        w = loft(Pt, ys, False, far=40000.0 * c)
        res, geo = solve(w)
        cl0, cla = 2.0 * res[0][1][0] / c, 2.0 * res[1][1][0] / c
        out['grids'].append(dict(nc=nc, cl0=cl0, cl_alpha=cla))
        print(f'  {2 * nc:4d} panels round the section: cl at zero angle {cl0:.4f} ({100 * (cl0 / exact0 - 1):+.1f} %), slope {cla:.4f} ({100 * (cla / exact_a - 1):+.1f} %)', flush=True)
    (HERE / 'results' / 'panel_camber.json').write_text(json.dumps(out, indent=1))


def run_pressure(key, grids=((24, 16), (32, 24), (48, 32), (64, 40)), alpha=4.0):
    sec, meta = sections(key)
    S = bodies.reference(key)['s_ref_m2']
    out = dict(variant=key, mach=0.0, alpha_deg=alpha, grids=[])
    for nc, ms in grids:
        Pt, ys, closed = points(sec, nc, ms)
        w = loft(Pt, ys, closed)
        res, geo = solve(w)
        cl_p, cd_p, cl_w, Q, cp = pressure_lift(w, geo, res, S, alpha)
        out['grids'].append(dict(nc=nc, ms=ms, panels=int(len(w['A'])), cl_pressure=cl_p, cd_pressure=cd_p, cl_wake=cl_w, cp_min=float(cp.min()), cp_max=float(cp.max())))
        print(f'{key} Mach 0, {alpha} deg, nc {nc} ms {ms}: CL from pressure {cl_p:.4f}, from the wake {cl_w:.4f} ({100 * (cl_p / cl_w - 1):+.1f} %), drag from pressure {cd_p:+.5f}, Cp {cp.min():.2f} to {cp.max():.2f}', flush=True)
        (HERE / 'results' / f'panel_pressure_{key}.json').write_text(json.dumps(out, indent=1))
        np.savez_compressed(HERE / 'results' / f'panel_cp_{key}_nc{nc}.npz', Q=Q, cp=cp, alpha_deg=alpha)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'check'
    if what == 'check':
        check()
    elif what == 'run':
        run(sys.argv[2].upper(), float(sys.argv[3]))
    elif what == 'wing':
        run_wing()
    elif what == 'camber':
        check_camber()
    elif what == 'wingq':
        run_wing(grids=((24, 16), (32, 24), (48, 32), (64, 40), (96, 48), (64, 56), (96, 24), (128, 64)), tcs=(0.12,), tag='_q12')
    elif what == 'wing12':
        run_wing(grids=((24, 16), (32, 24), (48, 32), (64, 40), (96, 48), (64, 56), (96, 24)), tcs=(0.12,), tag='_t12')
    elif what == 'pressure':
        run_pressure(sys.argv[2].upper())
