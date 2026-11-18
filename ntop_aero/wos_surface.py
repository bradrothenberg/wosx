"""Direct Walk on Stars estimates of the potential ON the body surface, and the trailing-edge (Kutta)
condition from them.

Near a thin trailing edge a flow that turns around the edge has a potential that behaves like
  u = regular + sigma * K * sqrt(s)        (s = distance ahead of the edge, sigma = +1 upper, -1 lower surface)
The flow leaves the edge smoothly exactly when K = 0. K of every channel is found by a least-squares fit over
many surface points per strip, so the noise of the single estimates averages out. No velocity estimate close
to the wall is needed (those are the noisy ones)."""
from __future__ import annotations

import math
import time

import numpy as np


def sample_on_mesh(V, F, mask, n, rng):
    """n random points on the triangles in mask (area weighted). Returns points, unit normals."""
    tri = V[F[mask]]
    nn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(nn, axis=1)
    pick = rng.choice(len(tri), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s1 = np.sqrt(r1)
    b = np.c_[1 - s1, s1 * (1 - r2), s1 * r2]
    p = (tri[pick] * b[:, :, None]).sum(1)
    nrm = nn[pick] / np.maximum(np.linalg.norm(nn[pick], axis=1), 1e-30)[:, None]
    return p, nrm


def boundary_potential(S, pts_m, normals_m, walks, eps=1e-4, max_walk=10000, rr=0.99):
    """u [n, channels] (normalised) at points on the body surface. normals_m: outward body normals."""
    wosx = S.wosx
    ch = dict(dim=3, channels=S.channels)
    x = S.to_n(pts_m)
    r2 = (x * x).sum(1)
    y = (x / r2[:, None])
    nd = -np.asarray(normals_m, float)                                   # outward from the fluid domain
    N = nd - 2.0 * (nd * x).sum(1)[:, None] * x / r2[:, None]            # Kelvin-transformed normal
    sp = []
    for q, m in zip(y.astype(np.float32), N.astype(np.float32)):
        p = wosx.Solvers.GPUSamplePoint(dim=3)
        p.pt = wosx.GPUFloat3(float(q[0]), float(q[1]), float(q[2]))
        p.normal = wosx.GPUFloat3(float(m[0]), float(m[1]), float(m[2]))
        p.type = wosx.Solvers.SampleType.OnReflectingBoundary
        p.estimation_quantity = wosx.Solvers.EstimationQuantity.Solution
        p.pdf = 1.0
        sp.append(p)
    sp = wosx.Solvers.GPUSamplePointList(sp, dim=3)
    ws = wosx.Solvers.GPUWalkSettings()
    ws.epsilon_shell_for_absorbing_boundary = eps
    ws.epsilon_shell_for_reflecting_boundary = eps
    ws.silhouette_precision = eps
    ws.russian_roulette_threshold = rr
    ws.max_walk_length = max_walk
    ws.steps_before_using_maximal_spheres = max_walk
    ws.solve_double_sided = 0
    ws.use_gradient_control_variates = 1
    ws.use_gradient_antithetic_variates = 1
    ws.ignore_absorbing_boundary_contribution = 1
    ws.ignore_reflecting_boundary_contribution = 0
    ws.ignore_source_contribution = 1
    t0 = time.perf_counter()
    solver = wosx.Solvers.GPUWalkOnStarsSolver(S.task, ws, 131072, True, False, **ch)
    solver.populate_sample_points(sp)
    t1 = time.perf_counter()
    solver.solve(walks)
    t2 = time.perf_counter()
    st = wosx.Solvers.GPUSampleStatisticsList(**ch)
    solver.get_sample_statistics(st)
    n = len(y)
    Vs = np.zeros((n, S.channels))
    wl = np.zeros(n)
    for i in range(n):
        Vs[i] = np.asarray(st[i].get_estimated_solution()).reshape(-1)
        wl[i] = st[i].get_mean_walk_length()
    ry = np.linalg.norm(y, axis=1)
    return Vs * ry[:, None], wl, dict(setup=t1 - t0, walks=t2 - t1, read=time.perf_counter() - t2)


def te_points(S, n_per, s_min, s_max, margin, rng):
    """Surface points in the band ahead of the trailing edge, both wings, per strip and side."""
    V, F, lat = S.V, S.F, S.lat
    tri = V[F]
    c = tri.mean(1)
    nz = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])[:, 2]
    xte = np.asarray(S.pf.xte(np.abs(c[:, 1])), float)
    s = xte - c[:, 0]
    chord = xte - np.asarray(S.pf.xle(np.abs(c[:, 1])), float)
    band = (s > s_min) & (s < np.minimum(s_max, 0.35 * chord))
    P, Nn, J, SG = [], [], [], []
    for j in range(lat.ns):
        ins = (np.abs(c[:, 1]) > lat.ya[j] + margin) & (np.abs(c[:, 1]) < lat.yb[j] - margin)
        for sg in (1.0, -1.0):
            m = band & ins & (nz * sg > 0)
            if m.sum() < 3:
                continue
            p, nrm = sample_on_mesh(V, F, m, n_per, rng)
            P.append(p); Nn.append(nrm); J.append(np.full(n_per, j)); SG.append(np.full(n_per, sg))
    P = np.vstack(P)
    return P, np.vstack(Nn), np.concatenate(J), np.concatenate(SG), np.asarray(S.pf.xte(np.abs(P[:, 1])), float) - P[:, 0]


def singular_coefficients(u, J, SG, s, ns):
    """K[j, channel]: coefficient of sigma * sqrt(s) per strip."""
    K = np.zeros((ns, u.shape[1]))
    resid = np.zeros(ns)
    for j in range(ns):
        m = J == j
        sj, sg = s[m], SG[m]
        B = np.c_[np.ones(m.sum()), sj, sg, sg * np.sqrt(sj), sg * sj]
        co, *_ = np.linalg.lstsq(B, u[m], rcond=None)
        K[j] = co[3]
        resid[j] = float(np.sqrt(np.mean((B @ co - u[m]) ** 2)))
    return K, resid


def kutta_from_K(K, ns):
    A = K[:, 3:3 + ns]
    gx = np.linalg.solve(A, -K[:, 0])
    gz = np.linalg.solve(A, -K[:, 1])
    return gx, gz, float(np.linalg.cond(A))


if __name__ == '__main__':
    import sys
    from wos_lift import WosLift
    from study_noise import ALPHAS
    mesh, n_per, walks, rep = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    s_max = float(sys.argv[5]) if len(sys.argv) > 5 else 0.6
    S = WosLift('V03', mesh, 16)
    rng = np.random.default_rng(1)
    for r in range(rep):
        P, Nn, J, SG, s = te_points(S, n_per, 0.03, s_max, 0.12, rng)
        u, wl, tm = boundary_potential(S, P, Nn, walks)
        K, resid = singular_coefficients(u, J, SG, s, S.lat.ns)
        gx, gz, cond = kutta_from_K(K, S.lat.ns)
        f = [S.forces(gx, gz, al) for al in ALPHAS]
        print(f'{len(P)} pts x {walks} walks: {tm["walks"]:.1f} s (+{tm["setup"]:.1f} setup, {tm["read"]:.1f} read) walk len {wl.mean():.0f} cond {cond:.1f}  '
              + '  '.join(f"{p['alpha_deg']:.0f}: {p['CL']:.4f}/{p['CDi']:.5f}" for p in f), flush=True)
        print('   Kx', np.round(K[:, 0], 3).tolist()); print('   Kz', np.round(K[:, 1], 2).tolist()); print('   Kdiag', np.round(np.diag(K[:, 3:]), 2).tolist(), 'resid', np.round(resid, 3).tolist())
        print('   gamma_z', np.round(gz * S.R, 2).tolist())
