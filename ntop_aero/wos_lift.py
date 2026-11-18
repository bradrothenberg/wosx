"""Lifting potential flow around a closed body with WoSX (Walk on Stars, GPU), from an nTop implicit model.

What the stock potential_flow demo does: one exterior Neumann problem for one free-stream direction. That
flow has no circulation, so it has no lift and no drag (d'Alembert). What this file adds:

1. A wake. The lift of a wing comes from circulation, which a single-valued potential cannot hold. A lattice
   of horseshoe vortices is placed INSIDE the body (bound lines on the chine plane z = 0, trailing legs
   leaving through the trailing edge). Their velocity field is known in closed form (Biot-Savart).
2. One GPU run, many right-hand sides. The equation is linear, so the walk-on-stars solve is done once with
   16 (or 64) channels on the same random walks: channel 0 and 1 are the body's answer to a unit free stream
   along x and along z, channel 2 along y, the remaining channels are the body's answer to each unit vortex
   strip (wall condition du/dn = -V_vortex . n).
3. A Kutta condition. The strip strengths are the ones for which the flow leaves the trailing edge smoothly
   (no flow through the wake plane just behind the edge). That is a small linear system, solved on the CPU.
   Because everything is linear in the free-stream direction, ONE run gives every angle of attack.
4. Forces from the wake (Trefftz plane): lift from the circulation, induced drag from the downwash.

Frame: the study's model frame (metres, x aft, y starboard, z up). Internally everything is shifted to the
inversion point and scaled to unit size, as the demo does.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WOSX_ROOT = HERE.parent
from paths import STUDY, STUDY_PY  # noqa: E402  (the study folder; a Python with torch and the study's field library)


# ---------------------------------------------------------------------------------------------- geometry
import bodies  # noqa: E402  (study variants and plain test wings, both implicit bodies)

planform, reference = bodies.planform, bodies.reference


def sdf_eval(key, pts, tag='q'):
    """Exact field value and gradient of the nTop implicit model at points (model frame, metres)."""
    tmp = HERE / 'tmp'
    tmp.mkdir(exist_ok=True)
    a, b = tmp / f'{tag}_pts.npy', tmp / f'{tag}_out.npz'
    np.save(a, np.asarray(pts, np.float32))
    t = time.perf_counter()
    subprocess.run([STUDY_PY, str(HERE / 'sdf_bridge.py'), 'eval', key, str(a), str(b)], check=True, capture_output=True)
    d = np.load(b)
    return d['value'], d['gradient'], dict(wall=time.perf_counter() - t, field_build=float(d['seconds_field_build']), evaluate=float(d['seconds_eval']))


class Stretched:
    """Planform stretched in x by 1 / beta (Goethert rule for the Mach number)."""

    def __init__(self, pf, beta):
        self.pf, self.beta, self.semispan = pf, beta, pf.semispan

    def xle(self, y):
        return np.asarray(self.pf.xle(y), float) / self.beta

    def xte(self, y):
        return np.asarray(self.pf.xte(y), float) / self.beta


class Lattice:
    """Horseshoe strips inside the body. ns strips per side between y = 0 and the tip, nc bound lines per
    strip at chord fractions `xi` with weights `w` (sum 1). All on z = z0. Geometry in metres, model frame."""

    def __init__(self, pf, ns, xi=(0.25,), w=(1.0,), z0=0.0, tip_inset=0.0, spacing='uniform'):
        s = pf.semispan - tip_inset
        if spacing == 'cosine':
            th = np.linspace(0.0, math.pi / 2, ns + 1)
            edges = s * np.sin(th)
        else:
            edges = np.linspace(0.0, s, ns + 1)
        self.ya, self.yb = edges[:-1], edges[1:]
        self.ym = 0.5 * (self.ya + self.yb)
        self.dy = self.yb - self.ya
        self.ns, self.nc = ns, len(xi)
        self.xi, self.w, self.z0 = np.asarray(xi, float), np.asarray(w, float), float(z0)
        xle, xte = (lambda y: np.asarray(pf.xle(y), float)), (lambda y: np.asarray(pf.xte(y), float))
        self.xte_mid = xte(self.ym)
        self.xle_mid = xle(self.ym)
        # bound line end points [strip, line, 3]
        self.A = np.zeros((ns, self.nc, 3))
        self.B = np.zeros((ns, self.nc, 3))
        for i, f in enumerate(self.xi):
            self.A[:, i] = np.c_[xle(self.ya) + f * (xte(self.ya) - xle(self.ya)), self.ya, np.full(ns, z0)]
            self.B[:, i] = np.c_[xle(self.yb) + f * (xte(self.yb) - xle(self.yb)), self.yb, np.full(ns, z0)]

    # ---- Biot-Savart of the unit strips (both sides), numpy, with the same core as the shader
    @staticmethod
    def _seg(p, a, b, core2, kernel='alg'):
        r1, r2 = p - a, p - b
        cr = np.cross(r1, r2)
        n1 = np.maximum(np.linalg.norm(r1, axis=-1), 1e-12)
        n2 = np.maximum(np.linalg.norm(r2, axis=-1), 1e-12)
        r0 = b - a
        if kernel == 'exp':
            c2 = (cr * cr).sum(-1)
            fac = -np.expm1(-c2 / (core2 * (r0 * r0).sum(-1)))
            k = ((r0 * (r1 / n1[..., None] - r2 / n2[..., None])).sum(-1)) * fac / (4 * math.pi * np.maximum(c2, 1e-30))
            return cr * k[..., None]
        d = (cr * cr).sum(-1) + core2 * (r0 * r0).sum(-1)
        k = ((r0 * (r1 / n1[..., None] - r2 / n2[..., None])).sum(-1)) / (4 * math.pi * np.maximum(d, 1e-30))
        return cr * k[..., None]

    @staticmethod
    def _leg(p, a, core2, kernel='alg'):
        r = p - a
        if kernel == 'exp':
            h2 = r[..., 1] ** 2 + r[..., 2] ** 2
            k = (1.0 + r[..., 0] / np.maximum(np.linalg.norm(r, axis=-1), 1e-12)) * -np.expm1(-h2 / core2) / (4 * math.pi * np.maximum(h2, 1e-30))
            return np.stack([np.zeros_like(k), -r[..., 2] * k, r[..., 1] * k], axis=-1)
        h2 = r[..., 1] ** 2 + r[..., 2] ** 2 + core2
        k = (1.0 + r[..., 0] / np.maximum(np.linalg.norm(r, axis=-1), 1e-12)) / (4 * math.pi * h2)
        return np.stack([np.zeros_like(k), -r[..., 2] * k, r[..., 1] * k], axis=-1)

    def velocity(self, p, core, scale=1.0, shift=0.0):
        """Velocity of each unit strip pair at points p: [n, ns, 3]. Lengths are (x - shift) * scale."""
        p = np.asarray(p, float)
        out = np.zeros((len(p), self.ns, 3))
        m = np.array([1.0, -1.0, 1.0])
        c2 = core * core
        for k in range(self.ns):
            for i in range(self.nc):
                a, b = (self.A[k, i] - shift) * scale, (self.B[k, i] - shift) * scale
                for (aa, bb) in ((a, b), (b * m, a * m)):
                    kn = getattr(self, 'kernel', 'alg')
                    out[:, k] += self.w[i] * (self._seg(p, aa, bb, c2, kn) - self._leg(p, aa, c2, kn) + self._leg(p, bb, c2, kn))
        return out


def write_pde(folder, lat, scale, shift, core, channels, kernel='alg'):
    """Writes wosx-pde-definition.slang with the lattice baked in as constants (normalised frame)."""
    ns, nc = lat.ns, lat.nc
    assert 3 + ns <= channels
    A = (lat.A - shift) * scale
    B = (lat.B - shift) * scale

    def arr(M):
        return ',\n        '.join('float3(%.9g, %.9g, %.9g)' % tuple(v) for v in M.reshape(-1, 3))

    src = f'''// Generated by ntop_aero/wos_lift.py. Lifting potential flow: free-stream channels plus one channel per
// horseshoe-vortex strip pair. Conforms to the IPDE interface of include/wosx/gpu/core/pde.slang.
module wosx_pde_definition;
import wosx;

static const int N_STRIP = {ns};
static const int N_LINE = {nc};
static const float CORE2 = {core * core:.9g};
static const bool KERNEL_EXP = {'true' if kernel == 'exp' else 'false'};
static const float INV4PI = 0.07957747154594767;
static const float3 HS_A[{ns * nc}] = {{
        {arr(A)}
}};
static const float3 HS_B[{ns * nc}] = {{
        {arr(B)}
}};
static const float HS_W[{nc}] = {{ {', '.join('%.9g' % v for v in lat.w)} }};

float3 segVel(float3 p, float3 a, float3 b)
{{
    float3 r1 = p - a;
    float3 r2 = p - b;
    float3 cr = cross(r1, r2);
    float3 r0 = b - a;
    float n1 = max(length(r1), 1e-12);
    float n2 = max(length(r2), 1e-12);
    float c2 = dot(cr, cr);
    if (KERNEL_EXP)
    {{
        float fac = 1.0 - exp(-c2 / (CORE2 * dot(r0, r0)));
        return cr * (dot(r0, r1 / n1 - r2 / n2) * INV4PI * fac / max(c2, 1e-30));
    }}
    float d = c2 + CORE2 * dot(r0, r0);
    return cr * (dot(r0, r1 / n1 - r2 / n2) * INV4PI / max(d, 1e-30));
}}

float3 legVel(float3 p, float3 a)
{{
    float3 r = p - a;
    float h0 = r.y * r.y + r.z * r.z;
    float h2 = h0 + CORE2;
    float k = (1.0 + r.x / max(length(r), 1e-12)) * INV4PI / h2;
    if (KERNEL_EXP)
    {{
        k = (1.0 + r.x / max(length(r), 1e-12)) * INV4PI * (1.0 - exp(-h0 / CORE2)) / max(h0, 1e-30);
    }}
    return float3(0.0, -r.z * k, r.y * k);
}}

float3 horseshoe(float3 p, float3 a, float3 b)
{{
    return segVel(p, a, b) - legVel(p, a) + legVel(p, b);
}}

public struct LiftingFlowPDE : IPDE<DataType, 3>
{{
    public typedef HarmonicGreensFnFreeSpace3D GreensFnFreeSpace;
    public typedef HarmonicGreensFnBall3D GreensFnBall;
    private float mUnused;

    public DataType source(float3 x)
    {{
        return DataType(0.0);
    }}

    public DataType dirichlet(float3 x, bool returnBoundaryNormalAlignedValue = false)
    {{
        return DataType(0.0);
    }}

    // no flow through the wall: du/dn = -(onset velocity) . n, one onset flow per channel
    public DataType robin(float3 x, float3 n, bool returnBoundaryNormalAlignedValue = false)
    {{
        DataType r = DataType(0.0);
        r[0] = -n.x;
        r[1] = -n.z;
        r[2] = -n.y;
        float3 m = float3(1.0, -1.0, 1.0);
        for (int k = 0; k < N_STRIP; k++)
        {{
            float3 v = float3(0.0);
            for (int i = 0; i < N_LINE; i++)
            {{
                float3 a = HS_A[k * N_LINE + i];
                float3 b = HS_B[k * N_LINE + i];
                v += HS_W[i] * (horseshoe(x, a, b) + horseshoe(x, b * m, a * m));
            }}
            r[3 + k] = -dot(v, n);
        }}
        return r;
    }}

    public float robinCoeff(float3 x, float3 n, bool returnBoundaryNormalAlignedValue = false)
    {{
        return 0.0;
    }}

    public float absorptionCoeff()
    {{
        return 0.0;
    }}

    public bool isSourceConstant()
    {{
        return true;
    }}

    public bool areRobinConditionsPureNeumann()
    {{
        return true;
    }}

    public bool areRobinCoeffsNonnegative()
    {{
        return true;
    }}
}};

public typedef KelvinPDE<DataType, LiftingFlowPDE, 3> PDEType;
'''
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'wosx-pde-definition.slang').write_text(src)


# ---------------------------------------------------------------------------------------------- solver
class Timer:
    def __init__(self):
        self.t, self.rec = time.perf_counter(), {}

    def lap(self, name):
        now = time.perf_counter()
        self.rec[name] = self.rec.get(name, 0.0) + now - self.t
        self.t = now


class WosLift:
    def __init__(self, key='V03', mesh='geo/V03_h20.npz', channels=16, ns=None, xi=(0.25,), w=(1.0,), core_m=0.05,
                 centre=None, spacing='uniform', backend='cuda', print_logs=False, mach=0.0, kernel='alg'):
        import wosx
        self.wosx = wosx
        self.key, self.channels = key, channels
        self.T = Timer()
        self.ref = reference(key)
        # Mach number by the Goethert rule: the incompressible problem is solved on the body stretched in x by
        # 1 / beta; the circulation of the real flow is that of the stretched flow at beta * alpha, divided by beta.
        self.mach, self.beta = float(mach), math.sqrt(1.0 - mach * mach)
        self.pf = Stretched(planform(key), self.beta)
        d = np.load(HERE / mesh)
        self.mesh_meta = json.loads(str(d['meta']))
        self.V, self.F = d['vertices'].astype(np.float64), d['faces'].astype(np.int32)
        self.V[:, 0] /= self.beta
        self.c = np.asarray(centre if centre is not None else bodies.centre(key), float) / np.array([self.beta, 1.0, 1.0])
        self.R = float(np.linalg.norm(self.V - self.c, axis=1).max())
        self.scale = 1.0 / self.R
        self.lat = Lattice(self.pf, ns or channels - 3, xi, w, spacing=spacing)
        self.core = core_m * self.scale
        self.kernel = self.lat.kernel = kernel
        self.pde_dir = HERE / 'pde_gen' / (f'{key}_c{channels}' + ('' if kernel == 'alg' else f'_{kernel}'))
        write_pde(self.pde_dir, self.lat, self.scale, self.c, self.core, channels, kernel)
        self.T.lap('setup_cpu')

        ch = dict(dim=3, channels=channels)
        self.task = wosx.GPUTaskHandle(str(WOSX_ROOT), str(self.pde_dir), backend, **ch)
        self.pos = wosx.Float3List(((self.V - self.c) * self.scale).astype(np.float32))
        self.idx = wosx.Int3List(self.F)
        self.inv_point = np.zeros(3, dtype=np.float32)
        self.kelvin = wosx.Solvers.KelvinTransform(self.inv_point, **ch)
        self.inv_pos = wosx.Float3List()
        self.kelvin.transform_points(self.pos, self.inv_pos)
        nf = len(self.F)
        zero = wosx.FloatList([0.0] * nf)
        self.rc_min, self.rc_max = wosx.FloatList(), wosx.FloatList()
        self.kelvin.compute_robin_coefficients(self.inv_pos, self.idx, zero, zero, self.rc_min, self.rc_max)
        self.bbox = wosx.Utils.compute_bounding_box(self.inv_pos, True, 1.0, dim=3)
        self.T.lap('invert_mesh')

        absorbing = wosx.Core.GPUEmptyAbsorbingBoundaryHandler()
        ignore = wosx.Utils.get_ignore_candidate_silhouette_callback(False)
        self.reflecting = wosx.Utils.GPUFcpwRobinBoundaryHandler(self.inv_pos, self.idx, ignore, self.rc_min, self.rc_max, dim=3)
        self.absorbing = absorbing
        self.gq = wosx.Core.GPUGeometricQueries(absorbing, self.reflecting, self.bbox[0], self.bbox[1], True, dim=3)

        class PDE(wosx.Core.GPUPDE):
            def allocate(self, context):
                pass

            def set_resources(self, cursor, print_logs):
                cursor['mUnused'].set_data_float(0.0)

            def get_reflection_type(self):
                return 'LiftingFlowPDE'

            def get_type(self):
                return wosx.Core.GPUPDEType.Poisson

        self.pde = PDE()
        self.kpde = wosx.Core.GPUKelvinPDE(self.pde, self.inv_point, **ch)
        inside = wosx.Utils.get_inside_bounding_box_callback(self.bbox[0], self.bbox[1], dim=3)
        self.sampler = wosx.Samplers.create_gpu_uniform_boundary_sampler(
            self.gq, self.inv_pos, self.idx, self._weights(), inside, 0.0, False, dim=3)
        self.task.set_geometric_queries(self.gq)
        self.task.set_reflecting_boundary_sampler(self.sampler)
        self.task.set_pde(self.kpde)
        self.task.init()
        self.T.lap('gpu_init_bvh_shaders')
        self.print_logs = print_logs

    def _weights(self):
        """Boundary samples uniform over the ORIGINAL surface area (1/|y|^4 per unit inverted area)."""
        P = self.wosx.convert_list_to_numpy_array(self.inv_pos)
        c = P[self.F].mean(axis=1)
        r2 = np.maximum((c * c).sum(1), 1e-12)
        return self.wosx.FloatList((1.0 / (r2 * r2)).astype(np.float32))

    # ---- model frame <-> normalised frame
    def to_n(self, p):
        return (np.asarray(p, float) - self.c) * self.scale

    def solve(self, pts_m, cache=250000, walks=8, eps=1e-4, max_walk=10000, rr=0.99, radius_clamp=0.0, regularization=0.0):
        """Body-response potentials and gradients of every channel at points (model frame, metres).
        Returns u [n, ch] (normalised length units, unit onset) and g [n, ch, 3] (velocity per unit onset)."""
        wosx = self.wosx
        ch = dict(dim=3, channels=self.channels)
        x = self.to_n(pts_m)
        r2 = (x * x).sum(1)
        y = (x / r2[:, None]).astype(np.float32)
        ev = []
        for q in y:
            e = wosx.Solvers.GPUBVCEvaluationPoint(dim=3)
            e.pt = wosx.GPUFloat3(float(q[0]), float(q[1]), float(q[2]))
            e.normal = wosx.GPUFloat3(0.0, 0.0, 0.0)
            e.type = wosx.Solvers.SampleType.InDomain
            ev.append(e)
        ev = wosx.Solvers.GPUBVCEvaluationPointList(ev, dim=3)
        self.T.lap('eval_points_cpu')
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
        bvc = wosx.Solvers.GPUBoundaryValueCachingSolver(self.task, ws, 131072, True, self.print_logs, **ch)
        bvc.populate_evaluation_points(ev)
        self.T.lap('gpu_solver_setup')
        bvc.generate_samples(0, cache, 0)
        self.T.lap('gpu_boundary_samples')
        bvc.compute_sample_estimates(walks, 0)
        self.T.lap('gpu_walks')
        bvc.splat(radius_clamp, regularization)
        self.T.lap('gpu_splat')
        out = wosx.Solvers.GPUBVCEvaluationOutputsList(**ch)
        bvc.get_evaluation_outputs(out)
        n, C = len(y), self.channels
        Vs = np.zeros((n, C))
        dV = np.zeros((n, C, 3))
        for i in range(n):
            o = out[i]
            Vs[i] = np.asarray(o.get_estimated_solution()).reshape(-1)
            for j in range(3):
                dV[i, :, j] = np.asarray(o.get_estimated_gradient(j)).reshape(-1)
        self.bvc = bvc
        # Kelvin transform back to the exterior domain (kelvin-transform.slang, transformGradientEstimate)
        ry = np.linalg.norm(y, axis=1)
        yy = y.astype(np.float64)
        u = Vs * ry[:, None]
        dU = dV * (ry ** 3)[:, None, None] + Vs[:, :, None] * (ry[:, None] * yy)[:, None, :]
        refl = np.eye(3)[None] - 2.0 * yy[:, :, None] * yy[:, None, :] / (ry ** 2)[:, None, None]
        g = np.einsum('nij,ncj->nci', refl, dU)
        self.T.lap('readback_cpu')
        return u, g

    # ---- total velocity per unit onset: columns [x onset, z onset, strips...]
    def basis_velocity(self, pts_m, g):
        n, ns = len(pts_m), self.lat.ns
        Vb = np.zeros((n, 2 + ns, 3))
        Vb[:, 0] = g[:, 0] + np.array([1.0, 0.0, 0.0])
        Vb[:, 1] = g[:, 1] + np.array([0.0, 0.0, 1.0])
        Vb[:, 2:] = g[:, 3:3 + ns] + self.lat.velocity(self.to_n(pts_m), self.core, 1.0, 0.0) if False else \
            g[:, 3:3 + ns] + Lattice.velocity(self._lat_n(), self.to_n(pts_m), self.core)
        return Vb

    def _lat_n(self):
        if not hasattr(self, '_ln'):
            import copy
            ln = copy.copy(self.lat)
            ln.A = (self.lat.A - self.c) * self.scale
            ln.B = (self.lat.B - self.c) * self.scale
            self._ln = ln
        return self._ln

    def kutta_points(self, d_m):
        lat = self.lat
        return np.c_[lat.xte_mid + d_m, lat.ym, np.full(lat.ns, lat.z0)]

    def kutta(self, Vb_k):
        """Vb_k: basis velocity at one Kutta point per strip [ns, 2 + ns, 3]. Returns gamma for unit x onset
        and unit z onset (normalised length units)."""
        A = Vb_k[:, 2:, 2]
        gx = np.linalg.solve(A, -Vb_k[:, 0, 2])
        gz = np.linalg.solve(A, -Vb_k[:, 1, 2])
        return gx, gz, float(np.linalg.cond(A))

    def forces(self, gx, gz, alpha_deg):
        """Lift and induced drag from the wake (Trefftz plane), on the study's reference area."""
        a = math.atan(self.beta * math.tan(math.radians(alpha_deg)))
        gam = (math.cos(a) * gx + math.sin(a) * gz) * self.R / self.beta          # m2/s per unit free-stream speed
        lat, S, b = self.lat, self.ref['s_ref_m2'], self.ref['b_ref_m']
        cl = float(4.0 * (gam * lat.dy).sum() / S)
        g2 = np.r_[gam, gam[::-1]]
        ya = np.r_[lat.ya, -lat.yb[::-1]]
        yb = np.r_[lat.yb, -lat.ya[::-1]]
        ym = 0.5 * (ya + yb)
        w = (g2[None] * (-1.0 / (ym[:, None] - ya[None]) + 1.0 / (ym[:, None] - yb[None]))).sum(axis=1) / (4 * math.pi)
        cdi = float(-(g2 * w * (yb - ya)).sum() * 2.0 / S)
        e = cl * cl / (math.pi * b * b / S * cdi) if cdi > 0 else float('nan')
        return dict(alpha_deg=alpha_deg, CL=cl, CDi=cdi, e=e, gamma=gam.tolist(), span_y=lat.ym.tolist(),
                    span_ccl=(2.0 * gam).tolist())


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', default='V03')
    ap.add_argument('--mesh', default='geo/V03_h20.npz')
    ap.add_argument('--channels', type=int, default=16)
    ap.add_argument('--cache', type=int, default=250000)
    ap.add_argument('--walks', type=int, default=8)
    ap.add_argument('--eps', type=float, default=1e-4)
    ap.add_argument('--d', type=float, nargs='+', default=[0.05, 0.10, 0.20])
    ap.add_argument('--core', type=float, default=0.05)
    ap.add_argument('--xi', type=float, nargs='+', default=[0.25])
    ap.add_argument('--tag', default='test')
    ap.add_argument("--logs", action="store_true")
    ap.add_argument("--maxwalk", type=int, default=10000)
    ap.add_argument("--rr", type=float, default=0.99)
    a = ap.parse_args()
    t0 = time.perf_counter()
    w = [1.0 / len(a.xi)] * len(a.xi)
    S = WosLift(a.key, a.mesh, a.channels, xi=a.xi, w=w, core_m=a.core, print_logs=a.logs)
    pts = np.vstack([S.kutta_points(d) for d in a.d])
    u, g = S.solve(pts, cache=a.cache, walks=a.walks, eps=a.eps, max_walk=a.maxwalk, rr=a.rr)
    Vb = S.basis_velocity(pts, g)
    ns = S.lat.ns
    res = dict(args=vars(a), timing=None, mesh=S.mesh_meta, sweeps={})
    for i, d in enumerate(a.d):
        gx, gz, cond = S.kutta(Vb[i * ns:(i + 1) * ns])
        rows = [S.forces(gx, gz, al) for al in (0.0, 2.0, 4.0, 6.0, 8.0, 12.0)]
        res['sweeps'][f'd{d}'] = dict(d_m=d, cond=cond, points=rows)
        print(f'd = {d:5.2f} m  cond {cond:8.1f}  ' + '  '.join(f"{r['alpha_deg']:.0f}: CL {r['CL']:.4f} CDi {r['CDi']:.5f}" for r in rows))
    S.T.rec['total_wall'] = time.perf_counter() - t0
    res['timing'] = S.T.rec
    print(json.dumps({k: round(v, 2) for k, v in S.T.rec.items()}))
    (HERE / 'results').mkdir(exist_ok=True)
    (HERE / 'results' / f'wos_{a.key}_{a.tag}.json').write_text(json.dumps(res, indent=1))
    np.savez(HERE / 'results' / f'wos_{a.key}_{a.tag}_kutta.npz', pts=pts, u=u, g=g, Vb=Vb)
