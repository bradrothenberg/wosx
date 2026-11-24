"""The WoSX wall from an implicit body (grids written by implicit_grid.py). No triangles.

The Slang side is include/wosx/gpu/utils/implicit-geometric-queries.slang (ImplicitBody3D,
ImplicitReflectingBoundaryGeometricQueries, PointListBoundarySampler). This file gives WoSX the two objects it
needs: a reflecting-wall handler that binds the grids, and a boundary sampler that binds a list of wall points.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


class Grid:
    def __init__(self, path):
        d = np.load(path)
        self.meta = json.loads(str(d['meta']))
        self.near, self.far = d['near'], d['far']
        self.pts, self.nrm, self.area = d['pts'].astype(np.float64), d['nrm'].astype(np.float64), d['area'].astype(np.float64)
        self.path = str(path)


def sphere_grid(path, a=5.0, c=(6.0, 0.0, 0.15), n=320, n_far=160, pad=0.4, n_pts=600_000):
    """Exact grids for a sphere (check case): the distance is known, so no field library is needed."""
    c = np.asarray(c, float)
    lo = np.array([c[0] - a - pad, 0.0, c[2] - a - pad]); hi = np.array([c[0] + a + pad, a + pad, c[2] + a + pad])
    shape = (n, n // 2, n)

    def dist(lo, hi, shape):
        ax = [lo[i] + (hi[i] - lo[i]) / shape[i] * (np.arange(shape[i]) + 0.5) for i in range(3)]
        X, Y, Z = np.meshgrid(*ax, indexing='ij')
        return (np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) - a).astype(np.float32)

    R = a * 1.0002
    flo = np.array([c[0] - 3 * R, 0.0, c[2] - 3 * R]); fhi = np.array([c[0] + 3 * R, 3 * R, c[2] + 3 * R])
    fshape = (n_far, n_far // 2, n_far)
    hf = (fhi - flo) / np.array(fshape)
    far = np.maximum(dist(flo, fhi, fshape) - float(np.linalg.norm(hf)), 0.5 * pad).astype(np.float32)
    k = np.arange(n_pts) + 0.5                           # Fibonacci points: equal areas
    z = 1.0 - 2.0 * k / n_pts
    ph = k * math.pi * (3.0 - math.sqrt(5.0))
    s = np.sqrt(1.0 - z * z)
    nrm = np.stack([s * np.cos(ph), s * np.sin(ph), z], 1)
    meta = dict(variant='sphere', source='exact distance to a sphere', mach=0.0, beta=1.0, shape=list(shape),
                cell_m=((hi - lo) / np.array(shape)).tolist(), near_lo=lo.tolist(), near_hi=hi.tolist(), far_shape=list(fshape),
                far_lo=flo.tolist(), far_hi=fhi.tolist(), pad_m=pad, kappa=1.0, wall_points=n_pts,
                wetted_area_m2_stretched=4 * math.pi * a * a, centre=c.tolist(), radius_m=R)
    np.savez(path, near=dist(lo, hi, shape), far=far, pts=(c + a * nrm).astype(np.float32), nrm=nrm.astype(np.float32),
             area=np.full(n_pts, 4 * math.pi * a * a / n_pts, np.float32), meta=json.dumps(meta))
    return meta


def morton(x, bits=16):
    """Morton (Z-order) key of points: neighbours on the curve are neighbours in space."""
    lo, hi = x.min(0), x.max(0)
    q = np.minimum(((x - lo) / np.maximum(hi - lo, 1e-30) * (1 << bits)).astype(np.uint64), (1 << bits) - 1)
    key = np.zeros(len(x), np.uint64)
    for b in range(bits):
        for a in range(3):
            key |= ((q[:, a] >> np.uint64(b)) & np.uint64(1)) << np.uint64(3 * b + a)
    return key


class Wall:
    """Handler and sampler for a body given as grids. c: inversion centre (m), scale: 1 / body radius."""

    def __init__(self, wosx, grid, c, scale, pool=1_000_000, seed=0, kelvin=True):
        self.wosx, self.grid = wosx, grid
        m = grid.meta
        c = np.asarray(c, float)

        def sdf(data, lo, hi):
            return wosx.Utils.GPUSdfGrid(np.ascontiguousarray(data * np.float32(scale)).ravel(), np.array(data.shape, np.int32),
                                         ((np.asarray(lo) - c) * scale).astype(np.float32), ((np.asarray(hi) - c) * scale).astype(np.float32),
                                         False, dim=3)

        near = sdf(grid.near, m['near_lo'], m['near_hi'])
        far = sdf(grid.far, m['far_lo'], m['far_hi'])
        radius = float(m['radius_m'] * scale)

        class Handler(wosx.Core.GPUCustomReflectingBoundaryHandler):
            def allocate(self, context):
                near.allocate(context)
                far.allocate(context)

            def set_resources(self, cursor, print_logs):
                near.set_resources(cursor['nearGrid'], print_logs)
                far.set_resources(cursor['farGrid'], print_logs)
                cursor['bodyRadius'].set_data_float(radius)
                cursor['mirrorY'].set_data_uint(1)
                cursor['useKelvin'].set_data_uint(1 if kelvin else 0)
                cursor['_padding'].set_data_uint(0)

            def get_reflection_type(self):
                return 'ImplicitBody3D'

            def get_backend_type(self):
                return 'Implicit'

            def get_name(self):
                return 'body'

        self._grids = (near, far)
        self.handler = Handler()

        # wall samples for boundary value caching: uniform over the area of the body (drawn by area weight),
        # given in the inverted space where the walk runs
        # The pool is in a space-filling (Morton) order and each entry stands for the same wall area, so the
        # sampler can take one entry per stratum of the list (as the mesh sampler does over its triangles).
        rng = np.random.default_rng(seed)
        order = np.argsort(morton((grid.pts - c) * scale), kind='stable')
        cdf = np.cumsum(grid.area[order]); cdf /= cdf[-1]
        j = order[np.minimum(np.searchsorted(cdf, (np.arange(pool) + rng.random(pool)) / pool), len(cdf) - 1)]
        x = (grid.pts[j] - c) * scale
        n_out = -grid.nrm[j]                             # out of the flow domain, into the body
        r2 = (x * x).sum(1)
        area_n = grid.area.sum() * scale * scale
        if kelvin:
            y = x / r2[:, None]
            ny = n_out - 2.0 * x * ((n_out * x).sum(1) / r2)[:, None]
            pdf = r2 * r2 / area_n
        else:
            y, ny, pdf = x, n_out, np.full(pool, 1.0 / area_n)
        ny /= np.linalg.norm(ny, axis=1, keepdims=True)
        self.samples = np.concatenate([y, ny, pdf[:, None]], 1).astype(np.float32)
        self.inv_pos = wosx.Float3List(self.samples[:, :3].copy())
        self.area_m2 = float(grid.area.sum())

    def sampler(self, bbox):
        wosx, data = self.wosx, self.samples
        buf = wosx.Utils.GPUFloatBuffer(False)
        flat = wosx.FloatList(np.ascontiguousarray(data.ravel()))
        lo = wosx.GPUFloat3(*[float(q) for q in bbox[0]]); hi = wosx.GPUFloat3(*[float(q) for q in bbox[1]])
        n = len(data)

        class Sampler(wosx.Samplers.GPUCustomBoundarySampler):
            def allocate(self, context):
                buf.allocate(context, flat)

            def reallocate(self, context, normal_offset_for_boundary):
                pass

            def set_resources(self, cursor, print_logs):
                buf.set_binding(cursor['samples'])
                cursor['nListSamples'].set_data_uint(n)
                cursor['domainMin'].set_data_float3(lo)
                cursor['domainMax'].set_data_float3(hi)

            def get_reflection_type(self):
                return 'PointListBoundarySampler<3>'

            def get_normal_offset_for_boundary(self):
                return 0.0

            def get_sample_count(self, n_total_samples, boundary_normal_aligned_samples=False):
                return 0 if boundary_normal_aligned_samples else n_total_samples

        self._buf = (buf, flat)
        self._sampler = Sampler()
        return self._sampler
