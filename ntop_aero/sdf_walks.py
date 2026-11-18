"""Random walks run DIRECTLY on the nTop signed distance field (no triangles), for the film.

This is an illustration of what the method does, written here in a few lines of torch. It is not WoSX's
kernel: WoSX walks in a Kelvin-inverted copy of the space and uses a triangle BVH for its star-shaped steps.
Here: far from the wall a walker jumps to a random point of the largest empty sphere (radius = field value);
at the wall it reflects (a random direction into the fluid, ray-marched through the field so it cannot
pass through the skin). Walkers that wander far away are absorbed.

  <study python> sdf_walks.py V03 600 240   -> media/sdf_walks.npz (paths [steps, walkers, 3], alive mask)
"""
import sys, time, json
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
from paths import STUDY  # noqa: E402
sys.path.insert(0, str(STUDY / 'aero'))
import common as C
key = C.use(sys.argv[1]); nw = int(sys.argv[2]); nsteps = int(sys.argv[3])
import oml, fieldlab as fl
field = oml.flow_body('faired')
dev = fl.DEV
g = torch.Generator(device=dev).manual_seed(3)

def sdf(p):
    with torch.no_grad():
        return fl.evaluate(field, p[:, 0], p[:, 1], p[:, 2])

def sdf_grad(p):
    q = p.clone().requires_grad_(True)
    f = fl.evaluate(field, q[:, 0], q[:, 1], q[:, 2])
    gr, = torch.autograd.grad(f.sum(), q)
    return f.detach(), gr / gr.norm(dim=1, keepdim=True).clamp_min(1e-9)

def rand_dir(n):
    v = torch.randn(n, 3, device=dev, generator=g)
    return v / v.norm(dim=1, keepdim=True)

# start points: on the skin (the cached surface samples of the real solver also start there)
d = np.load(HERE / 'geo' / f'{key}_h20_q150k.npz')
V, F = d['vertices'], d['faces']
rng = np.random.default_rng(5)
tri = V[F]; area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
pick = rng.choice(len(F), nw, p=area / area.sum())
x = torch.tensor(tri[pick].mean(1), dtype=torch.float32, device=dev)
LIP, EPS, RMAX, FAR = 1.6, 0.01, 0.6, 40.0
paths = np.zeros((nsteps + 1, nw, 3), np.float32); alive_rec = np.zeros((nsteps + 1, nw), bool); wall_rec = np.zeros((nsteps + 1, nw), bool)
alive = torch.ones(nw, dtype=torch.bool, device=dev)
paths[0] = x.cpu().numpy(); alive_rec[0] = True; wall_rec[0] = True
t0 = time.time()
for s in range(1, nsteps + 1):
    f, n = sdf_grad(x)
    near = f < EPS
    wall_rec[s - 1] |= near.cpu().numpy() if s > 1 else False
    # free step: uniform point on the empty sphere
    step_free = rand_dir(nw) * (f / LIP).clamp_min(0.0)[:, None]
    # wall step: random direction into the fluid, marched through the field
    dr = rand_dir(nw); flip = (dr * n).sum(1) < 0; dr[flip] = -dr[flip]
    t = torch.full((nw,), 2 * EPS, device=dev)
    for _ in range(24):
        ft = sdf(x + dr * t[:, None])
        t = torch.minimum(t + (ft / LIP).clamp_min(0.0), torch.full_like(t, RMAX))
    u = torch.rand(nw, device=dev, generator=g)
    step_wall = dr * (t * (0.35 + 0.65 * u))[:, None]
    xn = torch.where(near[:, None], x + step_wall, x + step_free)
    x = torch.where(alive[:, None], xn, x)
    alive = alive & (x.norm(dim=1) < FAR)
    paths[s] = x.cpu().numpy(); alive_rec[s] = alive.cpu().numpy()
np.savez_compressed(HERE / 'media' / 'sdf_walks.npz', paths=paths, alive=alive_rec, wall=wall_rec)
print(json.dumps(dict(walkers=nw, steps=nsteps, seconds=round(time.time() - t0, 2), still_alive=int(alive.sum()), device=str(dev),
                      field_evals=int(nw * nsteps * 26))))
