# ntop_aero: lifting potential flow on nTop implicit bodies with WoSX

This folder connects WoSX (Walk on Stars, GPU Monte Carlo) to implicit bodies from nTop and adds what the
stock `potential_flow` demo needs to give lift. It is a study tool. It is not a validated solver.

Lift, drag and the checks sit beside WoSX and use its Python bindings. One part changes the WoSX source: the wall
from an implicit (no triangles), see below. Without that part the WoSX source is unchanged.

## What it adds

| File | What it does |
|---|---|
| `bodies.py`, `sdf_bridge.py` | An implicit body (a field graph, negative inside) becomes surface triangles for WoSX. The field is also read directly for normals, masks and sections. |
| `wos_lift.py` | Lift. Horseshoe vortex strips sit inside the body; each strip is one more channel on the same random walks. A trailing-edge (Kutta) condition sets the strip strengths on the CPU. One GPU run gives every angle of attack. Mach number by the Goethert rule. |
| `study_noise.py`, `kutta_study.py` | Two ways to read the trailing-edge condition from the same runs. The `edge` reading is the corrected one. |
| `run_final.py` | One command per body: mesh if needed, solve, lift and induced drag from the wake. `--implicit auto` uses the wall from the field, with no triangles. |
| `implicit_grid.py`, `implicit_wall.py` | The wall from the field. `implicit_grid.py` samples the field on a grid and finds wall points from its sign changes. `implicit_wall.py` gives WoSX a wall handler and a wall sampler made from them. |
| `check_sphere.py`, `check_sphere_implicit.py`, `diag_wall.py` | Flow past a sphere against the exact answer, with triangles and with the implicit wall. `diag_wall.py` runs the same walks on both walls. |
| `drag_model.py` | Drag from the lift curve, the wake and the body shape: fixed leading-edge suction share, flat-plate friction. Compared with CFD of a separate study. The suction share is a fit to that CFD, not a prediction. |
| `wos_surface.py`, `run_fields.py`, `sdf_walks.py` | Potential on the skin, velocity and pressure fields, streamlines. |
| `vlm_baseline.py` | Flat vortex lattice for comparison (needs the lattice code of the study). |
| `panel_check.py`, `panel_sections.py`, `panel_body.py` | A small CPU panel code (thick body, flat wake) as an independent check, with its own checks: sphere, 2-D section, cambered section. |

## The wall from an implicit: changes to WoSX

WoSX takes a solid (reflecting) wall as triangles. This fork adds a wall that is a field on a grid:

| File | Change |
|---|---|
| `include/wosx/gpu/utils/implicit-geometric-queries.slang` (new) | Every wall query of Walk on Stars from field values: distance, ray hit (sphere tracing and bisection), normal (gradient), wall sample (tangent disc), star radius (probes on the wall), with or without the Kelvin transform. Also a wall sampler that reads a list of points. |
| `include/wosx/gpu/wosx.slang` | Two more choices for the wall type and the sampler type. |
| `include/wosx/gpu/entrypoints/interop/geometric_queries.h`, `samplers.h` | One enum value each. |
| `python/wosx_gpu_bindings.h` | `Core.GPUCustomReflectingBoundaryHandler` and `Samplers.GPUCustomBoundarySampler`: a wall handler and a sampler that Python can subclass. |

Rebuild the Python module after these changes. The field is read from 3D textures. Next to the wall the shader
interpolates the grid itself, because the filtered texture lookup of the GPU has only 8 bits of position precision
per cell. The body must be symmetric about y = 0 (the grid holds one half).

Limits of this wall. It is checked on three bodies only (a sphere, a tapered wing, a delta body), where lift agrees
with the triangle wall within the scatter of the walks. The star radius next to the wall is an approximation: 12 probes
of the field on the tangent plane per ball size. It is not proven. Walks take about 7 times more steps than with
triangles (each step is cheaper, so the run is still faster); the cause is not found. The WoSX authors have not
reviewed this change.

```
python implicit_grid.py W12T12 --mach 0 --shape 768 512 128   # (in the Python with the field library) field -> grid
python check_sphere_implicit.py                                # sphere with the implicit wall
python run_final.py --key W12T12 --mach 0 --implicit auto --cache 240000 --walks 10 --repeat 2 --tag _c240i
```

## What you need

- WoSX built with its Python bindings (see the top-level README), and an NVIDIA GPU.
- A Python with torch and the field library (`fieldlab`) that evaluates the nTop field graph. It is not part of this
  repository. `paths.py` says where to point to it: copy `local_paths.example.json` to `local_paths.json`.
- `numpy`, `scipy`, `matplotlib`, `fast_simplification`.

## Run

```
python run_final.py --key W12T12 --mach 0.0 --repeat 4   # test wing: Warren 12 planform, NACA 0012, Mach 0
python kutta_study.py W12T12 0.0                          # lift slope with both trailing-edge readings
python panel_check.py sphere            # checks of the panel code
python panel_body.py camber
python panel_sections.py W12T12         # (in the Python with the field library) sections of the body from its field
python panel_body.py check              # the panel code on the body from its field
```

The test wings (`W12T06`, `W12T12`, `W12T18`) are defined in `bodies.py`. Bodies named `V..` come from a separate
study and need its model files.

## What we found

- With the wake added, WoSX gives a straight lift curve and ranks swept deltas in the right order.
- One four-minute run on one GPU scatters by 1 to 3 % in lift. A vortex lattice or the panel code gives the same
  class of answer in seconds on one CPU, with no scatter.
- With the corrected trailing-edge reading, the WoSX lift slope is 1 to 3 % above the panel code on the bodies tested.
- Induced drag comes from the wake. Friction and pressure drag are not in this flow model.

## Limits

- Potential flow only: no friction, no separation, no shocks.
- The wake is flat and has 13 strips per side.
- The panel code is written for these checks. Its answer moves by about 0.5 % with the way the quads are cut.
