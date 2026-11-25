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
| `run_variants.py`, `speed_study.py`, `time_panel.py`, `strips_test.py` | Speed. `run_variants.py` times one cold command per body for four presets (below). `speed_study.py` compares settings on one body. `time_panel.py` times the panel code the same way. `strips_test.py` changes the number of wake strips. |
| `compare_methods.py` | Lift slope, lift at zero angle and wake drag factor of the lattice, the panel code and WoSX against the CFD of a separate study, with one definition for every method. |
| `denoise_test.py`, `denoise_select.py`, `denoise_maps.py` | De-noising of the velocity next to the skin with a small neural network trained on the noisy points of one run. A second, independent run chooses the network size. Needs torch. |

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

## Speed: a shader cache and four presets

One more change to the WoSX source makes a run start faster:

| File | Change |
|---|---|
| `include/wosx/gpu/utils/shader_file_cache.h` (new) | A cache on disk for the compiled shaders. |
| `include/wosx/gpu/entrypoints/interop/task_handle.h` | Two lines: the device gets the cache. |

Set the environment variable `WOSX_SHADER_CACHE` to a folder to use it. With the variable unset nothing changes.
On the workstation of this study the shader set-up of a run went from 8 to 19 s down to 0.2 s.

`run_final.py` has more options for speed: `--grid-shape`, `--grid-tag` and `--dist-block` build the grid of the
field when it is missing, `--importance` starts more walks near the trailing edge, `--max-walk` cuts long walks.
`run_variants.py` holds four presets of these:

| Preset | Fine grid | Points × walks | New body, one cold command | One more run |
|---|---|---|---|---|
| `fast` | 512 × 256 × 128 | 15,000 × 4 | 18 to 23 s | 3 to 4 s |
| `std` | 512 × 256 × 128 | 15,000 × 10 | 20 to 25 s | 6 to 7 s |
| `tall` | 512 × 256 × 256 | 15,000 × 10 | 25 to 27 s | 6 to 7 s |
| `full` | 1024 × 512 × 256 | 60,000 × 10 | 73 to 80 s | 38 to 41 s |

Times are measured on ten delta bodies at Mach 0.7 on one RTX 6000 Ada (`tall`: four bodies). The lift at zero
angle is not reliable on the `fast` and `std` grid; use `tall` or `full` for it. The lift slope is the same within
0.5 % on all four.

```
python run_variants.py fast                    # every V.. body; python run_variants.py full V03 for one
python compare_methods.py --wos _fast _std _tall _full --panel-grid 32 24 --tag _all
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
- A new body takes about 20 s on one GPU with the `fast` preset, the same as the panel code on a 32 × 24 grid. One
  run scatters by 0.2 to 0.8 % in lift slope. A flat vortex lattice takes 0.2 s.
- With the corrected trailing-edge reading, the WoSX lift slope is 2.6 to 3.9 % above the panel code on ten delta bodies.
- Against the CFD of a separate study, WoSX, the panel code and the lattice rank 42 of 45 pairs of those bodies in
  the same order. The WoSX lift slope is 8 % above that CFD, the panel code 5 %, the lattice 2 %. WoSX is not more
  accurate than the two simple methods. There is no test data to say which level is right.
- A small neural network takes the noise out of the pressure next to the skin better than a mean over neighbours.
  Forces summed from that pressure are still wrong; forces come from the wake.
- Induced drag comes from the wake. Friction and pressure drag are not in this flow model.

## Limits

- Potential flow only: no friction, no separation, no shocks.
- The wake is flat and has 13 strips per side.
- The panel code is written for these checks. Its answer moves by about 0.5 % with the way the quads are cut.
