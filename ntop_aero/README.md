# ntop_aero: lifting potential flow on nTop implicit bodies with WoSX

This folder connects WoSX (Walk on Stars, GPU Monte Carlo) to implicit bodies from nTop and adds what the
stock `potential_flow` demo needs to give lift. It is a study tool. It is not a validated solver.

Nothing in the WoSX source is changed. Everything here sits beside it and uses the Python bindings.

## What it adds

| File | What it does |
|---|---|
| `bodies.py`, `sdf_bridge.py` | An implicit body (a field graph, negative inside) becomes surface triangles for WoSX. The field is also read directly for normals, masks and sections. |
| `wos_lift.py` | Lift. Horseshoe vortex strips sit inside the body; each strip is one more channel on the same random walks. A trailing-edge (Kutta) condition sets the strip strengths on the CPU. One GPU run gives every angle of attack. Mach number by the Goethert rule. |
| `study_noise.py`, `kutta_study.py` | Two ways to read the trailing-edge condition from the same runs. The `edge` reading is the corrected one. |
| `run_final.py` | One command per body: mesh if needed, solve, lift and induced drag from the wake. |
| `wos_surface.py`, `run_fields.py`, `sdf_walks.py` | Potential on the skin, velocity and pressure fields, streamlines. |
| `vlm_baseline.py` | Flat vortex lattice for comparison (needs the lattice code of the study). |
| `panel_check.py`, `panel_sections.py`, `panel_body.py` | A small CPU panel code (thick body, flat wake) as an independent check, with its own checks: sphere, 2-D section, cambered section. |

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
