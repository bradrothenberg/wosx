# Results

Result files of the scripts in `ntop_aero`, as written by the runs of November 2026 on one RTX 6000 Ada GPU.
The code that wrote them is on branch `ntop-aero` (commit d131cd9). This branch adds only this folder.

## Bodies

| Key | Body |
|---|---|
| `V01` to `V10` | Ten delta-wing bodies of a separate design study. Their field graphs are not in this repository. |
| `W12Tnn` | Warren 12 planform, scaled by 5, with a NACA 00nn section (`bodies.py`). |
| sphere | Unit sphere, used to verify the potential and the surface speed against the exact solution. |

## Files

| Pattern | Content |
|---|---|
| `final_<body>_M<Mach>_<tag>.json` | WoSX runs: lift and wake drag for each angle of attack, span loading, timing, settings. Tags `fast`, `std`, `tall`, `full` are the presets of `run_variants.py`; the same tag with a `1` is the single run of the cold command. Tags `c60` and `c240` are triangle-wall runs with 60,000 and 240,000 start points. |
| `kutta/<run>_r<n>.npz` | Velocities read behind the trailing edge in repeat run `n`, with the read-out distances and the local chord. |
| `kutta_study_*.json` | Lift as a function of the trailing-edge read-out distance. |
| `variants_<preset>.json` | Wall-clock time of one cold command and of repeat runs, for each body. |
| `speed_*.json`, `strips_V03.json` | Timing of single settings on body V03. |
| `panel_*.json`, `vlm_*.json` | Panel code and vortex lattice results. `_cmp` is the 32 x 24 panel grid of the comparison. |
| `compare_methods_all.json` | Lift slope, lift at zero angle, drag-due-to-lift factor and run time of each method for each body. |
| `rank.json`, `close_pair.json` | Three bodies of different sweep, and two bodies that differ by a cropped tip: repeat runs and the differences between the bodies. |
| `noise_*.json` | Standard deviation between repeat runs against the number of start points and walks. |
| `sphere_check*.json` | Sphere verification, triangle wall and implicit wall. |
| `known_wing.json`, `span_w12.json`, `panel_wing*.json` | Warren 12 reference wing: lift slope against section thickness, span loading. |
| `denoise_*.json` | Error statistics of the de-noising test. The reference is the panel code. |
| `summary.json` | Lift curve, wake drag and timing of body V03. |

Lift slope is (CL(8 deg) - CL(0 deg)) / 8 per degree. K is CDi / CL squared. Scatter is one standard deviation
between repeat runs. Each body uses its own reference area, given in the file.

## What was removed

These files are a copy of the working results with three kinds of content taken out.

1. The values of the CFD that the study used as its reference, and every quantity computed from them: ratios,
   differences, scores, and the drag model output (`drag_model.json`), which is fitted to that CFD. The CFD data
   belong to the study. The summary of the comparison is in the README of `ntop_aero`: against that CFD, the
   lift slope of WoSX is 8 % high, of the panel code 5 %, of the lattice 2 %, and each method orders 42 of the 45
   pairs of bodies as the CFD does.
2. Arrays of points on or near the study bodies (surface pressure samples, field slices, panel corner points),
   because they describe the shape of the bodies. The figure scripts that need them cannot be run from this
   branch.
3. Run logs, machine paths and the study's names of the bodies.

The keys `cfd_on_gpu_during_cold_command` and `cfd_on_gpu_during_repeats` remain. They are flags that say whether
another job used the same GPU during a timing; `true` means the time is an upper bound.

There are no experimental data for the V bodies. The published value for the Warren 12 wing and the exact sphere
solution are the only external references in this folder.
