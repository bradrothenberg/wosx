"""WoSX with the wall from the field on every delta variant, one preset, with the wall-clock time of one cold command.

  python run_variants.py fast            -> results/final_V.._M0.7_fast.json, results/variants_fast.json
  python run_variants.py full V03 V05

For each variant:
  1. the grid file is removed and ONE command is timed from start to end (grid build, start of the solver, one run):
     this is the time a user waits for a new body
  2. the same command is run again with more repeats, for the scatter (the grid is there now)
Presets are below. Every timed step records if the program named by 'busy_process' (paths.py) ran at that moment.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from paths import other_job_running

HERE = Path(__file__).resolve().parent
PRESET = dict(
    fast=dict(shape=(512, 256, 128), gtag='_hb2', block=2, cache=15000, walks=4, cap=2000, repeat=6),
    std=dict(shape=(512, 256, 128), gtag='_hb2', block=2, cache=15000, walks=10, cap=2000, repeat=5),
    full=dict(shape=(1024, 512, 256), gtag='_f', block=4, cache=60000, walks=10, cap=4000, repeat=3),
    tall=dict(shape=(512, 256, 256), gtag='_hz', block=2, cache=15000, walks=10, cap=2000, repeat=5),   # half grid in x and y, full in z: for the lift at zero angle
)


if __name__ == '__main__':
    name = sys.argv[1]
    P = PRESET[name]
    keys = [k.upper() for k in sys.argv[2:]] or [f'V{i:02d}' for i in range(1, 11)]
    out_file = HERE / 'results' / f'variants_{name}.json'
    out = json.loads(out_file.read_text()) if out_file.exists() else dict(preset=P, variants={})
    for k in keys:
        grid = HERE / 'geo' / f'{k}_M0.70_grid{P["gtag"]}.npz'
        base = [sys.executable, str(HERE / 'run_final.py'), '--key', k, '--mach', '0.7', '--implicit', 'auto', '--grid-tag', P['gtag'], '--dist-block', str(P['block']), '--grid-shape'] + [str(q) for q in P['shape']] + \
               ['--importance', '--max-walk', str(P['cap']), '--cache', str(P['cache']), '--walks', str(P['walks'])]
        if grid.exists():
            grid.unlink()
        shared = other_job_running()
        t = time.perf_counter()
        r = subprocess.run(base + ['--repeat', '1', '--tag', f'_{name}1'], capture_output=True, text=True, cwd=HERE)
        cold = time.perf_counter() - t
        assert r.returncode == 0, r.stderr[-3000:]
        one = json.loads((HERE / 'results' / f'final_{k}_M0.7_{name}1.json').read_text())
        shared2 = other_job_running()
        r = subprocess.run(base + ['--repeat', str(P['repeat']), '--tag', f'_{name}'], capture_output=True, text=True, cwd=HERE)
        assert r.returncode == 0, r.stderr[-3000:]
        many = json.loads((HERE / 'results' / f'final_{k}_M0.7_{name}.json').read_text())
        out['variants'][k] = dict(cold_command_seconds=cold, grid_step_seconds=one['init']['field_to_grid'], grid_inside_seconds=one['mesh']['seconds_total'],
                                  solver_start_seconds=one['init']['wall'], run_seconds=one['runs'][0]['seconds'], run_timing=one['runs'][0]['timing'],
                                  warm_run_seconds=many['seconds_per_run'], cfd_on_gpu_during_cold_command=shared, cfd_on_gpu_during_repeats=shared2 or many.get('gpu_shared'))
        out_file.write_text(json.dumps(out, indent=1))
        v = out['variants'][k]
        print(f"{k} {name}: one cold command {cold:.1f} s (grid {v['grid_step_seconds']:.1f}, solver start {v['solver_start_seconds']:.1f}, run {v['run_seconds']:.1f}; walks {v['run_timing']['gpu_walks']:.1f})"
              f"  warm run {v['warm_run_seconds']:.1f} s  CFD on GPU: {shared}", flush=True)
