"""Wall-clock time of the panel code for a new body, as two cold commands: sections from the field, then the solve.

  python time_panel.py V03 V01 ...   -> results/panel_cold.json

The sections step is the one of panel_sections.py as it is: it makes the stations of every grid size (16 to 56 span
strips, 192 chord intervals), so it does more work than one grid needs. The existing sections file is copied to
tmp/ first and the new one is checked to be the same.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from paths import STUDY_PY, other_job_running

HERE = Path(__file__).resolve().parent

if __name__ == '__main__':
    keys = [k.upper() for k in sys.argv[1:]] or ['V03']
    f = HERE / 'results' / 'panel_cold.json'
    out = json.loads(f.read_text()) if f.exists() else {}
    for k in keys:
        sec = HERE / 'geo' / f'{k}_sections.npz'
        keep = HERE / 'tmp' / f'{k}_sections_before_timing.npz'
        shutil.copy2(sec, keep)
        busy = other_job_running()
        t = time.perf_counter()
        r = subprocess.run([STUDY_PY, str(HERE / 'panel_sections.py'), k], capture_output=True, text=True, cwd=HERE)
        t_sec = time.perf_counter() - t
        assert r.returncode == 0, r.stderr[-2000:]
        a, b = np.load(keep), np.load(sec)
        same = all(np.array_equal(a[q], b[q]) for q in ('ys', 'xi', 'xle', 'xte', 'zu', 'zl'))
        row = dict(sections_seconds=t_sec, sections_same_as_before=bool(same), cfd_on_gpu=busy, solve={})
        for nc, ms in ((24, 16), (32, 24), (48, 32)):
            t = time.perf_counter()
            r = subprocess.run([sys.executable, '-c', f"import panel_body as PB; PB.run('{k}', 0.7, grids=(({nc}, {ms}),), tag='_cold')"], capture_output=True, text=True, cwd=HERE)
            assert r.returncode == 0, r.stderr[-2000:]
            row['solve'][f'{nc}x{ms}'] = time.perf_counter() - t
        out[k] = row
        f.write_text(json.dumps(out, indent=1))
        print(k, 'sections %.1f s (same file: %s)' % (t_sec, same), {g: round(v, 1) for g, v in row['solve'].items()}, 'CFD on GPU:', busy, flush=True)
