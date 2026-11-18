"""Vortex-lattice baseline for the comparison: the trade study's own planar lattice (study/vlm.py, the same
class of method as VORLAX or AVL: flat lifting surface, horseshoe vortices, no thickness, no body).

  python vlm_baseline.py V03   -> results/vlm_V03.json

Runs on the CPU (numpy). Reads the study read-only.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from paths import STUDY as _STUDY  # noqa: E402
STUDY = _STUDY / 'study'
sys.path.insert(0, str(STUDY))
import vlm  # noqa: E402

import bodies  # noqa: E402

key = (sys.argv[1] if len(sys.argv) > 1 else 'V03').upper()
ref = bodies.reference(key)
S, c, b, xref = ref['s_ref_m2'], ref['c_ref_m'], ref['b_ref_m'], ref['moment_ref_m'][0]
pf = bodies.planform(key)
alphas = [0.0, 2.0, 4.0, 6.0, 8.0, 12.0]
out = dict(variant=key, reference=ref, method='planar vortex lattice, study/vlm.py (26 strips per side, 14 chordwise), flat plate, CPU numpy',
           alphas_deg=alphas, cases={})
for mach in (0.0, 0.7):
    t0 = time.perf_counter()
    L = vlm.Lattice(pf, mach, (), s_ref=S, c_ref=c, b_ref=b, x_ref=xref)
    rows = []
    for a in alphas:
        r = L.solve(alpha=math.radians(a))
        tz = L.trefftz(r['gamma'])
        ym, ccl = L.span_loading(r['gamma'])
        rows.append(dict(alpha_deg=a, CL=float(r['CL']), CDi_near=float(r['CDi']), CDi_trefftz=tz['CDi'], e=tz['e'],
                         Cm=float(r['Cm']), span_y=ym.tolist(), span_ccl=ccl.tolist()))
    dt = time.perf_counter() - t0
    h = math.radians(1.0)
    cla = (L.solve(alpha=h)['CL'] - L.solve(alpha=-h)['CL']) / (2 * h)
    out['cases'][f'mach_{mach:.1f}'] = dict(mach=mach, panels=int(L.n), seconds_sweep=dt, CL_alpha_per_rad=float(cla), points=rows)
    print(f'Mach {mach}: {L.n} panels, {dt:.3f} s, CL_alpha {cla:.4f}/rad', [(p['alpha_deg'], round(p['CL'], 4), round(p['CDi_trefftz'], 5)) for p in rows])
(HERE / 'results').mkdir(exist_ok=True)
(HERE / 'results' / f'vlm_{key}.json').write_text(json.dumps(out, indent=1))
