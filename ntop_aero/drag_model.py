"""Which drag can a potential-flow method give? Five ways to turn wake strengths and body shape into drag, each
checked against the CFD runs of the trade study (ten delta variants, Mach 0.7, 0 to 8 degrees).

  python drag_model.py   -> results/drag_model.json            (CPU, a few seconds)

Inputs (all read-only):
  CFD      <study>/aero/results/table.json: cl, cd, cd_pressure, cd_friction at 0, 2, 4, 6, 8, 12 degrees
  lattice   results/vlm_V..json (flat vortex lattice, Mach 0.7): lift curve and wake (Trefftz) induced drag
  WoSX      results/final_V.._M0.7*.json: lift curve and wake strengths (six variants)
  panel     results/panel_V03_M0.7.json: wake strengths of the panel code (V03 only)
  shape     geo/V.._sections.npz from panel_sections.py: chord, thickness, leading-edge radius, wetted area

Lift-dependent drag, as an increase over the drag at zero angle:
  full suction    K_wake * CL^2                      what the wake (Trefftz plane) gives; the lowest drag for the span
  zero suction    CL * tan(alpha)                    the leading edge gives no forward force at all
  fixed fraction  (1 - s) * zero + s * full          one number s for all variants and angles
  Polhamus        CL * tan(alpha), with vortex lift  suction analogy: Kv = (Kp - Kp^2 K_wake) / cos(sweep)
Zero-lift drag:
  flat-plate friction per span strip (Prandtl-Schlichting, turbulent, Mach factor) on the wetted area of the strip,
  times a form factor for thickness (Hoerner: 1 + 2 t/c + 60 (t/c)^4).

"Matched lift" means: the model is evaluated at the lift coefficient that CFD has at each angle, and the angle the
model needs for that lift comes from its own lift curve. Reference areas are each variant's own (the study's rule).
"""
from __future__ import annotations

import glob
import itertools
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from paths import STUDY  # noqa: E402

R, GEO = HERE / 'results', HERE / 'geo'
KEYS = [f'V{i:02d}' for i in range(1, 11)]
ALPHAS = (2.0, 4.0, 6.0, 8.0)           # the angles of the study's "turn set" above zero; 12 degrees is left out (see table.json honesty)
FIT_ALPHAS = (4.0, 6.0, 8.0)            # the suction fraction is fitted here; at 2 degrees the lift drag in CFD is 0.002 to 0.003, near its noise
WOS_FILE = dict(V01='final_V01_M0.7_c240.json', V02='final_V02_M0.7.json', V03='final_V03_M0.7_c240e.json', V04='final_V04_M0.7_c240.json',
                V05='final_V05_M0.7_quick.json', V07='final_V07_M0.7_onecmd.json')
M2FT2, M2FT = 10.7639, 3.28084
tz = np.trapezoid

tab = json.loads((STUDY / 'aero' / 'results' / 'table.json').read_text())
FC = tab['flight_condition']
MACH, VEL, NU = FC['mach'], FC['velocity'], FC['kinematic_viscosity']
sizing = json.loads((STUDY / 'study' / 'sizing.json').read_text())['variants']


# ------------------------------------------------------------------------------------------------ inputs
def cfd(k):
    v = tab['variants'][k]
    p = {q['alpha_deg']: q for q in v['points']}
    return dict(ref=v['reference'], a0=v['fits']['alpha_zero_lift_deg'], cla=v['fits']['cl_alpha_per_rad'], cruise=v['at_cruise_cl'],
                pts={a: dict(cl=q['cl'], cd=q['cd'], cdp=q['cd_pressure'], cdf=q['cd_friction']) for a, q in p.items()})


def lattice(k):
    c = json.loads((R / f'vlm_{k}.json').read_text())['cases']['mach_0.7']
    a = np.array([p['alpha_deg'] for p in c['points']]); cl = np.array([p['CL'] for p in c['points']])
    cdi = np.array([p['CDi_trefftz'] for p in c['points']])
    i4 = list(a).index(4.0)
    return dict(alpha=a, cl=cl, kp=float(c['CL_alpha_per_rad']), ki=float(cdi[i4] / cl[i4] ** 2))


def wake_k(gam, ya, yb, S):
    """Drag-due-to-lift factor CDi / CL^2 of a wake of horseshoe strips (one side given, mirrored here)."""
    g2 = np.r_[gam, gam[::-1]]; a = np.r_[ya, -yb[::-1]]; b = np.r_[yb, -ya[::-1]]; yc = 0.5 * (a + b)
    wv = (g2[None] * (-1.0 / (yc[:, None] - a[None]) + 1.0 / (yc[:, None] - b[None]))).sum(axis=1) / (4 * math.pi)
    cl = 2.0 * (g2 * (b - a)).sum() / S
    return float(-(g2 * wv * (b - a)).sum() * 2.0 / S / cl ** 2)


def wosx(k):
    """Lift slope (0 to 8 degrees, per rad) and wake factor from the MEAN wake strengths of the repeat runs."""
    if k not in WOS_FILE or not (R / WOS_FILE[k]).exists():
        return None
    w = json.loads((R / WOS_FILE[k]).read_text()); S = w['reference']['s_ref_m2']
    al = w['alphas_deg']; cl = np.array(w['CL_mean'])
    a = math.atan(w['beta'] * math.tan(math.radians(8.0)))
    gam = math.cos(a) * np.mean([r['gx'] for r in w['runs']], axis=0) + math.sin(a) * np.mean([r['gz'] for r in w['runs']], axis=0)
    ym = np.array(w['span_y']); dy = 2 * ym[0]
    return dict(file=WOS_FILE[k], runs=len(w['runs']), alpha=np.array(al), cl=cl - cl[al.index(0.0)],          # lift curve through zero, as the lattice
                kp=float((cl[al.index(8.0)] - cl[al.index(0.0)]) / math.radians(8.0)), ki=wake_k(gam, ym - dy / 2, ym + dy / 2, S),
                wetted_m2=w['mesh']['wetted_area_m2'], volume_m3=w['mesh']['volume_m3'])


def panel(k):
    f = R / f'panel_{k}_M0.7.json'
    if not f.exists():
        return None
    p = json.loads(f.read_text()); g = p['grids'][-1]; S = p['reference']['s_ref_m2']
    y, dy, gz = (np.array(g[q]) for q in ('span_y', 'dy', 'gamma_z'))
    return dict(panels=g['panels'], kp=float(g['cl_alpha']), ki=wake_k(gz, y - dy / 2, y + dy / 2, S))


def shape(k):
    """Strip geometry from the field sections: chord, thickness, leading-edge radius, sweep, wetted length."""
    s = np.load(GEO / f'{k}_sections.npz')
    ys, xi, xle, xte, zu, zl = (s[a] for a in ('ys', 'xi', 'xle', 'xte', 'zu', 'zl'))
    c = xte - xle
    X = xle[:, None] + c[:, None] * xi[None]
    arc = np.sqrt(np.diff(X, axis=1) ** 2 + np.diff(zu, axis=1) ** 2).sum(1) + np.sqrt(np.diff(X, axis=1) ** 2 + np.diff(zl, axis=1) ** 2).sum(1)
    t = zu - zl
    tc = t.max(1) / np.maximum(c, 1e-9); xt = xi[t.argmax(1)]
    m = (xi > 0) & (xi < 0.004)                                           # nose: half thickness^2 = 2 r x
    x = xi[m][None] * c[:, None]; h = 0.5 * t[:, m]
    r = (h * h * x).sum(1) / np.maximum(2 * (x * x).sum(1), 1e-30)
    sw = np.arctan(np.gradient(xle, ys))
    xm = xle + xt * c; swm = np.arctan(np.gradient(xm, ys))               # sweep of the line of greatest thickness
    S = 2 * tz(c, ys)
    wgt = lambda q: float(tz(q * c, ys) / tz(c, ys))
    return dict(ys=ys, c=c, arc=arc, tc=tc, xt=xt, r=r, sw=sw, swm=swm, planform_m2=float(S), wetted_m2=float(2 * tz(arc, ys)),
                base_m2=float(2 * tz(t[:, -1], ys)), tc_mean=wgt(tc), r_over_c_mean=wgt(r / np.maximum(c, 1e-9)), r_mean_m=wgt(r),
                le_sweep_deg=math.degrees(wgt(sw)), inv_cos_sweep=wgt(1.0 / np.cos(sw)), r_normal_over_c=wgt(r / np.maximum(c, 1e-9) / np.cos(sw) ** 2))


# ------------------------------------------------------------------------------------------------ models
def cf_turbulent(re, mach):
    """Flat-plate skin friction, fully turbulent (Prandtl-Schlichting), with the usual Mach factor (Raymer eq. 12.27)."""
    return 0.455 / np.log10(re) ** 2.58 / (1.0 + 0.144 * mach * mach) ** 0.65


def zero_lift(g, S):
    """Friction and form drag by span strips, on the reference area S."""
    ok = g['c'] > 1e-3
    cf = np.where(ok, cf_turbulent(np.maximum(VEL * g['c'] / NU, 1e3), MACH), 0.0)
    fric = 2 * tz(cf * g['arc'], g['ys']) / S
    hoerner = 1 + 2 * g['tc'] + 60 * g['tc'] ** 4
    raymer = (1 + 0.6 / np.maximum(g['xt'], 0.1) * g['tc'] + 100 * g['tc'] ** 4) * 1.34 * MACH ** 0.18 * np.cos(g['swm']) ** 0.28
    cmac = tz(g['c'] ** 2, g['ys']) / tz(g['c'], g['ys'])
    return dict(friction=float(fric), with_form_hoerner=float(2 * tz(cf * hoerner * g['arc'], g['ys']) / S),
                with_form_raymer=float(2 * tz(cf * raymer * g['arc'], g['ys']) / S),
                friction_one_chord=float(cf_turbulent(VEL * cmac / NU, MACH) * g['wetted_m2'] / S), reynolds_mac=float(VEL * cmac / NU),
                cf_mean=float(fric * S / g['wetted_m2']), form_factor_hoerner=float(tz(cf * hoerner * g['arc'], g['ys']) / tz(cf * g['arc'], g['ys'])))


def alpha_for(cl, src):
    """Angle (rad) that the potential-flow lift curve of `src` needs for the lift coefficient cl."""
    return math.radians(float(np.interp(cl, src['cl'], src['alpha'])))


def polhamus(cl, kp, ki, inv_cos):
    """Suction analogy: CL = Kp sin a cos^2 a + Kv cos a sin^2 a, Kv = (Kp - Kp^2 Ki) / cos(sweep); drag due to lift CL tan a."""
    kv = (kp - kp * kp * ki) * inv_cos
    f = lambda a: kp * math.sin(a) * math.cos(a) ** 2 + kv * math.cos(a) * math.sin(a) ** 2 - cl
    lo, hi = 0.0, math.radians(30.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    a = 0.5 * (lo + hi)
    return cl * math.tan(a), math.degrees(a), kv


def score(model, truth):
    """Level and trend of `model` against `truth` over the variants (same order)."""
    m, t = np.asarray(model, float), np.asarray(truth, float)
    ratio = m / t
    rank = lambda q: np.argsort(np.argsort(q)).astype(float)
    pairs = list(itertools.combinations(range(len(t)), 2))
    conc = [np.sign(m[i] - m[j]) == np.sign(t[i] - t[j]) for i, j in pairs]
    big = [c for c, (i, j) in zip(conc, pairs) if abs(t[i] - t[j]) > 0.05 * 0.5 * (t[i] + t[j])]
    return dict(n=len(t), level=float(ratio.mean() - 1), level_min=float(ratio.min() - 1), level_max=float(ratio.max() - 1),
                scatter=float(ratio.std(ddof=1) / ratio.mean()), spread_of_cfd=float(t.std(ddof=1) / t.mean()),
                spearman=float(np.corrcoef(rank(m), rank(t))[0, 1]), pearson=float(np.corrcoef(m, t)[0, 1]),
                pairs_right=int(sum(conc)), pairs=len(pairs), pairs_right_over_5pct=int(sum(big)), pairs_over_5pct=len(big))


# ------------------------------------------------------------------------------------------------ run
def main():
    D = {}
    for k in KEYS:
        sx, la, g = cfd(k), lattice(k), shape(k)
        S, b = sx['ref']['s_ref_m2'], sx['ref']['b_ref_m']
        cd0s = sx['pts'][0.0]['cd']
        z = zero_lift(g, S)
        row = dict(name=sizing[k]['name'], kind=sizing[k]['kind'], s_ref_m2=S, s_ref_ft2=S * M2FT2, span_ft=b * M2FT, aspect_ratio=b * b / S,
                   shape={q: g[q] for q in ('planform_m2', 'wetted_m2', 'base_m2', 'tc_mean', 'r_over_c_mean', 'r_mean_m', 'le_sweep_deg', 'inv_cos_sweep', 'r_normal_over_c')},
                   wetted_ft2=g['wetted_m2'] * M2FT2, wetted_over_ref=g['wetted_m2'] / S, le_radius_in_mean=g['r_mean_m'] * M2FT * 12,
                   zero_lift=dict(model=z, cfd_cd=cd0s, cfd_friction=sx['pts'][0.0]['cdf'], cfd_pressure=sx['pts'][0.0]['cdp'], cfd_cl=sx['pts'][0.0]['cl']),
                   lattice=dict(kp=la['kp'], ki=la['ki'], ideal_ki=1 / (math.pi * b * b / S)), cfd=dict(cla=sx['cla'], alpha_zero_lift_deg=sx['a0']), points={})
        srcs = dict(lattice=la)
        w, pn = wosx(k), panel(k)
        if w:
            srcs['wosx'] = w
            row['wosx'] = dict(file=w['file'], runs=w['runs'], kp=w['kp'], ki=w['ki'], mesh_wetted_m2=w['wetted_m2'], sections_wetted_vs_mesh=g['wetted_m2'] / w['wetted_m2'] - 1)
        if pn:
            row['panel'] = pn
            srcs['panel'] = dict(alpha=np.array([0.0, 12.0]), cl=np.array([0.0, pn['kp'] * math.radians(12.0)]), kp=pn['kp'], ki=pn['ki'])
        for a in ALPHAS:
            p = sx['pts'][a]; cl = p['cl']
            dcd = p['cd'] - cd0s                                          # lift-dependent drag in CFD: increase over zero angle
            ar = math.radians(a - sx['a0'])
            q = dict(cl=cl, cd=p['cd'], dcd=dcd, dcd_pressure_only=p['cdp'] - sx['pts'][0.0]['cdp'], k=dcd / cl ** 2, friction=p['cdf'],
                     zero_suction_cfd_angle=cl * math.tan(ar), suction_cfd=(cl * math.tan(ar) - dcd) / (cl * math.tan(ar) - la['ki'] * cl * cl))
            for name, src in srcs.items():
                am = alpha_for(cl, src)
                full, zero = src['ki'] * cl * cl, cl * math.tan(am)
                pol, apol, kv = polhamus(cl, src['kp'], src['ki'], g['inv_cos_sweep'])
                clp = float(np.interp(a, src['alpha'], src['cl']))        # matched angle: the model's own lift at the CFD angle
                q[name] = dict(alpha_deg=math.degrees(am), full=full, zero=zero, polhamus=pol, polhamus_alpha_deg=apol, kv=kv,
                               suction_needed=(zero - dcd) / (zero - full), same_angle=dict(cl=clp, full=src['ki'] * clp * clp, zero=clp * math.tan(math.radians(a))))
            row['points'][f'{a:.0f}'] = q
        D[k] = row

    # one suction fraction for all variants: least squares on the lattice basis, and leave one variant out
    def fit(keys, alphas, basis='lattice'):
        num = den = 0.0
        for k in keys:
            for a in alphas:
                q = D[k]['points'][f'{a:.0f}']; m = q[basis]
                num += (m['zero'] - q['dcd']) * (m['zero'] - m['full']); den += (m['zero'] - m['full']) ** 2
        return num / den

    s_all = fit(KEYS, FIT_ALPHAS)
    s_loo = {k: fit([j for j in KEYS if j != k], FIT_ALPHAS) for k in KEYS}
    s_by_alpha = {f'{a:.0f}': fit(KEYS, (a,)) for a in ALPHAS}
    for k in KEYS:
        for a in ALPHAS:
            q = D[k]['points'][f'{a:.0f}']
            for name in ('lattice', 'wosx', 'panel'):
                if name in q:
                    m = q[name]
                    m['fixed'] = (1 - s_all) * m['zero'] + s_all * m['full']
                    m['fixed_left_out'] = (1 - s_loo[k]) * m['zero'] + s_loo[k] * m['full']
                    m['fixed_per_angle'] = (1 - s_by_alpha[f'{a:.0f}']) * m['zero'] + s_by_alpha[f'{a:.0f}'] * m['full']

    # does the suction that CFD implies follow a shape number? (ten variants, mean over 4, 6 and 8 degrees)
    s_var = np.array([np.mean([D[k]['points'][f'{a:.0f}']['lattice']['suction_needed'] for a in FIT_ALPHAS]) for k in KEYS])
    preds = dict(le_sweep_deg=[D[k]['shape']['le_sweep_deg'] for k in KEYS], aspect_ratio=[D[k]['aspect_ratio'] for k in KEYS],
                 r_over_c_mean=[D[k]['shape']['r_over_c_mean'] for k in KEYS], r_normal_over_c=[D[k]['shape']['r_normal_over_c'] for k in KEYS],
                 tc_mean=[D[k]['shape']['tc_mean'] for k in KEYS], le_radius_m=[D[k]['shape']['r_mean_m'] for k in KEYS])
    corr = {}
    for name, x in preds.items():
        x = np.array(x); r = float(np.corrcoef(x, s_var)[0, 1])
        loo = []
        for i in range(len(KEYS)):                                        # leave one out: straight line in this number against a constant
            msk = np.arange(len(KEYS)) != i
            c1, c0 = np.polyfit(x[msk], s_var[msk], 1)
            loo.append((c0 + c1 * x[i] - s_var[i], s_var[msk].mean() - s_var[i]))
        loo = np.array(loo)
        corr[name] = dict(r=r, rms_error_line=float(np.sqrt((loo[:, 0] ** 2).mean())), rms_error_constant=float(np.sqrt((loo[:, 1] ** 2).mean())))

    # scores over the variants
    SC = {}
    for a in ALPHAS:
        key = f'{a:.0f}'
        truth = [D[k]['points'][key]['dcd'] for k in KEYS]
        sc = {m: score([D[k]['points'][key]['lattice'][m] for k in KEYS], truth) for m in ('full', 'zero', 'polhamus', 'fixed', 'fixed_left_out', 'fixed_per_angle')}
        sc['full_same_angle'] = score([D[k]['points'][key]['lattice']['same_angle']['full'] for k in KEYS], truth)
        sc['zero_same_angle'] = score([D[k]['points'][key]['lattice']['same_angle']['zero'] for k in KEYS], truth)
        wk = [k for k in KEYS if 'wosx' in D[k]['points'][key]]
        tw = [D[k]['points'][key]['dcd'] for k in wk]
        for m in ('full', 'zero', 'fixed'):
            sc[f'wosx_{m}'] = dict(score([D[k]['points'][key]['wosx'][m] for k in wk], tw), variants=wk)
            sc[f'lattice_{m}_same_six'] = score([D[k]['points'][key]['lattice'][m] for k in wk], tw)
        SC[key] = sc

    # zero-lift drag
    Z = dict(friction_vs_cfd_friction=score([D[k]['zero_lift']['model']['friction'] for k in KEYS], [D[k]['zero_lift']['cfd_friction'] for k in KEYS]),
             hoerner_vs_cfd_total=score([D[k]['zero_lift']['model']['with_form_hoerner'] for k in KEYS], [D[k]['zero_lift']['cfd_cd'] for k in KEYS]),
             raymer_vs_cfd_total=score([D[k]['zero_lift']['model']['with_form_raymer'] for k in KEYS], [D[k]['zero_lift']['cfd_cd'] for k in KEYS]),
             friction_vs_cfd_total=score([D[k]['zero_lift']['model']['friction'] for k in KEYS], [D[k]['zero_lift']['cfd_cd'] for k in KEYS]),
             drag_area_ft2=dict(cfd={k: D[k]['zero_lift']['cfd_cd'] * D[k]['s_ref_ft2'] for k in KEYS},
                                model={k: D[k]['zero_lift']['model']['with_form_hoerner'] * D[k]['s_ref_ft2'] for k in KEYS}))
    Z['drag_area_score'] = score(list(Z['drag_area_ft2']['model'].values()), list(Z['drag_area_ft2']['cfd'].values()))
    Z['cfd_pressure_share_at_zero'] = {k: D[k]['zero_lift']['cfd_pressure'] / D[k]['zero_lift']['cfd_cd'] for k in KEYS}

    # total polar and lift-to-drag ratio: model = friction and form (Hoerner) + fixed suction fraction on the lattice wake
    T = {}
    for a in ALPHAS:
        key = f'{a:.0f}'
        cd_m = [D[k]['zero_lift']['model']['with_form_hoerner'] + D[k]['points'][key]['lattice']['fixed'] for k in KEYS]
        cd_h = [D[k]['zero_lift']['cfd_cd'] + D[k]['points'][key]['lattice']['fixed'] for k in KEYS]
        cd_f = [D[k]['zero_lift']['model']['with_form_hoerner'] + D[k]['points'][key]['lattice']['full'] for k in KEYS]
        cd_s = [D[k]['points'][key]['cd'] for k in KEYS]; cl = [D[k]['points'][key]['cl'] for k in KEYS]
        ld = lambda cd: [c / d for c, d in zip(cl, cd)]
        T[key] = dict(cd=score(cd_m, cd_s), cd_with_cfd_zero_lift=score(cd_h, cd_s), cd_full_suction=score(cd_f, cd_s),
                      l_over_d=score(ld(cd_m), ld(cd_s)), l_over_d_full_suction=score(ld(cd_f), ld(cd_s)),
                      values={k: dict(cl=c, cfd_cd=s_, model_cd=m_, cfd_l_over_d=c / s_, model_l_over_d=c / m_, full_suction_l_over_d=c / f_)
                              for k, c, s_, m_, f_ in zip(KEYS, cl, cd_s, cd_m, cd_f)})
    # the study's point of use: cruise lift coefficient of each variant
    cr = {}
    for k in KEYS:
        c = cfd(k)['cruise']; cl = c['cl']; la = lattice(k)
        am = alpha_for(cl, la); full, zero = la['ki'] * cl * cl, cl * math.tan(am)
        cdm = D[k]['zero_lift']['model']['with_form_hoerner'] + (1 - s_all) * zero + s_all * full
        cdf = D[k]['zero_lift']['model']['with_form_hoerner'] + full
        cr[k] = dict(cl=cl, cfd_cd=c['cd'], cfd_l_over_d=c['lift_to_drag'], model_cd=cdm, model_l_over_d=cl / cdm, full_suction_l_over_d=cl / cdf,
                     cfd_drag_area_ft2=c['cd'] * D[k]['s_ref_ft2'], model_drag_area_ft2=cdm * D[k]['s_ref_ft2'])
    T['cruise'] = dict(values=cr, cd=score([cr[k]['model_cd'] for k in KEYS], [cr[k]['cfd_cd'] for k in KEYS]),
                       l_over_d=score([cr[k]['model_l_over_d'] for k in KEYS], [cr[k]['cfd_l_over_d'] for k in KEYS]),
                       l_over_d_full_suction=score([cr[k]['full_suction_l_over_d'] for k in KEYS], [cr[k]['cfd_l_over_d'] for k in KEYS]),
                       drag_area=score([cr[k]['model_drag_area_ft2'] for k in KEYS], [cr[k]['cfd_drag_area_ft2'] for k in KEYS]))

    # surface pressure drag from the panel code on V03 (measured earlier, results/panel_pressure_V03.json)
    pp = json.loads((R / 'panel_pressure_V03.json').read_text())
    pres = dict(mach=pp['mach'], alpha_deg=pp['alpha_deg'],
                grids=[dict(panels=g['panels'], cd_pressure=g['cd_pressure'], cl_pressure=g['cl_pressure'], cl_wake=g['cl_wake'], cp_min=g['cp_min']) for g in pp['grids']])
    v0 = json.loads((R / 'vlm_V03.json').read_text())['cases']['mach_0.0']['points']
    p4 = [p for p in v0 if p['alpha_deg'] == 4.0][0]
    pres['wake_value_lattice_mach0'] = p4['CDi_trefftz'] * (pp['grids'][-1]['cl_wake'] / p4['CL']) ** 2

    out = dict(
        what='Drag from potential-flow results and body shape against CFD, ten delta variants, Mach 0.7', keys=KEYS, alphas_deg=list(ALPHAS),
        cfd_source=str(STUDY / 'aero' / 'results' / 'table.json').replace('\\', '/'), cfd_written=tab['written'],
        cfd_note='grid sensitivity of the CFD drag: see cl_grid_uncertainty_is in table.json',
        flight=dict(mach=MACH, velocity_m_s=VEL, kinematic_viscosity=NU, altitude_m=FC['altitude_m']),
        suction=dict(fixed=s_all, fit_alphas_deg=list(FIT_ALPHAS), left_out=s_loo, by_alpha=s_by_alpha, per_variant=dict(zip(KEYS, s_var.tolist())),
                     per_variant_min=float(s_var.min()), per_variant_max=float(s_var.max()), per_variant_std=float(s_var.std(ddof=1)), against_shape=corr,
                     cfd_angle_basis={k: {f'{a:.0f}': D[k]['points'][f'{a:.0f}']['suction_cfd'] for a in ALPHAS} for k in KEYS}),
        lift_drag_scores=SC, zero_lift_scores=Z, total=T, surface_pressure=pres, variants=D)
    (R / 'drag_model.json').write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))

    # ------------------------------------------------------------------------------------------- print
    pc = lambda x: f'{100 * x:+.0f} %'
    print(f'suction fraction, one number for all: {s_all:.3f}; left-one-out range {min(s_loo.values()):.3f} to {max(s_loo.values()):.3f}; by angle', {a: round(v, 3) for a, v in s_by_alpha.items()})
    print('per variant (4, 6, 8 deg mean):', {k: round(v, 3) for k, v in zip(KEYS, s_var)})
    print('against shape:', {n: (round(c['r'], 2), round(c['rms_error_line'], 3), round(c['rms_error_constant'], 3)) for n, c in corr.items()})
    for key, sc in SC.items():
        print(f'--- lift-dependent drag at the CFD lift of {key} degrees (10 variants unless noted)')
        for m, s in sc.items():
            print(f'   {m:24s} n {s["n"]:2d} level {pc(s["level"])} ({pc(s["level_min"])} to {pc(s["level_max"])}) scatter {100 * s["scatter"]:.1f} % (CFD spread {100 * s["spread_of_cfd"]:.1f} %)'
                  f' rank {s["spearman"]:+.2f} pairs {s["pairs_right"]}/{s["pairs"]} clear pairs {s["pairs_right_over_5pct"]}/{s["pairs_over_5pct"]}')
    print('--- zero lift')
    for m in ('friction_vs_cfd_friction', 'friction_vs_cfd_total', 'hoerner_vs_cfd_total', 'raymer_vs_cfd_total', 'drag_area_score'):
        s = Z[m]
        print(f'   {m:26s} level {pc(s["level"])} ({pc(s["level_min"])} to {pc(s["level_max"])}) scatter {100 * s["scatter"]:.1f} % (CFD spread {100 * s["spread_of_cfd"]:.1f} %) rank {s["spearman"]:+.2f} pairs {s["pairs_right"]}/{s["pairs"]} clear {s["pairs_right_over_5pct"]}/{s["pairs_over_5pct"]}')
    for k in KEYS:
        z = D[k]['zero_lift']
        print(f'   {k} {D[k]["name"]:14s} Swet/S {D[k]["wetted_over_ref"]:.3f} t/c {D[k]["shape"]["tc_mean"]:.3f} r_le {D[k]["le_radius_in_mean"]:.2f} in sweep {D[k]["shape"]["le_sweep_deg"]:.1f} | model friction {z["model"]["friction"]:.5f} +form {z["model"]["with_form_hoerner"]:.5f} (Raymer {z["model"]["with_form_raymer"]:.5f})'
              f' | CFD friction {z["cfd_friction"]:.5f} pressure {z["cfd_pressure"]:.5f} total {z["cfd_cd"]:.5f}')
    print('--- total drag and lift-to-drag ratio')
    for key, t in T.items():
        for m in [q for q in t if q != 'values']:
            s = t[m]
            print(f'   {key:6s} {m:24s} level {pc(s["level"])} ({pc(s["level_min"])} to {pc(s["level_max"])}) scatter {100 * s["scatter"]:.1f} % (CFD spread {100 * s["spread_of_cfd"]:.1f} %) rank {s["spearman"]:+.2f} pairs {s["pairs_right"]}/{s["pairs"]} clear {s["pairs_right_over_5pct"]}/{s["pairs_over_5pct"]}')
    print('--- surface pressure drag, panel code, V03:', [round(g['cd_pressure'], 4) for g in pres['grids']], 'wake value', round(pres['wake_value_lattice_mach0'], 4))
    for k in KEYS:
        q = D[k]['points']['8']
        print(k, f"AR {D[k]['aspect_ratio']:.2f} K cfd {q['k']:.3f} full {D[k]['lattice']['ki']:.3f} zero {q['lattice']['zero'] / q['cl'] ** 2:.3f} polhamus {q['lattice']['polhamus'] / q['cl'] ** 2:.3f} fixed {q['lattice']['fixed'] / q['cl'] ** 2:.3f}"
              + (f" | wosx kp {D[k]['wosx']['kp']:.3f} ki {D[k]['wosx']['ki']:.4f} (lattice kp {D[k]['lattice']['kp']:.3f} ki {D[k]['lattice']['ki']:.4f})" if 'wosx' in D[k] else '')
              + (f" | panel kp {D[k]['panel']['kp']:.3f} ki {D[k]['panel']['ki']:.4f}" if 'panel' in D[k] else ''))


if __name__ == '__main__':
    main()
