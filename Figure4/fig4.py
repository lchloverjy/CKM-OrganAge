"""Figure 4. Multi-organ ageing burden and disease risk within CKM stages.
Reads only ../source_data/*.csv.  Run:  python fig4.py
Panels: a standardised cumulative incidence within stage; b dose-response and cross-stage comparison;
c outcome-specific HRs and absolute risk differences; d specification analysis; e spread to multisystem
disease; f stage-associated risk shared with organ age gaps.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from style4 import *
import pandas as pd, numpy as np
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
SD = os.path.join(HERE, '..', 'source_data') + '/'
ROOT = os.path.join(HERE, '..')
CU = pd.read_csv(SD + 'Fig4a_std_cuminc_composite.csv')
NR = pd.read_csv(SD + 'Fig4a_numbers_at_risk.csv')
WS = pd.read_csv(SD + 'Fig4a_within_stage_hr.csv')
DR = pd.read_csv(SD + 'Fig4b_dose_response_10y_risk.csv')
DH = pd.read_csv(SD + 'Fig4b_dose_response_hr.csv')
EQ = pd.read_csv(SD + 'Fig4b_risk_equivalence_composite.csv').iloc[0]
OC = pd.read_csv(SD + 'Fig4c_outcome_hr_interaction.csv').set_index('outcome')
RD = pd.read_csv(SD + 'Fig4c_risk_difference_by_stage.csv')
SP = pd.read_csv(SD + 'Fig4d_specification_composite.csv')
MS = pd.read_csv(SD + 'Fig4e_multistate_hr.csv')
OCC = pd.read_csv(SD + 'Fig4e_state_occupation_12y.csv')
RM = pd.read_csv(SD + 'Fig4e_rmst_multisystem_free.csv')
DC = pd.read_csv(SD + 'Fig4f_stage_hr_attenuation.csv').set_index('outcome')

OUTS = ['CKMC', 'HF', 'ASCVD', 'CKD', 'T2D', 'CLD', 'DEM']
OLAB = {'CKMC': 'CKM composite', 'HF': 'Heart failure', 'ASCVD': 'ASCVD', 'CKD': 'CKD',
        'T2D': 'Type 2 diabetes', 'CLD': 'Chronic liver disease', 'DEM': 'Dementia'}
SGL = ['0', '1', '2-3']
SGT = {'0': 'Stage 0', '1': 'Stage 1', '2-3': 'Stage 2–3'}
B3 = ['0', '1', '>=2']
B3L = {'0': '0', '1': '1', '>=2': '≥2'}
B3C = {'0': '#9C9C9C', '1': '#E39B5B', '>=2': '#A4161A'}
B4 = ['0', '1', '2', '>=3']
B4L = {'0': '0', '1': '1', '2': '2', '>=3': '≥3'}
B4C = {'0': '#9C9C9C', '1': '#EBAA74', '2': '#D9732B', '>=3': '#A4161A'}
SGC = {'0': '#B3BECA', '1': '#6F8399', '2-3': '#1F3349'}
STC = ['#E9EEF3', '#A9BED3', '#2F5D8C']
f2 = lambda v: f'{v:.2f}'
ci2 = lambda h, l, u: um(f'{h:.2f} ({l:.2f}–{u:.2f})')


def logx(ax, lo, hi, ticks):
    ax.set_xscale('log'); ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f'{t:g}' for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())


# ------------------------------------------------------------------ a
def panel_a(C):
    x0s = [11, 45, 79]; W_, Y0, H_ = 30, 9, 31
    C.text(6, 1.6, 'Aged target organs', fontsize=5.8, color=MID)
    for j, b in enumerate(B3):
        xx = 27 + j * 11
        C.line([xx, xx + 4], [2.9, 2.9], color=B3C[b], lw=1.0)
        C.text(xx + 5, 1.6, B3L[b], fontsize=5.8, color=B3C[b])
    for i, sg in enumerate(SGL):
        ax = C.ax(x0s[i], Y0, W_, H_)
        clean(ax, grid='y')
        for b in B3:
            d = CU[(CU.stage == sg) & (CU.burden == b)]
            ax.fill_between(d.years, d.ci_lo * 100, d.ci_hi * 100, color=B3C[b], alpha=0.14, lw=0, zorder=1)
            ax.plot(d.years, d.std_cumulative_incidence * 100, color=B3C[b], lw=0.95, zorder=3)
        ax.set_xlim(0, 13); ax.set_ylim(0, 35)
        ax.set_xticks([0, 4, 8, 12]); ax.set_yticks([0, 10, 20, 30])
        if i == 0:
            ax.set_ylabel('Standardised cumulative\nincidence (%)', linespacing=0.95, labelpad=1.5)
        else:
            ax.set_yticklabels([])
        ax.set_title(SGT[sg], fontsize=6.3, pad=2, loc='left', fontweight='bold')
        w = WS[WS.stage == sg].set_index('contrast')
        ax.text(0.03, 0.98, 'HR vs 0 aged organs', transform=ax.transAxes, fontsize=5.1, color=MID, va='top')
        for k, (lab, key, col) in enumerate([('≥2', '>=2 vs 0 aged organs', B3C['>=2']),
                                               ('1', '1 vs 0 aged organs', B3C['1'])]):
            r = w.loc[key]
            ax.text(0.03, 0.885 - k * 0.105, f'{lab}: ' + ci2(r.hr, r.lo, r.hi), transform=ax.transAxes,
                    fontsize=5.1, color=col, va='top')
        for k, b in enumerate(B3):
            nr = NR[NR.group == f'S{sg}/B{b}'].set_index('years').n_risk
            for t in [0, 4, 8, 12]:
                C.text(x0s[i] + W_ * t / 13 - (0.4 if t == 0 else 0), 48.6 + k * 2.4, f'{nr[t]:,}', fontsize=5,
                       ha='left' if t == 0 else 'center', color=B3C[b])
    C.text(x0s[1] + W_ / 2, 43.2, 'Years since baseline', fontsize=6.3, ha='center')
    C.text(0.6, 46.0, 'No. at risk', fontsize=5.3, color=MID)
    for k, b in enumerate(B3):
        C.text(6.6, 48.6 + k * 2.4, B3L[b], fontsize=5, ha='right', color=B3C[b])


# ------------------------------------------------------------------ b
def panel_b(C):
    ax = C.ax(11, 8, 50, 35)
    clean(ax, grid='y')
    off = {'0': -0.27, '1': -0.09, '2': 0.09, '>=3': 0.27}
    ref = DR[(DR.stage == '2-3') & (DR.n_aged == '0')].risk10.iloc[0] * 100
    ax.axhline(ref, color=MID, lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax.text(0.40, 25.2, 'Reference: stage 2–3,\n0 aged organs', fontsize=5, color=MID, va='top', linespacing=0.95)
    ax.plot([0.63, 0.63], [21.6, ref + 0.25], color=MID, lw=0.4)
    for i, sg in enumerate(SGL):
        d = DR[DR.stage == sg].set_index('n_aged')
        xs = [i + off[b] for b in B4]
        ax.plot(xs, d.loc[B4].risk10 * 100, color=LIGHT, lw=0.5, zorder=2)
        for b in B4:
            r = d.loc[b]
            ci(ax, i + off[b], r.risk10 * 100, r.lo * 100, r.hi * 100, color=B4C[b], horizontal=False, ms=2.6, lw=0.7)
    ax.set_xlim(-0.55, 2.45); ax.set_ylim(0, 33)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels([SGT[s] for s in SGL])
    ax.tick_params(axis='x', length=0)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_ylabel('Standardised 10-year risk,\nCKM composite (%)', linespacing=0.95, labelpad=1.5)
    t = DH[(DH.outcome == 'CKMC') & (DH.contrast.str.startswith('per'))].iloc[0]
    ax.text(-0.5, 32.5, 'Per aged organ, adjusted for stage', fontsize=5.1, color=MID, va='top')
    ax.text(-0.5, 29.7, 'HR ' + ci2(t.hr, t.lo, t.hi), fontsize=5.1, color=INK, va='top')
    C.text(14, 1.6, 'Aged target organs', fontsize=5.8, color=MID)
    for j, b in enumerate(B4):
        xx = 35.5 + j * 6.4
        C.text(xx, 1.6, '●', fontsize=5.6, color=B4C[b])
        C.text(xx + 2.0, 1.6, B4L[b], fontsize=5.8, color=B4C[b])
    C.text(3, 47.4, 'Stage 0–1 with ≥2 vs stage 2–3 with 0 aged organs', fontsize=5.2, color=MID)
    C.text(3, 50.0, um(f'HR {EQ.hr_LH_vs_HL:.2f} (90% CI {EQ.lo90:.2f}–{EQ.hi90:.2f}); '
                       + (f'P$_{{equivalence}}$ < 0.001' if EQ.p_tost < 0.001 else f'P$_{{equivalence}}$ = {EQ.p_tost:.3f}')), fontsize=5.2, color=INK)
    C.text(3, 52.8, 'Equivalence margins 0.80–1.25', fontsize=5.0, color=LIGHT)


# ------------------------------------------------------------------ c
def panel_c(C):
    Y0, H_ = 9, 39
    n = len(OUTS)
    yrow = lambda i: Y0 + H_ * (i + 0.5) / n
    for i, k in enumerate(OUTS):
        C.text(1, yrow(i), OLAB[k], fontsize=5.8, va='center', fontweight='bold' if k == 'CKMC' else 'normal')
    C.line([1, 99], [Y0 + H_ / n, Y0 + H_ / n], color=FAINT, lw=0.4)
    ax = C.ax(22, Y0, 24, H_)
    clean(ax, left=False, grid='x')
    for i, k in enumerate(OUTS):
        r = OC.loc[k]
        ci(ax, r.hr_b1, i - 0.16, r.lo_b1, r.hi_b1, color=B3C['1'], ms=2.2, lw=0.6, filled=False)
        ci(ax, r.hr_b2, i + 0.16, r.lo_b2, r.hi_b2, color=B3C['>=2'], ms=2.5, lw=0.7)
    ax.axvline(1, color=MID, lw=0.5, zorder=1)
    logx(ax, 0.8, 5.5, [1, 2, 4])
    ax.set_ylim(n - 0.5, -0.5); ax.set_yticks([])
    ax.set_xlabel('HR (95% CI)', labelpad=1)
    C.text(22, 1.6, 'vs 0 aged organs', fontsize=5.5, color=MID)
    C.text(22, 4.2, '○', fontsize=5.6, color=B3C['1']); C.text(24.4, 4.2, '1', fontsize=5.5, color=B3C['1'])
    C.text(28.5, 4.2, '●', fontsize=5.6, color=B3C['>=2']); C.text(30.9, 4.2, '≥2', fontsize=5.5, color=B3C['>=2'])
    C.text(47.5, 4.2, 'HR, ≥2 aged', fontsize=5.5, color=MID)
    for i, k in enumerate(OUTS):
        r = OC.loc[k]
        C.text(47.5, yrow(i), ci2(r.hr_b2, r.lo_b2, r.hi_b2), fontsize=5.2, va='center')
    ax = C.ax(66, Y0, 24, H_)
    clean(ax, left=False, grid='x')
    offs = {'0': -0.24, '1': 0.0, '2-3': 0.24}
    for i, k in enumerate(OUTS):
        d = RD[(RD.outcome == k) & (RD.contrast == '>=2 vs 0')].set_index('stage')
        for sg in SGL:
            r = d.loc[sg]
            ci(ax, r.rd * 100, i + offs[sg], r.lo * 100, r.hi * 100, color=SGC[sg], ms=2.0, lw=0.6)
    ax.axvline(0, color=MID, lw=0.5, zorder=1)
    ax.set_xlim(-2.5, 13); ax.set_xticks([0, 5, 10])
    ax.set_ylim(n - 0.5, -0.5); ax.set_yticks([])
    ax.set_xlabel('10-year risk difference,\n≥2 vs 0 aged organs (%)', labelpad=1, linespacing=0.95)
    for j, sg in enumerate(SGL):
        xx = 64 + j * 8.2
        C.text(xx, 1.6, '●', fontsize=5.6, color=SGC[sg])
        C.text(xx + 2.0, 1.6, SGT[sg].replace('Stage ', 'S'), fontsize=5.5, color=SGC[sg])
    C.text(66, 4.2, 'by CKM stage', fontsize=5.5, color=MID)
    C.text(99.5, 1.6, 'P$_{int}$', fontsize=5.5, color=MID, ha='right')
    C.text(99.5, 4.2, 'HR × stage', fontsize=5.0, color=LIGHT, ha='right')
    for i, k in enumerate(OUTS):
        C.text(99.5, yrow(i), f2(OC.loc[k].p_int_stage_mult), fontsize=5.2, va='center', ha='right')


# ------------------------------------------------------------------ d
SPLAB = {'Primary analysis': 'Primary analysis',
         '+ Stage-defining traits': '+ Stage-defining traits',
         '+ eGFRcr-cys splines and UACR': '+ eGFRcr-cys splines and UACR',
         '+ Kidney age gap (continuous)': '+ Kidney age gap',
         '+ PREVENT risk': '+ PREVENT risk',
         '+ Model d covariates': '+ Model d covariates',
         'Excluding the kidney clock (>=2 of 5)': 'Kidney clock excluded (≥2 of 5)',
         'Heart, kidney and brain only (>=2 of 3)': 'Heart, kidney, brain (≥2 of 3)',
         'All ten organ clocks (>=2 of 10)': 'Ten organ clocks (≥2 of 10)',
         'Kidney-function-residualised age gaps': 'Kidney-residualised age gaps',
         'Threshold: stage 0 mean + 1.5 s.d.': 'Threshold mean + 1.5 s.d.',
         'Threshold: pooled 90th percentile': 'Threshold pooled 90th pct',
         'Original stage assignment': 'Original CKM stage',
         'Excluding events in first 2 years': 'First 2 years excluded',
         'Men': 'Men', 'Women': 'Women'}
GH = {'Adjustment': 'Additional adjustment', 'Definition': 'Burden definition', 'Cohort': 'Cohort and subgroup'}


def panel_d(C):
    lines = []
    for g in ['Primary', 'Adjustment', 'Definition', 'Cohort']:
        if g != 'Primary':
            lines.append(('h', GH[g], None))
        for _, r in SP[SP.group == g].iterrows():
            lines.append(('r', SPLAB[r.specification], r))
    n = len(lines); Y0, H_ = 6.5, 43
    ax = C.ax(37, Y0, 22, H_)
    clean(ax, left=False, grid='x')
    prim = SP[SP.group == 'Primary'].iloc[0]
    ax.axvline(1, color=MID, lw=0.5, zorder=1)
    ax.axvline(prim.hr, color=ACCENT, lw=0.5, ls=(0, (2, 1.5)), zorder=1)
    for i, (typ, lab, r) in enumerate(lines):
        y = Y0 + H_ * (i + 0.5) / n
        if typ == 'h':
            C.text(1, y, lab, fontsize=5.4, va='center', color=MID, fontstyle='italic')
            continue
        isp = r.group == 'Primary'
        C.text(1 if isp else 2.6, y, lab, fontsize=5.4, va='center', fontweight='bold' if isp else 'normal')
        if isp:
            ci(ax, r.hr, i, r.lo, r.hi, color=ACCENT, ms=3.0, lw=0.8, marker='D')
        else:
            ci(ax, r.hr, i, r.lo, r.hi, color=INK, ms=2.2, lw=0.6)
        C.text(60.5, y, ci2(r.hr, r.lo, r.hi), fontsize=5.2, va='center', color=ACCENT if isp else INK)
    logx(ax, 1.0, 2.6, [1, 1.5, 2, 2.5])
    ax.set_ylim(n - 0.5, -0.5); ax.set_yticks([])
    ax.set_xlabel('HR, ≥2 vs 0 aged organs\n(CKM composite)', labelpad=1, linespacing=0.95)
    C.text(60.5, 1.6, 'HR (95% CI)', fontsize=5.5, color=MID)


# ------------------------------------------------------------------ e
def panel_e(C):
    bx = [2, 41, 80]; bw, by, bh = 18, 3.0, 8.6
    names = ['No new disease', 'One system', '≥2 systems']
    _ms = pd.read_csv(SD + 'Fig4e_multistate_hr.csv')
    _ab = _ms[_ms.transition == 'A->B'].iloc[0]; _bc = _ms[_ms.transition == 'B->C'].iloc[0]
    nums = [f'n = {int(_ab.n):,}', f'{int(_ab.events):,} entered', f'{int(_bc.events):,} entered']
    for j in range(3):
        C.rect(bx[j], by, bw, bh, fc=STC[j], ec=MID if j < 2 else STC[2], lw=0.4)
        col = 'white' if j == 2 else INK
        C.text(bx[j] + bw / 2, by + 1.6, names[j], fontsize=5.8, ha='center', color=col, fontweight='bold')
        C.text(bx[j] + bw / 2, by + 4.8, nums[j], fontsize=5.1, ha='center', color=col)
    for j, tr in enumerate(['A->B', 'B->C']):
        xa, xb = bx[j] + bw + 0.8, bx[j + 1] - 0.8
        C.arrow(xa, by + bh / 2, xb, by + bh / 2, lw=0.6, color=INK)
        m = MS[MS.transition == tr].set_index('exposure')
        for k, (lab, key, col) in enumerate([('≥2', 'Burden >=2', B3C['>=2']), ('1', 'Burden 1', B3C['1'])]):
            r = m.loc[key]
            C.text((xa + xb) / 2, by + bh / 2 + 1.0 + k * 2.2, f'{lab}: ' + ci2(r.hr, r.lo, r.hi),
                   fontsize=5.0, ha='center', color=col)
    C.text(50, 14.4, 'Incident disease in 12 organ systems; transition HR vs 0 aged organs, adjusted for stage',
           fontsize=5.1, ha='center', color=MID)
    ypos = {}
    for i, sg in enumerate(SGL):
        for j, b in enumerate(B3):
            ypos[(sg, b)] = i * 3.6 + j
    ax = C.ax(17, 20.5, 60, 27)
    clean(ax, left=False)
    for (sg, b), y in ypos.items():
        r = OCC[(OCC.stage == sg) & (OCC.burden == b)].iloc[0]
        vals = [r.p_free * 100, r.p_one * 100, r.p_multi * 100]
        left = 0
        for v, col in zip(vals, STC):
            ax.barh(y, v, left=left, height=0.78, color=col, edgecolor='white', lw=0.3)
            left += v
        ax.text(99, y, f'{r.p_multi * 100:.0f}%', fontsize=5.1, color='white', ha='right', va='center')
        ax.text(-1.5, y, B3L[b], fontsize=5.3, color=B3C[b], ha='right', va='center')
    ax.set_xlim(0, 100); ax.set_ylim(9.7, -0.6)
    ax.set_yticks([]); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('State occupied at 12 years (%), Kaplan–Meier', labelpad=1)
    for i, sg in enumerate(SGL):
        yy = 20.5 + 27 * (i * 3.6 + 1 + 0.6) / 10.3
        C.text(1, yy, SGT[sg], fontsize=5.6, va='center')
    C.text(12.2, 18.2, 'Aged', fontsize=5.1, color=MID, ha='center')
    C.text(79, 17.6, 'Multisystem-free', fontsize=5.1, color=MID)
    C.text(79, 19.6, 'years lost, ≥2 vs 0', fontsize=5.1, color=MID)
    for i, sg in enumerate(SGL):
        r = RM[RM.stage.astype(str) == sg].iloc[0]
        yy = 20.5 + 27 * (ypos[(sg, '>=2')] + 0.6) / 10.3
        C.text(79, yy, um(f'{r.loss_years:.2f} ({r.lo:.2f}–{r.hi:.2f})'), fontsize=5.1, va='center')


# ------------------------------------------------------------------ f
def panel_f(C):
    Y0, H_ = 9, 37
    n = len(OUTS)
    ax = C.ax(25, Y0, 30, H_)
    clean(ax, left=False, grid='x')
    for i, k in enumerate(OUTS):
        r = DC.loc[k]; y = Y0 + H_ * (i + 0.5) / n
        sig = r.lo_stage > 1
        col = INK if sig else LIGHT
        C.text(1, y, OLAB[k], fontsize=5.8, va='center', color=INK if sig else MID,
               fontweight='bold' if k == 'CKMC' else 'normal')
        ax.plot([r.hr_stage, r.hr_stage_adj], [i, i], color=FAINT if not sig else LIGHT, lw=1.1, zorder=2)
        ci(ax, r.hr_stage, i - 0.14, r.lo_stage, r.hi_stage, color=col, ms=2.5, lw=0.6)
        ci(ax, r.hr_stage_adj, i + 0.14, r.lo_adj, r.hi_adj, color=col, ms=2.5, lw=0.6, filled=False)
        txt = um(f'{r.prop_attenuated * 100:.0f} ({r.lo * 100:.0f}–{r.hi * 100:.0f})') if sig else '–'
        C.text(58, y, txt, fontsize=5.2, va='center', color=INK if sig else LIGHT)
    ax.axvline(1, color=MID, lw=0.5, zorder=1)
    logx(ax, 0.9, 2.3, [1, 1.5, 2])
    ax.set_ylim(n - 0.5, -0.5); ax.set_yticks([])
    ax.set_xlabel('HR per CKM stage (95% CI)', labelpad=1)
    C.text(25, 1.6, '●', fontsize=5.6); C.text(27.3, 1.6, 'Model a + stage', fontsize=5.5)
    C.text(25, 4.2, '○', fontsize=5.6); C.text(27.3, 4.2, '+ 12 organ age gaps', fontsize=5.5)
    C.text(58, 1.6, 'Attenuated,', fontsize=5.5, color=MID)
    C.text(58, 4.2, '% (95% CI)', fontsize=5.5, color=MID)
    C.text(1, 52.6, 'Dementia: stage HR 95% CI included 1; attenuation not estimated.', fontsize=5.0, color=LIGHT)


W, H = 180, 183
PANELS = {
    'a': (panel_a, 0, 0, 112, 57),
    'b': (panel_b, 116, 0, 64, 57),
    'c': (panel_c, 0, 63, 101, 58),
    'd': (panel_d, 104, 63, 76, 58),
    'e': (panel_e, 0, 126, 101, 57),
    'f': (panel_f, 104, 126, 76, 57),
}
if __name__ == '__main__':
    build(PANELS, W, H, 'Fig4', ROOT)
    print('Fig4 done')
