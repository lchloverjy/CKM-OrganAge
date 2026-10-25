"""Figure 5. Clinical value of organ age gaps beyond guideline-based assessment.
Reads only ../source_data/*.csv.   Run:  cd code && python fig5.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from style4 import *
import pandas as pd, numpy as np
from matplotlib.colors import TwoSlopeNorm
SD = os.path.join(HERE, '..', 'source_data') + '/'
ROOT = os.path.join(HERE, '..')
rd = lambda n: pd.read_csv(SD + n + '.csv')

OLAB = {'CKMC': 'CKM composite', 'HF': 'Heart failure', 'ASCVD': 'ASCVD', 'CKD': 'CKD', 'T2D': 'Type 2 diabetes',
        'CLD': 'Chronic liver disease', 'DEM': 'Dementia'}
PAIRED = {'HF': 'Heart', 'ASCVD': 'Artery', 'CKD': 'Kidney', 'T2D': 'Pancreas', 'CLD': 'Liver', 'DEM': 'Brain'}
COL_PREV, COL_G, COL_N, COL_D = '#B0B0B0', '#4D4D4D', ACCENT, BLUE
COL_DOWN, COL_SAME, COL_UP = '#3F6FA8', '#8A8A8A', ACCENT

A = rd('Fig5a_discrimination_gain')
ORDER = A[A.baseline == 'Model G'].sort_values('dC', ascending=False).outcome.tolist()


def fmt3(v):
    s = f'{v:.3f}'
    return '0.000' if s in ('-0.000', '0.000') else um(s)


# ---------------------------------------------------------------------------------------------- a
def pa(C):
    ax = C.ax(23, 9, 25, 40)
    n = len(ORDER)
    ax.axvline(0, color=INK, lw=0.5, zorder=1)
    for i, k in enumerate(ORDER):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=BAND, lw=0, zorder=0)
        for base, col, dy in [('Model G', COL_N, -0.17), ('Model D', COL_D, 0.17)]:
            r = A[(A.outcome == k) & (A.baseline == base)].iloc[0]
            ci(ax, r.dC, i + dy, r.lo, r.hi, color=col, filled=bool(r.lo > 0 or r.hi < 0), ms=2.4, lw=0.7)
    ax.set_ylim(n - 0.5, -0.5); ax.set_xlim(-0.024, 0.032)
    ax.set_xticks([-0.02, 0, 0.02]); ax.set_xticklabels([um('-0.02'), '0', '0.02'])
    ax.set_yticks(range(n)); ax.set_yticklabels([OLAB[k] for k in ORDER])
    ax.set_xlabel('ΔC-index (95% CI)')
    clean(ax, left=False, grid='x')
    C.text(53.5, 5.6, 'C, G', fontsize=5.4, ha='center', color=MID)
    C.text(60.5, 5.6, 'G + gaps', fontsize=5.4, ha='center', color=MID)
    for i, k in enumerate(ORDER):
        r = A[(A.outcome == k) & (A.baseline == 'Model G')].iloc[0]
        yy = 9 + 40 * (i + 0.5) / n
        C.text(53.5, yy, f'{r.C_base:.3f}', fontsize=5.4, ha='center', va='center', color=MID)
        C.text(60.5, yy, f'{r.C_new:.3f}', fontsize=5.4, ha='center', va='center',
               color=INK, fontweight='bold' if r.lo > 0 else 'normal')
    for j, (lab, col) in enumerate([('12 gaps added to model G', COL_N), ('12 gaps added to model D', COL_D)]):
        x0 = 5 + j * 29
        C.text(x0, 1.6, '●', fontsize=5.6, color=col, va='center')
        C.text(x0 + 2.2, 1.6, lab, fontsize=5.4, va='center')
    C.text(23, 56.5, 'Open symbols: 95% CI includes 0', fontsize=5.0, color=MID)


# ---------------------------------------------------------------------------------------------- b
B = rd('Fig5b_gain_by_clock')
COLS = CLOCKS + ['Aged-organ count', 'All 12 clocks']
COLLAB = {'Aged-organ count': 'Organ count'}


def pb(C):
    x0, y0, w, h = 22, 17.5, 84, 28
    ax = C.ax(x0, y0, w, h)
    M = np.full((len(ORDER), len(COLS)), np.nan); S = np.zeros_like(M, bool)
    for i, k in enumerate(ORDER):
        for j, c in enumerate(COLS):
            r = B[(B.outcome == k) & (B.added == c)].iloc[0]
            M[i, j] = r.dC; S[i, j] = bool(r.lo > 0)
    norm = TwoSlopeNorm(vmin=-0.01, vcenter=0, vmax=0.02)
    ax.imshow(M, cmap=DIV, norm=norm, aspect='auto', interpolation='nearest')
    for i, k in enumerate(ORDER):
        for j, c in enumerate(COLS):
            if S[i, j] or j >= 12:
                v = M[i, j]
                ax.text(j, i, fmt3(v), ha='center', va='center', fontsize=4.5,
                        color='white' if v > 0.013 else INK, fontweight='bold' if S[i, j] else 'normal')
        if k in PAIRED:
            j = COLS.index(PAIRED[k])
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc='none', ec=INK, lw=0.9, zorder=4))
    ax.axvline(11.5, color='white', lw=2.2); ax.axvline(12.5, color='white', lw=1.0)
    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels([COLLAB.get(c, c) for c in COLS], rotation=90, ha='center', va='bottom')
    ax.xaxis.tick_top()
    for t, l in zip(ax.get_xticklabels(), COLS):
        t.set_color(ORGAN_COL.get(l, INK))
        if l in ['Aged-organ count', 'All 12 clocks']:
            t.set_fontweight('bold')
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([OLAB[k] for k in ORDER])
    ax.tick_params(length=0, pad=1.5)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    C.text(x0 + w * 6 / 14, 1.6, 'Added to model G', fontsize=5.6, ha='center', color=MID)
    C.line([x0 + 0.3, x0 + w * 12 / 14 - 0.3], [3.4, 3.4], color=LIGHT, lw=0.4)
    hcolorbar(C, DIV, norm, x0, y0 + h + 4.6, 32, 1.8, 'ΔC-index over model G',
              ticks=[-0.01, 0, 0.01, 0.02], fmt=lambda t: um(f'{t:g}'))
    C.rect(x0 + 38, y0 + h + 4.2, 2.6, 2.4, ec=INK, lw=0.9)
    C.text(x0 + 41.6, y0 + h + 5.4, 'Paired clock', fontsize=5.3, va='center')
    C.text(x0 + 56, y0 + h + 5.4, 'Numbers: ΔC with 95% CI excluding 0 (bold);\nall values shown for the last two columns',
           fontsize=5.0, va='center', color=MID, linespacing=1.05)


# ---------------------------------------------------------------------------------------------- c
CC = rd('Fig5c_tier2_gain_two_baselines')
SYS = ['Cardiac', 'Vascular', 'Kidney', 'Metabolic', 'Liver', 'Brain', 'Lung', 'Immune/infection', 'Musculoskeletal',
       'Gastrointestinal', 'Eye', 'Cancer']
SYS_LAB = {'Immune/infection': 'Immune/\ninfection', 'Musculoskeletal': 'Musculo-\nskeletal', 'Gastrointestinal': 'Gastro-\nintestinal',
           'Metabolic': 'Meta-\nbolic'}
SHORT = {'AF': 'Atrial fibrillation', 'T2D': 'Type 2 diabetes', 'PANC': 'Pancreatitis', 'DEM': 'All-cause dementia',
         'PD': "Parkinson's disease", 'FALL': 'Falls', 'DR': 'Diabetic retinopathy'}


def pc(C):
    t = CC[CC.column_order.notna()].sort_values('column_order').copy()   # 46 tier-2 diseases + control, Fig. 3a order
    xs, groups, x, prev = {}, [], 0.0, None
    for r in t.itertuples():
        if prev is not None and r.column_group != prev:
            x += 0.9
        if r.column_group != prev:
            groups.append([r.column_group, x, x])
        xs[r.outcome] = x; groups[-1][2] = x; x += 1; prev = r.column_group
    XL = (-0.8, x - 1 + 0.8)
    AX0, AY0, AW, AH = 15.0, 7.0, 163.0, 24.0
    mm = lambda xd: AX0 + (xd - XL[0]) / (XL[1] - XL[0]) * AW
    ax = C.ax(AX0, AY0, AW, AH)
    for j, (g, a_, b_) in enumerate(groups):
        if j % 2 == 0:
            ax.axvspan(a_ - 0.5, b_ + 0.5, color=BAND, lw=0, zorder=0)
    ax.axhline(0, color=INK, lw=0.5, zorder=1)
    for r in t.itertuples():
        xx = xs[r.outcome]
        if r.outcome == 'NEG':
            ax.plot([xx, xx], [r.lo_G, r.hi_G], color=LIGHT, lw=0.5, zorder=2)
            ax.scatter([xx], [r.dC_min], marker='_', s=9, color='#7A7A7A', lw=0.7, zorder=3)
            ax.scatter([xx], [r.dC_G], marker='D', s=7, fc='white', ec=MID, lw=0.6, zorder=4)
            continue
        sig = r.lo_G > 0
        col = ORGAN_COL.get(r.clock, INK) if sig else LIGHT
        ax.plot([xx, xx], [r.lo_G, r.hi_G], color=col, lw=0.6 if sig else 0.5, zorder=2, solid_capstyle='butt')
        ax.scatter([xx], [r.dC_min], marker='_', s=9, color='#7A7A7A', lw=0.7, zorder=3)
        ax.scatter([xx], [r.dC_G], s=9 if sig else 6, fc=col if sig else 'white', ec=col, lw=0.6, zorder=4)
        if r.q_G < 0.05:
            ax.scatter([xx], [r.dC_G], s=34, fc='none', ec=INK, lw=0.6, zorder=5)
            pass  # v5: ring marks q < 0.05; per-point q labels omitted (28 diseases in the full cohort)
    ax.set_xlim(*XL); ax.set_ylim(-0.047, 0.1)
    ax.set_yticks([-0.04, 0, 0.04, 0.08]); ax.set_yticklabels([um('-0.04'), '0', '0.04', '0.08'])
    ax.set_xticks([])
    ax.set_ylabel('ΔC-index over\nmodel G (95% CI)', linespacing=1.0)
    clean(ax, bottom=False, grid='y')
    # group headers (as in Fig. 3a)
    for g, a_, b_ in groups:
        colg = ORGAN_COL.get(g, MID)
        C.line([mm(a_ - 0.42), mm(b_ + 0.42)], [AY0 - 1.0] * 2, color=colg, lw=0.9, solid_capstyle='butt')
        if g != 'Control':
            C.text((mm(a_) + mm(b_)) / 2, AY0 - 1.5, g, ha='center', va='bottom', fontsize=5.0, color=colg)
    # disease names (as in Fig. 3a): tier-1 bold; 95% CI > 0 in the colour of the paired clock; control grey italic
    for r in t.itertuples():
        kw = dict(fontweight='bold') if r.tier1 else {}
        if r.outcome == 'NEG':
            kw = dict(color=MID, fontstyle='italic')
        elif r.lo_G > 0:
            kw['color'] = ORGAN_COL.get(r.clock, INK)
        C.text(mm(xs[r.outcome]) + 0.6, AY0 + AH + 1.0, r.label, ha='right', va='top', fontsize=5.0, rotation=52,
               rotation_mode='anchor', **kw)
    tt = t[t.outcome != 'NEG']
    nci = int((tt.lo_G > 0).sum()); nq = int((tt.q_G < 0.05).sum())
    nci0 = int((tt.lo_min > 0).sum()); nq0 = int((tt.q_min < 0.05).sum())
    med, med0 = tt.dC_G.median(), tt.dC_min.median()
    # legend row
    C.text(15, 1.2, '●', fontsize=5.6, color=ORGAN_COL['Brain'], va='center')
    C.text(17.3, 1.2, '95% CI excludes 0 (colour, paired clock)', fontsize=5.3, va='center')
    C.text(60, 1.2, '○', fontsize=5.6, color=LIGHT, va='center')
    C.text(62.3, 1.2, '95% CI includes 0', fontsize=5.3, va='center')
    C.line([82.2], [1.2], marker='o', ms=3.3, mfc='none', mec=INK, mew=0.6, lw=0)
    C.text(84.3, 1.2, 'FDR < 0.05', fontsize=5.3, va='center')
    C.text(98, 1.2, '–', fontsize=7, color='#7A7A7A', va='center', fontweight='bold')
    C.text(100.3, 1.2, 'ΔC over the minimal model (model a + CKM stage)', fontsize=5.3, va='center')
    C.text(150, 1.2, 'Bold, tier-1 disease', fontsize=5.3, va='center', fontweight='bold')
    ax.text(0.995, 0.97, f'46 tier-2 diseases\nOver model G: {nci} with 95% CI > 0, {nq} at FDR < 0.05; median ΔC {med:.4f}'
            f'\nOver the minimal model: {nci0} with 95% CI > 0, {nq0} at FDR < 0.05; median ΔC {med0:.4f}',
            transform=ax.transAxes, fontsize=5.1, va='top', ha='right', color=MID, linespacing=1.15)
# ---------------------------------------------------------------------------------------------- d
CAP = rd('Fig5d_case_capture_curves'); C10 = rd('Fig5d_capture_at_10pct')
DOUT = ['DEM', 'T2D', 'CKMC']


def pd_(C):
    for m, k in enumerate(DOUT):
        x0 = 10 + m * 27.5
        ax = C.ax(x0, 10.2, 22, 20.8)
        g = CAP[CAP.outcome == k]
        ax.plot([0, 40], [0, 40], color=FAINT, lw=0.6, ls=(0, (2, 1.5)), zorder=1)
        for lab, col, ls, lw in [('PREVENT', COL_PREV, (0, (2.5, 1.2)), 0.8), ('Model G', COL_G, '-', 0.8),
                                 ('Model G + age gaps', COL_N, '-', 1.0)]:
            h = g[g.selector == lab]
            ax.plot(np.r_[0, h.fraction_selected * 100], np.r_[0, h.cases_captured * 100], color=col, ls=ls, lw=lw, zorder=3)
        ax.axvline(10, color=LIGHT, lw=0.4, ls=(0, (1, 1)), zorder=2)
        r = C10[(C10.outcome == k) & (C10.comparison == 'Model G + age gaps minus Model G')].iloc[0]
        ax.set_xlim(0, 40); ax.set_ylim(0, 100)
        ax.set_xticks([0, 10, 20, 40]); ax.set_yticks([0, 50, 100])
        if m == 0:
            ax.set_ylabel('10-year cases captured (%)')
        else:
            ax.set_yticklabels([])
        clean(ax, grid='y')
        C.text(x0 + 11, 4.9, OLAB[k], fontsize=5.8, fontweight='bold', ha='center')
        C.text(x0 + 11, 8.0, um(f'At 10%: {100 * r['diff']:+.1f} ({100 * r.lo:.1f} to {100 * r.hi:.1f})'), fontsize=4.9,
               color=COL_N if r.lo > 0 else MID, ha='center')
        C.text(x0 + 21.6, 29.4, f'{int(r.cases10_stage12):,} cases', fontsize=4.8, color=MID, ha='right')
    C.text(51, 35.8, 'Participants selected, ranked by predicted risk (%)', fontsize=6, ha='center')
    for j, (lab, col, ls) in enumerate([('PREVENT', COL_PREV, (0, (2.5, 1.2))), ('Model G', COL_G, '-'),
                                        ('Model G + 12 age gaps', COL_N, '-')]):
        xx = 9 + [0, 17, 34][j]
        C.line([xx, xx + 4], [1.6, 1.6], color=col, ls=ls, lw=0.9)
        C.text(xx + 5, 1.6, lab, fontsize=5.3, va='center')
    C.text(89, 1.6, 'CKM stages 1–2', fontsize=5.3, va='center', ha='right', color=MID)


# ---------------------------------------------------------------------------------------------- e
RC = rd('Fig5e_reclassification_observed_risk')
EOUT = ['DEM', 'T2D', 'CKMC']


def pe(C):
    for m, k in enumerate(EOUT):
        x0 = 11 + m * 26.5
        ax = C.ax(x0, 10.2, 21, 20.8)
        g = RC[RC.outcome == k]
        thr = sorted({float(s.replace('%', '').replace('≥', '').replace('<', '').split('–')[0]) for s in g.G_category})
        thr = [v for v in thr if v > 0]
        for v in thr:
            ax.axhline(v, color=LIGHT, lw=0.45, ls=(0, (2, 1.5)), zorder=1)
        for _, r in g.iterrows():
            if r.n < 30 or np.isnan(r.obs10):
                continue
            dx = {'Down': -0.26, 'Unchanged': 0.0, 'Up': 0.26}[r.movement]
            col = {'Down': COL_DOWN, 'Unchanged': COL_SAME, 'Up': COL_UP}[r.movement]
            xx = r.G_category_index + dx
            lo = max(r.obs10_lo * 100, 0.12)
            ax.plot([xx, xx], [lo, r.obs10_hi * 100], color=col, lw=0.6, zorder=2)
            ax.scatter([xx], [max(r.obs10 * 100, 0.12)], s=8, color=col, lw=0, zorder=3)
        ax.set_yscale('log'); ax.set_ylim(0.25, 30)
        ax.set_yticks([0.5, 1, 2, 5, 10, 20])
        ax.set_yticklabels(['0.5', '1', '2', '5', '10', '20'] if m == 0 else [])
        ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
        cats = g.drop_duplicates('G_category_index').sort_values('G_category_index').G_category.tolist()
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(cats, fontsize=5.0)
        ax.set_xlim(-0.55, 2.55)
        if m == 0:
            ax.set_ylabel('Observed 10-year risk (%)')
        clean(ax)
        C.text(x0 + 10.5, 4.9, OLAB[k], fontsize=5.8, fontweight='bold', ha='center')
    C.text(48, 35.8, '10-year risk category under model G (dashed lines, category thresholds)', fontsize=6, ha='center')
    C.text(4, 1.6, 'After adding the 12 gaps:', fontsize=5.3, va='center')
    for j, (lab, col) in enumerate([('moved down', COL_DOWN), ('unchanged', COL_SAME), ('moved up', COL_UP)]):
        xx = [30, 46, 60][j]
        C.text(xx, 1.6, '●', fontsize=5.6, color=col, va='center')
        C.text(xx + 2.2, 1.6, lab, fontsize=5.3, va='center')


# ---------------------------------------------------------------------------------------------- f
NB = rd('Fig5f_net_benefit_difference')
XMAX = {'CKMC': 30, 'ASCVD': 20}


def pf(C):
    for m, k in enumerate(ORDER):
        x0 = 13 + m * 23.6
        ax = C.ax(x0, 7, 19, 17)
        xm = XMAX.get(k, 10)
        g = NB[(NB.outcome == k) & (NB.pt * 100 <= xm + 1e-9)].sort_values('pt')
        step = {'CKMC': 3, 'ASCVD': 2}.get(k, 1)
        g = g.iloc[step - 1::step]
        ax.axhline(0, color=INK, lw=0.45, zorder=1)
        ax.fill_between(g.pt * 100, g.lo_per1000, g.hi_per1000, color=lighten(COL_N, 0.72), lw=0, zorder=2)
        ax.plot(g.pt * 100, g.dnb_per1000, color=COL_N, lw=0.8, zorder=3)
        pos = g[g.lo_per1000 > 0]
        if len(pos):
            ax.scatter(pos.pt * 100, np.full(len(pos), -2.6), marker='s', s=2.2, color=COL_N, lw=0, zorder=3)
        ax.set_xlim(0, xm); ax.set_ylim(-3, 4.5)
        ax.set_xticks([0, xm / 2, xm]); ax.set_xticklabels([f'{v:g}' for v in [0, xm / 2, xm]])
        ax.set_yticks([-2, 0, 2, 4])
        if m == 0:
            ax.set_yticklabels([um('-2'), '0', '2', '4'])
            ax.set_ylabel('ΔNet benefit\nper 1,000', linespacing=1.0)
        else:
            ax.set_yticklabels([])
        clean(ax, grid='y')
        C.text(x0 + 9.5, 4.2, OLAB[k], fontsize=5.6, fontweight='bold', ha='center')
    C.text(92, 30.0, 'Threshold probability (%)', fontsize=6, ha='center')
    C.text(176, 1.2, 'Model G + 12 gaps minus model G (95% CI band; 20 thresholds per outcome); squares: lower 95% bound above 0',
           fontsize=5.0, ha='right', color=MID, va='center')


PANELS = {
    'a': (pa, 0, 0, 64, 58),
    'b': (pb, 66, 0, 114, 58),
    'c': (pc, 0, 60, 180, 54),
    'd': (pd_, 0, 115, 92, 39),
    'e': (pe, 92, 115, 88, 39),
    'f': (pf, 0, 155, 180, 33),
}
if __name__ == '__main__':
    build(PANELS, 180, 188, 'Fig5', ROOT)
