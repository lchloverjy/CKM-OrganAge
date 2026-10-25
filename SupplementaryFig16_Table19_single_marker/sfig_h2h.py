"""Supplementary Figure: organ clocks versus single plasma biomarkers (head-to-head).
Reads ../results/*.csv (from h2h_analysis.py); writes source data and panels.
Run: cd /home/ubuntu/work/v5/h2h/code && python3 sfig_h2h.py [a b c d e]
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/ubuntu/work/v5/figure6/code')
from style4 import *
import pandas as pd, numpy as np
R = os.path.join(HERE, '..', 'results') + '/'
FIG = os.path.join(HERE, '..', 'figure')
SD = os.path.join(FIG, 'source_data') + '/'; os.makedirs(SD, exist_ok=True)

MCOL = '#4A4E57'
NAME = {'m_NTproBNP': 'NT-proBNP', 'm_NPPB': 'NPPB†', 'm_TNNI3': 'Troponin I', 'm_GDF15': 'GDF15',
        'm_cysc': 'Cystatin C, clinical', 'm_egfrcc': 'eGFRcr-cys (lower)', 'm_CST3': 'Cystatin C, Olink',
        'm_NEFL': 'NfL†', 'm_GFAP': 'GFAP†'}
OUTS = [('HF', 'Heart failure', 'Heart', 'gap_Heart', ['m_NTproBNP', 'm_NPPB', 'm_TNNI3', 'm_GDF15']),
        ('CKD', 'Chronic kidney disease', 'Kidney', 'gap_Kidney', ['m_egfrcc', 'm_cysc', 'm_CST3', 'm_GDF15']),
        ('DEM', 'Dementia', 'Brain', 'gap_Brain', ['m_NEFL', 'm_GFAP', 'm_GDF15'])]

# ------------------------------------------------------------------ source data
Cc = pd.read_csv(R + 'H0_correlations.csv')
D = pd.read_csv(R + 'H1_dC.csv')
AJ = pd.read_csv(R + 'H1_aj.csv'); AJ = AJ[AJ.clock_aged.notna()]
MD = pd.read_csv(R + 'H4_multi_dC.csv'); MA = pd.read_csv(R + 'H4_multi_aj.csv')
def dc(k, new, base):
    r = D[(D.outcome == k) & (D.model == new) & (D.baseline == base)]
    assert len(r) == 1, (k, new, base); return r.iloc[0]
rows_b = []
for k, ttl, org, clock, mk in OUTS:
    r = dc(k, 'b+clock', 'b'); r2 = dc(k, 'b+panel+clock', 'b+panel')
    rows_b.append(dict(outcome=k, exposure=org + ' age gap', kind='clock', dC_alone=r.dC, lo=r.lo, hi=r.hi,
                       dC_clock_added=r2.dC, lo2=r2.lo, hi2=r2.hi, added_to='organ marker panel'))
    for m in mk:
        r = dc(k, 'b+' + m, 'b'); r2 = dc(k, 'b+clock+' + m, 'b+' + m)
        rows_b.append(dict(outcome=k, exposure=NAME[m], kind='marker', dC_alone=r.dC, lo=r.lo, hi=r.hi,
                           dC_clock_added=r2.dC, lo2=r2.lo, hi2=r2.hi, added_to=NAME[m]))
B_ = pd.DataFrame(rows_b)
PAIRS_C = [('HF', 'm_NTproBNP', 'Heart failure', 'Heart', 'NT-proBNP'),
           ('CKD', 'm_egfrcc', 'Chronic kidney disease', 'Kidney', 'eGFRcr-cys'),
           ('DEM', 'm_NEFL', 'Dementia', 'Brain', 'NfL'),
           ('DEM', 'm_GFAP', 'Dementia', 'Brain', 'GFAP')]
C_ = pd.concat([AJ[(AJ.outcome == k) & (AJ.marker == m)].assign(marker_label=ml) for k, m, _, _, ml in PAIRS_C])
ROWS_D = [('b+burden', 'b', 'Aged-organ burden', 'alone'), ('b+organismal', 'b', 'Organismal age gap', 'alone'),
          ('b+gaps12', 'b', 'All 12 age gaps', 'alone'), ('b+GDF15', 'b', 'GDF15', 'alone'),
          ('b+panel6', 'b', 'Six-marker panel', 'alone'),
          ('b+panel6+burden', 'b+panel6', '+ Aged-organ burden', 'beyond'),
          ('b+panel6+gaps12', 'b+panel6', '+ All 12 age gaps', 'beyond')]
D_ = pd.concat([MD[(MD.model == n) & (MD.baseline == b)].assign(row=lab, block=bl) for n, b, lab, bl in ROWS_D])
with pd.ExcelWriter(SD + 'SFig_h2h_source_data.xlsx') as w:
    Cc.to_excel(w, 'a_correlations', index=False); B_.to_excel(w, 'b_discrimination', index=False)
    C_.to_excel(w, 'c_absolute_risk', index=False); D_.to_excel(w, 'd_multiorgan_dC', index=False)
    MA.to_excel(w, 'e_multiorgan_risk', index=False)
for nm, df in [('a_correlations', Cc), ('b_discrimination', B_), ('c_absolute_risk', C_), ('d_multiorgan_dC', D_),
               ('e_multiorgan_risk', MA)]:
    df.to_csv(SD + f'SFig_h2h_{nm}.csv', index=False)

def head(C, x, y, t):
    C.text(x, y, t, fontsize=6.6, fontweight='bold')

# ------------------------------------------------------------------ a: correlation heatmap
def pa(C):
    head(C, 4.5, 0.3, 'Overlap with single markers')
    clocks = [('gap_Heart', 'Heart'), ('gap_Kidney', 'Kidney'), ('gap_Brain', 'Brain'), ('gap_Organismal', 'Organismal')]
    mk = ['m_NTproBNP', 'm_NPPB', 'm_TNNI3', 'm_egfrcc', 'm_cysc', 'm_CST3', 'm_NEFL', 'm_GFAP', 'm_GDF15']
    lab = ['NT-proBNP', 'NPPB', 'Troponin I', 'eGFRcr-cys', 'Cys C, clin.', 'Cys C, Olink', 'NfL', 'GFAP', 'GDF15']
    M = np.array([[Cc[(Cc.clock == c) & (Cc.marker == m)].rho.iloc[0] for m in mk] for c, _ in clocks])
    ax = C.ax(17, 8.5, 41, 22)
    ax.imshow(M, cmap=SEQ, vmin=0, vmax=0.9, aspect='auto')
    comp = {(0, 1), (2, 6), (2, 7), (3, 8)}
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f'{v:.2f}'.replace('0.', '.').replace('-.', '\u2212.'), ha='center', va='center',
                    fontsize=5.0, color='white' if v > 0.45 else INK)
            if (i, j) in comp:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=0.8))
    ax.set_yticks(range(4)); ax.set_yticklabels([n for _, n in clocks])
    for t, (_, n) in zip(ax.get_yticklabels(), clocks):
        t.set_color(ORGAN_COL[n])
    ax.set_xticks(range(len(mk))); ax.set_xticklabels(lab, rotation=55, ha='right', rotation_mode='anchor')
    ax.tick_params(length=0, labelsize=5.4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    hcolorbar(C, SEQ, mpl.colors.Normalize(0, 0.9), 20, 52.5, 22, 1.5, 'Spearman ρ', ticks=[0, 0.3, 0.6, 0.9],
              fmt=lambda t: f'{t:.1f}')
    C.rect(45.2, 52.7, 2.2, 1.6, ec=INK, lw=0.8)
    C.text(48.3, 52.5, 'clock\ncomponent', fontsize=5.2, va='top', linespacing=0.95)

# ------------------------------------------------------------------ b: discrimination head-to-head + clock added
def pb(C):
    head(C, 4.5, 0.3, 'Head-to-head discrimination, organ-matched outcomes')
    rows = []
    for k, ttl, org, clock, mk in OUTS:
        rows.append(('h', ttl, org, None))
        g = B_[B_.outcome == k]
        for _, r in g.iterrows():
            rows.append(('r', r.exposure, org, r))
    n = len(rows); rh = 2.62; y0 = 9.5
    xa, wa = 33, 40; xb, wb = 80, 38
    axA = C.ax(xa, y0, wa, rh * n); axB = C.ax(xb, y0, wb, rh * n)
    for ax in (axA, axB):
        ax.set_ylim(n - 0.5, -0.5); ax.set_yticks([]); clean(ax, left=False, grid='x')
        ax.axvline(0, color=INK, lw=0.45)
    for i, (kind, lab, org, r) in enumerate(rows):
        yy = y0 + rh * (i + 0.5)
        if kind == 'h':
            C.text(4.5, yy, lab, fontsize=5.8, fontweight='bold', va='center', color=ORGAN_COL[org])
            for ax in (axA, axB):
                ax.axhspan(i - 0.5, i + 0.5, color=BAND, lw=0, zorder=0)
            continue
        isc = r.kind == 'clock'; col = ORGAN_COL[org] if isc else MCOL
        C.text(6.5, yy, lab, fontsize=5.5, va='center', color=col, fontweight='bold' if isc else 'normal')
        ci(axA, np.array([r.dC_alone]), np.array([i]), np.array([r.lo]), np.array([r.hi]), color=col,
           ms=2.6 if isc else 2.3, lw=0.65, marker='D' if isc else 'o')
        ci(axB, np.array([r.dC_clock_added]), np.array([i]), np.array([r.lo2]), np.array([r.hi2]),
           color=ORGAN_COL[org], ms=2.3, lw=0.65, filled=not isc, marker='o')
    axA.set_xlim(-0.003, 0.092); axA.set_xticks([0, 0.03, 0.06, 0.09])
    axA.set_xticklabels(['0', '0.03', '0.06', '0.09'])
    axB.set_xlim(-0.004, 0.028); axB.set_xticks([0, 0.01, 0.02]); axB.set_xticklabels(['0', '0.01', '0.02'])
    axA.set_xlabel('ΔC, each exposure over model b'); axB.set_xlabel('ΔC, clock added to the marker')
    C.text(xa + wa / 2, 6.0, 'Each alone', fontsize=5.8, ha='center', color=MID)
    C.text(xb + wb / 2, 6.0, 'Clock beyond marker (open: beyond organ panel)', fontsize=5.8, ha='center', color=MID)

# ------------------------------------------------------------------ c: 2x2 absolute risk
def pc(C):
    head(C, 4.5, 0.3, '10-year risk by aged clock and elevated marker')
    cats = [(0, 0, 'Neither'), (0, 1, 'Marker'), (1, 0, 'Clock'), (1, 1, 'Both')]
    W = 43.5
    for j, (k, m, ttl, org, ml) in enumerate(PAIRS_C):
        x0 = 4.5 + j * (W + 2.0)
        g = C_[(C_.outcome == k) & (C_.marker == m)]
        ax = C.ax(x0 + 8.5, 11.5, W - 10.5, 24)
        ymax = 0; ns = []
        for i, (hc, hm, cl) in enumerate(cats):
            r = g[(g.clock_aged == hc) & (g.marker_high == hm)].iloc[0]
            col = {0: LIGHT, 1: MCOL, 2: lighten(ORGAN_COL[org], 0.25), 3: ORGAN_COL[org]}[i]
            ax.bar(i, r.risk10, width=0.62, color=lighten(col, 0.55), lw=0, zorder=1)
            ci(ax, np.array([i]), np.array([r.risk10]), np.array([r.lo]), np.array([r.hi]), color=col,
               horizontal=False, ms=2.3, lw=0.65)
            ns.append(f'{int(r.n):,}')
            ymax = max(ymax, r.hi)
        ax.set_xticks(range(4)); ax.set_xticklabels([f'{c[2]}\n{n_}' for c, n_ in zip(cats, ns)], fontsize=5.2,
                                                    linespacing=1.15)
        ax.tick_params(axis='x', length=0, pad=1.5)
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%g'))
        ax.set_ylim(0, ymax * 1.18); clean(ax, grid='y')
        if j == 0:
            ax.set_ylabel('10-y cumulative incidence (%)')
        C.text(x0 + 8.5 + (W - 10.5) / 2, 5.6, ttl, fontsize=5.8, fontweight='bold', ha='center',
               color=ORGAN_COL[org])
        C.text(x0 + 8.5 + (W - 10.5) / 2, 8.4, f'{org} clock × {ml}', fontsize=5.4, ha='center', color=MID)

# ------------------------------------------------------------------ d: multi-organ discrimination
def pd_(C):
    head(C, 4.5, 0.3, 'Multi-organ outcomes: clocks versus single markers')
    labs = [r[2] for r in ROWS_D]
    pos = [0, 1, 2, 3, 4, 6, 7]
    n = 8; rh = 4.0; y0 = 10.5
    ax = C.ax(36, y0, 50, rh * n)
    ax.set_ylim(n - 0.5, -0.5); ax.set_yticks([]); clean(ax, left=False, grid='x')
    ax.axvline(0, color=INK, lw=0.45)
    ax.axhspan(5.5, 7.5, color=BAND, lw=0, zorder=0)
    for kk, (key, col, off, nm) in enumerate([('CKMC', ACCENT, -0.17, 'CKM composite'),
                                              ('MSYS', BLUE, 0.17, '≥2 new organ systems')]):
        for i, (nw, bs, lab, bl) in enumerate(ROWS_D):
            r = D_[(D_.outcome == key) & (D_.row == lab)].iloc[0]
            isclock = 'gap' in lab or 'burden' in lab
            ci(ax, np.array([r.dC]), np.array([pos[i] + off]), np.array([r.lo]), np.array([r.hi]), color=col, ms=2.4,
               lw=0.65, filled=not isclock)
        ax.scatter([], [], s=2.4 ** 2, color=col, label=nm)
    for i, lab in enumerate(labs):
        isclock = 'gap' in lab or 'burden' in lab
        C.text(6.5 if not lab.startswith('+') else 8.5, y0 + rh * (pos[i] + 0.5), lab, fontsize=5.6, va='center',
               fontweight='bold' if isclock else 'normal')
    C.text(4.5, y0 - 0.6, 'Each over model b', fontsize=5.6, color=MID, va='bottom')
    C.text(4.5, y0 + rh * 5.5, 'Beyond the six-marker panel', fontsize=5.6, color=MID, va='center')
    ax.set_xlim(-0.001, 0.0262); ax.set_xticks([0, 0.01, 0.02]); ax.set_xticklabels(['0', '0.01', '0.02'])
    ax.set_xlabel('Cross-fitted ΔC')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=5.4, handletextpad=0.2, borderaxespad=0.1)
    C.text(88, y0 + 9.5, 'open: clock based\nfilled: single marker', fontsize=5.1, color=MID, va='top',
           linespacing=1.0)

# ------------------------------------------------------------------ e: multi-organ absolute risk
def pe(C):
    head(C, 4.5, 0.3, 'CKM composite by burden and markers')
    g = MA[MA.outcome == 'CKMC']
    ax = C.ax(14, 10.5, 52, 30)
    for hm, col, nm, off in [(0, LIGHT, 'No marker elevated', -0.08), (1, ACCENT, '≥1 of six elevated', 0.08)]:
        gg = g[g.any_marker_high == hm].sort_values('burden')
        x = gg.burden.values + off
        ax.plot(x, gg.risk10, color=col, lw=0.8, zorder=2)
        ci(ax, x, gg.risk10.values, gg.lo.values, gg.hi.values, color=col, horizontal=False, ms=2.6, lw=0.7)
        for xi, (_, r) in zip(x, gg.iterrows()):
            ax.annotate(f'{r.risk10:.1f}', (xi, r.risk10), xytext=(-4, 4) if hm else (4, -5),
                        textcoords='offset points', fontsize=5.2, ha='right' if hm else 'left',
                        va='bottom' if hm else 'top', color=col if hm else MID)
        ax.plot([], [], color=col, marker='o', ms=2.6, lw=0.8, label=nm)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['0', '1', '≥2']); ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(5, 26); clean(ax, grid='y')
    ax.set_xlabel('Aged target organs'); ax.set_ylabel('10-y cumulative incidence (%)')
    ax.legend(loc='upper left', fontsize=5.4, borderaxespad=0.1)

W, H = 184, 170
PANELS = {'a': (pa, 0, 0, 62, 62), 'b': (pb, 62, 0, 122, 62),
          'c': (pc, 0, 64, 184, 46),
          'd': (pd_, 0, 112, 114, 56), 'e': (pe, 116, 112, 68, 56)}
if __name__ == '__main__':
    sel = sys.argv[1:] or list(PANELS)
    for lab in sel:
        fn, x, y, w, h = PANELS[lab]
        single(fn, w, h, f'SFig_h2h_{lab}', os.path.join(FIG, 'panels'), lab=lab)
    if not sys.argv[1:]:
        compose([(fn, x, y, lab) for lab, (fn, x, y, w, h) in PANELS.items()], W, H, 'SFig_h2h', FIG)
    print('done', sel)
