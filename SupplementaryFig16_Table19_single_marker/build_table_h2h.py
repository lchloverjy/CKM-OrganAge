"""Build formatted Supplementary Table (head-to-head clocks vs single markers) from h2h/results."""
import os, pandas as pd, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..', 'results') + '/'
OUT = os.path.join(HERE, '..', 'figure', 'Supplementary_Table_H2H.xlsx')
EN = '\u2013'; MN = '\u2212'
f2 = lambda x: (f'{x:.2f}').replace('-', MN)
f3 = lambda x: (f'{x:.3f}').replace('-', MN)
f4 = lambda x: (f'{x:.4f}').replace('-', MN)
hrs = lambda r: f'{f2(r.hr)} ({f2(r.lo)}{EN}{f2(r.hi)})'
dcs = lambda r: f'{f4(r.dC)} ({f4(r.lo)} to {f4(r.hi)})'
OUTN = {'HF': 'Heart failure', 'CKD': 'Chronic kidney disease', 'DEM': 'Dementia',
        'CKMC': 'CKM composite', 'MSYS': '≥2 new organ systems'}
MK = {'m_NTproBNP': 'NT-proBNP (Olink)', 'm_NPPB': 'NPPB (Olink)', 'm_TNNI3': 'Troponin I (Olink TNNI3)',
      'm_GDF15': 'GDF15 (Olink)', 'm_cysc': 'Cystatin C (clinical, log)', 'm_egfrcc': 'eGFRcr-cys (per s.d. lower)',
      'm_CST3': 'Cystatin C (Olink CST3)', 'm_NEFL': 'NfL (Olink NEFL)', 'm_GFAP': 'GFAP (Olink)'}
CK = {'HF': 'Heart age gap', 'CKD': 'Kidney age gap', 'DEM': 'Brain age gap'}

# a: associations
H = pd.read_csv(R + 'H1_hr.csv')
H['Outcome'] = H.outcome.map(OUTN); H['HR per s.d. (95% CI)'] = H.apply(hrs, axis=1)
H['Exposure'] = H.label
H['Adjusted together with'] = H.pair_with.fillna('').map(lambda s: ' + '.join(MK.get(x, x) for x in s.split('+')) if s else '')
H['Model'] = H.analysis
A = H[['Outcome', 'Model', 'Exposure', 'Adjusted together with', 'n', 'events', 'HR per s.d. (95% CI)']]
A = A.rename(columns={'n': 'n', 'events': 'Events'})

# b: discrimination
D = pd.read_csv(R + 'H1_dC.csv'); HO = pd.read_csv(R + 'H1_head_to_head_holm.csv')
def lab(m):
    if m == 'b':
        return 'Model b'
    m = m.replace('b+', 'Model b + ', 1).replace('panel', 'organ panel').replace('clock', CKX)
    for k, v in MK.items():
        m = m.replace(k, v)
    return m.replace('+', ' + ').replace('  ', ' ')
rows = []
for _, r in D.iterrows():
    global CKX
    CKX = CK[r.outcome]
    ho = HO[(HO.outcome == r.outcome) & (HO.model == r.model) & (HO.baseline == r.baseline)]
    rows.append({'Outcome': OUTN[r.outcome], 'Comparison model': lab(r.model), 'Reference model': lab(r.baseline),
                 'n': int(r.n), 'Events': int(r.events), 'C, comparison': f3(r.C_new), 'C, reference': f3(r.C_base),
                 'ΔC (95% CI)': dcs(r), 'Holm-adjusted P (head-to-head)': (f'{ho.p_holm.iloc[0]:.2g}' if len(ho) else '')})
B = pd.DataFrame(rows)

# c: absolute risk 2x2 + conditional HRs
AJ = pd.read_csv(R + 'H1_aj.csv')
G = AJ[AJ.clock_aged.notna()].copy()
grp = {(0, 0): 'Neither', (0, 1): 'Marker elevated only', (1, 0): 'Clock aged only', (1, 1): 'Both'}
G['Group'] = [grp[(int(a), int(b))] for a, b in zip(G.clock_aged, G.marker_high)]
G['10-y cumulative incidence, % (95% CI)'] = G.apply(lambda r: f'{r.risk10:.1f} ({r.lo:.1f}{EN}{r.hi:.1f})', axis=1)
G['Outcome'] = G.outcome.map(OUTN); G['Clock'] = G.outcome.map(CK); G['Marker'] = G.marker.map(MK)
C1 = G[['Outcome', 'Clock', 'Marker', 'Group', 'n', 'events', '10-y cumulative incidence, % (95% CI)']].rename(columns={'events': 'Events'})
K = AJ[AJ.clock_aged.isna()].copy()
K['Outcome'] = K.outcome.map(OUTN); K['Marker'] = K.marker.map(MK)
K['HR (95% CI), model b'] = K.apply(hrs, axis=1)
C2 = K[['Outcome', 'Marker', 'analysis', 'n', 'events', 'HR (95% CI), model b']].rename(columns={'analysis': 'Contrast', 'events': 'Events'})

# d: multi-organ
MD = pd.read_csv(R + 'H4_multi_dC.csv'); MH = pd.read_csv(R + 'H4_multi_hr.csv'); MA = pd.read_csv(R + 'H4_multi_aj.csv')
def mlab(m):
    if m in ('b', 'G'):
        return 'Model ' + m
    rep = [('b+', 'Model b + '), ('G+', 'Model G + '), ('panel6', 'six-marker panel'), ('gaps12', 'all 12 age gaps'),
           ('burden', 'aged-organ burden'), ('organismal', 'organismal age gap'), ('brainmk', 'GDF15 + NfL + GFAP')]
    for a, b in rep:
        m = m.replace(a, b)
    return m.replace('+', ' + ').replace('  ', ' ').strip()
D1 = pd.DataFrame([{'Outcome': OUTN[r.outcome], 'Comparison model': mlab(r.model), 'Reference model': mlab(r.baseline),
                    'n': int(r.n), 'Events': int(r.events), 'C, comparison': f3(r.C_new), 'C, reference': f3(r.C_base),
                    'ΔC (95% CI)': dcs(r)} for _, r in MD.iterrows()])
MH['term'] = MH.label.replace({'bur1': '1 aged organ vs 0', 'bur2': '≥2 aged organs vs 0'})
D2 = pd.DataFrame({'Outcome': MH.outcome.map(OUTN), 'Model': MH.analysis, 'Term': MH.term,
                   'HR (95% CI)': MH.apply(hrs, axis=1)})
D3 = pd.DataFrame({'Outcome': MA.outcome.map(OUTN), 'Aged target organs': MA.burden.map({0: '0', 1: '1', 2: '≥2'}),
                   '≥1 of six markers elevated': MA.any_marker_high.map({0: 'No', 1: 'Yes'}), 'n': MA.n, 'Events': MA.events,
                   '10-y cumulative incidence, % (95% CI)': MA.apply(lambda r: f'{r.risk10:.1f} ({r.lo:.1f}{EN}{r.hi:.1f})', axis=1)})

# e: sensitivity + time split
S = pd.read_csv(R + 'H1_s1_dC.csv')
E1 = pd.DataFrame([{'Outcome': OUTN[r.outcome], 'Marker': MK[r.marker],
                    'Contrast': {('b+clock', 'b+m'): 'Clock vs marker (head-to-head)', ('b+clock+m', 'b+m'): 'Clock added to marker',
                                 ('b+clock+m', 'b+clock'): 'Marker added to clock'}.get((r.model, r.baseline), f'{r.model} vs {r.baseline}'),
                    'n': int(r.n), 'Events': int(r.events), 'ΔC (95% CI)': dcs(r)} for _, r in S.iterrows()])
T = pd.read_csv(R + 'H1_tsplit.csv')
E2 = pd.DataFrame({'Outcome': T.outcome.map(OUTN), 'Follow-up window': T.analysis, 'Exposure': T.label,
                   'Events in window': T.events, 'HR per s.d. (95% CI), model b': T.apply(hrs, axis=1)})

NOTES = {
 'a': 'Cox proportional hazards models with time since baseline as the time scale; model b covariates (age, age², sex, ethnicity, Townsend index, education, smoking, alcohol intake and CKM stage). Organ panels: heart failure, NT-proBNP + troponin I; CKD, eGFRcr-cys + Olink cystatin C; dementia, NfL + GFAP. Model c adds stage-defining traits; for CKD, model c contains eGFR and is collinear with the kidney markers and is not interpreted.',
 'b': 'Harrell C from five-fold cross-fitting repeated twice; ΔC 95% CIs from 500 paired bootstrap samples. Holm correction across the 11 head-to-head comparisons (model b + clock vs model b + marker).',
 'c': 'Clock aged and marker elevated: above the sex- and age-band-specific 90th percentile in stage 0 (identical rule for clocks and markers; lowest decile for eGFRcr-cys). 10-year Aalen–Johansen cumulative incidence with death as a competing event, 200 bootstrap samples. Conditional HRs, model b.',
 'd': 'CKM composite: first of incident type 2 diabetes, CKD or cardiovascular disease among participants free of all three at baseline (n = 37,904; 6,652 events). Six-marker panel: NT-proBNP, troponin I, clinical cystatin C, NfL, GFAP and GDF15. Model G as in Fig. 5.',
 'e': 'S1: Olink values exactly equal to the cohort median (0 after centring; 1.6–5.4% of values, consistent with median imputation) set to missing, complete-case analysis. Time-split: follow-up split at 5 years, model b.'}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with pd.ExcelWriter(OUT, engine='openpyxl') as w:
    for nm, df in [('a_associations', A), ('b_discrimination', B), ('c1_absolute_risk_2x2', C1), ('c2_conditional_HR', C2),
                   ('d1_multiorgan_dC', D1), ('d2_multiorgan_HR', D2), ('d3_multiorgan_risk', D3),
                   ('e1_sensitivity_S1', E1), ('e2_time_split', E2)]:
        df.to_excel(w, sheet_name=nm, index=False, startrow=1)
        ws = w.sheets[nm]; ws.cell(row=1, column=1, value=NOTES[nm[0]])
        for col in ws.columns:
            L = max(len(str(c.value or '')) for c in col[1:]) if len(col) > 1 else 10
            ws.column_dimensions[col[0].column_letter].width = min(max(10, L + 2), 48)
print('written', OUT, len(A), len(B), len(C1), len(C2), len(D1), len(D2), len(D3), len(E1), len(E2))
