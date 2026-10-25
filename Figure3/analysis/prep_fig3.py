"""Export source data for Figure 3 (organ age gaps and incident disease of the corresponding organ).

Panels
  a  Association atlas: 12 clocks x 46 tier-2 diseases (+ negative control), model b
  b  Tier-1 prespecified pairs under models a-d, with bootstrap probability that the paired clock ranks first
  c  Organ correspondence index (43 organ-paired tier-2 diseases; 200 bootstrap resamples)
  d  Phenome-wide summary by clock (155 phecodes, model c)
  e  Kidney-function residualisation of the 43 paired associations
Panels moved to Supplementary Fig. 9: system-composite specificity (9b) and phenome-wide Manhattan plot (9c).
"""
import os, json
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

R = './results/v3/'
R4 = './results/v4/'
OUT = './v4/Figure3/source_data/'
SOUT = './v4/Supplementary/Supplementary_Figures/source_data/'
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    if f.endswith('.csv') or f.endswith('.xlsx'):
        os.remove(OUT + f)

PAIRS = [('HF', 'Heart'), ('CKD', 'Kidney'), ('DEM', 'Brain'), ('ASCVD', 'Artery'), ('T2D', 'Pancreas'), ('CLD', 'Liver')]
TIER1 = [k for k, _ in PAIRS]
ORG10 = ['Heart', 'Artery', 'Kidney', 'Pancreas', 'Liver', 'Brain', 'Lung', 'Immune', 'Muscle', 'Intestine']
ROWS = ORG10 + ['Organismal', 'Conventional']
cols = ['outcome', 'clock', 'model', 'hr', 'lo', 'hi', 'p', 'n', 'events']

OM = pd.read_csv('./results/V3_outcome_meta.csv')
T2 = pd.read_csv(R + 'M2_tier2_hr.csv').drop_duplicates(['outcome', 'clock', 'model'])
K46 = OM[OM.key != 'NEG'].key.tolist()
assert len(K46) == 46

# ------------------------------------------------------------------ a: atlas (model b)
SHORT = {'HF': 'Heart failure', 'AF': 'Atrial fibrillation', 'CMP': 'Cardiomyopathy', 'AVD': 'Aortic valve disease',
         'COND': 'Conduction disorders', 'ASCVD': 'ASCVD', 'CHD': 'Coronary heart disease',
         'MI': 'Myocardial infarction', 'ISTR': 'Ischaemic stroke', 'HSTR': 'Haemorrhagic stroke',
         'PAD': 'Peripheral artery disease', 'AAD': 'Aortic aneurysm', 'VTE': 'Venous thromboembolism',
         'DR': 'Diabetic retinopathy', 'CKD': 'CKD', 'AKI': 'Acute kidney injury', 'URO': 'Urolithiasis',
         'T2D': 'Type 2 diabetes', 'PANC': 'Pancreatitis', 'GOUT': 'Gout', 'CLD': 'Chronic liver disease',
         'GALL': 'Gallstone disease', 'DEM': 'Dementia', 'AD': "Alzheimer's disease", 'VAD': 'Vascular dementia',
         'PD': "Parkinson's disease", 'DEP': 'Depression', 'ANX': 'Anxiety', 'COPD': 'COPD', 'ASTH': 'Asthma',
         'ILD': 'Interstitial lung disease', 'OSA': 'Sleep apnoea', 'PNEU': 'Pneumonia', 'SEPS': 'Sepsis',
         'UTI': 'Urinary tract infection', 'CELL': 'Cellulitis', 'HIPF': 'Hip fracture', 'OSTP': 'Osteoporosis',
         'OA': 'Osteoarthritis', 'FALL': 'Falls', 'DIV': 'Diverticular disease', 'GORD': 'Reflux disease',
         'CRC': 'Colorectal cancer', 'CAT': 'Cataract', 'GLAU': 'Glaucoma', 'OBCA': 'Obesity-related cancers',
         'NEG': 'Transport accidents (control)'}
B = T2[(T2.model == 'b') & T2.outcome.isin(OM.key)].copy()
B['q_bh_552'] = np.nan
i46 = B.outcome.isin(K46)
B.loc[i46, 'q_bh_552'] = multipletests(B.loc[i46, 'p'], method='fdr_bh')[1]
ineg = B.outcome == 'NEG'
B.loc[ineg, 'q_bh_552'] = multipletests(B.loc[ineg, 'p'], method='fdr_bh')[1]   # control: FDR across its 12 clocks
meta = OM.set_index('key')
B['paired_clock'] = B.outcome.map(meta.clock)
B['paired'] = B.clock == B.paired_clock
B['log_hr'] = np.log(B.hr)
B['disease'] = B.outcome.map(meta.disease)
B['label'] = B.outcome.map(SHORT)
B['tier1'] = B.outcome.isin(TIER1)
B['events'] = B.outcome.map(meta.n_events).astype(int)
# rank of the paired clock among the ten organ clocks (point estimates)
rk = {}
for k in K46:
    pc = meta.loc[k, 'clock']
    if pc not in ORG10:
        continue
    d = B[(B.outcome == k) & B.clock.isin(ORG10)].set_index('clock').hr
    rk[k] = int((d > d[pc]).sum()) + 1
B['paired_rank_among_10'] = B.outcome.map(rk)
group = {k: (meta.loc[k, 'clock'] if meta.loc[k, 'clock'] != '-' else 'Control') for k in OM.key}
B['column_group'] = B.outcome.map(group)
# column order: by paired clock, tier 1 first, then events (descending)
order = []
for g in ORG10 + ['Organismal', 'Control']:
    ks = [k for k in OM.key if group[k] == g]
    ks = sorted(ks, key=lambda k: (k not in TIER1, -meta.loc[k, 'n_events']))
    order += ks
B['column_order'] = B.outcome.map({k: i for i, k in enumerate(order)})
B['row_order'] = B.clock.map({c: i for i, c in enumerate(ROWS)})
B = B.sort_values(['column_order', 'row_order'])
B[['column_order', 'column_group', 'outcome', 'disease', 'label', 'tier1', 'events', 'row_order', 'clock',
   'paired_clock', 'paired', 'hr', 'lo', 'hi', 'log_hr', 'p', 'q_bh_552', 'paired_rank_among_10']].to_csv(
    OUT + 'Fig3a_atlas_tier2_model_b.csv', index=False)
pp = B[B.paired & B.outcome.isin(K46)]
print('a: paired FDR-significant', int((pp.q_bh_552 < 0.05).sum()), 'of 46; cells significant',
      int((B[i46].q_bh_552 < 0.05).sum()), 'of 552; ranked first', int((pp.paired_rank_among_10 == 1).sum()))
print('a: significant diseases per clock', B[i46 & (B.q_bh_552 < 0.05)].groupby('clock').size().to_dict())

# model c paired significance on the same 552-test family (for text)
Cc = T2[(T2.model == 'c') & T2.outcome.isin(K46)].copy()
Cc['q_bh_552'] = multipletests(Cc.p, method='fdr_bh')[1]
Cc['paired'] = Cc.clock == Cc.outcome.map(meta.clock)
print('model c paired FDR-significant', int(((Cc.q_bh_552 < 0.05) & Cc.paired).sum()))

# ------------------------------------------------------------------ b: tier-1 prespecified pairs
T1 = pd.read_csv(R + 'M2_tier1_hr.csv')
pa = pd.concat([T1[(T1.outcome == k) & (T1.clock == o)] for k, o in PAIRS])
pa[cols + ['p_holm']].to_csv(OUT + 'Fig3b_tier1_paired_hr.csv', index=False)
RB = pd.read_csv(R + 'M2_tier1_rankboot.csv')
rb = pd.concat([RB[(RB.model == 'b') & (RB.outcome == k) & (RB.clock == o)] for k, o in PAIRS])
rb[['outcome', 'clock', 'model', 'p_rank1', 'p_top3']].to_csv(OUT + 'Fig3b_tier1_rank_probability.csv', index=False)

# ------------------------------------------------------------------ c: organ correspondence index
DC = pd.read_csv(R4 + 'F3_disease_correspondence.csv')
HR = pd.read_csv(R4 + 'F3_tier2_raw_resid_hr.csv')
CI = pd.read_csv(R4 + 'F3_clock_correspondence_index.csv')
CI.to_csv(OUT + 'Fig3c_clock_correspondence_index.csv', index=False)
q552 = pp.set_index('outcome').q_bh_552
rows = []
for x in DC.itertuples():
    r = HR[(HR.outcome == x.key) & (HR.clock == x.clock)].set_index('version')
    rows.append({'outcome': x.key, 'disease': x.disease, 'system': x.system, 'paired_clock': x.clock,
                 'events': int(x.n_events), 'tier1': x.key in TIER1,
                 'hr_raw': r.hr['raw'], 'lo_raw': r.lo['raw'], 'hi_raw': r.hi['raw'], 'q_bh_552_raw': q552[x.key],
                 'hr_resid': r.hr['resid'], 'lo_resid': r.lo['resid'], 'hi_resid': r.hi['resid'],
                 'attenuation_pct': (1 - np.log(r.hr['resid']) / np.log(r.hr['raw'])) * 100,
                 'excess_log_hr_raw': x.excess_raw, 'excess_lo_raw': x.excess_raw_lo, 'excess_hi_raw': x.excess_raw_hi,
                 'excess_log_hr_resid': x.excess_resid, 'rank_among_10_raw': x.rank_raw,
                 'rank_among_10_resid': x.rank_resid})
E = pd.DataFrame(rows)
assert (E.set_index('outcome').rank_among_10_raw == pp.set_index('outcome').paired_rank_among_10.reindex(E.outcome)).all()
E.drop(columns=['hr_resid', 'lo_resid', 'hi_resid', 'attenuation_pct']).to_csv(
    OUT + 'Fig3c_disease_correspondence.csv', index=False)

# ------------------------------------------------------------------ d: phenome-wide summary by clock (model c)
P = pd.read_csv(R + 'M2_tier3_phenome.csv')
P = P[P.model == 'c'][['phecode', 'description', 'group', 'clock', 'hr', 'lo', 'hi', 'p', 'q_bh', 'p_bonf', 'n_events']]
P.to_csv(OUT + 'Fig3d_tier3_phenome_model_c.csv', index=False)
CHAP = {'Heart': ['Cardiovascular'], 'Artery': ['Cardiovascular'], 'Kidney': ['Genitourinary'],
        'Pancreas': ['Endocrine/Metab'], 'Liver': ['Gastrointestinal'], 'Brain': ['Neurological', 'Mental'],
        'Lung': ['Respiratory'], 'Immune': ['Infections', 'Blood/Immune'], 'Muscle': ['Muscloskeletal'],
        'Intestine': ['Gastrointestinal']}
CHAP_LAB = {'Cardiovascular': 'circulatory', 'Genitourinary': 'genitourinary', 'Endocrine/Metab': 'endocrine/metabolic',
            'Gastrointestinal': 'digestive', 'Neurological': 'neurological', 'Mental': 'mental',
            'Respiratory': 'respiratory', 'Infections': 'infectious', 'Blood/Immune': 'blood/immune',
            'Muscloskeletal': 'musculoskeletal'}
PLAB = {'Cardiac arrhythmia and conduction disorders': 'Arrhythmia/conduction disorders',
        'Dementias and cerebral degeneration': 'Dementias', 'Renal failure': 'Renal failure',
        'Pneumonia': 'Pneumonia', 'Osteoarthritis': 'Osteoarthritis',
        'Extrapyramidal and movement disorders': 'Extrapyramidal disorders',
        'Osteoporosis and low bone density': 'Osteoporosis', 'Diseases of the pancreas': 'Pancreatic disease',
        'Infections': 'Infections', 'Bacteremia, Sepsis, and SIRS': 'Sepsis',
        'Disorders of fluid, electrolyte and acid-base balance': 'Fluid/electrolyte disorders'}
rows = []
for c in ROWS:
    x = P[P.clock == c].copy()
    x['sig'] = x.q_bh < 0.05
    top = x.sort_values('p').iloc[0]
    ch = CHAP.get(c)
    own = x.group.isin(ch) if ch else pd.Series(False, index=x.index)
    r = {'clock': c, 'corresponding_chapters': '/'.join(CHAP_LAB[g] for g in ch) if ch else '',
         'n_phecodes': len(x), 'n_phecodes_corresponding': int(own.sum()) if ch else np.nan,
         'n_sig': int(x.sig.sum()), 'n_sig_corresponding': int((x.sig & own).sum()) if ch else np.nan,
         'top_phecode': top.phecode, 'top_description': top.description,
         'top_label': PLAB.get(top.description, top.description), 'top_chapter': top.group,
         'top_hr': top.hr, 'top_lo': top.lo, 'top_hi': top.hi, 'top_p': top.p, 'top_q': top.q_bh,
         'top_in_corresponding_chapter': bool(top.group in ch) if ch else np.nan}
    if ch:
        a_ = int((x.sig & own).sum()); b_ = int((x.sig & ~own).sum())
        c_ = int((~x.sig & own).sum()); d_ = int((~x.sig & ~own).sum())
        orr, pf = fisher_exact([[a_, b_], [c_, d_]])
        r.update({'fisher_or': orr, 'fisher_p': pf})
    rows.append(r)
D = pd.DataFrame(rows)
m = D.fisher_p.notna()
D.loc[m, 'fisher_q_bh'] = multipletests(D.loc[m, 'fisher_p'], method='fdr_bh')[1]
D.to_csv(OUT + 'Fig3d_phenome_by_clock.csv', index=False)
print(D[['clock', 'n_sig', 'n_sig_corresponding', 'top_label', 'top_hr', 'top_p', 'top_in_corresponding_chapter',
         'fisher_or', 'fisher_p', 'fisher_q_bh']].round(4).to_string())

# ------------------------------------------------------------------ e: kidney-function residualisation
E[['outcome', 'disease', 'system', 'paired_clock', 'events', 'tier1', 'hr_raw', 'lo_raw', 'hi_raw', 'q_bh_552_raw',
   'hr_resid', 'lo_resid', 'hi_resid', 'attenuation_pct']].to_csv(OUT + 'Fig3e_paired_raw_vs_residualised.csv',
                                                                   index=False)
HR.to_csv(OUT + 'Fig3e_all_clocks_raw_resid_hr.csv', index=False)
S = E[E.q_bh_552_raw < 0.05]
print('e: significant', len(S), 'median attenuation by clock',
      S.groupby('paired_clock').attenuation_pct.agg(['median', 'size']).round(1).to_dict('index'))

# ------------------------------------------------------------------ Supplementary Fig. 9b, 9c
SYS = ['Cardiac', 'Vascular', 'Kidney', 'Metabolic', 'Liver', 'Brain', 'Lung', 'Immune', 'Musculoskeletal', 'Gastrointestinal']
SYS_PAIR = {'Cardiac': 'Heart', 'Vascular': 'Artery', 'Kidney': 'Kidney', 'Metabolic': 'Pancreas', 'Liver': 'Liver',
            'Brain': 'Brain', 'Lung': 'Lung', 'Immune': 'Immune', 'Musculoskeletal': 'Muscle', 'Gastrointestinal': 'Intestine'}
rows = []
for s in SYS:
    d = T2[(T2.outcome == 'SYS_' + s) & (T2.model == 'b') & T2.clock.isin(SYS_PAIR.values())].set_index('clock')
    v = d.loc[list(SYS_PAIR.values()), 'beta']
    z = (v - v.mean()) / v.std(ddof=0)
    for o in SYS_PAIR.values():
        rows.append((s, o, SYS_PAIR[s] == o, d.loc[o, 'hr'], d.loc[o, 'lo'], d.loc[o, 'hi'], d.loc[o, 'p'], z[o],
                     int((v > v[SYS_PAIR[s]]).sum()) + 1, int(d.loc[o, 'events'])))
pd.DataFrame(rows, columns=['system', 'clock', 'paired', 'hr', 'lo', 'hi', 'p', 'z_within_system', 'paired_clock_rank',
                            'events']).to_csv(SOUT + 'SFig9b_system_specificity.csv', index=False)
SP = json.load(open(R + 'M2_system_specificity.json'))['b']
pd.Series({'n_top1': SP['n_top1'], 'n_top3': SP['n_top3'], 'contrast_obs': SP['obs'], 'perm_mean': SP['perm_mean'],
           'perm_sd': SP['perm_sd'], 'p_perm': SP['p_perm'], 'n_perm': 10000}).to_csv(
    SOUT + 'SFig9b_permutation_test.csv', header=['value'])
P.to_csv(SOUT + 'SFig9c_tier3_phenome_model_c.csv', index=False)
print('done')
