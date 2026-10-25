"""Rebuild the Supplementary Information with the new Supplementary Table 1 (study population; parts a-e).

The v5 build inputs are copied to ../integrated/_si_build (the v5 files are not modified); then
 * Supplementary Table 1 = a participant selection, b analytic samples, c CKM stage definitions (former ST1),
   d included versus excluded participants, e availability of staging measurements;
 * Supplementary Note 1 gains a paragraph on participant selection;
 * STROBE items 6, 10, 13 and 14 (Supplementary Table 23) point to Fig. 1a and Supplementary Table 1;
 * Supplementary Data 1 receives sheets ST1a-ST1e in place of ST1;
and the v5 script build_si.py is run unchanged on the copy.
"""
import os, json, shutil, subprocess
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SUPD = os.path.join(HERE, '..', 'supplementary')
INT = os.path.abspath(os.path.join(HERE, '..', 'integrated'))
RUN = '/home/ubuntu/work/v5/run/'
V4S = RUN + 'v4/Supplementary/'
B = os.path.join(INT, '_si_build')
if os.path.exists(B):
    shutil.rmtree(B)
os.makedirs(B + '/v4/Supplementary', exist_ok=True)
shutil.copytree(V4S + 'Supplementary_Tables', B + '/v4/Supplementary/Supplementary_Tables')
os.symlink(V4S + 'Supplementary_Figures', B + '/v4/Supplementary/Supplementary_Figures')
shutil.copytree(RUN + 'v4/manuscript', B + '/v4/manuscript')
TAB = B + '/v4/Supplementary/Supplementary_Tables/'
F = json.load(open(RUN + 'results/v5_fig1v2/F1_facts.json'))


def n_(x):
    if pd.isna(x) or x == '':
        return ''
    try:
        return f'{int(float(x)):,}'
    except ValueError:
        return str(x)


# ---------------------------------------------------------------- a selection
a = pd.read_csv(os.path.join(SUPD, 'Supplementary_Table_1a_selection.csv'))
a_d = pd.DataFrame({
    'Step': a.step.astype(str),
    'Criterion': a.description.str.replace('2006-2010', '2006–2010').str.replace('0-3', '0–3'),
    'Remaining, n': a.n_remaining.map(n_),
    'Excluded, n': a.n_excluded.map(n_),
    'Reason for exclusion or comment': a.reason.fillna('').str.replace('1000 participants', '1,000 participants'),
})
a_d.loc[a_d.Step == '1', 'Excluded, n'] = '–'
a_d['Reason for exclusion or comment'] = a_d['Reason for exclusion or comment'].replace(
    {'No baseline proteomic data; the protein and organ age gap files match one to one (52,995 participants)':
     'No baseline proteomic data; every participant with proteomic data had the 12 organ age gaps',
     '1,000 participants (2.5%) reclassified': '1,000 participants (2.5%) changed stage (Supplementary Table 6)',
     'Missing data required for CKM staging': 'Missing data required for CKM staging (Supplementary Table 1d,e)'})
a_d['Criterion'] = a_d['Criterion'].str.replace(' (no exclusion)', '', regex=False)

# ---------------------------------------------------------------- b analytic samples
b = pd.read_csv(os.path.join(SUPD, 'Supplementary_Table_1b_analytic_samples.csv'))
b_d = pd.DataFrame({
    'Figure': b.figure.str.replace('-', '–'),
    'Analysis': b.analysis.str.replace('>= 80', '≥80'),
    'Sample': b['sample'].str.replace('1-2', '1–2'),
    'n': b.n.map(lambda x: x.replace('-', '–') if isinstance(x, str) and '-' in x else n_(x)),
    'Events or comment': b.events_or_note.map(lambda x: x.replace('-', '–') if isinstance(x, str) and '-' in x else n_(x)),
})

# ---------------------------------------------------------------- c stage definitions (former ST1)
c_d = pd.read_csv(TAB + 'display/ST1.csv', dtype=str, keep_default_na=False)

# ---------------------------------------------------------------- d included vs excluded
d = pd.read_csv(os.path.join(SUPD, 'Supplementary_Table_1d_included_excluded.csv'))
col = lambda key, kind: [x for x in d.columns if key in x and kind in x][0]
inc_v, ns_v, s4_v = col('Included', 'value'), col('no CKM staging', 'value'), col('stage 4', 'value')
ns_s, s4_s = col('no CKM staging', 'SMD'), col('stage 4', 'SMD')
ns_m = col('no CKM staging', 'missing')
fmt_smd = lambda x: '' if pd.isna(x) else ('0.00' if abs(x) < 0.005 else f'{x:.2f}'.replace('-', '−'))
d_d = pd.DataFrame({
    'Characteristic': d.characteristic.str.replace('>=', '≥').str.replace('kg/m2', 'kg m⁻²').str.replace('umol/L', 'µmol l⁻¹')
                       .str.replace('mmol/L', 'mmol l⁻¹').str.replace('mg/L', 'mg l⁻¹').str.replace('mmol/mol', 'mmol mol⁻¹'),
    'Analytic cohort (n = 40,674)': d[inc_v].astype(str).str.replace('-', '–'),
    'No staging data (n = 9,253)': d[ns_v].astype(str).str.replace('-', '–'),
    'SMD vs analytic cohort': d[ns_s].map(fmt_smd),
    'Missing in no-staging group, %': d[ns_m].map(lambda x: f'{x:.1f}'),
    'CKM stage 4 (n = 3,068)': d[s4_v].astype(str).str.replace('-', '–'),
    'SMD vs analytic cohort ': d[s4_s].map(fmt_smd),
})
for cname in ['Analytic cohort (n = 40,674)', 'No staging data (n = 9,253)', 'CKM stage 4 (n = 3,068)']:
    d_d[cname] = d_d[cname].str.replace('–0.', '−0.', regex=False).str.replace('(–', '(−', regex=False)

# ---------------------------------------------------------------- e availability of staging measurements
e = pd.read_csv(os.path.join(SUPD, 'Supplementary_Table_1e_staging_missingness.csv'))
e_d = e.copy()
e_d.columns = ['Staging measurement', 'Missing in analytic cohort, %', 'Missing in no-staging group, %', 'Missing in CKM stage 4, %']
for cname in e_d.columns[1:]:
    e_d[cname] = e[e.columns[list(e_d.columns).index(cname)]].map(lambda x: f'{x:.1f}')

for lab, df in zip('abcde', [a_d, b_d, c_d, d_d, e_d]):
    df.to_csv(TAB + f'display/ST1{lab}.csv', index=False)
with pd.ExcelWriter(os.path.join(SUPD, 'Supplementary_Table_1.xlsx'), engine='openpyxl') as xw:
    for lab, df in zip('abcde', [a_d, b_d, c_d, d_d, e_d]):
        df.to_excel(xw, sheet_name=f'ST1{lab}', index=False)
c_d.to_csv(os.path.join(SUPD, 'Supplementary_Table_1c_stage_definitions.csv'), index=False)

TB = json.load(open(TAB + 'tables.json'))
old1 = TB[0]
assert old1['id'] == '1'
anyc = F['nostage_any_core_component_missing_pct']
TB[0] = {
    'id': '1',
    'title': 'Study population: participant selection, analytic samples, CKM stage definitions and comparison of included '
             'and excluded participants',
    'note': ('a, Participant selection (summarised in Fig. 1a). All 52,995 participants with baseline proteomic data (Olink Explore '
             '3072, 2,923 proteins; UK Biobank Pharma Proteomics Project) had the 12 organ age gaps, derived as in Wang et al. '
             '(ref. 10 of the main text); the protein and age gap files match one to one. b, Sample and '
             'number of events for each analysis; participants with prevalent disease were excluded outcome by outcome, and '
             'imaging analyses used participants with the phenotype and complete model covariates (number measured in '
             'brackets). c, ' + old1['note'] + ' d, Baseline characteristics of the analytic cohort and of participants '
             'excluded for missing staging data or for stage 4. Data are mean (s.d.), median (interquartile range) or %; values '
             'refer to participants with the measurement. SMD, standardised mean difference (excluded minus analytic cohort; '
             'pooled s.d.); absolute values below 0.10 are conventionally regarded as negligible and above 0.20 as small. e, '
             f'Percentage of participants without each staging measurement; {anyc:.1f}% of the 9,253 participants without '
             'staging lacked at least one of BMI, systolic blood pressure, HbA1c, triglycerides, HDL cholesterol, total '
             'cholesterol, creatinine and cystatin C.'),
    'parts': [
        {'label': 'a', 'file': 'ST1a.csv', 'landscape': True, 'rows': len(a_d), 'cols': a_d.shape[1]},
        {'label': 'b', 'file': 'ST1b.csv', 'landscape': True, 'rows': len(b_d), 'cols': b_d.shape[1]},
        {'label': 'c', 'file': 'ST1c.csv', 'landscape': True, 'rows': len(c_d), 'cols': c_d.shape[1]},
        {'label': 'd', 'file': 'ST1d.csv', 'landscape': True, 'rows': len(d_d), 'cols': d_d.shape[1]},
        {'label': 'e', 'file': 'ST1e.csv', 'landscape': True, 'rows': len(e_d), 'cols': e_d.shape[1]},
    ]}
json.dump(TB, open(TAB + 'tables.json', 'w'), indent=1, ensure_ascii=False)

# ---------------------------------------------------------------- STROBE locations (ST23)
st = pd.read_csv(TAB + 'display/ST23.csv', dtype=str, keep_default_na=False)
loc = 'Location in manuscript'
upd = {'6': 'Methods; Fig. 1a; Supplementary Table 1a', '10': 'Fig. 1a; Supplementary Table 1a,b',
       '13': 'Fig. 1a; Results; Supplementary Table 1a,b', '14': 'Table 1; Results; Supplementary Table 1d,e'}
for item, v in upd.items():
    m = st.Item == item
    assert m.sum() == 1, item
    st.loc[m, loc] = v
st.to_csv(TAB + 'display/ST23.csv', index=False)

# ---------------------------------------------------------------- Supplementary Note 1
NP = B + '/v4/manuscript/supp_notes.json'
N = json.load(open(NP))
assert N[0][0].startswith('Analytic cohort')
N[0][0] = 'Analytic cohort, participant selection, follow-up and baseline-verified staging'
sel = (
    'Of the 502,411 UK Biobank participants, 52,995 had baseline plasma proteomic data (Olink Explore 3072, 2,923 proteins; '
    'UK Biobank Pharma Proteomics Project), and all of them had the 12 proteomic organ age gaps. A further 9,253 lacked at least one input required for CKM staging '
    'and 3,068 had clinical cardiovascular disease (stage 4), leaving 40,674 participants in stages 0–3, all of whom contributed '
    'to both the cross-sectional and the prospective analyses (Fig. 1a and Supplementary Table 1a); the sample of each analysis '
    'is listed in Supplementary Table 1b. In the characteristics that were measured, participants excluded for missing staging '
    'data closely resembled the analytic cohort in age, sex, ethnicity, deprivation, education, smoking, adiposity, blood '
    'pressure, glycaemia, lipids, kidney function, inflammation and organ age gaps (all absolute standardised mean differences '
    f'≤0.14; Supplementary Table 1d). Among them, {anyc:.1f}% lacked at least one core staging measurement, most often HDL '
    'cholesterol (71.2%), followed by systolic blood pressure and the serum biochemistry panel (26.6–27.0% each) '
    '(Supplementary Table 1e), which indicates that exclusion '
    'reflected missing measurements rather than selection on health status. As expected, participants excluded for stage 4 '
    'were older and more often men and had higher HbA1c, creatinine, cystatin C and organ age gaps and lower LDL cholesterol, '
    'probably reflecting lipid-lowering treatment, than the analytic cohort.')
N[0][1].insert(0, sel)
json.dump(N, open(NP, 'w'), indent=1, ensure_ascii=False)

# ---------------------------------------------------------------- build SI with the unchanged v5 script
subprocess.run(['python3', RUN + 'code/build_si.py'], cwd=B, check=True)
OUT_SI = os.path.join(INT, 'Supplementary_Information_v5_fig1v2.docx')
shutil.copy(B + '/v4/Supplementary/Supplementary_Information.docx', OUT_SI)

# ---------------------------------------------------------------- Supplementary Data 1
import openpyxl
src = V4S + 'Supplementary_Data_1.xlsx'
dst = os.path.join(INT, 'Supplementary_Data_1_v5_fig1v2.xlsx')
full = {'a': a.rename(columns={'n_remaining': 'remaining_n', 'n_excluded': 'excluded_n'}), 'b': b,
        'c': pd.read_excel(src, sheet_name='ST1'), 'd': d, 'e': e}
sheets = pd.read_excel(src, sheet_name=None)
idx = sheets.pop('Index')
with pd.ExcelWriter(dst, engine='openpyxl') as xw:
    rows = []
    for _, r in idx.iterrows():
        if r.Sheet == 'ST1':
            for lab in 'abcde':
                rows.append([f'Supplementary Table 1{lab}', TB[0]['title'] + f' ({lab})', f'ST1{lab}'])
        else:
            rows.append(list(r.values))
    pd.DataFrame(rows, columns=idx.columns).to_excel(xw, sheet_name='Index', index=False)
    for name, df in sheets.items():
        if name == 'ST1':
            for lab in 'abcde':
                full[lab].to_excel(xw, sheet_name=f'ST1{lab}', index=False)
        else:
            df.to_excel(xw, sheet_name=name, index=False)
print('SI ->', OUT_SI)
print('Data ->', dst)
for lab, df in zip('abcde', [a_d, b_d, c_d, d_d, e_d]):
    print(lab, df.shape)
