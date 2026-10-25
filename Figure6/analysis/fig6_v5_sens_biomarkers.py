"""Fig 6 sensitivity (v5): does the aged-organ burden remain a risk enhancer beyond cardiac biomarkers
(NT-proBNP, troponin I; Olink) and the full guideline kidney assessment (eGFRcr-cys spline, UACR category)?
Outputs results/v5_fig6/F6_sens_biomarkers.csv (used in Supplementary Table and Results text)."""
import sys
sys.path.insert(0, './analysis'); sys.path.insert(0, './code')
from fig6_v5_analysis import *   # module-level frame `s`, hr(), COV_B, SPEC helpers

KID = [f'egfrcc_s{j}' for j in range(3)] + ['uacr_a2', 'uacr_a3']
ADJ = {
    'PREVENT + stage (main)': ['log_prevent'] + COV_B,
    '+ NT-proBNP + troponin I': ['log_prevent'] + COV_B + ['nppb', 'tnni3_p'],
    '+ NT-proBNP + troponin I + eGFRcr-cys + UACR': ['log_prevent'] + COV_B + ['nppb', 'tnni3_p'] + KID,
    'Model G covariates': ['log_prevent'] + COV_B + ['nppb', 'tnni3_p'] + KID + ['htn_med', 'statin'],
}
rows = []
for key, lab in [('CVD', 'Total CVD (PREVENT definition)'), ('CKMC', 'CKM composite'), ('HF', 'Heart failure'),
                 ('MSYS', '>=2 newly affected organ systems'), ('OVL', 'CKM overlap')]:
    for an, cv in ADJ.items():
        r = hr(s, key, ['b1', 'b2'], cv)
        for i, t_ in enumerate(['b1', 'b2']):
            rows.append(dict(outcome=key, label=lab, adjustment=an, term=t_, hr=r.hr[i], lo=r.lo[i], hi=r.hi[i], p=r.p[i],
                             n=int(r.n[i]), events=int(r.events[i])))
# hidden heart risk: heart-aged vs not among PREVENT-HF <5%, additionally adjusted for NT-proBNP and troponin I
d = s[s.prev_hf < 5].copy(); d['aged_h'] = d.aged_Heart.astype(float)
for an, cv in [('model a', COV_A), ('+ NT-proBNP + troponin I', COV_A + ['nppb', 'tnni3_p'])]:
    r = hr(d, 'HF', ['aged_h'], cv)
    rows.append(dict(outcome='HF', label='Heart failure, PREVENT-HF <5%', adjustment=an, term='aged_Heart', hr=r.hr[0], lo=r.lo[0],
                     hi=r.hi[0], p=r.p[0], n=int(r.n[0]), events=int(r.events[0])))
out = pd.DataFrame(rows); save(out, 'F6_sens_biomarkers')
pd.set_option('display.width', 220)
print(out[out.term != 'b1'][['outcome', 'adjustment', 'term', 'hr', 'lo', 'hi', 'p', 'events']].round(3).to_string())
