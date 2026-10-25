"""Text edits accompanying Figure 1 (v2) and the new Supplementary Table 1 (participant selection).

Input : /home/ubuntu/work/v5/ms/ms_v5_markup.txt         (v5 manuscript, paragraph markup)
Output: ../integrated/ms_v5_fig1v2_markup.txt            (v5 + Figure 1 v2 text; v5 file is not modified)
All numbers are read from ../source_data and ../supplementary (produced by fig1v2_analysis.py / prep_figure1.py).
"""
import os, json, re
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SD = os.path.join(HERE, '..', 'source_data')
SUPD = os.path.join(HERE, '..', 'supplementary')
SRC = '/home/ubuntu/work/v5/ms/ms_v5_markup.txt'
OUTD = os.path.join(HERE, '..', 'integrated')
os.makedirs(OUTD, exist_ok=True)
OUT = os.path.join(OUTD, 'ms_v5_fig1v2_markup.txt')
s = open(SRC, encoding='utf-8').read()

# ------------------------------------------------------------------ numbers
A = pd.read_csv(os.path.join(SD, 'Fig1a_cohort_data_layers.csv')).set_index('key').value
IND = pd.read_csv(os.path.join(SD, 'Fig1d_independence.csv'))
ia = IND[IND['sample'].str.startswith('All (')].set_index('metric')
i0 = IND[IND['sample'].str.startswith('Stage 0')].set_index('metric')
PT = pd.read_csv(os.path.join(SD, 'Fig1d_pair_types.csv')).set_index('pair_type')
P = pd.read_csv(os.path.join(SD, 'Fig1d_corr_pearson.csv'), index_col=0)
S = pd.read_csv(os.path.join(SD, 'Fig1c_stage_profile.csv'))
C = pd.read_csv(os.path.join(SUPD, 'Supplementary_Table_1d_included_excluded.csv'))
M = pd.read_csv(os.path.join(SUPD, 'Supplementary_Table_1e_staging_missingness.csv'))
F = json.load(open('/home/ubuntu/work/v5/run/results/v5_fig1v2/F1_facts.json'))
smd_col = [c for c in C.columns if 'no CKM staging' in c and 'SMD' in c][0]
max_smd = C[smd_col].abs().max()
rmin, rmax = P.where(~np.eye(len(P), dtype=bool)).min().min(), P.where(~np.eye(len(P), dtype=bool)).max().max()
mean_r = P.where(~np.eye(len(P), dtype=bool)).mean()
assert mean_r.idxmin() == 'Heart', mean_r.sort_values().head(3)
diff = json.load(open('/home/ubuntu/work/v5/run/results/v5_fig1v2/F1_facts.json'))['r_median_diff_all_minus_stage0']
f = lambda x, d=1: f'{x:.{d}f}'
pc1 = ia.loc['pc1_pct']
print(f"r range {rmin:.3f}-{rmax:.3f}; max |SMD| {max_smd:.3f}; PC1 {pc1.estimate:.1f} ({pc1.lo:.1f}-{pc1.hi:.1f})")

# numbers quoted in Fig. 1f must already be in the v5 Results
# numbers printed in Fig. 1f (v2 evidence chain) that the v5 Results already report
for chk in ['P = 3.9 × 10^{−73}', 'HR 1.00, 90% CI 0.93–1.09', '0.025', '(0.014, 0', '1.00 (0.94–1.05), 1.19 (1.13–1.25) and 1.67 (1.58–1.76)']:
    assert chk in s, chk


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (n, old[:90])
    s = s.replace(old, new)


def rep_line(start, new, count=1):
    """Replace every whole line that starts with `start`."""
    global s
    lines = s.split('\n')
    idx = [i for i, l in enumerate(lines) if l.startswith(start)]
    assert len(idx) == count, (len(idx), start[:80])
    old = lines[idx[0]]
    for i in idx:
        lines[i] = new
    s = '\n'.join(lines)
    return old


# ------------------------------------------------------------------ Results: first paragraph
women = S.women_pct.values
prev = S.prevent_median.values
res1 = (
    f"The analytic cohort included {int(float(A.n_cohort)):,} participants in CKM stages 0–3 with 12 proteomic organ age gaps "
    f"(Fig. 1a; participant selection in Supplementary Table 1a): 8,770 (21.6%) in stage 0, 17,304 (42.5%) in stage 1, "
    f"13,827 (34.0%) in stage 2 and 773 (1.9%) in stage 3 (Fig. 1c, Table 1 and Supplementary Table 1c). Mean age was 56.4 years, "
    f"and 55.3% were women; age and PREVENT risk rose and the proportion of women fell with increasing stage (median 10-year total "
    f"CVD risk {f(prev[0])}%, {f(prev[1])}%, {f(prev[2])}% and {f(prev[3])}%; women {f(women[0])}% to {f(women[3])}%; Fig. 1c). "
    f"In the characteristics that were measured, the 9,253 participants excluded for missing staging data resembled the analytic cohort (all absolute standardised mean "
    f"differences ≤{max_smd:.2f}; Supplementary Table 1d), and {F['nostage_any_core_component_missing_pct']:.1f}% of them lacked at "
    f"least one core staging measurement (Supplementary Table 1e), indicating that exclusion reflected missing measurements rather "
    f"than participant characteristics. Exposures and covariates are defined in Supplementary Table 2, and the sample of each "
    f"analysis is listed in Supplementary Table 1b. One s.d. of age gap corresponded to 2.0 (heart) to 4.1 (artery) years "
    f"(Fig. 1b and Supplementary Table 3). The ten organ clocks and two whole-body clocks (organismal, based on non-organ-enriched "
    f"proteins, and conventional, based on all proteins) were weakly to moderately correlated (Pearson r {rmin:.2f}–{rmax:.2f}; "
    f"median {PT.loc['organ-organ', 'r_median']:.2f} between organ clocks), with the strongest correlations involving the whole-body "
    f"clocks and the lowest the heart clock (Fig. 1d). The first principal component explained {pc1.estimate:.1f}% "
    f"(95% CI {pc1.lo:.1f}–{pc1.hi:.1f}%) of the joint variance, {int(ia.loc['k80', 'estimate'])} of 12 components were needed to "
    f"explain 80%, and the effective number of independent clocks was {ia.loc['meff_liji', 'estimate']:.0f} (Li–Ji) to "
    f"{ia.loc['meff_nyholt', 'estimate']:.1f} (Nyholt). This structure was almost identical in stage 0 participants (median r "
    f"{i0.loc['r_median', 'estimate']:.3f} versus {ia.loc['r_median', 'estimate']:.3f}; difference {diff[0]:.3f}, 95% CI "
    f"{diff[1]:.3f} to {diff[2]:.3f}; Fig. 1d). The organ clocks therefore capture largely distinct axes of ageing rather than a "
    f"single general ageing factor, which motivated the organ-specific analyses that follow (Fig. 1f).")
res1 = res1.replace('-0.', '−0.')
old_res1 = rep_line('The cross-sectional cohort included 40,674 participants', res1)

# ------------------------------------------------------------------ Figure 1 legend (in-text copy and legend section)
old_leg = [l for l in s.split('\n') if l.startswith('**a**, Design. Plasma proteomic')][0]
title_new = '**Fig. 1 | Study overview and analytical roadmap.**'
leg = (
    f"**a**, Follow-up via hospital and death records to 30 September 2021. Selection, Supplementary Table 1a; "
    f"analytic samples, Supplementary Table 1b. "
    f"**b**, Ten organ and two whole-body clocks built as in Wang et al.^{{10}} (organ-enriched proteins, ≥4-fold GTEx "
    f"expression; LightGBM with SHAP-Boruta selection; fivefold out-of-fold prediction). An organ was aged when its age gap exceeded the 90th centile of stage 0 participants "
    f"of the same sex and age band. "
    f"**c**, Participants by baseline-verified stage, with mean age and median PREVENT 10-year CVD risk and eGFRcr-cys "
    f"(ml min⁻¹ per 1.73 m²). Stage 4 (n = 3,068) was excluded; baseline-only re-staging reclassified 1,000 participants "
    f"(Supplementary Fig. 1). "
    f"**d**, Pearson correlations ordered by hierarchical clustering (r ≥ 0.30 printed), variance explained by principal "
    f"components and effective number of independent clocks. "
    f"**e**, Incident events per outcome by organ system (n = {int(float(A.n_cohort)):,}); tier 3 includes phecodes with "
    f"≥80 events; negative control, transport accidents. Right, MRI and arterial-stiffness phenotypes for cross-sectional "
    f"corroboration. "
    f"**f**, Each column poses the question answered by one later figure; the arrow label carries its answer to the next. "
    f"Thumbnails reproduce source data: Fig. 2, adjusted stage differences in target-organ age gaps (equal-onset Wald test); "
    f"Fig. 3, Cox HR per s.d. for tier-1 diseases, paired clock coloured, with its rank of ten; Fig. 4, 10-year CKM "
    f"composite risk by stage and burden (dashed line, stage 2–3 with none aged; equivalence by two one-sided tests); "
    f"Fig. 5, cross-fitted ΔC over models G and D; Fig. 6, observed versus PREVENT-predicted 10-year CVD risk. "
    f"Risks, Aalen–Johansen estimates with death as a competing event; error bars, 95% CI. All estimates are "
    f"observational; decision points (Fig. 6) are proposed for prospective evaluation. "
    f"ASCVD, atherosclerotic cardiovascular disease; ASI, arterial stiffness index; CKD, chronic kidney disease; CKM, "
    f"cardiovascular–kidney–metabolic; CLD, chronic liver disease; CVD, cardiovascular disease; DEM, dementia; eGFRcr-cys, "
    f"estimated glomerular filtration rate from creatinine and cystatin C; FDR, false discovery rate; HF, heart failure; "
    f"HR, hazard ratio; Meff, effective number of tests; O/E, observed-to-expected ratio; PC, principal component; "
    f"PREVENT, Predicting Risk of cardiovascular disease EVENTs; T2D, type 2 diabetes; ΔC, gain in C-index.")
import re as _re
_wc = lambda x: len(_re.findall(r"[A-Za-z0-9][^\s]*", x))
NW = _wc(title_new.replace('**', '')) + _wc(leg.replace('**', ''))
print('Fig. 1 legend words (title + body):', NW)
assert NW <= 350, NW
rep('**Fig. 1 | Study design, analytic samples and outcome system.**', title_new, count=2)
rep_line('**a**, Design. Plasma proteomic', leg, count=2)

# ------------------------------------------------------------------ Methods: study population
rep("Plasma proteins were measured in baseline samples of 54,219 participants in the UK Biobank Pharma Proteomics Project^{17}. "
    "Of 52,995 participants with proteomic organ age gaps,",
    "Plasma proteins (Olink Explore 3072; 2,923 proteins) were measured in baseline samples in the UK Biobank Pharma Proteomics "
    "Project^{17}; all 52,995 participants with proteomic data had the 12 organ age gaps. Of these,")
rep("leaving 40,674 participants in stages 0–3 for the cross-sectional analyses (Fig. 1b).",
    "leaving 40,674 participants in stages 0–3 (Fig. 1a and Supplementary Table 1a). Characteristics of included and excluded "
    "participants were compared with standardised mean differences, and the availability of each staging measurement was "
    "tabulated (Supplementary Table 1d,e).")
rep("and imaging analyses included participants who attended the first imaging visit (see below).",
    "and imaging analyses included participants who attended the first imaging visit (see below); the sample of every analysis "
    "is listed in Supplementary Table 1b.")
# v5 cross-reference error: the STROBE checklist is Supplementary Table 23
rep("Reporting follows the STROBE guideline^{35} (Supplementary Table 21)", "Reporting follows the STROBE guideline^{35} (Supplementary Table 23)")
# CKM stage definitions are now Supplementary Table 1c
rep("as implemented by Zhu et al.^{5} (Supplementary Table 1).", "as implemented by Zhu et al.^{5} (Supplementary Table 1c).")
# v5 cross-reference error: disease-to-clock assignments are listed in Supplementary Table 4a
rep("to the organismal clock (Supplementary Table 2)", "to the organismal clock (Supplementary Table 4a)")

# ------------------------------------------------------------------ Methods: structure of the age gaps (end of proteomics section)
anchor = "Age gaps were standardised to a mean of 0 and an s.d. of 1; positive values indicate an organ that is older than expected for chronological age."
rep(anchor, anchor + " The structure of the 12 age gaps was described with Pearson correlations (Spearman correlations in "
    "sensitivity analysis), principal component analysis of the correlation matrix and the effective number of independent "
    "tests by the methods of Nyholt (Am. J. Hum. Genet. 2004) and of Li and Ji (Heredity 2005) [reference numbers to be added], in all participants and in stage 0, with percentile 95% CIs from "
    "200 bootstrap samples.")

open(OUT, 'w', encoding='utf-8').write(s)
open(os.path.join(OUTD, '_old_results_para1.txt'), 'w').write(old_res1 + '\n\n' + old_leg + '\n')
print('written', OUT)
print(res1)
