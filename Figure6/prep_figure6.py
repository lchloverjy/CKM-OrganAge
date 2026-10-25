"""Export all inputs of Figure 6 (v2) and Supplementary Fig. 18 to figure6/source_data (CSV + one xlsx)."""
import os, json, shutil
import pandas as pd
RUN = '/home/ubuntu/work/v5/run/results/'
V5, V2 = RUN + 'v5_fig6/', RUN + 'v5_fig6v2/'
HERE = os.path.dirname(os.path.abspath(__file__))
SD = os.path.join(HERE, '..', 'source_data'); os.makedirs(SD, exist_ok=True)
MAP = {
    'Fig6a_calibration_deciles': V2 + 'F6a_calibration_deciles.csv',
    'Fig6a_oe_by_burden': V2 + 'F6a_oe_by_burden.csv',
    'Fig6a_hr_overall': V5 + 'F6a_overall.csv',
    'Fig6a_sensitivity_biomarkers': V5 + 'F6_sens_biomarkers.csv',
    'Fig6b_threshold_upgrade': V2 + 'F6b_threshold_upgrade.csv',
    'Fig6b_threshold_upgrade_comparators': V2 + 'F6b_threshold_upgrade_comparators.csv',
    'Fig6c_enhancer_paf': V2 + 'F6c_enhancer_paf.csv',
    'Fig6c_dc_beyond_enhancers': V2 + 'F6c_dc_beyond_enhancers.csv',
    'Fig6d_hidden_organ_risk': V5 + 'F6c_hidden_organ_risk.csv',
    'Fig6d_hidden_curves': V2 + 'F6d_hidden_curves.csv',
    'Fig6d_hidden_sensitivity': V2 + 'F6d_hidden_sensitivity.csv',
    'Fig6e_risk_grid': V5 + 'F6d_coordinated_care.csv',
    'Fig6e_dc_corrected': V2 + 'F6e_dc_corrected.csv',
    'Fig6f_frontier_curves': V2 + 'F6f_frontier_curves.csv',
    'Fig6f_named_strategies': V2 + 'F6f_named_strategies.csv',
    'Fig6g_imaging_associations': V5 + 'F6f_imaging.csv',
    'Fig6g_damage_index': V5 + 'F6f_damage_index.csv',
    'Fig6g_domain_rank': V5 + 'F6f_domain_rank.csv',
    'SFig18_subgroups': V2 + 'SFig18_subgroups.csv',
}
sheets = {}
for k, p in MAP.items():
    df = pd.read_csv(p); df.to_csv(os.path.join(SD, k + '.csv'), index=False); sheets[k] = df
facts = {}
for nm, p in [('cohort', V5 + 'fig6_facts.json'), ('imaging_n', V5 + 'F6f_counts.json'), ('testing', V2 + 'F6f_counts.json')]:
    for kk, vv in json.load(open(p)).items():
        if not isinstance(vv, dict):
            facts[f'{nm}.{kk}'] = vv
F = pd.Series(facts, name='value'); F.index.name = 'key'; F.to_csv(os.path.join(SD, 'Fig6_facts.csv'))
sheets['Fig6_facts'] = F.reset_index()
with pd.ExcelWriter(os.path.join(SD, 'Fig6_source_data.xlsx')) as xw:
    pd.DataFrame({'sheet': list(sheets), 'content': [
        'PREVENT deciles x burden: mean predicted and AJ observed 10-y total CVD risk (bootstrap 95% CI)',
        'Observed/expected ratio by burden (bootstrap 95% CI)',
        'Cox HR of burden for total CVD and CKM composite beyond log PREVENT + stage + model a covariates',
        'Burden HR after additional adjustment for NT-proBNP, troponin I, eGFRcr-cys, UACR',
        'Observed risk below each guideline threshold by burden; observational net benefit of upgrading',
        'Same, for upgrading by single guideline enhancers',
        'Mutually adjusted HR, prevalence and population attributable fraction of guideline enhancers',
        'Cross-fitted C-index gain beyond PREVENT + all available enhancers; LR test',
        '10-y risk among people with a normal routine organ assessment, aged vs not aged paired clock',
        'Aalen-Johansen cumulative incidence 0-12 y with bootstrap 95% band',
        'Hidden-risk HRs with additional biomarker/genetic adjustment',
        '10-y risk of multisystem disease and CKM overlap by stage group x burden',
        'Cross-fitted C-index gain for coordinated-care outcomes (corrected linear predictor)',
        'Testing efficiency curves in CKM stages 1-2',
        'Named testing strategies (bootstrap 95% CI)',
        'Clock-imaging phenotype associations (beta per SD, adverse direction positive; BH q)',
        'Subclinical damage index by burden',
        'Rank of the paired clock within each imaging domain',
        'Subgroup HRs of >=2 vs 0 aged organs and interaction P (Supplementary Fig. 18)',
        'Cohort and sample-size facts']}).to_excel(xw, sheet_name='README', index=False)
    for k, df in sheets.items():
        df.to_excel(xw, sheet_name=k[:31], index=False)
print('exported', len(sheets), 'tables to', SD)
