from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "source_data"

def close(a,b,tol=5e-3):
    assert abs(float(a)-float(b)) <= tol, (a,b)

checks=[]

def ok(label, detail):
    checks.append((label, detail))

# Clean virtual performance.
clean=pd.read_csv(SD/'clean_performance_with_ci.csv')
prim=clean[clean.task.isin(['knee_two','ankle_two'])]
close(prim.mean_participant_rmse_deg.min(),0.8924511413)
close(prim.mean_participant_rmse_deg.max(),1.3974580481)
ok('Virtual clean RMSE','0.89-1.40 deg')

# Full virtual matrix and 10-deg claims.
vm=pd.read_csv(SD/'virtual_rotation_matrix_summary.csv')
u10=vm[(vm.train_augmentation_max_deg==0)&(vm.test_rotation_deg==10)]
a10=vm[(vm.train_augmentation_max_deg==10)&(vm.test_rotation_deg==10)]
close(u10.mean_participant_rmse_deg.min(),8.078447,1e-3)
close(u10.mean_participant_rmse_deg.max(),8.240646,1e-3)
close(a10.mean_participant_rmse_deg.min(),3.664335,1e-3)
close(a10.mean_participant_rmse_deg.max(),4.273407,1e-3)
close(a10.ci_high.max(),4.455609,1e-3)
assert (a10.ci_high < 4.5).all()
ok('Virtual 10-deg unaugmented','8.08-8.24 deg RMSE')
ok('Virtual 10-deg matched augmentation','3.66-4.27 deg RMSE; worst upper CI 4.456 deg')

rule=pd.read_csv(SD/'virtual_deployment_rule.csv')
r10=rule[rule.field_q95_rotation_deg==10].iloc[0]
assert r10.decision=='DEPLOY' and float(r10.selected_train_augmentation_max_deg)==10
r125=rule[rule.field_q95_rotation_deg==12.5].iloc[0]
r15=rule[rule.field_q95_rotation_deg==15].iloc[0]
assert r125.decision.startswith('NO_DEPLOY') and r15.decision.startswith('NO_DEPLOY')
ok('Virtual operating rule','10-deg cap qualifies through 10 deg; no cap qualifies through 12.5 or 15 deg')

rescue=pd.read_csv(SD/'virtual_augmentation_rescue_v2.csv')
r=rescue[(rescue.train_augmentation_max_deg==10)&(rescue.test_rotation_deg==10)].set_index('target')
for target,val in {
    'ankle_dorsiflex_deg':4.4614917,
    'ankle_plantarflex_deg':4.3775469,
    'knee_stance_flex_deg':3.8698801,
    'knee_swing_flex_deg':3.9105679,
}.items():
    close(r.loc[target,'mean_rescue_deg'],val,2e-3)
    assert r.loc[target,'ci_low']>0
ok('Virtual matched-augmentation rescue','4.46, 4.38, 3.87, 3.91 deg; all paired CIs > 0')

pen=pd.read_csv(SD/'virtual_clean_penalties_v2.csv')
p=pen[pen.train_augmentation_max_deg==10]
assert p.mean_clean_penalty_deg.min()>1.45 and p.mean_clean_penalty_deg.max()<1.60
ok('Virtual clean penalty for 10-deg augmentation','1.46-1.59 deg')

vb=pd.read_csv(SD/'virtual_orientation_baselines.csv')
t10=vb[(vb.method=='template_registration')&(vb.test_rotation_deg==10)]
assert t10.mean_participant_rmse_deg.min()>3.18 and t10.mean_participant_rmse_deg.max()<3.93
ok('Virtual template registration at 10 deg','3.19-3.92 deg RMSE')

vp=pd.read_csv(SD/'virtual_orientation_paired_vs_aug10.csv')
tp=vp[vp.method.eq('template_registration')]
assert (tp[tp.test_rotation_deg==0].ci95_low_deg>0).all()
assert (tp[tp.test_rotation_deg==10].ci95_high_deg<0).all()
ok('Virtual template vs aug10 trade-off','template worse at 0 deg, better at 10 deg for all four endpoints')

# Relative-frame mechanism.
decomp=pd.read_csv(SD/'figure4_perturbation_decomposition.csv')
clean10=decomp[(decomp.training_regime=='clean')&(decomp.angle_deg==10)]
common=clean10[clean10.perturbation_family=='common']
indep=clean10[clean10.perturbation_family=='independent']
assert len(common)==4 and len(indep)==4
assert common.change_from_0deg.max() < 1.0
assert indep.change_from_0deg.min() > 6.80 and indep.change_from_0deg.max() < 7.16
ok('Virtual perturbation decomposition','common-mode 10-deg change <1 deg; independent two-sensor change 6.81-7.15 deg')

geom=pd.read_csv(SD/'relative_rotation_geometry.csv')
g10=geom[geom.nominal_per_sensor_rotation_deg==10].iloc[0]
close(g10['mean'],13.317343,2e-3)
close(g10['p97_5'],19.744783,2e-3)
assert 19.99 < g10['max'] <= 20.0
ok('Relative-frame geometry','10-deg per-sensor pair angle mean 13.317 deg, 97.5th percentile 19.745 deg, max ~20 deg')

# External physical-IMU replication.
ed=pd.read_csv(SD/'external_unaugmented_degradation.csv')
e10=ed[ed.test_rotation_deg==10].set_index('target')
for target,val in {
    'ankle_dorsiflex_deg':0.3823575,
    'ankle_plantarflex_deg':0.4820858,
    'knee_stance_flex_deg':0.6634664,
    'knee_swing_flex_deg':0.6707565,
}.items():
    close(e10.loc[target,'mean_paired_degradation_deg'],val,2e-3)
    assert e10.loc[target,'ci95_low_deg']>0
ok('External unaugmented 10-deg degradation','0.382, 0.482, 0.663, 0.671 deg; all paired CIs > 0')

ea=pd.read_csv(SD/'external_augmentation_vs_unaugmented.csv')
ea10=ea[(ea.test_rotation_deg==10)&(ea.train_augmentation_max_deg==10)].set_index('target')
expected={'ankle_dorsiflex_deg':0.78093,'ankle_plantarflex_deg':0.33748,'knee_stance_flex_deg':1.14653,'knee_swing_flex_deg':1.63626}
for target,val in expected.items():
    close(ea10.loc[target,'mean_improvement_unaug_minus_aug_deg'],val,3e-3)
assert sum(ea10.ci95_low_deg>0)==3
ok('External matched 10-deg augmentation','improves 3/4 endpoints with CI excluding zero')

ep=pd.read_csv(SD/'external_orientation_paired_comparisons.csv')
et=ep[(ep.method=='template_registration')&(ep.reference=='unaugmented')&(ep.test_rotation_deg==10)].set_index('target')
for target,val in {'ankle_dorsiflex_deg':0.91754,'ankle_plantarflex_deg':0.72483,'knee_stance_flex_deg':1.14294,'knee_swing_flex_deg':2.17207}.items():
    close(et.loc[target,'mean_improvement_reference_minus_baseline_deg'],val,3e-3)
    assert et.loc[target,'ci95_low_deg']>0
ok('External template registration vs unaugmented at 10 deg','improves all 4 endpoints; all paired CIs > 0')

ev=ep[(ep.method=='template_registration')&(ep.reference=='aug10')&(ep.test_rotation_deg==10)].set_index('target')
close(ev.loc['knee_swing_flex_deg','mean_improvement_reference_minus_baseline_deg'],0.535805,3e-3)
assert ev.loc['knee_swing_flex_deg','ci95_low_deg']>0
for target in ['ankle_dorsiflex_deg','ankle_plantarflex_deg','knee_stance_flex_deg']:
    assert ev.loc[target,'ci95_low_deg'] <= 0 <= ev.loc[target,'ci95_high_deg']
ok('External template vs aug10','only knee swing has a clear template advantage: 0.536 deg, CI 0.031-1.288')

i=pd.read_csv(SD/'external_orientation_invariance.csv')
mx=i[i.method.eq('template_registration')].max_abs_registered_signal_difference_vs_clean.max()
assert mx <= 2e-6
ok('External template invariance',f'max signal discrepancy {mx:.2e}')

print('CLAIM AUDIT: PASS')
for label, detail in checks:
    print(f'- {label}: {detail}')
