from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.confirmatory.data import load_study_data
from src.confirmatory.folds import load_outer_folds, inner_splits_for_outer_train
from src.confirmatory.constants import PRIMARY_TASKS
from src.confirmatory.features import engineered_features
from src.confirmatory.orientation import make_augmented_training_set, perturb_by_subject
from src.confirmatory.metrics import composite_participant_rmse

ANGLES=np.array([0.0,2.5,5.0,7.5,10.0,12.5,15.0],float)
ALPHAS=np.array([0.1,1.0,10.0,100.0],float)
TASKS=('knee_two','ankle_two')
MC=50; COPIES=5; SEED=20260907; BOOT=5000; TOL=4.5
OUT=ROOT/'strengthening'/'rotation_matrix_v2'
OUT.mkdir(parents=True,exist_ok=True)

class WeightedRidge:
    def __init__(self, alpha):
        self.alpha=float(alpha); self.scaler=StandardScaler(); self.model=Ridge(alpha=self.alpha)
    def fit(self,X,Y,w=None):
        if w is None: w=np.ones(len(X),float)
        self.scaler.fit(X,sample_weight=w)
        Z=self.scaler.transform(X)
        self.model.fit(Z,Y,sample_weight=w)
        return self
    def predict(self,X): return np.asarray(self.model.predict(self.scaler.transform(X)),dtype=np.float32)

def augmented_for_indices(d, idx, task, cap, seed):
    X,Y,G,D,S=make_augmented_training_set(
        d.X[idx], d.T[idx][:,task['targets']], d.subjects[idx], d.durations[idx], d.speeds[idx],
        task['sensors'], float(cap), COPIES, seed=seed)
    if cap>0:
        # Original + five augmented copies; total weight per source cycle = 1.
        w=np.full(len(X),1.0/(COPIES+1),dtype=float)
    else:
        w=np.ones(len(X),dtype=float)
    return X,Y,G,D,S,w

def select_alpha_for_cap(d, outer, ofold, cap):
    outer_train=np.flatnonzero(outer!=ofold)
    g_outer=d.subjects[outer_train]
    scores={a:[] for a in ALPHAS}
    rows=[]
    for ifold,(itr_local,iva_local) in enumerate(inner_splits_for_outer_train(g_outer,3),start=1):
        itr=outer_train[itr_local]; iva=outer_train[iva_local]
        # Build each augmented task representation once, then reuse it for all alphas.
        prepared=[]
        for ti,task_name in enumerate(TASKS):
            task=PRIMARY_TASKS[task_name]
            seed=SEED+ofold*1_000_000+int(round(cap*10))*10_000+ifold*100+ti
            Xtr,Ytr,Gtr,Dtr,Str,w=augmented_for_indices(d,itr,task,cap,seed)
            Ftr=engineered_features(Xtr,task['sensors'],Dtr,Str)
            Fva=engineered_features(d.X[iva],task['sensors'],d.durations[iva],d.speeds[iva])
            prepared.append((task,Ftr,Ytr,w,Fva))
        for a in ALPHAS:
            ts=[]
            for task,Ftr,Ytr,w,Fva in prepared:
                m=WeightedRidge(a).fit(Ftr,Ytr,w)
                pred=m.predict(Fva)
                ts.append(composite_participant_rmse(d.T[iva][:,task['targets']],pred,d.subjects[iva]))
            sc=float(np.mean(ts)); scores[a].append(sc)
            rows.append({'outer_fold':ofold,'train_augmentation_max_deg':cap,'alpha':a,'inner_fold':ifold,
                         'inner_mean_rmse_across_primary_tasks':sc})
    means={a:float(np.mean(v)) for a,v in scores.items()}
    best=min(means,key=means.get)
    for r in rows:
        r['mean_inner_score_for_alpha']=means[r['alpha']]
        r['selected_alpha']=best
        r['selection_metric']='clean inner participant-composite RMSE across knee_two and ankle_two; weight-normalized augmentation'
    return float(best),rows

def bootstrap(vals,seed):
    vals=np.asarray(vals,float); n=len(vals); rng=np.random.default_rng(seed)
    # chunked to keep memory small
    b=np.empty(BOOT,float)
    chunk=500
    for s in range(0,BOOT,chunk):
        m=min(chunk,BOOT-s)
        ix=rng.integers(0,n,size=(m,n))
        b[s:s+m]=vals[ix].mean(axis=1)
    return float(vals.mean()),float(np.quantile(b,.025)),float(np.quantile(b,.975))

def main():
    t0=time.time(); d=load_study_data(ROOT/'data/processed/strict_parameters.npz')
    outer=load_outer_folds(ROOT/'protocol/outer_folds.csv',d.subjects)
    alpha_rows=[]; alpha_map={}
    for ofold in sorted(np.unique(outer)):
        for cap in ANGLES:
            best,rows=select_alpha_for_cap(d,outer,int(ofold),float(cap))
            alpha_map[(int(ofold),float(cap))]=best; alpha_rows.extend(rows)
            print('alpha',ofold,cap,best,flush=True)
    pd.DataFrame(alpha_rows).to_csv(OUT/'selected_alphas_by_fold.csv',index=False)
    pd.DataFrame([{'outer_fold':k[0],'train_augmentation_max_deg':k[1],'selected_alpha':v} for k,v in alpha_map.items()]).to_csv(OUT/'alpha_map.csv',index=False)

    # accum keyed task,target,cap,test_angle,subject -> [sum_sq_error,count]
    accum={}; fitlog=[]
    for ofold in sorted(np.unique(outer)):
        ofold=int(ofold); tr=np.flatnonzero(outer!=ofold); te=np.flatnonzero(outer==ofold)
        for ti,task_name in enumerate(TASKS):
            task=PRIMARY_TASKS[task_name]; models={}
            for cap in ANGLES:
                seed=SEED+ofold*2_000_000+int(round(cap*10))*10_000+ti
                Xtr,Ytr,Gtr,Dtr,Str,w=augmented_for_indices(d,tr,task,float(cap),seed)
                Ftr=engineered_features(Xtr,task['sensors'],Dtr,Str)
                a=alpha_map[(ofold,float(cap))]
                models[float(cap)]=WeightedRidge(a).fit(Ftr,Ytr,w)
                fitlog.append({'outer_fold':ofold,'task':task_name,'train_augmentation_max_deg':float(cap),'alpha':a,
                               'n_training_rows':len(Xtr),'total_training_weight':float(w.sum()),'feature_dim':Ftr.shape[1]})
            y=d.T[te][:,task['targets']]; g=d.subjects[te]
            unique=np.unique(g)
            for ang in ANGLES:
                reps=1 if ang==0 else MC
                for rep in range(reps):
                    if ang==0: Xte=d.X[te]
                    else:
                        pseed=SEED+ofold*3_000_000+ti*100_000+int(round(ang*10))*100+rep
                        Xte,_=perturb_by_subject(d.X[te],g,task['sensors'],float(ang),pseed,exact_angle=True)
                    Fte=engineered_features(Xte,task['sensors'],d.durations[te],d.speeds[te])
                    for cap,m in models.items():
                        p=m.predict(Fte); e=(p-y)**2
                        for sid in unique:
                            ii=(g==sid)
                            sums=e[ii].sum(axis=0); cnt=int(ii.sum())
                            for j,target in enumerate(task['target_names']):
                                k=(task_name,target,cap,float(ang),int(sid)); s,n=accum.get(k,(0.0,0)); accum[k]=(s+float(sums[j]),n+cnt)
                print('eval',ofold,task_name,ang,flush=True)
    pd.DataFrame(fitlog).to_csv(OUT/'fit_log.csv',index=False)
    rows=[]
    for (task,target,cap,ang,sid),(s,n) in accum.items():
        rows.append({'task':task,'target':target,'train_augmentation_max_deg':cap,'test_rotation_deg':ang,'subject':sid,
                     'rmse_deg':float(np.sqrt(s/n)),'pooled_error_count':n})
    part=pd.DataFrame(rows).sort_values(['task','target','train_augmentation_max_deg','test_rotation_deg','subject'])
    part.to_csv(OUT/'participant_matrix.csv',index=False)
    sums=[]
    for i,(keys,q) in enumerate(part.groupby(['task','target','train_augmentation_max_deg','test_rotation_deg'],sort=True)):
        mean,lo,hi=bootstrap(q.rmse_deg.to_numpy(),SEED+5_000_000+i)
        task,target,cap,ang=keys
        sums.append({'task':task,'target':target,'train_augmentation_max_deg':cap,'test_rotation_deg':ang,
                     'mean_participant_rmse_deg':mean,'ci_low':lo,'ci_high':hi,'n_subjects':len(q)})
    sm=pd.DataFrame(sums);sm.to_csv(OUT/'matrix_summary.csv',index=False)

    # paired clean penalties vs train cap 0
    penalties=[]; clean=part[part.test_rotation_deg==0]
    base=clean[clean.train_augmentation_max_deg==0][['task','target','subject','rmse_deg']].rename(columns={'rmse_deg':'base'})
    for cap in ANGLES[1:]:
        qq=clean[clean.train_augmentation_max_deg==cap][['task','target','subject','rmse_deg']].rename(columns={'rmse_deg':'aug'})
        mm=base.merge(qq,on=['task','target','subject'])
        for j,((task,target),gq) in enumerate(mm.groupby(['task','target'])):
            diff=gq.aug.to_numpy()-gq.base.to_numpy(); mean,lo,hi=bootstrap(diff,SEED+6_000_000+int(cap*100)+j)
            penalties.append({'task':task,'target':target,'train_augmentation_max_deg':cap,'mean_clean_penalty_deg':mean,'ci_low':lo,'ci_high':hi})
    pd.DataFrame(penalties).to_csv(OUT/'clean_penalties.csv',index=False)

    dep=[]
    for delta in ANGLES:
        cand=[]
        for cap in ANGLES:
            q=sm[(sm.train_augmentation_max_deg==cap)&(sm.test_rotation_deg<=delta+1e-9)]
            expected=4*int(np.sum(ANGLES<=delta+1e-9))
            if len(q)!=expected: continue
            max_hi=float(q.ci_high.max()); max_mean=float(q.mean_participant_rmse_deg.max())
            if max_hi<=TOL:
                cleanq=sm[(sm.train_augmentation_max_deg==cap)&(sm.test_rotation_deg==0)]
                cand.append((float(cap),max_hi,max_mean,float(cleanq.mean_participant_rmse_deg.max()),float(cleanq.ci_high.max())))
        if cand:
            cand.sort(key=lambda x:x[0]); cap,max_hi,max_mean,cm,ch=cand[0]
            dep.append({'field_q95_rotation_deg':delta,'decision':'DEPLOY','selected_train_augmentation_max_deg':cap,
                        'worst_ci_high_deg_through_q95':max_hi,'worst_mean_rmse_deg_through_q95':max_mean,
                        'worst_clean_mean_rmse_deg':cm,'worst_clean_ci_high_deg':ch,'n_qualifying_rows':len(cand)})
        else:
            dep.append({'field_q95_rotation_deg':delta,'decision':'NO_DEPLOY_AUGMENTATION','selected_train_augmentation_max_deg':np.nan,
                        'worst_ci_high_deg_through_q95':np.nan,'worst_mean_rmse_deg_through_q95':np.nan,
                        'worst_clean_mean_rmse_deg':np.nan,'worst_clean_ci_high_deg':np.nan,'n_qualifying_rows':0})
    depdf=pd.DataFrame(dep);depdf.to_csv(OUT/'deployment_rule.csv',index=False)
    worst=sm.groupby(['train_augmentation_max_deg','test_rotation_deg'],as_index=False).agg(worst_mean_rmse_deg=('mean_participant_rmse_deg','max'),worst_ci_high_deg=('ci_high','max'))
    worst.to_csv(OUT/'worst_endpoint_matrix.csv',index=False)
    print('\nALPHAS\n',pd.read_csv(OUT/'alpha_map.csv').pivot(index='outer_fold',columns='train_augmentation_max_deg',values='selected_alpha').to_string())
    print('\nDEPLOY\n',depdf.to_string(index=False))
    print('\nWORST CI\n',worst.pivot(index='train_augmentation_max_deg',columns='test_rotation_deg',values='worst_ci_high_deg').round(3).to_string())
    print('elapsed',time.time()-t0)
if __name__=='__main__': main()
