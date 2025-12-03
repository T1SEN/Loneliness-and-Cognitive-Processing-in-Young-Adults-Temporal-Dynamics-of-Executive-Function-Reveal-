#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import arviz as az
import pymc as pm
from scipy import stats

from analysis.utils.data_loader_utils import load_master_dataset
from analysis.utils.trial_data_loader import load_wcst_trials

BASE = Path(__file__).resolve().parent.parent
RES = BASE / 'results'
OUT = RES / 'analysis_outputs'
OUT.mkdir(parents=True, exist_ok=True)

def tost_means(x, y, low, high, alpha=0.05):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    nx, ny = len(x), len(y)
    if nx < 5 or ny < 5: return {'nx':nx, 'ny':ny, 'p_lower':np.nan, 'p_upper':np.nan, 'equivalent':False}
    diff = x.mean() - y.mean()
    se = np.sqrt(x.var(ddof=1)/nx + y.var(ddof=1)/ny)
    df = nx+ny-2
    t_low = (diff - low)/se; t_up = (diff - high)/se
    p_low = 1 - stats.t.cdf(t_low, df)
    p_up = stats.t.cdf(t_up, df)
    return {'nx':nx,'ny':ny,'diff':float(diff),'p_lower':float(p_low),'p_upper':float(p_up),'equivalent':bool(p_low<alpha and p_up<alpha)}

def build_df():
    import sys; sys.path.append('analysis')
    import loneliness_exec_models as lem
    df = lem.build_analysis_dataframe()
    df = lem.add_meta_control(df)
    return df

def run_tost(df):
    data = df.copy()
    q1 = data['ucla_total'].quantile(0.25); q3 = data['ucla_total'].quantile(0.75)
    group = pd.Series(index=data.index, dtype='object')
    group[data['ucla_total']>=q3] = 'high'
    group[data['ucla_total']<=q1] = 'low'
    data = data.assign(group=group).dropna(subset=['group'])
    rows=[]
    for col, lo, hi in [('stroop_effect',-10,10), ('prp_bottleneck',-10,10), ('wcst_total_errors',-0.5,0.5)]:
        x = data.loc[data.group=='high', col]
        y = data.loc[data.group=='low', col]
        res = tost_means(x,y,lo,hi)
        rows.append({'outcome':col,'low_bound':lo,'high_bound':hi, **res})
    out = pd.DataFrame(rows)
    out.to_csv(OUT/'tost_summary.csv', index=False)
    return out

def fit_model(y,X):
    with pm.Model() as m:
        beta = pm.Normal('beta',0,1,shape=X.shape[1])
        sigma = pm.HalfNormal('sigma',1)
        mu = (X*beta).sum(axis=1)
        pm.Normal('obs', mu=mu, sigma=sigma, observed=y)
        idata = pm.sample(1000, tune=1000, chains=2, target_accept=0.9, progressbar=False, return_inferencedata=True, idata_kwargs={"log_likelihood": True})
    return idata

def run_model_compare(df):
    rows=[]
    for col in ['stroop_effect','prp_bottleneck','wcst_total_errors']:
        data = df[[col,'z_dass_dep','z_dass_anx','z_dass_stress','age','gender','z_ucla']].dropna()
        if len(data)<50:
            rows.append({'outcome':col,'n':len(data),'waic_diff':np.nan,'loo_diff':np.nan}); continue
        g = (data['gender'].astype(str)=='male').astype(int).values.reshape(-1,1)
        age = pd.to_numeric(data['age'], errors='coerce').fillna(data['age'].median()).values.reshape(-1,1)
        cov = data[['z_dass_dep','z_dass_anx','z_dass_stress']].values
        X0 = np.hstack([cov, age, g]); X1 = np.hstack([cov, age, g, data[['z_ucla']].values])
        y = data[col].values
        id0=fit_model(y,X0); id1=fit_model(y,X1)
        waic0,waic1 = az.waic(id0), az.waic(id1)
        loo0,loo1 = az.loo(id0), az.loo(id1)
        rows.append({'outcome':col,'n':len(data),'waic0':float(waic0.elpd_waic),'waic1':float(waic1.elpd_waic),'elpd_waic_diff':float(waic1.elpd_waic-waic0.elpd_waic),
                     'loo0':float(loo0.elpd_loo),'loo1':float(loo1.elpd_loo),'elpd_loo_diff':float(loo1.elpd_loo-loo0.elpd_loo)})
    out=pd.DataFrame(rows); out.to_csv(OUT/'model_compare_waic_loo.csv', index=False); return out

def wcst_window():
    df, _ = load_wcst_trials(use_cache=True)
    if 'reactionTimeMs' not in df.columns and 'rt_ms' in df.columns:
        df = df.rename(columns={'rt_ms': 'reactionTimeMs'})
    if 'reactionTimeMs' not in df.columns or 'ruleAtThatTime' not in df.columns:
        return pd.DataFrame()
    df=df.dropna(subset=['participant_id']).copy()
    rt=pd.to_numeric(df['reactionTimeMs'], errors='coerce')
    df=df[(rt>=200)&(rt<=5000)]; df=df.sort_values(['participant_id', df.get('trialIndex','trialIndexInBlock')])
    rows=[]
    for pid,grp in df.groupby('participant_id'):
        grp=grp.copy(); grp['prev_rule']=grp['ruleAtThatTime'].shift(1)
        switches=grp.index[(grp['ruleAtThatTime']!=grp['prev_rule']) & grp['prev_rule'].notna()]
        for idx in switches:
            i=grp.index.get_loc(idx)
            if i<5: continue
            pre=grp.iloc[max(0,i-3):i]['reactionTimeMs'].mean()
            post=grp.iloc[i+1:i+4]['reactionTimeMs'].mean()
            if np.isfinite(pre) and np.isfinite(post): rows.append({'participant_id':pid,'pre3_mean':float(pre),'post3_mean':float(post),'diff':float(post-pre)})
    out=pd.DataFrame(rows); 
    if not out.empty: out.to_csv(OUT/'wcst_switch_cost_window.csv', index=False)
    return out

def invariance_time(df):
    tr, _ = load_wcst_trials(use_cache=True)
    if 'timestamp' not in tr.columns: return '[Invariance] timestamp missing\n'
    tr=tr.dropna(subset=['participant_id','timestamp']).copy(); tr['ts']=pd.to_datetime(tr['timestamp'], errors='coerce'); tr=tr.dropna(subset=['ts'])
    tr=tr.dropna(subset=['participant_id','timestamp']).copy(); tr['ts']=pd.to_datetime(tr['timestamp'], errors='coerce'); tr=tr.dropna(subset=['ts'])
    tr['hour']=tr['ts'].dt.hour
    ef=df[['participant_id','stroop_effect','prp_bottleneck','wcst_total_errors','z_ucla','z_dass_dep','z_dass_anx','z_dass_stress','age','gender']].dropna()
    med=tr.groupby('participant_id')['hour'].median().reset_index().rename(columns={'hour':'median_hour'})
    ef2=ef.merge(med,on='participant_id',how='left').dropna(subset=['median_hour'])
    if len(ef2)<40: return '[Invariance] insufficient N with timestamp\n'
    lines=['# Invariance (time-of-day)']
    for col in ['stroop_effect','prp_bottleneck','wcst_total_errors']:
        X=ef2[['z_ucla','z_dass_dep','z_dass_anx','z_dass_stress','age']].astype(float).values
        y=ef2[col].astype(float).values; X=np.hstack([np.ones((len(X),1)),X])
        beta=np.linalg.lstsq(X,y,rcond=None)[0]; resid=y-X.dot(beta)
        r=stats.pearsonr(resid, ef2['median_hour'].astype(float).values)
        lines.append(f'- {col}: resid vs hour r={r.statistic:.03f}, p={r.pvalue:.3f}')
    Path(OUT/'invariance_summary.txt').write_text('\n'.join(lines), encoding='utf-8'); return '\n'.join(lines)

if __name__=='__main__':
    df=build_df()
    tost=run_tost(df); print('TOST summary\n', tost.to_string(index=False))
    mc=run_model_compare(df); print('\nModel compare\n', mc.to_string(index=False))
    win=wcst_window(); 
    if not win.empty: print('\nWCST window sample\n', win.head().to_string(index=False))
    print('\n', invariance_time(df))


