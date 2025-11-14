import pandas as pd
import numpy as np
from statsmodels.stats.power import FTestPower
from pathlib import Path

coef_path = Path('results/analysis_outputs/loneliness_models_coefficients_py.csv')
fit_path = Path('results/analysis_outputs/loneliness_models_fit_py.csv')

coef = pd.read_csv(coef_path)
fit = pd.read_csv(fit_path)

lon = coef.query("term == 'z_ucla'").copy()
param_counts = coef.groupby('outcome').size().rename('k')
lon = lon.merge(param_counts, on='outcome', how='left')
lon = lon.merge(fit[['outcome','nobs']], on='outcome', how='left')

res = []
for _, row in lon.iterrows():
    outcome = row['outcome']
    est = row['estimate']
    se = row['std_error']
    nobs = int(row['nobs'])
    k = int(row['k'])
    df2 = nobs - k
    t = est / se if se != 0 else 0.0
    f2 = (t**2) / df2 if df2 > 0 else np.nan
    ftp = FTestPower()
    try:
        power_now = ftp.power(effect_size=f2, df_num=1, df_denom=df2, alpha=0.05)
    except Exception:
        power_now = np.nan
    n_target = 150
    df2_150 = n_target - k
    try:
        power_150 = ftp.power(effect_size=f2, df_num=1, df_denom=df2_150, alpha=0.05)
    except Exception:
        power_150 = np.nan
    try:
        f2_mde = ftp.solve_power(effect_size=None, df_num=1, df_denom=df2_150, alpha=0.05, power=0.80)
    except Exception:
        f2_mde = np.nan
    pr2 = f2/(1+f2) if f2>0 else 0.0
    pr2_mde = f2_mde/(1+f2_mde) if f2_mde>0 else np.nan
    res.append({
        'outcome': outcome,
        'nobs_now': nobs,
        'k_params': k,
        'df2_now': df2,
        't_lon': t,
        'f2_now': f2,
        'partialR2_now': pr2,
        'power_now': power_now,
        'power_at_150': power_150,
        'f2_MDE_at_150': f2_mde,
        'partialR2_MDE_at_150': pr2_mde,
    })

out = pd.DataFrame(res)
order = ['Stroop interference (ms)','PRP bottleneck (short-long RT)','WCST total errors','Latent meta-control factor']
out['order'] = out['outcome'].map({v:i for i,v in enumerate(order)})
out = out.sort_values(['order']).drop(columns=['order'])
print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
