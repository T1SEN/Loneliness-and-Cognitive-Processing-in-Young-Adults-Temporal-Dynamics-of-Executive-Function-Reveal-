import pandas as pd
import numpy as np
from pathlib import Path
import math

coef = pd.read_csv('results/analysis_outputs/loneliness_models_coefficients_py.csv')
fit = pd.read_csv('results/analysis_outputs/loneliness_models_fit_py.csv')

lon = coef.query("term == 'z_ucla'").copy()
param_counts = coef.groupby('outcome').size().rename('k')
lon = lon.merge(param_counts, on='outcome', how='left')
lon = lon.merge(fit[['outcome','nobs']], on='outcome', how='left')

n_target = 150
rows = []
for _, r in lon.iterrows():
    k = int(r['k'])
    df2_now = int(r['nobs']) - k
    df2_new = n_target - k
    t_now = r['estimate'] / r['std_error'] if r['std_error'] != 0 else 0.0
    pr2 = (t_now**2) / (t_now**2 + df2_now) if df2_now>0 else 0.0
    # predicted t at N=150 using constant partial R^2 approximation
    t2_new = df2_new * pr2 / (1 - pr2) if pr2>0 else 0.0
    t_new = math.sqrt(t2_new)
    # normal approx p-value
    def norm_cdf(z):
        return 0.5*(1+math.erf(z/math.sqrt(2)))
    p_new = 2*(1 - norm_cdf(abs(t_new)))
    # thresholds
    tcrit = 1.976  # approx t_{0.975, 143}
    pr2_sig = (tcrit*tcrit) / (df2_new + tcrit*tcrit)
    t_needed = 1.96 + 0.84  # ~80% power, two-sided
    pr2_power = (t_needed*t_needed) / (df2_new + t_needed*t_needed)
    rows.append({
        'outcome': r['outcome'],
        'partialR2_now': pr2,
        'pred_p_at_N150': p_new,
        'pr2_needed_sig_at_150': pr2_sig,
        'pr2_needed_80power_at_150': pr2_power,
    })

out = pd.DataFrame(rows)
order = ['Stroop interference (ms)','PRP bottleneck (short-long RT)','WCST total errors','Latent meta-control factor']
out['order'] = out['outcome'].map({v:i for i,v in enumerate(order)})
out = out.sort_values(['order']).drop(columns=['order'])
print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
