import pandas as pd
import numpy as np
from pathlib import Path

coef = pd.read_csv('results/analysis_outputs/loneliness_models_coefficients_py.csv')
fit = pd.read_csv('results/analysis_outputs/loneliness_models_fit_py.csv')
lon = coef.query("term == 'z_ucla'").copy()
param_counts = coef.groupby('outcome').size().rename('k')
lon = lon.merge(param_counts, on='outcome', how='left')
lon = lon.merge(fit[['outcome','nobs']], on='outcome', how='left')

rows = []
for _, r in lon.iterrows():
    k = int(r['k'])
    df2_now = int(r['nobs']) - k
    t_now = r['estimate'] / r['std_error'] if r['std_error'] != 0 else 0.0
    pr2 = (t_now**2) / (t_now**2 + df2_now) if df2_now>0 else 0.0
    t_needed = 1.96 + 0.84
    df2_req = (t_needed**2) * (1 - pr2) / pr2 if pr2>0 else np.inf
    n_req = df2_req + k
    rows.append({'outcome': r['outcome'], 'partialR2_now': pr2, 'N_needed_80power_alpha05': n_req})

out = pd.DataFrame(rows)
print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
