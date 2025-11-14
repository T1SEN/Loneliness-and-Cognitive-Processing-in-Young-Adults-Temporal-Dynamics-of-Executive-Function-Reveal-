import pandas as pd
coef = pd.read_csv('results/analysis_outputs/loneliness_models_coefficients_py.csv')
print(list(coef.columns))
print(coef.query("term=='z_ucla'")[[c for c in coef.columns if c!='outcome']].head())
