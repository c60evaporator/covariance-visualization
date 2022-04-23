# %% Importing data
import pandas as pd
from seaborn_analyzer import regplot
df = pd.read_csv('./../nba_height_weight.csv')
regplot.linear_plot(x='weight', y='height', data=df)
# %%
import pandas as pd
import numpy as np
from scipy import stats
P_threshold = 0.05 # 信頼区間
df = pd.read_csv('./../nba_height_weight.csv')
n = len(df)
# zの計算
r = np.corrcoef(df['weight'], df['height'])[0,1]
z = np.log((1 + r) / (1 - r)) / 2
# ηの信頼区間下限
eta_min = z - stats.norm.ppf(q=1-P_threshold/2, loc=0, scale=1) / np.sqrt(n - 3)
# ηの信頼区間上限
eta_max = z - stats.norm.ppf(q=P_threshold/2, loc=0, scale=1) / np.sqrt(n - 3)
# 母相関係数ρの信頼区間下限
rho_min = (np.exp(2 * eta_min) - 1) / (np.exp(2 * eta_min) + 1)
# 母相関係数ρの信頼区間上限
rho_max = (np.exp(2 * eta_max) - 1) / (np.exp(2 * eta_max) + 1)
print(f'母相関係数の{P_threshold*100}%信頼区間 = {rho_min}〜{rho_max}')
# %%
