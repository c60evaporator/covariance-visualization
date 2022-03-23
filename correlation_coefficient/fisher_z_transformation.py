# %% Importing data
import pandas as pd
from seaborn_analyzer import regplot
df = pd.read_csv('./../nba_height_weight.csv')
regplot.linear_plot(x='weight', y='height', data=df)
# %%
