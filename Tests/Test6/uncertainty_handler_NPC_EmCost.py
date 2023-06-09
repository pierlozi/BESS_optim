#%%
import os
import sys
sys.path.append(r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model")

from Core import best_polyfit_degree

RES_data_file_path = r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model\InputData\RESData_option-2.csv"
load_data_file_path = r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model\InputData\load_data.xlsx"

import numpy as np
import pandas as pd


import altair as alt

import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter, DayLocator

plt.rcParams.update({'font.size': 12})

df = pd.read_excel('test_NSGAII_NPC_EmCost_2.xlsx', index_col = 0, header= 0)
df_p = pd.read_excel('test_NSGAII_NPC_EmCost_plus10_2.xlsx', index_col = 0, header= 0)
df_m = pd.read_excel('test_NSGAII_NPC_EmCost_minus10_2.xlsx', index_col = 0, header= 0)

df['gamma'] = df.Er/df.Pr
df_p['gamma'] = df_p.Er/df_p.Pr
df_m['gamma'] = df_m.Er/df_m.Pr

#%% Design Space
chart_f0 = alt.Chart(df, title = "Design Space").mark_circle(color = 'blue').encode(
        alt.X('Er').scale(zero=False),
        alt.Y('Pr').scale(zero=False),
        size = 'gamma'
        )
chart_fp = alt.Chart(df_p, title = "Design Space").mark_circle(color = 'red').encode(
        alt.X('Er').scale(zero=False),
        alt.Y('Pr').scale(zero=False),
        size = 'gamma'
        )
chart_fm = alt.Chart(df_m, title = "Design Space").mark_circle(color = 'green').encode(
        alt.X('Er').scale(zero=False),
        alt.Y('Pr').scale(zero=False),
        size = 'gamma'
        )
alt.layer(
    chart_f0,
    chart_fp,
    chart_fm
)
#%% Objective Space
chart_f0 = alt.Chart(df, title = "Objective Space").mark_circle(color = 'blue').encode(
        alt.X('NPC').scale(zero=False),
        alt.Y('EmCost').scale(zero=False),
        size = 'gamma'
        )
chart_fp = alt.Chart(df_p, title = "Objective Space").mark_circle(color = 'red').encode(
        alt.X('NPC').scale(zero=False),
        alt.Y('EmCost').scale(zero=False),
        size = 'gamma'
        )
chart_fm = alt.Chart(df_m, title = "Design Space").mark_circle(color = 'green').encode(
        alt.X('NPC').scale(zero=False),
        alt.Y('EmCost').scale(zero=False),
        size = 'gamma'
        )
alt.layer(
    chart_f0,
    chart_fp,
    chart_fm
)
# %%
