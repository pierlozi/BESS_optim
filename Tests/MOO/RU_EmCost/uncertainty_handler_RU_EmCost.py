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

df = pd.read_excel('test_NSGAII_RU_EmCost_2.xlsx', index_col = 0, header= 0)
df_p = pd.read_excel('test_NSGAII_RU_EmCost_plus10.xlsx', index_col = 0, header= 0)
df_m = pd.read_excel('test_NSGAII_RU_EmCost_minus10.xlsx', index_col = 0, header= 0)

df['gamma'] = df.Er/df.Pr
df_p['gamma'] = df_p.Er/df_p.Pr
df_m['gamma'] = df_m.Er/df_m.Pr

#%% Design Space
chart_f0 = alt.Chart(df[df.gamma<100], title = "Design Space").mark_circle(color = 'blue').encode(
        alt.X('Er').scale(zero=False),
        alt.Y('Pr').scale(zero=False),
        size = 'gamma'
        )
chart_fp = alt.Chart(df_p[df_p.gamma<100], title = "Design Space").mark_circle(color = 'red').encode(
        alt.X('Er').scale(zero=False),
        alt.Y('Pr').scale(zero=False),
        size = 'gamma'
        )
chart_fm = alt.Chart(df_m[df_m.gamma<100], title = "Design Space").mark_circle(color = 'green').encode(
        alt.X('Er').scale(zero=False),
        alt.Y('Pr').scale(zero=False),
        size = 'gamma'
        )
chart_des = alt.layer(
    chart_f0,
    chart_fp,
    chart_fm
)

chart_des
#%% Objective Space
chart_f0 = alt.Chart(df[df.gamma<100], title = "Objective Space").mark_circle(color = 'blue').encode(
        alt.X('RU').scale(zero=False),
        alt.Y('EmCost').scale(zero=False),
        size = 'gamma'
        )
chart_fp = alt.Chart(df_p[df_p.gamma<100], title = "Objective Space").mark_circle(color = 'red').encode(
        alt.X('RU').scale(zero=False),
        alt.Y('EmCost').scale(zero=False),
        size = 'gamma'
        )
chart_fm = alt.Chart(df_m[df_m.gamma<100], title = "Design Space").mark_circle(color = 'green').encode(
        alt.X('RU').scale(zero=False),
        alt.Y('EmCost').scale(zero=False),
        size = 'gamma'
        )
chart_obj = alt.layer(
    chart_f0,
    chart_fp,
    chart_fm
)

chart_obj
# %%
chart_des.save('uncertainty_RU_EmCost_des.png')
chart_obj.save('uncertainty_RU_EmCost_obj.png')
# %%
