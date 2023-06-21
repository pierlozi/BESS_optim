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

df = pd.read_excel('test_NSGAII_LCOS_EmCost_2.xlsx', index_col = 0, header= 0)
df_p = pd.read_excel('test_NSGAII_LCOS_EmCost_plus10.xlsx', index_col = 0, header= 0)
df_m = pd.read_excel('test_NSGAII_LCOS_EmCost_minus10.xlsx', index_col = 0, header= 0)

df['gamma'] = df.Er/df.Pr
df_p['gamma'] = df_p.Er/df_p.Pr
df_m['gamma'] = df_m.Er/df_m.Pr

#%%
fontsize = 20
width = 700
height = 3/4*width

#%% Design Space
chart_f0 = alt.Chart(df, title = "Design Space").mark_circle(color = 'blue').encode(
        alt.X('Er').scale(zero=False).title('Capacity rating [MWh]'),
        alt.Y('Pr').scale(zero=False).title('Power rating [MW]'),
        size = 'gamma'
        )
chart_fp = alt.Chart(df_p, title = "Design Space").mark_circle(color = 'red').encode(
        alt.X('Er').scale(zero=False).title('Capacity rating [MWh]'),
        alt.Y('Pr').scale(zero=False).title('Power rating [MW]'),
        size = 'gamma'
        )
chart_fm = alt.Chart(df_m, title = "Design Space").mark_circle(color = 'green').encode(
        alt.X('Er').scale(zero=False).title('Capacity rating [MWh]'),
        alt.Y('Pr').scale(zero=False).title('Power rating [MW]'),
        size = 'gamma'
        )
chart_des = alt.layer(
    chart_f0,
    chart_fp,
    chart_fm
        ).configure_axis(
            labelFontSize = fontsize,
            titleFontSize = fontsize
        ).configure_legend(
            labelFontSize = fontsize,
            titleFontSize = fontsize
        ).configure_title(
            fontSize = fontsize
        ).properties(
            width = width,
            height = height
        )

chart_des

#%% Objective Space
chart_f0 = alt.Chart(df, title = "Objective Space").mark_circle(color = 'blue').encode(
        alt.X('LCOS').scale(zero=False).title('LCOS [€/MWh]'),
        alt.Y('EmCost').scale(zero=False).title('EC [million €]'),
        size = 'gamma'
        )
chart_fp = alt.Chart(df_p, title = "Objective Space").mark_circle(color = 'red').encode(
        alt.X('LCOS').scale(zero=False).title('LCOS [€/MWh]'),
        alt.Y('EmCost').scale(zero=False).title('EC [million €]'),
        size = 'gamma'
        )
chart_fm = alt.Chart(df_m, title = "Objective Space").mark_circle(color = 'green').encode(
        alt.X('LCOS').scale(zero=False).title('LCOS [€/MWh]'),
        alt.Y('EmCost').scale(zero=False).title('EC [million €]'),
        size = 'gamma'
        )
chart_obj = alt.layer(
    chart_f0,
    chart_fp,
    chart_fm
        ).configure_axis(
            labelFontSize = fontsize,
            titleFontSize = fontsize
        ).configure_legend(
            labelFontSize = fontsize,
            titleFontSize = fontsize
        ).configure_title(
            fontSize = fontsize
        ).properties(
            width = width,
            height = height
        )

chart_obj
# %%
chart_des.save('uncertainty_LCOS_EmCost_des.png')
chart_obj.save('uncertainty_LCOS_EmCost_obj.png')
