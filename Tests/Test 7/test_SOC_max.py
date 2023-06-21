# %% 
import os
import sys
sys.path.append(r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model")
from Core import dispatcher_pareto
from Core import microgrid_design
from Core import rain_deg_funct, LCOS_funct
from Core import best_polyfit_degree

RES_data_file_path = r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model\InputData\RESData_option-2.csv"
load_data_file_path = r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model\InputData\load_data.xlsx"

import numpy as np
import pandas as pd
import time


import altair as alt

import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter, DayLocator

plt.rcParams.update({'font.size': 12})

import importlib
importlib.reload(dispatcher_pareto)
importlib.reload(microgrid_design)
importlib.reload(LCOS_funct)
importlib.reload(rain_deg_funct)

#%%
''' Reading the data from csv files'''

P_ren_read = pd.read_csv(RES_data_file_path, header=0, nrows = 8760) #W
P_load = pd.read_excel(load_data_file_path, sheet_name='Yearly Load', header=0)

design = microgrid_design.MG(Pr_BES=17.7, \
                Er_BES=173, \
                P_load=P_load, \
                P_ren=P_ren_read
                )

P_lim = round(max(abs(P_ren_read['Power']*design.RES_fac/1e6-P_load['Load [MW]'])))
E_lim = P_lim*10

#%%


data = pd.DataFrame()
data_time = pd.DataFrame()

P_ren_read = pd.read_csv(RES_data_file_path, header=0, nrows = 8760) #W
P_load = pd.read_excel(load_data_file_path, sheet_name='Yearly Load', header=0)

design = microgrid_design.MG(Pr_BES=17.7, \
            Er_BES=173, \
            P_load=P_load, \
            P_ren=P_ren_read,
            SOC_min = 40,
            SOC_max= 60
            )

data, data_time = dispatcher_pareto.MyFun(design, False)

design.DG_CAPEX = data['DG cost [million euros]']
design.DG_OPEX = data['Fuel Cost [million euros]']

design.cyclelife, _ = rain_deg_funct.MyFun(SOC_profile = data_time['SOC'].values)

LCOS, _ , _ , _ = LCOS_funct.MyFun(design, \
                        E_dch = sum(data_time['P_dch']),\
                        res_val_bin = True
                        )

emissions_cost = data['Emissions Cost [million euros]'].values[0]

# %%
day_start_display = 0
days_display = 7

fig, ax = plt.subplots()
 
ax.plot(data_time['P_dg'][24*day_start_display:24*(day_start_display+days_display)], color=None, label='DG')
ax.plot(data_time['P_curt'][24*day_start_display:24*(day_start_display+days_display)],color=None, label='Curt')
ax.plot(data_time['P_prod'][24*day_start_display:24*(day_start_display+days_display)], color=None, label='Prod')
ax.plot(data_time['P_load'][24*day_start_display:24*(day_start_display+days_display)], color=None, label='Load')
ax.plot(data_time['P_BES'][24*day_start_display:24*(day_start_display+days_display)], color=None, label='BES')

ax.set_ylabel("Power [MW]")
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticks(), ha='center')
ax.xaxis.set_major_locator(MonthLocator())
ax.xaxis.set_major_formatter(DateFormatter('%d/%m'))

ax_SOC = ax.twinx()
ax_SOC.set_ylabel("SOC [-]")
ax_SOC.plot(data_time['SOC'][24*day_start_display:24*(day_start_display+days_display)], color='purple')

ax_SOC.legend(['SOC'], fontsize=8, frameon=False, loc = 'lower right')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=8, frameon=False, loc = 'best')

   

plt.show()
# %%
