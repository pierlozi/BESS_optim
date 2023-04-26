#%%
import os
import sys
sys.path.append(r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model")
from Core import dispatcher_dsctd
from Core import microgrid_design, day_of_year

RES_data_file_path = r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model\InputData\RESData_option-2.csv"
load_data_file_path = r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model\InputData\load_data.xlsx"

import importlib
importlib.reload(dispatcher_dsctd)
importlib.reload(microgrid_design)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter, DayLocator

plt.rcParams.update({'font.size': 12})

P_ren_read = pd.read_csv(RES_data_file_path, header=0, nrows = 8760) #W

P_ren_read['Datetime'] =  pd.to_datetime(P_ren_read['Datetime'], format = '%Y-%m-%d %H:%M:%S')

P_load = pd.read_excel(load_data_file_path, sheet_name='Yearly Load', header=0)

design = microgrid_design.MG(P_ren=P_ren_read, P_load=P_load, RES_fac=8, price_f=1.85)

data, data_time = dispatcher_dsctd.MyFun(design, True)
data_time['Datetime'] = pd.to_datetime(data_time['Datetime'], format = '%Y-%m-%d %H:%M:%S')
data_time.set_index('Datetime', inplace=True)
#%% Plots

day_start_display = day_of_year.MyFun("15-08")
days_display = 3

fig, ax = plt.subplots()
 
ax.plot(data_time['P_dg'][24*day_start_display:24*(day_start_display+days_display)], color=None, label='DG')
ax.plot(data_time['P_curt'][24*day_start_display:24*(day_start_display+days_display)],color=None, label='Curt')
ax.plot(data_time['P_prod'][24*day_start_display:24*(day_start_display+days_display)], color=None, label='Prod')
ax.plot(data_time['P_load'][24*day_start_display:24*(day_start_display+days_display)], color=None, label='Load')
ax.plot(data_time['P_BES'][24*day_start_display:24*(day_start_display+days_display)], color=None, label='BES')

ax.set_ylabel("Power [MW]")
# ax.set_xticks(ax.get_xticks())
# ax.set_xticklabels(ax.get_xticks(), ha='center')
ax.xaxis.set_major_locator(DayLocator())
ax.xaxis.set_major_formatter(DateFormatter('%d-%m'))

ax_SOC = ax.twinx()
ax_SOC.set_ylabel("SOC [-]")
ax_SOC.plot(data_time['SOC'][24*day_start_display:24*(day_start_display+days_display)], color='purple')

ax_SOC.legend(['SOC'], fontsize=8, frameon=True, framealpha = 0,  loc = 'lower right')

handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(handles, labels, fontsize=8, frameon=True, loc = 'upper left', framealpha = 0)
legend.get_frame().set_facecolor('white')
plt.show()
# %%
