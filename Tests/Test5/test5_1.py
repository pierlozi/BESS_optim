#%%
import os
import sys
sys.path.append(r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model")
from Core import dispatcher_SOC_pen, dispatcher_dsctd
from Core import microgrid_design
from Core import day_of_year

RES_data_file_path = r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model\InputData\RESData_option-2.csv"

import numpy as np
import pandas as pd
import time


import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter, DayLocator

plt.rcParams.update({'font.size': 12})



P_ren = pd.read_csv(RES_data_file_path, header=0) #W
P_ren['Datetime'] = pd.to_datetime(P_ren['Datetime'], format = '%Y-%m-%d %H:%M:%S')
leap_years = P_ren[P_ren['Datetime'].dt.year % 4 == 0]
P_ren_read = P_ren[~((P_ren['Datetime'].dt.month == 2) & (P_ren['Datetime'].dt.day == 29) & (P_ren['Datetime'].dt.year.isin(leap_years['Datetime'].dt.year)))]

design = microgrid_design.MG(P_ren = P_ren_read, optim_horiz=len(P_ren_read))

P_load_read = pd.DataFrame()
P_load_read['Load [MW]'] = [P_ren_read['Power'][0:design.optim_horiz].min()+0.5*(P_ren_read['Power'][0:design.optim_horiz].max()-P_ren_read['Power'][0:design.optim_horiz].min())]*np.ones(design.optim_horiz)/1e6 #MW
design.P_load = P_load_read
design.optim_horiz = 8760

data, data_time = dispatcher_dsctd.MyFun(design, True)

df = pd.DataFrame()
dfs_time = []

i = 0
for design.SOC_w in np.linspace(0,1,11):
    data, data_time = dispatcher_SOC_pen.MyFun(design, True)
    df = pd.concat([df,data], ignore_index=True)
    df.loc[i, 'SOC weight [-]'] = design.SOC_w

    data_time.set_index('Datetime', inplace=True)
    data_time.index = pd.MultiIndex.from_product([[df.loc[i, 'SOC weight [-]']], data_time.index], names=['SOC weight [-]', 'Datetime'])
    dfs_time.append(data_time)
    i += 1

dfs_time_tot = pd.concat(dfs_time)
df.set_index('SOC weight [-]', inplace=True)


       

   


#%% Plots
data_time = dfs_time_tot.loc[0]
day_start_display = day_of_year.MyFun("24-10")
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
ax.xaxis.set_major_locator(DayLocator())
ax.xaxis.set_major_formatter(DateFormatter('%d/%m'))

ax_SOC = ax.twinx()
ax_SOC.set_ylabel("SOC [-]")
ax_SOC.plot(data_time['SOC'][24*day_start_display:24*(day_start_display+days_display)], color='purple')

ax_SOC.legend(['SOC'], fontsize=8, frameon=False, loc = 'lower right')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=8, frameon=False, loc = 'best')

   

plt.show()
# %%