#%%
import os
import sys
sys.path.append(r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model")
from Core import dispatcher_copy
from Core import dispatcher
from Core import microgrid_design

RES_data_file_path = r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model\InputData\RESData_option-2.csv"

import numpy as np
import pandas as pd
import time
import importlib

import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter, DayLocator

plt.rcParams.update({'font.size': 12})

#%%
tt = np.linspace(0,8759, 8760)
A = 3
B = 1
sin_ren = A*1e6 + B*1e6*np.sin(2*np.pi*tt/24)

P_ren_read = pd.read_csv(RES_data_file_path, header=0, nrows = 8760) #W
P_ren_read['Datetime'] =  pd.to_datetime(P_ren_read['Datetime'], format = '%Y-%m-%d %H:%M:%S')
P_ren_read['Power'] = sin_ren

design = microgrid_design.MG(P_ren=P_ren_read, RES_fac=1, price_f=2, eff=1, DoD=100)

design.P_load['Load [MW]'] = [P_ren_read['Power'].min()+1*(P_ren_read['Power'].max()-P_ren_read['Power'].min())]*np.ones(8760)/1e6 #MW

#%%
design.optim_time = 24*7

df = []

start = time.time()
data, data_time = dispatcher.MyFun(design, True)
deltaT = time.time() - start
data_time.set_index('Datetime', inplace=True)

importlib.reload(dispatcher_copy)
start = time.time()
data_update, data_time_update = dispatcher_copy.MyFun(design, True)
deltaT_update = time.time() - start 

data_time_update.set_index('Datetime', inplace=True)

df = pd.concat([data,data_update])
df['Simulation Time [s]'] = [deltaT, deltaT_update]



#%% Plots

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
#%% 
