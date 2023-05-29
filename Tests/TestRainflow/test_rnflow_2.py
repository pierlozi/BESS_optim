#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter, DayLocator
plt.rcParams.update({'font.size': 12})

import os
import sys
sys.path.append(r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model")
RES_data_file_path = r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model\InputData\RESData_option-2.csv"
load_data_file_path = r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model\InputData\load_data.xlsx"

from Core import dispatcher_dsctd, microgrid_design, LCOS_funct, rain_deg_funct

import importlib
importlib.reload(dispatcher_dsctd)
importlib.reload(microgrid_design)
importlib.reload(rain_deg_funct)


P_ren_read = pd.read_csv(RES_data_file_path, header=0, nrows = 8760) #W
P_load = pd.read_excel(load_data_file_path, sheet_name='Yearly Load', header=0)


design = microgrid_design.MG(Pr_BES=30, \
                        Er_BES=275, \
                        P_load=P_load, \
                        P_ren=P_ren_read, \
                        DoD=75
                        )
data, data_time = dispatcher_dsctd.MyFun(design, False)

data_time['Datetime'] = pd.to_datetime(data_time['Datetime'], format = '%Y-%m-%d %H:%M:%S')
data_time.set_index('Datetime', inplace=True)
data_time['SOC'].to_csv('SOC.csv') # to then be analyzed in MATLAB

#%% Plots

day_start_display = 0 #day_of_year.MyFun("15-08")
days_display = 30

fig, ax_SOC = plt.subplots()



ax_SOC.set_ylabel("SOC [-]")
ax_SOC.plot(data_time['SOC'][24*day_start_display:24*(day_start_display+days_display)], color='purple')

 

# ax.set_xticks(ax.get_xticks())
# ax.set_xticklabels(ax.get_xticks(), ha='center')
ax_SOC.xaxis.set_major_locator(DayLocator())
ax_SOC.xaxis.set_major_formatter(DateFormatter('%d-%m'))

ax_SOC.legend(['SOC'], fontsize=8, frameon=True, framealpha = 0,  loc = 'lower right')

handles, labels = ax_SOC.get_legend_handles_labels()
legend = ax_SOC.legend(handles, labels, fontsize=8, frameon=True, loc = 'upper left', framealpha = 0)
legend.get_frame().set_facecolor('white')
plt.show()

# %%
cyclelife, SOH = rain_deg_funct.MyFun(data_time['SOC'])
L_sei_MATLAB = pd.read_csv('L_sei_MATLAB.csv', header = None)
SOH_MATLAB = 1 - L_sei_MATLAB[0].values

#%%

plt.plot(SOH, label='Python')
plt.plot(SOH_MATLAB, label = 'MATLAB')
plt.grid(True)
plt.xlabel('Years')
plt.ylabel('State Of Healt [-]')
plt.legend(['Python', 'MATLAB'])

# %%
DSOH = abs(SOH_MATLAB[SOH_MATLAB>=0.8]-SOH[SOH_MATLAB>=0.8])/SOH_MATLAB[SOH_MATLAB>=0.8]*100
print("The average difference between the Python and MATLAB SOH [SOH>=0.8] is of", round(DSOH.mean(),2), "%.")
# %%
