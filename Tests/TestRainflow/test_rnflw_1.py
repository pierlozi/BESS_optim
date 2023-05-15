#%% 
import rainflow as rf
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter, DayLocator
plt.rcParams.update({'font.size': 12})

import os
import sys
sys.path.append(r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model")
RES_data_file_path = r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model\InputData\RESData_option-2.csv"
load_data_file_path = r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model\InputData\load_data.xlsx"

from Core import dispatcher_dsctd, microgrid_design, day_of_year

import importlib
importlib.reload(dispatcher_dsctd)
importlib.reload(microgrid_design)



tt = np.linspace(0,8759, 8760)
A = 3
B = 1
sin_ren = A*1e6 + B*1e6*np.sin(2*np.pi*tt/24)

P_ren_read = pd.read_csv(RES_data_file_path, header=0, nrows = 8760) #W
P_ren_read['Datetime'] =  pd.to_datetime(P_ren_read['Datetime'], format = '%Y-%m-%d %H:%M:%S')
# P_ren_read['Power'] = sin_ren

P_load = pd.read_excel(load_data_file_path, sheet_name='Yearly Load', header=0)
# P_load['Load [MW]'] = np.ones(P_ren_read['Power'].size)*P_ren_read['Power'].mean()/1e6


design = microgrid_design.MG(P_ren=P_ren_read, P_load=P_load, RES_fac=7)

data, data_time = dispatcher_dsctd.MyFun(design, True)
data_time['Datetime'] = pd.to_datetime(data_time['Datetime'], format = '%Y-%m-%d %H:%M:%S')
data_time.set_index('Datetime', inplace=True)

#%% Plots

day_start_display = 0 #day_of_year.MyFun("15-08")
days_display = 30

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


#%% Rainflow

# parameters 
# non-linear degradation
a_sei = 5.75e-2
b_sei = 121
#DoD stress
k_d1 = 1.4e5
k_d2 = -5.01e-1
k_d3 = -1.23e5
# SoC stress
k_s = 1.04
s_ref = 0.5 
# temperature stress
k_T = 6.93e-2
T_ref = 25 #degC
#calenar ageing
k_t = 4.14e-10 # 1/second

# functions
funct_S_d = lambda d: (k_d1 * d ** k_d2 + k_d3)**(-1)  #DoD degradation
funct_S_s = lambda s: np.exp(k_s*(s-s_ref))          #SOC degradation
funct_S_T = lambda T: np.exp(k_T*(T-T_ref)*T_ref/T)  #Temperature degradation
funct_S_t = lambda t: t*k_t                            #time degradation

funct_f_cyc_i = lambda d, s, T: funct_S_d(d)* funct_S_s(s) * funct_S_T(T)   #cyclic ageing
funct_f_cal = lambda s, t, T: funct_S_s(s) * funct_S_t(t) * funct_S_T(T)  #calendar ageing


L = np.array([])
L_sei = np.array([])

for i in range(1, 21):

    rnflow_data = pd.DataFrame(columns=['Range', 'Mean', 'Count', 'Start', 'End'])

    for rng, mean, count, i_start, i_end in rf.extract_cycles(np.tile(data_time['SOC'].values, i)): 
        new_row = pd.DataFrame({'Range': [rng], 'Mean': [mean], 'Count': [count], 'Start': [i_start], 'End': [i_end]})
        rnflow_data = pd.concat([rnflow_data, new_row], ignore_index=True)

    rf.count_cycles(data_time['SOC'].values)

    DoD = rnflow_data['Range']
    SOC = rnflow_data['Mean']
    f_cyc = funct_f_cyc_i(DoD, SOC, T_ref)*rnflow_data['Count'] #I multiply the weight of the cycle by the degradation of that cycle
    SOC_avg = data_time['SOC'].mean()
    f_cal = funct_f_cal(SOC_avg, 3600*data_time['SOC'].shape[0], T_ref)
    f_d = f_cyc.sum() + f_cal
    L = np.append(L, [1-np.exp(-f_d)])
    L_sei = np.append(L_sei, [1 - a_sei * np.exp(-b_sei*f_d) - (1-a_sei)*np.exp(-f_d)])


# %%
