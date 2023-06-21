'''In this script indago sul perché il degrado non influenzi il DoD. Runno il dispatcher tenendo fissi i rating
e cambiando il DoD max e min. Dimostro che dato il modello di degrado che uso, no matter how big the DoD, il degrado rimane pressocche
lo stesso. per grandi dod lo SOC average é basso , il che porta il degrado calendar a essere a sua volta basso, mentre dato che DOd aumenta degrado 
cyclico aumenta ma leffetto combinato di questi due porta il degrado totale a essere pressocche costante'''
# %% 
import os
import sys
sys.path.append(r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model")
from Core import dispatcher_DoD
from Core import microgrid_design
from Core import rain_deg_funct, LCOS_funct
from Core import best_polyfit_degree

RES_data_file_path = r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model\InputData\RESData_option-2.csv"
load_data_file_path = r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model\InputData\load_data.xlsx"

import numpy as np
import pandas as pd
import time
import rainflow as rf

import altair as alt

import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter, DayLocator

plt.rcParams.update({'font.size': 12})

import importlib
importlib.reload(dispatcher_DoD)
importlib.reload(microgrid_design)
importlib.reload(LCOS_funct)
importlib.reload(rain_deg_funct)

#%%
''' Reading the data from csv files'''

P_ren_read = pd.read_csv(RES_data_file_path, header=0, nrows = 8760) #W
P_load = pd.read_excel(load_data_file_path, sheet_name='Yearly Load', header=0)

#%%
data = pd.DataFrame()
data_time = pd.DataFrame()

P_ren_read = pd.read_csv(RES_data_file_path, header=0, nrows = 8760) #W
P_load = pd.read_excel(load_data_file_path, sheet_name='Yearly Load', header=0)

design = microgrid_design.MG(Pr_BES=55, \
            Er_BES=434, \
            P_load=P_load, \
            P_ren=P_ren_read, \
            )

DoDs = [10,20,30,40,50,60,70,80,90,100]
res = []

df = pd.DataFrame()
dfs_time = []
i = 0

for DoD in DoDs:
    
    design.DoD = DoD

    data, data_time = dispatcher_DoD.MyFun(design, False)

    df = pd.concat([df,data], ignore_index=True)
    df.loc[i, 'DoD'] = DoD
    data_time.set_index('Datetime', inplace=True)
    data_time.index = pd.MultiIndex.from_product([[df.loc[i, 'DoD']], data_time.index], names=['DoD', 'Datetime'])
    dfs_time.append(data_time)
    i += 1 

    cyclelife, _ = rain_deg_funct.MyFun(SOC_profile = data_time['SOC'].values)
    
    RU = (sum(data_time['P_RES'] + data_time['P_dch']))/sum(data_time['P_load'])

    res.append([cyclelife, RU])

dfs_time_tot = pd.concat(dfs_time)
df.set_index('DoD', inplace=True)

#%%
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

L_sei = []
L_sei_no_cyc = []
f_ratio = []
for DoD in DoDs:

    SOC_profile = dfs_time_tot.loc[DoD]['SOC'].values

    rainflow = pd.DataFrame(columns=['Range', 'Mean', 'Count', 'Start', 'End'])

    for rng, mean, count, i_start, i_end in rf.extract_cycles(np.tile(SOC_profile, 1)): 
        new_row = pd.DataFrame({'Range': [rng], 'Mean': [mean], 'Count': [count], 'Start': [i_start], 'End': [i_end]})
        rainflow = pd.concat([rainflow, new_row], ignore_index=True)

    rnflow_data = rainflow
    DoD = rnflow_data['Range']
    SOC = rnflow_data['Mean']
    f_cyc = funct_f_cyc_i(DoD, SOC, T_ref)*rnflow_data['Count'] #I multiply the weight of the cycle by the degradation of that cycle
    SOC_avg = SOC_profile.mean()
    f_cal = funct_f_cal(SOC_avg, 3600*8760, T_ref)
    f_d = f_cyc.sum() + f_cal
    L_sei.append(1 - a_sei * np.exp(-b_sei*f_d) - (1-a_sei)*np.exp(-f_d))
    L_sei_no_cyc.append(1 - a_sei * np.exp(-b_sei*f_cal) - (1-a_sei)*np.exp(-f_cal))

fig, ax = plt.subplots()

ax.plot(L_sei, label='Tot')
ax.plot(L_sei_no_cyc, label='No Cyc') #non considero degrado ciclico

ax.legend(loc='best')
#nel grafico si vede che piú il Dod concesso é grande piu il calendar ageing ad un anno di dispatch diminuisce: questo perche piu il dod e grande 
# piu il dispatcher produce un profilo di carica nel quale l'average SOC é mediamente piu basso
# %%
