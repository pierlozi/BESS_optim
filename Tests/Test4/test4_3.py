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

import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter, DayLocator

plt.rcParams.update({'font.size': 12})

tt = np.linspace(0,8759, 8760)
A = 3
B = 1
sin_ren = A*1e6 + B*1e6*np.sin(2*np.pi*tt/24)

P_ren_read = pd.read_csv(RES_data_file_path, header=0, nrows = 8760) #W
P_ren_read['Datetime'] =  pd.to_datetime(P_ren_read['Datetime'], format = '%Y-%m-%d %H:%M:%S')
P_ren_read['Power'] = sin_ren

design = microgrid_design.MG(P_ren=P_ren_read, RES_fac=1, price_f=2, eff=1, DoD=100)

design.P_load['Load [MW]'] = [P_ren_read['Power'].min()+0.45*(P_ren_read['Power'].max()-P_ren_read['Power'].min())]*np.ones(8760)/1e6 #MW
#%%
df = pd.DataFrame()
dfs = []
dfs_time = []

sim_times = [8760, 9*30*24, 6*30*24, 3*30*24, 30*24, 14*24, 7*24, 24]

i = 0
for sim_hor in sim_times:
    design.optim_time = sim_hor

    start = time.time()
    data, data_time = dispatcher.MyFun(design, True)
    deltaT = time.time() - start

    start = time.time()
    data_update, data_time_update = dispatcher_copy.MyFun(design, True)
    deltaT_update = time.time() - start 

    data_time_update.set_index('Datetime', inplace=True)

    df = pd.concat([data,data_update], ignore_index=True)
    df['Simulation time [s]'] = [deltaT, deltaT_update]
    #df.insert(loc=0, column='Simulation time [s]', value=df.pop('Simulation time [s]'))
    df.set_index('Simulation time [s]', inplace=True)
    df.index = pd.MultiIndex.from_product([[sim_hor], df.index], names=['Simulation horizon [s]', 'Simulation time [s]'])
    dfs.append(df)
    i += 1

dfs_tot = pd.concat(dfs)

time_ratios = []
for i in np.arange(0, len(dfs_tot.index.get_level_values(level=1)), 2):
    time_ratios.append(dfs_tot.index.get_level_values(level=1)[i+1]/dfs_tot.index.get_level_values(level=1)[i]) 

# %%
