#%%
import os
import sys
sys.path.append(r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model")
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
dfs_time = []

sim_times = [8760, 9*30*24, 6*30*24, 3*30*24, 30*24, 14*24, 7*24, 24]

i = 0
for sim_time in sim_times:
    design.optim_time = sim_time
    start = time.time()
    data, data_time = dispatcher.MyFun(design, True)
    deltaT = time.time() - start
    df = pd.concat([df,data], ignore_index=True)
    df.loc[i, 'Simulation horizon [s]'] = sim_time
    df.loc[i, 'Simulation time [s]'] = deltaT
    data_time.set_index('Datetime', inplace=True)
    data_time.index = pd.MultiIndex.from_product([[df.loc[i, 'Simulation horizon [s]']], data_time.index], names=['Simulation horizon [s]', 'Datetime'])
    dfs_time.append(data_time)
    i += 1

dfs_time_tot = pd.concat(dfs_time)
df.set_index('Simulation horizon [s]', inplace=True)
df.insert(loc=0, column='Simulation time [s]', value=df.pop('Simulation time [s]'))

df.describe() #don't look at the LCOS, it does not make sense