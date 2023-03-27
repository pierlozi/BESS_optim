#%%
import os
import sys
sys.path.append(r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model")
from Core import dispatcher
from Core import microgrid_design

RES_data_file_path = r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model\InputData\RESData_option-2.csv"

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter, DayLocator

plt.rcParams.update({'font.size': 12})

P_ren_read = pd.read_csv(RES_data_file_path, header=0, nrows = 8760) #W
P_ren_read['Datetime'] =  pd.to_datetime(P_ren_read['Datetime'], format = '%Y-%m-%d %H:%M:%S')

#in P_ren_read there are 89 zeros. I substitute them with the smallest value of generation higher than zero. 
#I need this change to do testing
P_ren_read[P_ren_read['Power']==0] = P_ren_read[P_ren_read['Power'] > 0]['Power'].min()

res_share = np.linspace(0,200,41)

load = np.ones(8760)*10 #MW
design = microgrid_design.MG()
design.P_load['Load [MW]'] = load
design.P_ren = P_ren_read

design.RES_fac = np.mean(load)/(P_ren_read['Power'].min()/1e6)
data, data_time = dispatcher.MyFun(design, True)
# %%
