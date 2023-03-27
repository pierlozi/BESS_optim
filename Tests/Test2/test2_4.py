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

tt = np.linspace(0,8759, 8760)
P_ren_read['Power']= 10*1e6 + 2*1e6*np.sin(2*np.pi*tt/24)

load = np.ones(8760)*P_ren_read['Power'].min()/1e6 #MW
design = microgrid_design.MG(RES_fac=1)
design.P_load['Load [MW]'] = load
design.P_ren = P_ren_read

data, data_time = dispatcher.MyFun(design, True)
# %%
