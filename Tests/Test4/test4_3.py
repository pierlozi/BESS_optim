#%%
import os
import sys
sys.path.append(r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model")
from Core import dispatcher_DG
from Core import dispatcher
from Core import microgrid_design

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


tt = np.linspace(0,len(P_ren_read)-1, len(P_ren_read))
A = 3
B = 1
sin_ren = A*1e6 + B*1e6*np.sin(2*np.pi*tt/24)

P_ren_read['Power'] = sin_ren
P_load_read = pd.DataFrame()
P_load_read['Load [MW]'] = [P_ren_read['Power'].min()+1*(P_ren_read['Power'].max()-P_ren_read['Power'].min())]*np.ones(len(P_ren_read))/1e6 #MW

design = microgrid_design.MG(P_ren=P_ren_read, P_load= P_load_read, RES_fac=1, price_f=2, eff=1, DoD=100, 
                             optim_horiz=len(P_ren_read))


#%%
df = pd.DataFrame()
dfs = []
dfs_time = []

sim_times = [8760 , 9*30*24, 6*30*24, 3*30*24, 30*24, 14*24, 7*24, 24]
sim_times = sim_times[-1::-1]
sim_times.extend([2*8760,4*8760,8*8760,16*8760,21*8760])

i = 0
for sim_hor in sim_times[0:10]:
    design.optim_horiz = sim_hor

    start = time.time()
    data, data_time = dispatcher.MyFun(design, True)
    deltaT = time.time() - start

    start = time.time()
    data_update, data_time_update = dispatcher_DG.MyFun(design, True)
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

#%% To save table as a Microsoft Word table
from docx import Document

# Create a new Word document
doc = Document()

# Add a table to the document
table = doc.add_table(rows=dfs_tot.shape[0]+1, cols=dfs_tot.shape[1]+2)
table.style = 'Table Grid'

# Add the header row
for i, column in enumerate(dfs_tot.columns):
    table.cell(0, i+2).text = column

k = 0
# Add the data rows
for i, row in dfs_tot.iterrows():
    table.cell(k+1,0).text = str(int(i[0]))
    table.cell(k+1,1).text = str(round(i[1],2))
    for j, value in enumerate(row):
        table.cell(k+1, j+2).text = str(round(value,2))
    k += 1

table.cell(0,0).text = dfs_tot.index.names[0]
table.cell(0,1).text = dfs_tot.index.names[1]

# Save the document
doc.save('test4_3_table.docx')
# %%
