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

tt = np.linspace(0,8759, 8760)
sin_ren = 3*1e6 + 1e6*np.sin(2*np.pi*tt/24)

P_ren_read = pd.read_csv(RES_data_file_path, header=0, nrows = 8760) #W
P_ren_read['Datetime'] =  pd.to_datetime(P_ren_read['Datetime'], format = '%Y-%m-%d %H:%M:%S')
P_ren_read['Power'] = sin_ren

design = microgrid_design.MG(P_ren=P_ren_read, RES_fac=1)
design.P_load['Load [MW]'] = max(P_ren_read['Power'])*np.ones(8760)/1e6 #MW

df = pd.DataFrame()
dfs_time = []

prcs_fuel = np.linspace(0, 2, 21)

i = 0
for prc_f in prcs_fuel:
    
    design.price_f = prc_f
    data, data_time = dispatcher.MyFun(design, True)

    df = pd.concat([df,data], ignore_index=True)
    df.loc[i, 'Fuel price [€/L]'] = prc_f
    data_time.set_index('Datetime', inplace=True)
    data_time.index = pd.MultiIndex.from_product([[df.loc[i, 'Fuel price [€/L]']], data_time.index], names=['Fuel price [€/L]', 'Datetime'])
    dfs_time.append(data_time)
    i += 1 


dfs_time_tot = pd.concat(dfs_time)
df.set_index('Fuel price [€/L]', inplace=True)

#%% 
fig,  ax_cost= plt.subplots()



ax_cost.set_xlabel("Fuel price [€/L]")
ax_cost.set_ylabel("Cost [million €]")
ax_cost.grid(True)

ax_cost.plot(df.index.get_level_values('Fuel price [€/L]'), \
            df['BES cost [million euros]'], color= 'black')
ax_cost.plot(df.index.get_level_values('Fuel price [€/L]'), \
            df['DG cost [million euros]'], color= 'blue')
ax_cost.plot(df.index.get_level_values('Fuel price [€/L]'), \
            df['Fuel Cost [million euros]'], color= 'yellow')
ax_cost.plot(df.index.get_level_values('Fuel price [€/L]'), \
            df['Total Cost [million euros]'], color= 'red')

# ax_cost.plot(df['Load [%]'], df['DG cost [million euros]'], color= 'blue')
# ax_cost.plot(df['Load [%]'], df['Total cost [million euros]'], color= 'red')

ax_cost.legend(['BES', 'DG', 'Fuel', 'Total'])



ax_LCOS = ax_cost.twinx()
ax_LCOS.set_ylabel("LCOS [€/MWh]")

ax_LCOS.plot(df.index.get_level_values('Fuel price [€/L]'), \
            df['LCOS [€/MWh]'], color= 'green')
ax_LCOS.legend(['LCOS'])


ax_cost.spines['right'].set_color("black")
ax_LCOS.spines['right'].set_color("green")



#plt.savefig("test1_year_"+str(P_ren_read.index[4380].year)+".png")
plt.show()
plt.close()
# %%
fig,  ax_cost = plt.subplots()



ax_cost.set_xlabel("Fuel price [€/L]")
ax_cost.set_ylabel("Ratio to Total Cost")
ax_cost.grid(True)

ax_cost.plot(df.index.get_level_values('Fuel price [€/L]'), \
            df['BES cost [million euros]']/df['Total Cost [million euros]'], color= 'black')
ax_cost.plot(df.index.get_level_values('Fuel price [€/L]'), \
            df['DG cost [million euros]']/df['Total Cost [million euros]'], color= 'blue')
ax_cost.plot(df.index.get_level_values('Fuel price [€/L]'), \
            df['Fuel Cost [million euros]']/df['Total Cost [million euros]'], color= 'yellow')

# ax_cost.plot(df['Load [%]'], df['DG cost [million euros]'], color= 'blue')
# ax_cost.plot(df['Load [%]'], df['Total cost [million euros]'], color= 'red')

ax_cost.legend(['BES', 'DG', 'Fuel'])

plt.show()
plt.close()
# %%
