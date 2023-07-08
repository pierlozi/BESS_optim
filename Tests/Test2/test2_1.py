#%%
import os
import sys
sys.path.append(r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model")
from Core import dispatcher_DoD
from Core import microgrid_design

RES_data_file_path = r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model\InputData\RESData_option-2.csv"

import numpy as np
import pandas as pd
import math

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

load_avg = np.linspace(min(P_ren_read['Power']), max(P_ren_read['Power']) + 1/2*(max(P_ren_read['Power']) - min(P_ren_read['Power'])) , 16)
#%%

design = microgrid_design.MG(P_ren=P_ren_read, DoD = 100, RES_fac=1, eff=1, sigma=0)
design.optim_horiz = 24*3

df = pd.DataFrame()
dfs_time = []

i = 0
for load in load_avg:
    design.P_load['Load [MW]'] = load*np.ones(8760)/1e6 #MW
    data, data_time = dispatcher_DoD.MyFun(design, True)

    df = pd.concat([df,data], ignore_index=True)
    df.loc[i, 'Load [%]'] = (load - load_avg[0])/(max(P_ren_read['Power'])-min(P_ren_read['Power']))*100
    data_time.set_index('Datetime', inplace=True)
    data_time.index = pd.MultiIndex.from_product([[df.loc[i, 'Load [%]']], data_time.index], names=['Load [%]', 'Datetime'])
    dfs_time.append(data_time)
    i += 1 


dfs_time_tot = pd.concat(dfs_time)
df.set_index('Load [%]', inplace=True)

#%%
# df.to_excel("test2_2_df_load_0_145.xlsx")
# dfs_time_tot.to_excel("test2_2_dfs_time_tot_load_0_300.xlsx")

#%%GIGA plot

day_start_display = 1
days_display = 2
col_num = 3

fig, axes = plt.subplots(nrows=math.ceil(len(df.index.get_level_values('Load [%]'))/col_num), ncols=col_num, figsize=(25, 50))
plt.subplots_adjust(wspace=0.55, hspace=0.4)


#The enumerate() function returns an iterator with both the index number and value of each element
# in the axes.flat object, which represents a flattened version of the subplots. 
# The i variable stores the index number of the current subplot, 
# while the ax variable stores the current subplot object itself.

for i, ax in enumerate(axes.flat):
    if i < len(df.index.get_level_values('Load [%]')):
        load = df.index.get_level_values('Load [%]')[i]
        ax.plot(dfs_time_tot.loc[load]['P_dg'][24*day_start_display:24*(day_start_display+days_display)], color=None, label='DG')
        ax.plot(dfs_time_tot.loc[load]['P_curt'][24*day_start_display:24*(day_start_display+days_display)],color=None, label='Curt')
        ax.plot(dfs_time_tot.loc[load]['P_prod'][24*day_start_display:24*(day_start_display+days_display)], color=None, label='Prod')
        ax.plot(dfs_time_tot.loc[load]['P_load'][24*day_start_display:24*(day_start_display+days_display)], color=None, label='Load')
        ax.plot(dfs_time_tot.loc[load]['P_BES'][24*day_start_display:24*(day_start_display+days_display)], color=None, label='BES')

        ax.set_ylabel("Power [MW]")
        ax.set_title("Load " + str(load) + "%")  # add a title to the subplot
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticks(), rotation=30, ha='right')
        ax.xaxis.set_major_locator(DayLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%d/%m'))
        
        ax_SOC = ax.twinx()
        ax_SOC.set_ylabel("SOC [-]")
        ax_SOC.plot(dfs_time_tot.loc[load]['SOC'][24*day_start_display:24*(day_start_display+days_display)], color='purple')
        ax_SOC.legend(['SOC'], loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=8, frameon=False)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='center left', fontsize=8, frameon=False, bbox_to_anchor=(1.15, 0.87))
        
    else:
        # Remove empty subplot
        fig.delaxes(ax)

# Show the figure
plt.show()
plt.savefig('test_trivial.png')
#%% 
fig,  ax_cost= plt.subplots()

ax_cost.set_xlabel("Load [% max RES]")
ax_cost.set_ylabel("Cost [million €]")
ax_cost.set_title("Costs of " + str(P_ren_read.index[4380].year))
ax_cost.grid(True)

ax_cost.plot(df.index.get_level_values('Load [%]'), \
            df['BES cost [million euros]'], color= 'black')
ax_cost.plot(df.index.get_level_values('Load [%]'), \
            df['DG cost [million euros]'], color= 'blue')
ax_cost.plot(df.index.get_level_values('Load [%]'), \
            df['Fuel Cost [million euros]'], color= 'yellow')
ax_cost.plot(df.index.get_level_values('Load [%]'), \
            df['Total Cost [million euros]'], color= 'red')

# ax_cost.plot(df['Load [%]'], df['DG cost [million euros]'], color= 'blue')
# ax_cost.plot(df['Load [%]'], df['Total cost [million euros]'], color= 'red')

ax_cost.legend(['BES', 'DG', 'Fuel', 'Total'])


ax_LCOS = ax_cost.twinx()
ax_LCOS.set_ylabel("LCOS [€/MWh]")

ax_LCOS.plot(df.index.get_level_values('Load [%]'), \
            df['LCOS [€/MWh]'], color= 'green')
ax_LCOS.legend(['LCOS'])


ax_cost.spines['right'].set_color("black")
ax_LCOS.spines['right'].set_color("green")



#plt.savefig("test1_year_"+str(P_ren_read.index[4380].year)+".png")
plt.show()
plt.close()
# %%
