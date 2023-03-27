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
#vector of periods for the sinusioidal wave
# 1 hour, day, week, month, 4 months, 6 months
periods = np.array([1, 24, 24*7, 24*31, 24*31*4, 24*31*6]) 
dfs = []

P_ren_read = pd.read_csv(RES_data_file_path, header=0, nrows = 8760) #W
P_ren_read['Datetime'] =  pd.to_datetime(P_ren_read['Datetime'], format = '%Y-%m-%d %H:%M:%S')

for period in periods:

    sin_ren = 10*1e6 + 5*1e6*np.sin(2*np.pi*tt/period)
    
    P_ren_read['Power'] = sin_ren
    load_avg = np.linspace(min(P_ren_read['Power']),max(P_ren_read['Power']), 21) # 5% jumps from min to max ren power

    design = microgrid_design.MG(P_ren=P_ren_read, RES_fac=1)

    df = pd.DataFrame(columns=['Load [%]','Er_BES [MWh]','Pr_BES [MW]','Pr_diesel [MW]','BES cost [million euros]',\
                    'DG cost [million euros]','LCOS [€/MWh]','Fuel Consumption [L]', 'Total cost [million euros]'])

    i = 0
    for load in load_avg[0:]:
        design.P_load['Load [MW]'] = load*np.ones(8760)/1e6 #MW
        data, data_time = dispatcher.MyFun(design, True)

        df = pd.concat([df,data], ignore_index=True)
        df.loc[i, 'Load [%]'] = (load - load_avg[0])/(load_avg[-1] -  load_avg[0])*100
        df['Total cost [million euros]'] = df['BES cost [million euros]'] + df['DG cost [million euros]']
        i += 1 

    df.set_index('Load [%]', inplace=True)
    df.index = pd.MultiIndex.from_product([[period], df.index], names=['Period [h]', 'Load [%]'])
    dfs.append(df)

df_tot = pd.concat(dfs)
df_tot['Total cost [million euros]'] = df_tot['BES cost [million euros]'] + df_tot['DG cost [million euros]']
#%% 
P_ren_read = P_ren_read.set_index('Datetime')
for period in periods: 

    sin_ren = 10*1e6 + 5*1e6*np.sin(2*np.pi*tt/period)
    P_ren_read['Power'] = sin_ren

    

    fig,  ax_cost = plt.subplots()


    ax_cost.set_xlabel("Load [% max RES]")
    ax_cost.set_ylabel("Cost [million €]")
    ax_cost.set_title("Costs with period " + str(period) + " h.")
    ax_cost.grid(True)

    ax_cost.plot(df_tot.loc[period].index.get_level_values('Load [%]'), \
                df_tot.loc[period, 'BES cost [million euros]'], color= 'black')
    ax_cost.plot(df_tot.loc[period].index.get_level_values('Load [%]'), \
                df_tot.loc[period, 'DG cost [million euros]'], color= 'blue')
    ax_cost.plot(df_tot.loc[period].index.get_level_values('Load [%]'), \
                df_tot.loc[period, 'Total cost [million euros]'], color= 'red')

    # ax_cost.plot(df['Load [%]'], df['DG cost [million euros]'], color= 'blue')
    # ax_cost.plot(df['Load [%]'], df['Total cost [million euros]'], color= 'red')

    ax_cost.legend(['BES', 'DG', 'Total'],loc=4)


    
    ax_LCOS = ax_cost.twinx()
    ax_LCOS.set_ylabel("LCOS [€/MWh]")

    ax_LCOS.plot(df_tot.loc[period].index.get_level_values('Load [%]'), \
                df_tot.loc[period, 'LCOS [€/MWh]'], color= 'green')
    ax_LCOS.legend(['LCOS'])


    ax_cost.spines['right'].set_color("black")
    ax_LCOS.spines['right'].set_color("green")
    

    fig.tight_layout()
    #plt.savefig("test1_year_"+str(P_ren_read.index[4380].year)+".png")
    plt.show()
    plt.close()