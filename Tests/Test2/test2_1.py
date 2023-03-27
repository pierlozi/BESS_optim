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

#%% 

P_ren_read = P_ren_read.set_index('Datetime')

RES_data_info = [['Parameter', 'Value'],
        ['Energy [MWh/year]', round(P_ren_read['Power'].sum()/1e6, ndigits=2)],
        ['Mean [MWh]', round(P_ren_read['Power'].mean()/1e6, ndigits=2)],
        ['Std dev. [MWh]', round(P_ren_read['Power'].std()/1e6, ndigits=2)]]

print("The renewable energy generated in year ", P_ren_read.index[4380].year,\
        " amounts to ", sum(P_ren_read['Power'])/1e6, " MWh.\
        \nIt has a mean of ", P_ren_read['Power'].mean()/1e6,\
            " MWh and a standard deviation of ", P_ren_read['Power'].std()/1e6, " MWh.")


fig, (ax_pow, ax_cost) = plt.subplots(ncols=2, figsize=(35, 10), \
                                    gridspec_kw={'width_ratios':[1.5,1]}, \
                                    sharex= False, sharey= False)


ax_pow.set_xlabel("Month")
ax_pow.set_ylabel("Power [MW]")
ax_pow.set_title("RES curve of " + str(P_ren_read.index[4380].year))
ax_pow.grid(True)

# Create the table
table = ax_pow.table(cellText=RES_data_info, cellColours= [['w', 'w'],
    ['w', 'w'],
    ['w', 'w'],
    ['w', 'w']], loc='upper right', bbox=[0.8, 0.8, 0.18, 0.18])

# Set the table properties
table.set_fontsize(30)
table.scale(1.5, 1.5)

# Make the text in the first column and first row bold
for i in range(0, 4):
    cell = table.get_celld()[i, 0]
    cell.set_text_props(weight='bold')
    
for j in range(0, 2):
    cell = table.get_celld()[0, j]
    cell.set_text_props(weight='bold')

ax_pow.plot(P_ren_read['Power']/1e6, color='black')

ax_pow.set_xticks(ax_pow.get_xticks())
ax_pow.set_xticklabels(ax_pow.get_xticks(), rotation=30, ha='right')

ax_pow.xaxis.set_major_locator(MonthLocator(bymonthday=15))
ax_pow.xaxis.set_major_formatter(DateFormatter('%B-%Y'))


ax_cost.set_xlabel("Load [% max RES]")
ax_cost.set_ylabel("Cost [million €]")
ax_cost.set_title("Costs of " + str(P_ren_read.index[4380].year))
ax_cost.grid(True)

ax_cost.plot(df.index.get_level_values('Load [%]'), \
            df['BES cost [million euros]'], color= 'black')
ax_cost.plot(df.index.get_level_values('Load [%]'), \
            df['DG cost [million euros]'], color= 'blue')
ax_cost.plot(df.index.get_level_values('Load [%]'), \
            df['Total cost [million euros]'], color= 'red')

# ax_cost.plot(df['Load [%]'], df['DG cost [million euros]'], color= 'blue')
# ax_cost.plot(df['Load [%]'], df['Total cost [million euros]'], color= 'red')

ax_cost.legend(['BES', 'DG', 'Total'],loc=4)



ax_LCOS = ax_cost.twinx()
ax_LCOS.set_ylabel("LCOS [€/MWh]")

ax_LCOS.plot(df.index.get_level_values('Load [%]'), \
            df['LCOS [€/MWh]'], color= 'green')
ax_LCOS.legend(['LCOS'])


ax_cost.spines['right'].set_color("black")
ax_LCOS.spines['right'].set_color("green")


fig.tight_layout()
#plt.savefig("test1_year_"+str(P_ren_read.index[4380].year)+".png")
plt.show()
plt.close()
# %%
