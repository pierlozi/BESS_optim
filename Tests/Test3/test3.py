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


res_share = np.linspace(0,200,41)

load = np.ones(8760)*10 #MW
design = microgrid_design.MG()
design.P_load['Load [MW]'] = load
design.P_ren = P_ren_read

df = pd.DataFrame(columns=['Er_BES [MWh]','Pr_BES [MW]','Pr_diesel [MW]','BES cost [million euros]',\
                'DG cost [million euros]','LCOS [€/MWh]','Fuel Consumption [L]', 'Total cost [million euros]'])



df_time = pd.DataFrame(columns= ['Datetime','SOC','P_BES','P_curt','P_dg','P_prod','P_load'])


for i in range(len(res_share)):
    
    design.RES_fac = res_share[i]/100*np.mean(load)/(P_ren_read['Power'].mean()/1e6)
    data, data_time = dispatcher.MyFun(design, True)
    
    df = pd.concat([df,data], ignore_index=True)
    # df.loc[i, 'Res share [%]'] = res_share[i]
    df['Total cost [million euros]'] = df['BES cost [million euros]'] + df['DG cost [million euros]']

df['RES share [%]'] = res_share
df.set_index('RES share [%]', inplace=True)

#%% 

# P_ren_read = P_ren_read.set_index('Datetime')

# RES_data_info = [['Parameter', 'Value'],
#         ['Energy [MWh/year]', round(P_ren_read['Power'].sum()/1e6, ndigits=2)],
#         ['Mean [MWh]', round(P_ren_read['Power'].mean()/1e6, ndigits=2)],
#         ['Std dev. [MWh]', round(P_ren_read['Power'].std()/1e6, ndigits=2)]]

# print("The renewable energy generated in year ", P_ren_read.index[4380].year,\
#         " amounts to ", sum(P_ren_read['Power'])/1e6, " MWh.\
#         \nIt has a mean of ", P_ren_read['Power'].mean()/1e6,\
#             " MWh and a standard deviation of ", P_ren_read['Power'].std()/1e6, " MWh.")


# fig, (ax_pow, ax_cost) = plt.subplots(ncols=2, figsize=(35, 10), \
#                                     gridspec_kw={'width_ratios':[1.5,1]}, \
#                                     sharex= False, sharey= False)

fig, ax_cost = plt.subplots()

# ax_pow.set_xlabel("Month")
# ax_pow.set_ylabel("Power [MW]")
# ax_pow.set_title("RES curve of " + str(P_ren_read.index[4380].year))
# ax_pow.grid(True)

# ax_pow.plot(P_ren_read['Power']/1e6, color='black')

# ax_pow.set_xticks(ax_pow.get_xticks())
# ax_pow.set_xticklabels(ax_pow.get_xticks(), rotation=30, ha='right')

# ax_pow.xaxis.set_major_locator(MonthLocator(bymonthday=15))
# ax_pow.xaxis.set_major_formatter(DateFormatter('%B-%Y'))


ax_cost.set_xlabel("RES mean share [%]")
ax_cost.set_ylabel("Cost [million €]")
#ax_cost.set_title("Costs of " + str(P_ren_read.index[4380].year))
ax_cost.grid(True)

ax_cost.plot(df.index.get_level_values('RES share [%]'), \
            df['BES cost [million euros]'], color= 'black')
ax_cost.plot(df.index.get_level_values('RES share [%]'), \
            df['DG cost [million euros]'], color= 'blue')
ax_cost.plot(df.index.get_level_values('RES share [%]'), \
            df['Total cost [million euros]'], color= 'red')

# ax_cost.plot(df['Load [%]'], df['DG cost [million euros]'], color= 'blue')
# ax_cost.plot(df['Load [%]'], df['Total cost [million euros]'], color= 'red')

ax_cost.legend(['BES', 'DG', 'Total'],loc=4)


ax_LCOS = ax_cost.twinx()
ax_LCOS.set_ylabel("LCOS [€/MWh]")

ax_LCOS.plot(df.index.get_level_values('RES share [%]'), \
            df['LCOS [€/MWh]'], color= 'green')
ax_LCOS.legend(['LCOS'])


ax_cost.spines['right'].set_color("black")
ax_LCOS.spines['right'].set_color("green")


#fig.tight_layout()
#plt.savefig("test1_year_"+str(P_ren_read.index[4380].year)+".png")
plt.show()
plt.close()
# %%

fig, ax_pow = plt.subplots(figsize=(10,8))

ax_pow.set_ylabel("Power [MW]")
ax_pow.set_xlabel("time [h]")
ax_pow.plot(data_time['P_load'][0:24*14], color='black')
ax_pow.plot(data_time['P_BES'][0:24*14], color = 'blue')
ax_pow.plot(data_time['P_prod'][0:24*14], color = 'green')
ax_pow.plot(data_time['P_curt'][0:24*14], color='red')
ax_pow.plot(data_time['P_dg'][0:24*14], color='yellow')
ax_pow.legend(['P_Load', 'P_BES', 'P_ren', 'P_curt', 'P_dg'],loc=4)


ax_pow.spines['right'].set_color("black")


plt.grid(True)

fig.set_size_inches(10,8)
fig.set_dpi(200)
ax_pow.tick_params(axis='both', which='major')

# plt.savefig('powers_price.png',bbox_inches='tight', dpi=150)

plt.show()
plt.close()
# %%
