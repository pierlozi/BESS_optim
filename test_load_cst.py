#%%
import microgrid_design
import dispatcher

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter, DayLocator

from matplotlib.pyplot import figure


''' Reading the data from csv files'''
P_load_data = pd.read_excel('load_data.xlsx', sheet_name='Yearly Load', header=0) #MW
P_ren_read = pd.read_csv('RESData_option-2.csv', header=0, nrows = 8760) #W
P_ren_read['Datetime'] =  pd.to_datetime(P_ren_read['Datetime'], format = '%Y-%m-%d %H:%M:%S')

RES_fac = 7

# data from 'Optimal sizing of battery energy storage in a microgrid considering capacity degradation and replacement year'
# C_P = 320 #$/kW
# C_E = 360 #$/kWh
# C_inst = 15 #$/kWh
# C_POM = 5 #$/kW operation cost related to power
# C_EOM = 0 #$/Mwh operation cost related to energy
# sigma = 0,2/100 #original daily self discharge is 0,2% -> we need an hourly self discharge
# m.IR = 5/100

#data from 'Projecting the Future Levelized Cost of Electricity Storage Technologies'
C_P = 678 #$/kW
C_E = 802 #$/kWh
C_inst = 0 #$/kWh (the reference doesnt take into account installation)
C_POM = 10 #$/kW operation cost related to power
C_EOM = 3 #$/Mwh operation cost related to energy
sigma = 0 #original daily self discharge is 0,2% -> we need an hourly self discharge
IR = 8/100

floatlife = 10 #years
mine_life = 13 #years

price_f = 1.66 # €/L

DoD = 75 # %
cyclelife = 2700 #cycles


load_avg = np.linspace(min(P_ren_read['Power']),max(P_ren_read['Power']), 20) # 5% jumps from min to max ren power

load = load_avg[1]*np.ones(8760)/1e6 #MW
P_load_data['Load [MW]'] = load 

#design = microgrid(Pr_BES=20,Er_BES=250, P_load=P_load_data, P_ren=P_ren_read)
design = microgrid_design.MG(Pr_BES=20, \
                   Er_BES=250, \
                   P_load=P_load_data, \
                   P_ren=P_ren_read, \
                   mine_life= mine_life,\
                   RES_fac= RES_fac, \
                   floatlife= floatlife, \
                   C_P= C_P, \
                   C_E= C_E,\
                   C_inst= C_inst, \
                   C_POM= C_POM, \
                   C_EOM= C_EOM, \
                   sigma= sigma, \
                   IR= IR, \
                   DoD= DoD, \
                   cyclelife= cyclelife, \
                   price_f= price_f)

df = pd.DataFrame(columns=['Load [%]','Er_BES [MWh]','Pr_BES [MW]','Pr_diesel [MW]','BES cost [million euros]',\
                   'Fuel cost [million euros]','LCOS [€/MWh]','Fuel Consumption [L]'])

i = 0
for load in load_avg[1:3]:
    design.P_load['Load [MW]'] = load*np.ones(8760)/1e6 #MW
    data, _ = dispatcher.MyFun(design, True)

    df = pd.concat([df,data], ignore_index=True)
    df['Load [%]'][i] = (load - load_avg[1])/(load_avg[-1] - load_avg[1])*100
    i += 1 

#this sections calculates how much it is spent if there is no BES with minimum load
load = load_avg[1]
design.P_load['Load [MW]'] = load*np.ones(8760)/1e6 #MW
design.Er_BES = 0
design.Pr_BES = 0
data, _ = dispatcher.MyFun(design, False)
df = pd.concat([df,data], ignore_index=True)
df['Load [%]'][i] = (load - load_avg[1])/(load_avg[-1] - load_avg[1])*100

df['Total cost [million euros]'] = df['BES cost [million euros]'] + df['Fuel cost [million euros]']

# %%

fig, ax_cost = plt.subplots(figsize=(15,7))
fig.set_dpi(200)

ax_cost.set_xlabel("Load [% max RES]")
ax_cost.set_ylabel("Cost [million €]")
ax_cost.set_title("Costs")

ax_cost.plot(df['Load [%]'], df['BES cost [million euros]'], color= 'black')
ax_cost.plot(df['Load [%]'], df['Fuel cost [million euros]'], color= 'blue')
ax_cost.plot(df['Load [%]'], df['Total cost [million euros]'], color= 'red')
ax_cost.legend(['BES', 'Fuel', 'Total'],loc=4)

ax_LCOS = ax_cost.twinx()
ax_LCOS.set_ylabel("LCOS [€/MWh]")

ax_LCOS.plot(df['Load [%]'], df['LCOS [€/MWh]'], color= 'green')
ax_LCOS.legend(['LCOS'])


ax_cost.spines['right'].set_color("black")
ax_LCOS.spines['right'].set_color("green")

plt.grid(True)
plt.show()
plt.close()

# %% TIME SERIES PLOT
# plt.rcParams["figure.figsize"] = (20,3)
# plt.rcParams.update({'font.size': 12})

# display_start = 0
# display_end= 24*31

# time_horizon = range(display_start, display_end)

# fig, ax_pow = plt.subplots()

# ax_pow.set_ylabel("Power [MW]")
# ax_pow.set_xlabel("Time [h]")
# ax_pow.plot(data_time['Datetime'][display_start:display_end], data_time['P_load'][display_start:display_end], color='black')
# ax_pow.plot(data_time['Datetime'][display_start:display_end], data_time['P_BES'][display_start:display_end], color = 'blue')
# ax_pow.plot(data_time['Datetime'][display_start:display_end], data_time['P_prod'][display_start:display_end], color = 'green')
# ax_pow.plot(data_time['Datetime'][display_start:display_end], data_time['P_curt'][display_start:display_end], color='red')
# ax_pow.plot(data_time['Datetime'][display_start:display_end], data_time['P_dg'][display_start:display_end], color='orange')
# ax_pow.legend(['P_Load', 'P_BES', 'P_RES', 'P_curt','P_diesel'],loc=4)


# ax_pow.set_title("Energy dispatch diesel + PV + BESS")

# ax_SOC = ax_pow.twinx()  # instantiate a second axes that shares the same x-axis

# ax_SOC.set_ylabel('SOC [-]')  # we already handled the x-label with ax1
# ax_SOC.plot(data_time['Datetime'][display_start:display_end], data_time['SOC'][display_start:display_end], color='cyan')
# ax_SOC.legend(['SOC'],loc=2)

# ax_pow.spines['right'].set_position(('axes',0.15))

# ax_pow.spines['right'].set_color("black")
# ax_SOC.spines['right'].set_color("cyan")



# fig.set_dpi(200)
# ax_pow.tick_params(axis='both', which='major')
# ax_SOC.tick_params(axis='both', which='major')

# ax_pow.xaxis.set_major_locator(DayLocator())
# ax_pow.xaxis.set_major_formatter(DateFormatter('%m-%d'))

# # plt.savefig('powers_SOC.png',bbox_inches='tight', dpi=150)

# plt.grid(True)
# plt.show()
# plt.close()

#%% Plotting the renewable generation
P_ren_read = P_ren_read.set_index('Datetime')
fig = plt.figure(figsize=[25, 7])

ax_pow = fig.add_subplot()

ax_pow.set_xlabel("Month")
ax_pow.set_ylabel("Power [MW]")

ax_pow.plot(P_ren_read['Power']/1e6, color='black')

plt.title("RES generation curve")
plt.xticks(rotation=45)

plt.grid(True)

ax_pow.xaxis.set_major_locator(MonthLocator(bymonthday=15))
ax_pow.xaxis.set_major_formatter(DateFormatter('%B'))

plt.savefig('RES_curve.png',bbox_inches='tight', dpi=150)

# %%
