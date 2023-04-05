#%%
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import MonthLocator, DateFormatter, DayLocator

plt.rcParams.update({'font.size': 15}) 
plt.rcParams.update({'figure.figsize' : [20, 7]})


#%% Load plot
P_load = pd.read_excel('load_data.xlsx', sheet_name='Yearly Load', header=0)

ax_load = plt.gca()

P_load.iloc[0:25].plot(kind="line", color= 'black',y = 'Load [MW]', ax = ax_load)
ax_load.set_xlabel("Hour")
ax_load.set_ylabel("Power [MW]")
# P_prod_data.plot(kind="line", y = 'Power', ax = ax_load)

ax_load.get_legend().remove()
ax_load.set_xticks(list(range(0, 24+1,6)), labels = list(range(0, 24+1,6)))



plt.title("Daily Load Curve")
plt.ylim([25,40])
plt.grid()
plt.savefig('load_curve.png',bbox_inches='tight', dpi=150)

plt.show()

#%% RES generation
P_ren = pd.read_csv('RESData_option-2.csv', header=0, nrows = 8760) #W
P_ren['Datetime'] =  pd.to_datetime(P_ren['Datetime'], format = '%Y-%m-%d %H:%M:%S')
P_ren = P_ren.set_index('Datetime')
fig = plt.figure()#figsize=[25, 7])

ax_ren = fig.add_subplot()

ax_ren.set_xlabel("Month")
ax_ren.set_ylabel("Power [MW]")


ax_ren.plot(P_ren['Power']/1e6, color='black')


plt.title("RES generation curve")
plt.xticks(rotation=45)

plt.grid(True)

ax_ren.xaxis.set_major_locator(MonthLocator(bymonthday=1))
ax_ren.xaxis.set_major_formatter(DateFormatter('%B-%Y'))

plt.savefig('RES_curve.png',bbox_inches='tight', dpi=150)
# %%