#%%
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import MonthLocator, DateFormatter, DayLocator
import altair as alt

alt.data_transformers.disable_max_rows()

plt.rcParams.update({'font.size': 15}) 
plt.rcParams.update({'figure.figsize' : [20, 7]})


#%% Load plot
P_load = pd.read_excel('load_data.xlsx', sheet_name='Yearly Load', header=0)
P_ren_load = pd.read_csv('RESData_option-2.csv', header=0, nrows = 8760) #W
P_ren_load['Power'] = P_ren_load['Power']/1e6

ax_load = plt.subplot()


day_start = 2

ax_load.plot(P_load.iloc[25*day_start:25*day_start+24]['Load [MW]'], color= 'black', label = "Load")
ax_load.plot(P_ren_load.iloc[25*day_start:25*day_start+24]['Power']*7, color= 'green', label= "RES")
ax_load.set_xlabel("Hour")
ax_load.set_ylabel("Power [MW]")




ax_load.set_xticks(list(range(25*day_start, 25*day_start+24+1,6)), labels = list(range(0, 24+1,6)))

ax_load.legend(loc = 'best')

plt.title("Day Power Curve")
#plt.ylim([25,40])
plt.grid()
#plt.savefig('load_curve.png',bbox_inches='tight', dpi=150)

plt.show()

#%% RES generation
P_ren = pd.read_csv('RESData_option-2.csv', header=0, nrows = 8760) #W
P_ren['Datetime'] =  pd.to_datetime(P_ren['Datetime'], format = '%Y-%m-%d %H:%M:%S')
P_ren = P_ren.set_index('Datetime')
fig = plt.figure() #figsize=[25, 7])

ax_ren = fig.add_subplot()

ax_ren.set_xlabel("Month")
ax_ren.set_ylabel("Power [MW]")


ax_ren.plot(P_ren['Power']/1e6*7, color='black')


plt.title("RES generation curve")
plt.xticks(rotation=45)

plt.grid(True)

ax_ren.xaxis.set_major_locator(MonthLocator(bymonthday=1))
ax_ren.xaxis.set_major_formatter(DateFormatter('%B-%Y'))

plt.savefig('RES_curve.png',bbox_inches='tight', dpi=150)
# %%
P_diff = pd.DataFrame(columns = ['Datetime', 'Power'])
P_diff.Datetime = P_ren.index
P_diff.Power = P_ren['Power'].values*7/1e6 - P_load['Load [MW]'].values

alt.Chart(P_diff).mark_bar(width=1).encode(
    x = 'Datetime:T',
    y = 'Power',
    color = alt.condition(alt.datum.Power > 0, alt.ColorValue('green'), alt.ColorValue('red'))
).properties(
    width = 'container',
    height = 400
)
# %%
