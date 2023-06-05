#%%
import os
import sys
sys.path.append(r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model")

from Core import best_polyfit_degree

RES_data_file_path = r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model\InputData\RESData_option-2.csv"
load_data_file_path = r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model\InputData\load_data.xlsx"

import numpy as np
import pandas as pd


import altair as alt

import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter, DayLocator

plt.rcParams.update({'font.size': 12})

df = pd.read_excel('test_NSGAII_NPC_EmCost.xlsx', index_col = 0, header= 0)

X = np.array([list(elements) for elements in zip(df.iloc[:,0] ,df.iloc[:,1],df.iloc[:,2])])
F = np.array([list(elements) for elements in zip(df.iloc[:,3], df.iloc[:,4])])

#%%
coefficients = np.polyfit(df.iloc[:,3], df.iloc[:,4], 4)#best_polyfit_degree.MyFun(df.iloc[:,3], df.iloc[:,4]))

xl, xu = [X.min(axis=0), X.max(axis=0)]

plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], s=40, facecolors='none', edgecolors='r', label = "Pareto optimal solutions")
plt.plot(X[:, 0], X[:, 0]/10, label= "10 hours storage")
plt.xlim(xl[0], xu[0])
plt.ylim(xl[1], xu[1])
plt.title("Design Space")
plt.xlabel("Energy rating [MWh]")
plt.ylabel("Power rating [MW]")
plt.legend(loc = "best")
plt.show()

approx_ideal = F.min(axis=0) # gives an array with the minimum for every column
approx_nadir = F.max(axis=0)

plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.scatter(approx_ideal[0], approx_ideal[1], facecolors='none', edgecolors='red', marker="*", s=100, label="Ideal Point (Approx)")
plt.scatter(approx_nadir[0], approx_nadir[1], facecolors='none', edgecolors='black', marker="p", s=100, label="Nadir Point (Approx)")
plt.plot(np.linspace(df.iloc[:,3].min(), df.iloc[:,3].max(),100),np.polyval(coefficients, np.linspace(df.iloc[:,3].min()-15, df.iloc[:,3].max()+15,100)), color = 'green',label="PolyFit")

plt.title("Objective Space")
xstr = df.columns[-1]
plt.xlabel(xstr)
ystr = df.columns[-2]
plt.ylabel(ystr)
plt.legend(loc = "best")
plt.show()

#%%
df['gamma'] = df.iloc[:,0]/df.iloc[:,1]

#%%
fontsize = 20
chart1 = alt.Chart(df, title = "Objective Space").mark_circle(size = 350).encode(
        alt.X('NPC').scale(zero=False),
        alt.Y('EmCost').scale(zero=False),
        color = 'DoD',
        size = 'gamma'
        ).properties(
            width = 'container',
            height = 500
        ).configure_axis(
            labelFontSize = fontsize,
            titleFontSize = fontsize
        ).configure_legend(
            labelFontSize = fontsize,
            titleFontSize = fontsize
        ).configure_title(
            fontSize = fontsize
        )

chart2 = alt.Chart(pd.DataFrame({'Er': np.linspace(0,2000), 'Pr':1/10*np.linspace(0,2000)}))\
                  .mark_line().encode(
                                alt.X('Er').scale(domain = (0, df.Er.max()), clamp=True),
                                alt.Y('Pr').scale(domain = (0,df.Pr.max()), clamp=True)
                                ).properties(
            width = 'container',
            height = 500
        ).configure_axis(
            labelFontSize = fontsize,
            titleFontSize = fontsize
        ).configure_legend(
            labelFontSize = fontsize,
            titleFontSize = fontsize
        ).configure_title(
            fontSize = fontsize
        )
chart3  = alt.Chart(df, title = "Design Space").mark_circle().encode(
    alt.X('Er').scale(zero=False).title('Energy rating [MWh]'),
    alt.Y('Pr').scale(zero=False).title('Power rating [MW]'),
    size = 'gamma',
    color = 'DoD'
).properties(
            width = 'container',
            height = 500
        ).configure_axis(
            labelFontSize = fontsize,
            titleFontSize = fontsize
        ).configure_legend(
            labelFontSize = fontsize,
            titleFontSize = fontsize
        ).configure_title(
            fontSize = fontsize
        )
alt.vconcat(chart2+chart3, chart1)
# %%
