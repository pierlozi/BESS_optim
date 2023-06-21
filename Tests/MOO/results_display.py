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

df = pd.read_excel('LCOS_1-RU/test_NSGAII_LCOS_1-RU.xlsx', index_col = 0, header= 0)

X = np.array([list(elements) for elements in zip(df.iloc[:,0] ,df.iloc[:,1],df.iloc[:,2])])
F = np.array([list(elements) for elements in zip(df.iloc[:,3], df.iloc[:,4])])

df#%%
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

plt.title("Objective Space")
xstr = df.columns[-1]
plt.xlabel(xstr)
ystr = df.columns[-2]
plt.ylabel(ystr)
plt.legend(loc = "best")
plt.show()

#%%
df['gamma'] = df.iloc[:,0]/df.iloc[:,1]
df_sorted = df.sort_values(by=[df.columns[-3]], ignore_index=True)

#%%
fontsize = 20
width = 700
height = 3/4*width

#%%
chart1 =  alt.Chart(df_sorted, title = "Design Space").mark_circle(size = 350).encode(
        alt.X('Er').scale(zero=False).title('Capacity rating [MWh]'),
        alt.Y('Pr').scale(zero=False).title('Power rating [MW]'),
        color = 'DoD',
        size = alt.Size('gamma').title(['Hours','Of','Storage']),
        tooltip = 'gamma'
        )

chart2 = alt.Chart(df_sorted[df_sorted.index==21]).mark_point(filled=True, size = 50, color = 'red').encode(
        alt.X('Er').scale(zero=False),
        alt.Y('Pr').scale(zero=False)
)

chart_des = alt.layer(chart1, chart2).configure_axis(
            labelFontSize = fontsize,
            titleFontSize = fontsize
        ).configure_legend(
            labelFontSize = fontsize,
            titleFontSize = fontsize
        ).configure_title(
            fontSize = fontsize
        ).properties(
            width = width,
            height = height
        )
chart_des

#%%
chart_des.save('pareto_NPC_EmCost_des.png')

#%%

# chart2 = alt.Chart(pd.DataFrame({'Er': np.linspace(0,2000), 'Pr':1/10*np.linspace(0,2000)}))\
#                   .mark_line().encode(
#                                 alt.X('Er').scale(domain = (0, df.Er.max()), clamp=True),
#                                 alt.Y('Pr').scale(domain = (0,df.Pr.max()), clamp=True)
#                                 ).properties(
#             width = width,
#             height = height
#         )

chart1 = alt.Chart(df_sorted, title = "Objective Space").mark_circle(size = 350).encode(
        alt.X(df_sorted.columns[-3]).scale(zero=False).title('NPC [million €]'),
        alt.Y(df_sorted.columns[-2]).scale(zero=False).title('EC [million €]'),
        color = 'DoD',
        size = alt.Size('gamma').title(['Hours','Of','Storage']),
        tooltip = 'gamma'
        )
chart2 = alt.Chart(df_sorted[df_sorted.index==21]).mark_point(filled=True, size = 50, color = 'red').encode(
        alt.X(df_sorted.columns[-3]).scale(zero=False),
        alt.Y(df_sorted.columns[-2]).scale(zero=False)
)

chart_obj = alt.layer(chart1+chart2).properties(
            width = width,
            height = height
        ).configure_axis(
            labelFontSize = fontsize,
            titleFontSize = fontsize
        ).configure_legend(
            labelFontSize = fontsize,
            titleFontSize = fontsize
        ).configure_title(
            fontSize = fontsize
        )
chart_obj
# %%
chart_obj.save('pareto_NPC_EmCost_obj.png')
# %%
