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

df = pd.read_excel('res_NSGAII_LCOS_EMCost_30pop_05ftol.xlsx', index_col = 0, usecols="A:F", header= 0)
df.columns = ['Er', 'Pr', 'DoD', 'LCOS','EmCost']

X = np.array([list(elements) for elements in zip(df.Er.values,df.Pr.values,df.DoD.values)])
F = np.array([list(elements) for elements in zip(df.LCOS,df.EmCost)])

coefficients = np.polyfit(df.LCOS.values, df.EmCost.values, best_polyfit_degree.MyFun(df.LCOS.values, df.EmCost.values ))

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
plt.plot(np.linspace(df.LCOS.min(), df.LCOS.max(),100),np.polyval(coefficients, np.linspace(df.LCOS.min()-15, df.LCOS.max()+15,100)), color = 'green',label="PolyFit")

plt.title("Objective Space")
plt.xlabel("LCOS [€/MWh]")
plt.ylabel("Emissions cost [mil€]")
plt.legend(loc = "best")
plt.show()

alt.Chart(df, title = "Objective Space").mark_circle().encode(
    alt.X('LCOS').scale(zero=False),
    alt.Y('EmCost').scale(zero=False),
    color = 'DoD'
)

# alt.Chart(df[df.gamma<=10], title = "Less than 10hrs storage").mark_circle().encode(
#     alt.X('LCOS').scale(zero=False),
#     alt.Y('EmCost').scale(zero=False),
#     size = 'dist_id',
#     color = 'DoD'
# )