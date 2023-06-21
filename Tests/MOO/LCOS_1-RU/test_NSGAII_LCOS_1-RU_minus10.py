# %% 
import os
import sys
sys.path.append(r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model")
from Core import dispatcher_dsctd
from Core import microgrid_design
from Core import rain_deg_funct, LCOS_funct
from Core import best_polyfit_degree

RES_data_file_path = r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model\InputData\RESData_option-2.csv"
load_data_file_path = r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model\InputData\load_data.xlsx"

import numpy as np
import pandas as pd
import time

from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.termination.robust import RobustTermination
from pymoo.operators.sampling.rnd import IntegerRandomSampling

import altair as alt

import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter, DayLocator

plt.rcParams.update({'font.size': 12})

import importlib
importlib.reload(dispatcher_dsctd)
importlib.reload(microgrid_design)
importlib.reload(LCOS_funct)
importlib.reload(rain_deg_funct)

#%%
''' Reading the data from csv files'''

P_ren_read = pd.read_csv(RES_data_file_path, header=0, nrows = 8760) #W
P_load = pd.read_excel(load_data_file_path, sheet_name='Yearly Load', header=0)

design = microgrid_design.MG(Pr_BES=17.7, \
                Er_BES=173, \
                P_load=P_load, \
                P_ren=P_ren_read,
                price_f = 1.494
                )

P_lim = round(max(abs(P_ren_read['Power']*design.RES_fac/1e6-P_load['Load [MW]'])))
E_lim = P_lim*10
#%%

class ProblemWrapper(Problem):

    def _evaluate(self, designs, out, *args, **kwargs):
        res = []
        data = pd.DataFrame()
        data_time = pd.DataFrame()

        P_ren_read = pd.read_csv(RES_data_file_path, header=0, nrows = 8760) #W
        P_load = pd.read_excel(load_data_file_path, sheet_name='Yearly Load', header=0)

        design = microgrid_design.MG(Pr_BES=17.7, \
                        Er_BES=173, \
                        P_load=P_load, \
                        P_ren=P_ren_read
                        )

        for x in designs:

            design.Er_BES = x[0] 
            design.Pr_BES = x[1]
            design.DoD = x[2]

            data, data_time = dispatcher_dsctd.MyFun(design, False)

            design.DG_CAPEX = data['DG cost [million euros]']
            design.DG_OPEX = data['Fuel Cost [million euros]']

            design.cyclelife, _ = rain_deg_funct.MyFun(SOC_profile = data_time['SOC'].values)

            LCOS, _ , _ , _ = LCOS_funct.MyFun(design, \
                                    E_dch = sum(data_time['P_dch']),\
                                    res_val_bin = True
                                    )


            RU = (sum(data_time['P_RES'] + data_time['P_dch']))/sum(data_time['P_load'])

            res.append([LCOS, 1-RU])
        
        out['F'] = np.array(res)

#the variables are in order Er_BES, Pr_BES, DoD

problem = ProblemWrapper(n_var=3, n_obj=2, xl=[0.,0.,20.], xu = [E_lim, P_lim ,80.], vtype=int)

algorithm = NSGA2(pop_size=50,
                  sampling = IntegerRandomSampling(),
                  eliminate_duplicates=True
                  )

#termination = get_termination("n_gen", 1) # | get_termination("tolx", 1) # | get_termination("f_tol", 0.01)

termination = RobustTermination(MultiObjectiveSpaceTermination(tol = 0.4), period=5) #period is the number of generations to consider for the termination


results = minimize(problem,
               algorithm,
               termination)  

print('Time:', results.exec_time)

#%% 
X = results.X
F = results.F

df = pd.DataFrame(np.concatenate((X,F), axis = 1), columns = ['Er', 'Pr', 'DoD', 'LCOS','1-RU'])
df.to_excel('test_NSGAII_LCOS_1-RU_minus10.xlsx')


#%% Display 
coefficients = np.polyfit(df.RU.values, df.EmCost.values, best_polyfit_degree.MyFun(df.RU.values, df.EmCost.values ))

xl, xu = problem.bounds()
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
#%%
plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.scatter(approx_ideal[0], approx_ideal[1], facecolors='none', edgecolors='red', marker="*", s=100, label="Ideal Point (Approx)")
plt.scatter(approx_nadir[0], approx_nadir[1], facecolors='none', edgecolors='black', marker="p", s=100, label="Nadir Point (Approx)")
#plt.plot(np.linspace(df.RU.min(), df.RU.max(),100),np.polyval(coefficients, np.linspace(df.RU.min()-15, df.RU.max()+15,100)), color = 'green',label="PolyFit")

plt.title("Objective Space")
plt.xlabel("RU [-]")
plt.ylabel("Emissions cost [mil€]")
plt.ylim([0,40])
plt.legend(loc = "best")
plt.show()

# %%

df['gamma'] = df.Er/df.Pr

alt.Chart(df, title = "Objective Space").mark_circle().encode(
    alt.X('LCOS').scale(zero=False),
    alt.Y('1-RU').scale(zero=False),
    size = 'gamma',
    color = 'DoD'
)

# alt.Chart(df[df.gamma<=10], title = "Less than 10hrs storage").mark_circle().encode(
#     alt.X('RU').scale(zero=False),
#     alt.Y('EmCost').scale(zero=False),
#     color = 'DoD'
# )
# %%
