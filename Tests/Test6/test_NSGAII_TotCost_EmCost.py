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
                        P_ren=P_ren_read, \
                        )

        for x in designs:

            design.Er_BES = x[0]
            design.Pr_BES = x[1]
            design.DoD = x[2]

            data, data_time = dispatcher_dsctd.MyFun(design, False)

            design.cyclelife, _ = rain_deg_funct.MyFun(SOC_profile = data_time['SOC'].values)

            _ , _ , TotCost = LCOS_funct.MyFun(design, \
                                    capex = data['BES CAPEX [million euros]'].values[0]*1e6, \
                                    opex = data['BES OPEX [million euros]'].values[0]*1e6,\
                                    E_dch = sum(data_time['P_dch']),\
                                    res_val_bin = True
                                    )

            emissions_cost = data['Emissions Cost [million euros]'].values[0]
            res.append([TotCost/1e6, emissions_cost]) #Total Cost is also in million euros
        
        out['F'] = np.array(res)

#the variables are in order Er_BES, Pr_BES, DoD

problem = ProblemWrapper(n_var=3, n_obj=2, xl=[0.,0.,20.], xu = [2000.,200.,80.], vtype=int)

algorithm = NSGA2(pop_size=1,
                  sampling = IntegerRandomSampling(),
                  eliminate_duplicates=True
                  )

termination = get_termination("n_gen", 1) # | get_termination("tolx", 1) # | get_termination("f_tol", 0.01)

#termination = RobustTermination(MultiObjectiveSpaceTermination(tol = 0.5), period=5) #period is the number of generations to consider for the termination


results = minimize(problem,
               algorithm,
               termination)  

print('Time:', results.exec_time)

#%%

X = results.X
F = results.F

df = pd.DataFrame(np.concatenate((X,F), axis = 1), columns = ['Er', 'Pr', 'DoD', 'LCOS','EmCost'])
# df.to_excel('res_GA_LCOS_rnflw_10pop_10gen.xlsx')

coefficients = np.polyfit(df.LCOS.values, df.EmCost.values, best_polyfit_degree.MyFun(df.LCOS.values, df.EmCost.values ))


pareto_opt_gen = pd.DataFrame({'Capacity [MWh]': results.X[:,0],
                         'Power [MW]': results.X[:,1] ,
                         'DoD [%]': results.X[:,2],
                         'LCOS [€/MWh]': results.F[:,0],
                         'Emissions Costs [million €]': results.F[:,1]
                        })


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


# %%
