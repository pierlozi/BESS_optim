# %% 
import os
import sys
sys.path.append(r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model")
from Core import dispatcher_dsctd
from Core import microgrid_design
from Core import rain_deg_funct, LCOS_funct

RES_data_file_path = r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model\InputData\RESData_option-2.csv"
load_data_file_path = r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model\InputData\load_data.xlsx"

import numpy as np
import pandas as pd
import time

from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
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

            cyclelife, _ = rain_deg_funct.MyFun(SOC_profile = data_time['SOC'].values)

            LCOS, _ = LCOS_funct.MyFun(Er = data['Er_BES [MWh]'].values[0],\
                                    Pr = data['Pr_BES [MW]'].values[0], \
                                    cyclelife = cyclelife, minelife = design.mine_life, floatlife = design.floatlife,\
                                    DR = design.IR, \
                                    capex = data['BES CAPEX [million euros]'].values[0]*1e6, \
                                    opex = data['BES OPEX [million euros]'].values[0]*1e6,\
                                    E_dch = sum(data_time['P_dch']),\
                                    res_val_bin = True
                                    )

            emissions_cost = data['Emissions Cost [million euros]'].values[0]
            res.append([LCOS, emissions_cost])
        
        out['F'] = np.array(res)

#the variables are in order Er_BES, Pr_BES, DoD

problem = ProblemWrapper(n_var=3, n_obj=2, xl=[0.,0.,20.], xu = [2000.,200.,80.], vtype=int)

algorithm = NSGA2(pop_size=10,
                  sampling = IntegerRandomSampling(),
                  eliminate_duplicates=True
                  )

termination = get_termination("n_gen", 10) # | get_termination("tolx", 1) # | get_termination("f_tol", 0.01)

results = minimize(problem,
               algorithm,
               termination)  

print('Time:', results.exec_time)

#%%

X = results.X
F = results.F

# df = pd.DataFrame(np.concatenate((X,F), axis = 1), columns = ['Er [MWh]', 'Pr [MW]', 'DoD [%]', 'LCOS [€/MWh]','Emissions Cost [million€]' ])
# df.to_excel('res_GA_LCOS_rnflw_10pop_10gen.xlsx')

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

plt.title("Objective Space")
plt.xlabel("LCOS [€/MWh]")
plt.ylabel("Emissions cost [mil€]")
plt.legend(loc = "best")
plt.show()

# nF = (F - approx_ideal) / (approx_nadir - approx_ideal)

# plt.figure(figsize=(7, 5))
# plt.scatter(nF[:, 0], nF[:, 1], s=30, facecolors='none', edgecolors='blue')
# plt.title("Normalized Objective Space")
# plt.show()
# %%
