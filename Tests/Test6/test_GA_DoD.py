# %% 
import os
import sys
sys.path.append(r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model")
from Core import dispatcher_SOC_pen, dispatcher_dsctd
from Core import microgrid_design
from Core import day_of_year

RES_data_file_path = r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model\InputData\RESData_option-2.csv"

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


#%%
''' Reading the data from csv files'''

P_ren_read = pd.read_csv(RES_data_file_path, header=0, nrows = 8760) #W
P_load_data = pd.DataFrame()
P_load_data['Load [MW]'] = P_ren_read['Power'].mean()*np.ones(len(P_ren_read))/1e6

design = microgrid_design.MG(Pr_BES=17.7, \
                   Er_BES=173, \
                   P_load=P_load_data, \
                   P_ren=P_ren_read, \
                   )




#%%


# x and y data
cyclelife = [170000, 48000, 21050, 11400, 6400, 4150, 3500, 3000, 2700, 2500]
DoD = [10, 20, 30, 40, 50, 60, 65, 70, 75, 80]

# polynomial interpolation of degree 2
n = 6
coefficients = np.polyfit(DoD, cyclelife, n)

class ProblemWrapper(Problem):

    def _evaluate(self, designs, out, *args, **kwargs):
        res = []
        data = pd.DataFrame()

        for x in designs:

            design.Er_BES = x[0]
            design.Pr_BES = x[1]
            design.DoD = x[2]

            design.cyclelife = int(np.polyval(coefficients, x[2]))

            data, _ = dispatcher_SOC_pen.MyFun(design, False)
            LCOS = data['LCOS [€/MWh]'].values[0]
            lifetime_cost = data['Lifetime cost [million euros]'].values[0]
            res.append([LCOS, lifetime_cost])
        
        out['F'] = np.array(res)

#the variables are in order Er_BES, Pr_BES, DoD

problem = ProblemWrapper(n_var=3, n_obj=2, xl=[0.,0.,20.], xu = [2000.,200.,80.], vtype=int)

algorithm = NSGA2(pop_size=2,
                  sampling = IntegerRandomSampling(),
                  eliminate_duplicates=True
                  )

termination = get_termination("n_gen", 1) # | get_termination("tolx", 1) # | get_termination("f_tol", 0.01)

results = minimize(problem,
               algorithm,
               termination)  

print('Time:', results.exec_time)

#%%

X = results.X
F = results.F

pareto_opt_gen = pd.DataFrame({'Capacity [MWh]': results.X[:,0],
                         'Power [MW]': results.X[:,1] ,
                         'DoD [%]': results.X[:,2],
                         'LCOS [€/MWh]': results.F[:,0],
                         'Lifetime cost [mil€]': results.F[:,1]
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
plt.ylabel("Lifetime cost [mil€]")
plt.legend(loc = "best")
plt.show()

# nF = (F - approx_ideal) / (approx_nadir - approx_ideal)

# plt.figure(figsize=(7, 5))
# plt.scatter(nF[:, 0], nF[:, 1], s=30, facecolors='none', edgecolors='blue')
# plt.title("Normalized Objective Space")
# plt.show()
# %%
