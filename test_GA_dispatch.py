# %% 
import microgrid_design
import dispatcher

import time

from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import sqrt




#%%
''' Reading the data from csv files'''
P_load_data = pd.read_excel('InputData/load_data.xlsx', sheet_name='Yearly Load', header=0) #MW
P_ren_read = pd.read_csv('InputData/RESData_option-2.csv', header=0, nrows = 8760) #W

RES_fac = 7

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

design = microgrid_design.MG(Pr_BES=17.7, \
                   Er_BES=173, \
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

data, _ = dispatcher.MyFun(design, False)


#%%

class ProblemWrapper(Problem):

    def _evaluate(self, designs, out, *args, **kwargs):
        res = []
        data = pd.DataFrame()

        for x in designs:
            design.Pr_BES = x[1]
            design.Er_BES = x[0]
            data, _ = dispatcher.MyFun(design, False)
            LCOS = data['LCOS [€/MWh]'].values[0]
            fuel_cons = data['Fuel Consumption [L]'].values[0]
            res.append([LCOS, fuel_cons])
        
        out['F'] = np.array(res)

prob = ProblemWrapper(n_var=2, n_obj=2, xl=[0.,0.], xu = [2000.,116.])

algo = NSGA2(pop_size=1)

stop_criterium = ('n_gen',1)

results = minimize(
    problem=prob,
    algorithm=algo,
    termination=stop_criterium
)


X = results.X
F = results.F

pareto_opt_gen = pd.DataFrame({'Capacity [MWh]': results.X[:,0],
                         'Power [MW]': results.X[:,1] ,
                         'LCOS [€/MWh]': results.F[:,0],
                         'Fuel [L]': results.F[:,1]
                        })

pareto_opt_gen.to_excel('pareto_GA_60p_20g.xlsx')

xl, xu = prob.bounds()
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
plt.ylabel("Fuel consumption [L]")
plt.legend(loc = "best")
plt.show()

# nF = (F - approx_ideal) / (approx_nadir - approx_ideal)

# plt.figure(figsize=(7, 5))
# plt.scatter(nF[:, 0], nF[:, 1], s=30, facecolors='none', edgecolors='blue')
# plt.title("Normalized Objective Space")
# plt.show()
# %%
