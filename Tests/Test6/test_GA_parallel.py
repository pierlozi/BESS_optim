#%%
import os
import sys
sys.path.append(r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model")
from Core import dispatcher_GA, dispatcher_SOC_pen
from Core import microgrid_design
from Core import day_of_year

RES_data_file_path = r"C:\Users\SEPILOS\OneDrive - ABB\Documents\Projects\Model\InputData\RESData_option-2.csv"


from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.core.problem import ElementwiseProblem


from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.termination import get_termination

from pymoo.core.problem import StarmapParallelization

from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd

#I am parallelizing the code in a thread pool of as many threads as CPU cores there are
#pool = ThreadPool(os.cpu_count()) 
#n_threads = os.cpu_count()
pool = ThreadPool()

class MyProblem(Problem):

    def __init__(self, design, **kwargs):
        self.design = design  # store the microgrid design object as an attribute 
        super().__init__(n_var=3, n_obj=2, xl=[0.,0.,20.], xu = [2000.,200.,80.], vtype=int, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        #just prints of X components to see what's inside
        # print('X[:,0]:', X[:,0])
        # print('X[:,1]:', X[:,1])
        # print('X[:,2]:', X[:,2])

        # prepare the parameters for the pool
        params = [(self.design, False, X[k, 0], X[k, 1], X[k, 2]) for k in range(len(X))]
        #print('params:', params)

        F = pool.starmap(dispatcher_GA.MyFun, params)
        #print('F:', F)
        # store the function values and return them.
        out["F"] = np.array(F)

    # def __init__(self, **kwargs):
    #     super().__init__(n_var=3, n_obj=2, xl= -5, xu = 5, **kwargs)

    # def _evaluate(self, X, out, *args, **kwargs):

    #     # define the function
    #     def my_eval(x1, x2, x3):
    #         f1 = x1 ** 2 + x2 ** 2
    #         f2 = (x1 - 1) ** 2 + x3 ** 2
    #         return (f1, f2)

    #     '''prepare the parameters for the pool'''
    #     # params is a 'pop-size' long list of n_var long tuples. 
    #     # Each tuple carries the trial values for the N variables of the problem.
    #     # One tuple = one member of the population of the current generation
    #     # X is an array with 'pop-size' rows and N columns containing the same dat of params
    #     params = [(X[k, 0], X[k, 1], X[k, 2]) for k in range(len(X))]

    #     '''calculate the function values in a parallelized manner and wait until done'''
    #     # 'pool' is used to execute multiple threads of a function in parallel
    #     # 'starmap' maps a function to an iterable, and then applies the function to 
    #     #    each element in the iterable in parallel using the thread pool
    #     # F contains the objective function values for each parameter tuple
    #     # F is a 'pop-size' long list of n_obj long tuples
    #     F = pool.starmap(my_eval, params)

    #     '''store the function values and return them'''
    #     # out is a dictionary of which "F" is a key
    #     out["F"] = np.array(F)

''' Reading the data from csv files'''

P_ren_read = pd.read_csv(RES_data_file_path, header=0, nrows = 8760) #W
P_load_data = pd.DataFrame()
P_load_data['Load [MW]'] = P_ren_read['Power'].mean()*np.ones(len(P_ren_read))/1e6

problem = MyProblem(design = microgrid_design.MG(P_load=P_load_data, \
                                                 P_ren=P_ren_read, \
                                                )
                    )



algorithm = NSGA2(pop_size=2,
                  sampling = IntegerRandomSampling(),
                  eliminate_duplicates=True
                  )


results = minimize(problem, 
               algorithm,
               termination=("n_gen", 1), 
               seed=1
               )

print('Execution Time:', results.exec_time)
pool.close()

# %%
