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

from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd

''' Reading the data from csv files'''

P_ren_read = pd.read_csv(RES_data_file_path, header=0, nrows = 8760) #W
P_load_data = pd.DataFrame()
P_load_data['Load [MW]'] = P_ren_read['Power'].mean()*np.ones(len(P_ren_read))/1e6

#%%

#I am parallelizing the code in a thread pool of as many threads as CPU cores there are
# 'pool' is used to execute multiple threads of a function in parallel
# 'starmap' maps a function to an iterable, and then applies the function to 
#  each element in the iterable in parallel using the thread pool

#n_threads = os.cpu_count()
n_threads = 8
pool = ThreadPool(n_threads)
runner = StarmapParallelization(pool.starmap)

class MyProblem(ElementwiseProblem):

    def __init__(self, design, **kwargs):
        self.design = design  # store the microgrid design object as an attribute 
        super().__init__(n_var=3, n_obj=2, xl=[0.,0.,20.], xu = [2000.,200.,80.], elementwise_evaluation=True, vtype=int, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        #just prints of X components to see what's inside
        # print('X[0]:', X[0])
        # print('X[1]:', X[1])
        # print('X[2]:', X[2])

        # store the function values and return them.
        out["F"] = dispatcher_GA.MyFun(self.design, False, X[0], X[1], X[2])



problem = MyProblem(design = microgrid_design.MG(P_load=P_load_data, \
                                                 P_ren=P_ren_read, \
                                                ),
                    elementwise_runner=runner
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
