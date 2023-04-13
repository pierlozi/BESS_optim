from pymoo.core.problem import ElementwiseProblem

class MyProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        super().__init__(n_var=10, n_obj=1, n_ieq_constr=0, xl=-5, xu=5, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
         out["F"] = (x ** 2).sum()

from multiprocessing.pool import ThreadPool
from pymoo.core.problem import StarmapParallelization
from pymoo.algorithms.soo.nonconvex.ga import GA 
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


# initialize the thread pool and create the runner
n_threads = 8
pool = ThreadPool(n_threads)
runner = StarmapParallelization(pool.starmap)

# define the problem by passing the starmap interface of the thread pool
problem = MyProblem(elementwise_runner=runner)

res = minimize(problem, NSGA2(), termination=("n_gen", 200), seed=1)
print('Threads:', res.exec_time)

pool.close()