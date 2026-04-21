import numpy as np

import time

from joblib import dump, load

import pymoo
from pymoo.core.problem import Problem 
from pymoo.algorithms.moo.nsga2 import NSGA2 
from pymoo.optimize import minimize 
from pymoo.termination import get_termination 
from pymoo.operators.crossover.sbx import SBX 
from pymoo.operators.mutation.pm import PM

import matplotlib.pyplot as plt

from src.functions import simulate_and_compare, pack_params

def run_calibration(time_ode_train, X0_train, juveniles_train, adult_fems_train, T_ode_train, selected_parameters_indices):

    """
    Input: - time_ode_train (array-like): monthly time grid for the training period [months]
        - X0_train (list of float, len 2): initial state [J0, A0] for the training period [#crabs/1000m²]
        - juveniles_train (array-like): observed juvenile densities over the training years [#crabs/1000m²]
        - adult_fems_train (array-like): observed adult female densities over the training years [#crabs/1000m²]
        - T_ode_train (array-like): temperature time series over the training period [°C]
        - selected_parameters_indices (list of int): indices of the parameters to calibrate (0–7)
    
    Output: - results (dict) with keys: - params_est (ndarray): all Pareto-optimal parameter sets
                                        - pareto_front (ndarray): corresponding objective values [mse_J, mse_A]
                                        - distance_from_origin (ndarray): Euclidean distance of each Pareto point from the origin
                                        - opt_index (int): index of the selected optimal solution
                                        - params_opt (ndarray): optimal parameter vector
                                        - opt_cost (ndarray): objective values at the optimal solution
                                        - cost0 (ndarray): objective values at the nominal parameters
                                        - lb (list): lower bounds used for calibration
                                        - ub (list): upper bounds used for calibration
                                        - selected_parameters_indices (list of int): as provided in input
                                        - calibration_elapsed_time (int): computation time [minutes]

    Runs multi-objective calibration using NSGA-II. Only the parameters at selected_parameters_indices are optimized; their bounds are set to ±90% of the nominal values. The optimal solution is the Pareto point with minimum distance from the origin."""
    
    # initial parameters and bounds
    params0 = [20,       # p = predator densityp[#crabs/(1000m^2)]
          3,        # x = prey density at 0.5 k_max [#crabs/(1000m^2)]
          30.62/12, # alpha = maximum per capita reproduction rate [1/months]
          0.01256,  # b = density-dependent effect on reproduction [(1000m^2/#crabs)^2]
          0.9/12,   # m = adult mortality rate [1/months]
          1,        # beta_RR = maturation rate of juveniles to be multiplied by resp_rate [RR/months]
                    # beta = 0.0908 => anche beta_RR*resp_rate deve stare lì intorno => beta_RR = 1
          1/15.7,   # r_T = parametro per collegare recruitment a temperatura [1/°C]
                    # tra 1/T_max e 1/T_min (T_max = 29.3°C, T_min = 0.8°C, mean(T_ode) = 15.7°C)
          0.294/12] # f_l, linear fishing mortality rate [1/months]
    params_labels =['p','x',r'$\alpha$','b','m',r'$\beta_{RR}$',r'$r_T$',r'$f_l$']
    lb = params0.copy()
    ub = params0.copy()
    for i in selected_parameters_indices:
        lb[i] = 0.1*params0[i]
        ub[i] = 1.9*params0[i]
    cost0 = simulate_and_compare(pack_params(params0), time_ode_train, X0_train, juveniles_train, adult_fems_train, T_ode_train)
    
    # problem definition 
    class CrabProblem(Problem):
        def __init__(self, lb, ub):
            super().__init__(
                n_var = len(ub),
                n_obj = 2,
                n_constr = 0,
                xl = np.array(lb),
                xu = np.array(ub))
    
        def _evaluate(self, params_population, out, *args, **kwargs):
            results = []
    
            for p in params_population:
                params = pack_params(p)
                f = simulate_and_compare(params, time_ode_train, X0_train, juveniles_train, adult_fems_train, T_ode_train)
                results.append(f)
    
            out["F"] = np.array(results)
    
    # algorithm definition
    algorithm = NSGA2(pop_size = 100, crossover = SBX(prob=0.7), mutation = PM(prob=1))
    termination = ("n_gen", 200) 
    
    # optimization
    problem = CrabProblem(lb, ub)
    
    tic = time.time()
    res = minimize(problem, algorithm, termination, seed=1, verbose=True)
    toc = time.time()
    calibration_elapsed_time = int((toc-tic)/60)
    print('calibration elapsed time =',calibration_elapsed_time,'min')
    
    params_est = res.X # pareto solutions (estimated parameters)
    pareto_front = res.F # pareto front (objective values outputs)
    distance_from_origin = np.linalg.norm(pareto_front,axis=1) # distance from origin of each estimated parameter on pareto front
    opt_index = np.argmin(distance_from_origin)
    params_opt = params_est[opt_index]
    opt_cost = pareto_front[opt_index]
    
    results = dict(params_est = params_est,
             pareto_front = pareto_front,
             distance_from_origin = distance_from_origin,
             opt_index = opt_index,
             params_opt = params_opt,
             opt_cost = opt_cost,
             cost0 = cost0,
             ub = ub,
             lb = lb,
             selected_parameters_indices = selected_parameters_indices,
             calibration_elapsed_time = calibration_elapsed_time)
    return results



def save_calibration(calibration_results, path): 

    """
    Input: - calibration_results (dict): output of run_calibration
        - path (str): file path with .joblib extension

    Output: none
    
    Serializes the calibration results dictionary to disk using joblib.dump.""" 
    
    dump(calibration_results,path)
    return