from __future__ import division, absolute_import, print_function

import time
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from joblib import dump, load

import safepython.EET as EET # module to perform the EET
from safepython.sampling import OAT_sampling, Morris_sampling # module to perform the input
import safepython.plot_functions as pf # module to visualize the results
# sampling
from safepython.util import aggregate_boot # function to aggregate the bootstrap results

from src.functions import model_eval
from src.model import crab_model_ode


def run_sensitivity_analysis(time_ode, T_ode, X0_state, juveniles, adult_fems, thresh):

    """
    Input: - time_ode (array-like): monthly time grid for ODE integration [months]
        - T_ode (array-like): temperature time series over time_ode [°C]
        - X0_state (list of float, len 2): initial state [J0, A0] [#crabs/1000m²]
        - juveniles (array-like): observed juvenile densities [#crabs/1000m²]
        - adult_fems (array-like): observed adult female densities [#crabs/1000m²]
        - thresh (float): threshold fraction of max(μ) used to draw the significance cutoff line in plots

    Output: - results (dict) with keys: - thresh (float): significance threshold as provided in input
                                        - X_labels (list of str): parameter names ['p', 'x', 'alpha', 'b', 'm', 'beta_RR', 'r_T', 'f_l']
                                        - M (int): number of parameters (8)
                                        - lb (list of float): lower bounds of the parameter space
                                        - ub (list of float): upper bounds of the parameter space
                                        - sens_analysis_elapsed_time (int): computation time [minutes]
                                        - mi_J, mi_A (ndarray, shape (M,)): mean of Elementary Effects mi for juveniles and adults
                                        - sigma_J, sigma_A (ndarray, shape (M,)): standard deviation of Elementary Effects sigma for juveniles and adults
                                        - rr (ndarray, shape (5,)): sample sizes used for convergence analysis
                                        - mic_J, mic_A (ndarray, shape (5, M)): mi convergence values at each sample size, for juveniles and adults
                                        - mic_boot_J, mic_boot_A (ndarray, shape (Nboot, 5, M)): bootstrapped mi convergence values for juveniles and adults
                                        - mi_m_J, mi_m_A (ndarray, shape (M,)): bootstrap mean of mi for juveniles and adults
                                        - sigma_m_J, sigma_m_A (ndarray, shape (M,)): bootstrap mean of sigma for juveniles and adults
                                        - mi_lb_J, mi_ub_J, mi_lb_A, mi_ub_A (ndarray, shape (M,)): bootstrap lower and upper confidence bounds on mi for juveniles and adults
                                        - sigma_lb_J, sigma_ub_J, sigma_lb_A, sigma_ub_A (ndarray, shape (M,)): bootstrap lower and upper confidence bounds on sigma for juveniles and adults
                                        - mic_m_J, mic_m_A (ndarray, shape (5, M)): bootstrap mean of mi convergence for juveniles and adults
                                        - mic_lb_J, mic_ub_J, mic_lb_A, mic_ub_A (ndarray, shape (5, M)): bootstrap lower and upper confidence bounds on mi convergence for juveniles and adults
                                        - Y_J, Y_A (ndarray, shape (n_samples,)): MSE values on juveniles and adults across all sampled parameter sets
    
    Performs global sensitivity analysis using the Elementary Effects (Morris) method via SAFEpython. Samples 1000 parameter sets with LHS in a radial design, runs the model, and computes μ and σ indices for both juveniles and adults. Includes bootstrapping (Nboot=100) for confidence intervals and convergence analysis over 5 sample sizes."""
    
    p0 = [30.62/12, 0.01256, 0.9/12, 1, 1/15.7, 0.294/12]
    X0 = np.array([60, 5]+p0)
    X_labels =['p','x',r'$\alpha$','b','m',r'$\beta_{RR}$',r'$r_T$',r'$f_l$']
    # bounds
    lb = list(np.concatenate((np.array([10/X0[0], 1.01/X0[1]]), 0.1*np.ones(len(p0))),axis=0))
    ub = list(np.concatenate((np.array([1,1]), 1.9*np.ones(len(p0))),axis = 0))
    
    M = len(X_labels) # number of inputs ( = parameters)

    # Parameter distributions
    DistrFun = st.uniform # parameter uniform distribution
    
    DistrPar = [np.nan] * M
    for i in range(M):
        DistrPar[i] = [lb[i], ub[i]]
    
    # sample inputs space
    SampStrategy = 'lhs' # Latin Hypercube -> input space uniformely sampled
    r = 1000 # Number of samples 
    design_type ='radial' # or 'trajectory'
    X = OAT_sampling(r,M,DistrFun,DistrPar,SampStrategy,design_type)
    X = X*X0 #element-wise multiplication
    
    # run model
    tic = time.time()
    [Y, X_state_out] = model_eval(X, time_ode, T_ode, X0_state, juveniles, adult_fems)
    Y_J = Y[:,0] # juveniles
    Y_A = Y[:,1] # adults
    toc = time.time()
    
    sens_analysis_elapsed_time = int((toc-tic)/60)
    print('sensitivity analysis elapsed time =', sens_analysis_elapsed_time, 'min')
    
    # Compute Elementary Effects:
    [mi_J, sigma_J,_] = EET.EET_indices(r, lb, ub, X, Y_J, design_type)
    [mi_A, sigma_A,_] = EET.EET_indices(r, lb, ub, X, Y_A, design_type)
    
    # Use bootstrapping to derive confidence bounds:                               
    Nboot = 100
    [mi_boot_J, sigma_boot_J, EE_J] = EET.EET_indices(r, lb, ub, X, Y_J, design_type, Nboot=Nboot)
    mi_m_J, mi_lb_J, mi_ub_J = aggregate_boot(mi_boot_J)
    sigma_m_J, sigma_lb_J, sigma_ub_J = aggregate_boot(sigma_boot_J)  
    [mi_boot_A, sigma_boot_A, EE_A] = EET.EET_indices(r, lb, ub, X, Y_A, design_type, Nboot=Nboot)
    mi_m_A, mi_lb_A, mi_ub_A = aggregate_boot(mi_boot_A)
    sigma_m_A, sigma_lb_A, sigma_ub_A = aggregate_boot(sigma_boot_A)  
    
    # convergence
    rr = np.linspace(r/5, r, 5).astype(int) 
    mic_J, sigmac_J = EET.EET_convergence(EE_J, rr) 
    mic_A, sigmac_A = EET.EET_convergence(EE_A, rr) 
    
    # convergence with bootstrapping
    #juveniles_boot
    mic_boot_J, sigmac_boot_J = EET.EET_convergence(EE_J, rr, Nboot)
    mic_m_J, mic_lb_J, mic_ub_J = aggregate_boot(mic_boot_J) 
    sigmac_m_J, sigmac_lb_J, sigmac_ub_J = aggregate_boot(sigmac_boot_J)
    #adults
    mic_boot_A, sigmac_boot_A = EET.EET_convergence(EE_A, rr, Nboot)
    mic_m_A, mic_lb_A, mic_ub_A = aggregate_boot(mic_boot_A) 
    sigmac_m_A, sigmac_lb_A, sigmac_ub_A = aggregate_boot(sigmac_boot_A)

    # pack results
    results = dict(thresh=thresh,
         X_labels=X_labels,
         M=M,
         ub=ub,
         lb=lb,
         sens_analysis_elapsed_time=sens_analysis_elapsed_time,
         mi_J=mi_J,
         sigma_J=sigma_J,
         mi_A=mi_A,
         sigma_A=sigma_A,
         rr=rr,
         mic_J=mic_J,
         mic_A=mic_A,
         mic_boot_J=mic_boot_J,
         mic_boot_A=mic_boot_A,
         mi_m_J=mi_m_J,
         sigma_m_J=sigma_m_J,
         mi_lb_J=mi_lb_J,
         mi_ub_J=mi_ub_J,
         sigma_lb_J=sigma_lb_J,
         sigma_ub_J=sigma_ub_J,
         mi_m_A=mi_m_A,
         sigma_m_A=sigma_m_A,
         mi_lb_A=mi_lb_A,
         mi_ub_A=mi_ub_A,
         sigma_lb_A=sigma_lb_A,
         sigma_ub_A=sigma_ub_A,
         mic_m_J=mic_m_J,
         mic_lb_J=mic_lb_J,
         mic_ub_J=mic_ub_J,
         mic_m_A=mic_m_A,
         mic_lb_A=mic_lb_A,
         mic_ub_A=mic_ub_A,
         Y_J = Y_J,
         Y_A = Y_A)
        
    return results



def save_sensitivity_analysis(sensitivity_analysis_results,path):

    """
    Input: - sensitivity_analysis_results (dict): output of run_sensitivity_analysis
        - path (str): file path with .joblib extension

    Output: none
    
    Serializes the sensitivity analysis results dictionary to disk using joblib.dump."""
    
    dump(sensitivity_analysis_results,path)
    return