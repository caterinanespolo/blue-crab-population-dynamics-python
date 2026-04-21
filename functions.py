import numpy as np
import math
from scipy.integrate import solve_ivp

from src.model import crab_model_ode


def make_X_yearly(X,n):

    """
    Input: - X (ndarray, shape (2, n_months)): monthly ODE solution with rows [J, A]
        - n (int): number of yearly observations to return

    Output: - X_yearly (ndarray, shape (n, 2)): yearly aggregated values for juveniles and adults
    
    Converts the monthly ODE solution into yearly values comparable with observed data, which are collected between December and March. The first year is the mean of the first 3 months; subsequent years are the mean of the December–March window of each year."""
    
    X_yearly = np.zeros((n,2))
    X = np.transpose(X)
    X_yearly[0,:] = np.mean([X[0,:],X[1,:],X[2,:]],0)
    temp1 = [X[11:-2:12,0],X[12:-1:12,0],X[13:-1:12,0],X[14:-1:12,0]]
    temp2 = [X[11:-2:12,1],X[12:-1:12,1],X[13:-1:12,1],X[14:-1:12,1]]
    X_yearly[1:,0] = np.mean(temp1,0)
    X_yearly[1:,1] = np.mean(temp2,0)
    return X_yearly


def pack_params(params):

    """
    Input: - params (list of float, len 8): parameter vector [p, x, alpha, b, m, beta_RR, r_T, f_l]

    Output: - paramStruct (dict): parameter dictionary with keys p, x, k_max, alpha, b, m, beta_RR, r_T, f_l
    
    Builds the model parameter dictionary from a flat numeric vector. Derives k_max (maximum feeding rate [1/month]) analytically from p and x. The output is the format expected by crab_model."""
        
    p = params[0]
    x = params[1]
    paramStruct = {'p': p, # predator density [#crabs/(1000m^2)]
                'x': x, # prey density at 0.5 k_max [#crabs/(1000m^2)]
                'k_max': 34.5/math.log(x**2+52900/(x)**2)/12, # maximum feeding rate [1/months]
                'alpha': params[2], # maximum per capita reproduction rate [1/months]
                'b': params[3], # density-dependent effect on reproduction [(1000m^2/#crabs)^2]
                'm': params[4], # adult mortality rate [1/months]
                'beta_RR': params[5], # maturation rate of juveniles to be multiplied by resp_rate [RR/months]
                'r_T': params[6], # parametro per collegare recruitment a temperatura [1/°C]
                'f_l' : params[7]} # linear fishing mortality rate [1/months]
    return paramStruct


def model_eval(X, tspan, T, X0_state, J_obs, A_obs):

    """
    Input: - X (ndarray, shape (n_samples, 8)): population of parameter vectors to evaluate
        - tspan (array-like): monthly time grid for ODE integration [months]
        - T (array-like): temperature time series over tspan [°C]
        - X0_state (list of float, len 2): initial state [J0, A0] [#crabs/1000m²]
        - J_obs (array-like): observed juvenile densities [#crabs/1000m²]
        - A_obs (array-like): observed adult female densities [#crabs/1000m²]
    
    Output: - Y (ndarray, shape (n_samples, 2)): MSE values [mse_J, mse_A] for each parameter set
        - X_state (ndarray): ODE solution of the last evaluated parameter set
    
    Evaluates the model over a population of parameter vectors. Used in sensitivity analysis. For each set, integrates the ODE, aggregates the solution to yearly values, and computes MSE against observations."""

    Y = np.zeros((len(X),2))
    X_state = np.zeros((len(X),len(tspan),2))
    for k in range(0,len(X)):
        params = pack_params(X[k])
        sol = solve_ivp(lambda t,X_state: crab_model_ode(t, X_state, params, T, tspan)[0], [tspan[0],tspan[-1]], X0_state, method='BDF', t_eval = tspan, dense_output=True)
        X_state = sol.y 
        X_state_yearly = make_X_yearly(X_state,len(J_obs))
        mse = [np.mean((X_state_yearly[:,0] - J_obs)**2), np.mean((X_state_yearly[:,1] - A_obs)**2)] # mean squared error
        Y[k,0] = mse[0]
        Y[k,1] = mse[1]
    return Y, X_state


def simulate_and_compare(params, tspan, X0, J_obs, A_obs, T):

    """ 
    Input: - params (dict): model parameter dictionary, as returned by pack_params
        - tspan (array-like): monthly time grid for ODE integration [months]
        - X0 (list of float, len 2): initial state [J0, A0] [#crabs/1000m²]
        - J_obs (array-like): observed juvenile densities [#crabs/1000m²]
        - A_obs (array-like): observed adult female densities [#crabs/1000m²]
        - T (array-like): temperature time series over tspan [°C]
    
    Output: - mse (ndarray, shape (2,)): [mse_J, mse_A] — mean squared errors on juveniles and adults
    
    Objective function for calibration. Integrates the ODE model with the given parameters, aggregates the solution to yearly values, and returns the MSE on both state variables against observations."""
    
    sol = solve_ivp(lambda t,X: crab_model_ode(t, X, params, T, tspan)[0], [tspan[0],tspan[-1]], X0, method='BDF', t_eval = tspan, dense_output=True)
    X = sol.y
    X_state_yearly = make_X_yearly(X,len(J_obs))
    mse = np.array([np.mean((X_state_yearly[:,0] - J_obs)**2), 
                    np.mean((X_state_yearly[:,1] - A_obs)**2)]) # mean squared errors
    return mse