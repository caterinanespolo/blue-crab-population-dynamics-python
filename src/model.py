import numpy as np

def crab_model_ode(t, X, params, T, tspan):
   
    """
    Input: - t (float): current time point, provided by the ODE solver
        - X (list of float, len 2): state vector [J, A] — juveniles and adult females densities [#crabs/1000m²]
        - params (dict): model parameter dictionary, as returned by pack_params
        - T (array-like): temperature time series [°C], defined over tspan
        - tspan (array-like): time grid corresponding to T [months]
    
    Output: - [dXdt, F] (list): dXdt is a list of two floats [dJdt, dAdt]; F is the fishing catch at time t (float)
    
    ODE wrapper for numerical integration. Interpolates temperature at the current time t from the discrete series T, then calls crab_model to compute the state derivatives."""
    
    u = np.interp(t,tspan, T)
    dXdt = crab_model(X, params, u)
    return dXdt


def  crab_model(X, params, T):

    """
    Input: - X (list of float, len 2): state vector [J, A] — juveniles and adult females densities [#crabs/1000m²]
        - params (dict): model parameter dictionary with keys alpha, b, p, k_max, x, m, r_T, beta_RR, f_l
        - T (float): temperature at the current time step [°C]
    
    Output: - [dXdt, F] (list): dXdt is a list [dJdt, dAdt] (floats); F is the fishing catch (float)
    
    Core ODE system for blue crab population dynamics. Computes recruitment via a temperature-dependent Ricker model, juvenile predation via a type-II functional response, juvenile maturation weighted by a temperature-dependent metabolic rate (resp_rate), linear fishing mortality, and adult mortality. Saturation constraints are applied to ensure non-negativity of all fluxes."""
    
    J = X[0]
    A = X[1] 

    # parameters 
    alpha   = params['alpha']   # maximum per capita reproduction rate [1/months]
    b       = params['b']       # density-dependent effect on reproduction [(1000m^2/#crabs)^2]
    p       = params['p']       # predator density [#crabs/(1000m^2)]
    k_max   = params['k_max']   # maximum feeding rate [1/month]
    x       = params['x']       # prey density at 0.5 k_max [#crabs/(1000m^2)]
    m       = params['m']       # adult mortality rate [1/months]
    r_T     = params['r_T']     # parametro per collegare recruitment a temperatura [1/°C]
    beta_RR = params['beta_RR'] # maturation rate of juveniles to be multiplied by resp_rate [RR/months]
    f_l     = params['f_l']     # linear fishing mortality rate [1/months]

    # these are obtained through fitting made via MATLAB
    a_T =  70.59
    b_T = -5.244
    c_T =  0.108
    resp_rate = 1/(a_T+b_T*T+c_T*T*T)

    # recruitment (ricker)
    R = r_T*alpha*A/(1+b*(A*A))*T

    # juveniles predation
    P = (p+A)*k_max*(J*J)/(x**2+J*J)

    # fishing
    F = f_l*A

    # growth
    G = beta_RR*resp_rate*J

    # adults mortality
    M = m*A

    # juveniles saturation
    total_J = J + R
    P = min(P, total_J)
    G = min(G, total_J - P)

    # adults saturation
    total_A = A + G
    M = min(M, total_A)
    F = min(F, total_A - M)
        
    # ode
    dJdt = R - P - G
    dAdt = G - M - F 

    dXdt = [dJdt, dAdt]
    return [dXdt,F]    