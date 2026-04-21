import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import ode
import matplotlib.pyplot as plt

# sens analysis
import safepython.EET as EET # module to perform the EET
from safepython.sampling import OAT_sampling, Morris_sampling # module to perform the input
import safepython.plot_functions as pf # module to visualize the results

from src.model import crab_model_ode
from src.functions import pack_params


def plot_sensitivity_analysis(sensitivity_analysis_results):

    """
    Input: - sensitivity_analysis_results (dict): output of run_sensitivity_analysis, containing mi, sigma, convergence indices, confidence bounds, labels, and threshold

    Output: none (displays matplotlib figures)
    
    Generates 8 plots for the sensitivity analysis results: mi-sigma scatter plots for juveniles and adults (with and without confidence intervals), and convergence plots of mi over increasing sample sizes (with and without confidence intervals). A vertical dashed red line marks the significance threshold at thresh*max(mi)."""
    sens = sensitivity_analysis_results
    
    X_labels =['p','x',r'$\alpha$','b','m',r'$\beta_{RR}$',r'$r_T$',r'$f_l$']
    
    # results without confidence intervals
    # juveniles
    EET.EET_plot(sens['mi_J'], sens['sigma_J'], X_labels)
    plt.plot([sens['thresh']*max(sens['mi_J']),sens['thresh']*max(sens['mi_J'])],[min(sens['sigma_J']),max(sens['sigma_J'])],'r--')
    plt.title('Juveniles');
    
    # adults
    EET.EET_plot(sens['mi_A'], sens['sigma_A'], X_labels)
    plt.plot([sens['thresh']*max(sens['mi_A']),sens['thresh']*max(sens['mi_A'])],[min(sens['sigma_A']),max(sens['sigma_A'])],'r--')
    plt.title('Adults');
    
    # convergence without confidence intervals
    # juveniles
    plt.figure()
    pf.plot_convergence(sens['mic_J'], sens['rr']*(sens['M']+1), X_Label='# model evaluations',Y_Label='mean of EEs', labelinput=X_labels)
    plt.title('Juveniles');  
    # adults
    plt.figure()
    pf.plot_convergence(sens['mic_A'], sens['rr']*(sens['M']+1), X_Label='# model evaluations',Y_Label='mean of EEs', labelinput=X_labels)
    plt.title('Adults');
    
    # plots with confidence intervals
    #juveniles
    EET.EET_plot(sens['mi_m_J'],sens['sigma_m_J'],X_labels,sens['mi_lb_J'],sens['mi_ub_J'],sens['sigma_lb_J'],sens['sigma_ub_J'])
    plt.plot([sens['thresh']*max(sens['mi_m_J']),sens['thresh']*max(sens['mi_m_J'])],[min(sens['sigma_lb_J']),max(sens['sigma_ub_J'])],'r--')
    plt.title('Juveniles');
    #adults
    EET.EET_plot(sens['mi_m_A'],sens['sigma_m_A'],X_labels,sens['mi_lb_A'],sens['mi_ub_A'],sens['sigma_lb_A'],sens['sigma_ub_A'])   
    plt.plot([sens['thresh']*max(sens['mi_m_A']),sens['thresh']*max(sens['mi_m_A'])],[min(sens['sigma_lb_A']),max(sens['sigma_ub_A'])],'r--')
    plt.title('Adults');
    
    # convergenvce with confidence intervals
    plt.figure()
    pf.plot_convergence(sens['mic_m_J'], sens['rr']*(sens['M']+1), sens['mic_lb_J'], sens['mic_ub_J'], X_Label='# model evaluations', Y_Label='mean of EEs', labelinput=X_labels)
    plt.title('Juveniles');
    plt.figure()
    pf.plot_convergence(sens['mic_m_A'], sens['rr']*(sens['M']+1), sens['mic_lb_A'], sens['mic_ub_A'], X_Label='# model evaluations', Y_Label='mean of EEs', labelinput=X_labels)
    plt.title('Adults');
    
    return



def plot_calibration(calibration_results):

    """
    Input: - calibration_results (dict): output of run_calibration, containing the Pareto front, optimal solution, initial cost, bounds, and selected parameter indices

    Output: none (displays matplotlib figures)
    
    Generates 2 plots: the Pareto front in the (mse_J, mse_A) space with markers for the nominal parameters thrta_0 and the optimal solution theta_opt; and a dot-and-interval plot showing, for each calibrated parameter, its optimal value within its calibration bounds."""

    cal = calibration_results
    
   # pareto front
    fig1, ax1 = plt.subplots()
    ax1.set_title('Pareto front')
    ax1.scatter(cal['pareto_front'][:,0], cal['pareto_front'][:,1], color='tab:pink')
    ax1.set_xlabel(r'$mse_J$')
    ax1.set_ylabel(r'$mse_A$')
    ax1.scatter(cal['cost0'][0], cal['cost0'][1], color='r', marker='*')
    ax1.scatter(cal['opt_cost'][0], cal['opt_cost'][1], color='b', marker='*')
    ax1.legend(['Pareto front', r'$\theta_0$', r'$\theta_{opt}$'])
    
    # parameters intervals
    params_labels = ['p', 'x', r'$\alpha$', 'b', 'm', r'$\beta_{RR}$', r'$r_T$', r'$f_l$']
    colors = ['tab:orange', 'y', 'b', 'r', 'c']
    j = 0
    lgnd = []
    fig2, ax2 = plt.subplots()
    for i in cal['selected_parameters_indices']:
        c = colors[j]
        j += 1
        ax2.scatter(cal['params_opt'][i], -j, color=c)
        ax2.plot([cal['lb'][i], cal['ub'][i]], [-j, -j], color=c)
        lgnd = lgnd + [params_labels[i], '']
    ax2.set_yticks([])
    ax2.legend(lgnd)
    return



def plot_results(T_ode_train, time_ode_train, years_train, X0_train, juveniles_train, adult_fems_train, T_ode_val, time_ode_val, years_val, X0_val, juveniles_val, adult_fems_val, params_opt):

    """
    Input: - T_ode_train, T_ode_val (array-like): temperature time series for training and validation periods [°C]
        - time_ode_train, time_ode_val (array-like): monthly time grids for training and validation periods [months]
        - years_train, years_val (array-like): yearly observation timestamps for training and validation
        - X0_train, X0_val (list of float, len 2): initial states [J0, A0] for each period [#crabs/1000m²]
        - juveniles_train, juveniles_val (array-like): observed juvenile densities [#crabs/1000m²]
        - adult_fems_train, adult_fems_val (array-like): observed adult female densities [#crabs/1000m²]
        - params_opt (list of float, len 8): optimal parameter vector from calibration
       
    Output: none (displays matplotlib figures)
       
    Generates 2 time-series plots (training and validation) comparing observed data against simulated trajectories with both nominal (theta_0) and optimal (theta_opt) parameters, for juveniles and adults separately."""
    
    # initial parameters
    params0 = [20,3,30.62/12, 0.01256, 0.9/12, 1, 1/15.7, 0.294/12]
    p0 = pack_params(params0)
    sol_sim_train0 = solve_ivp(lambda t,X: crab_model_ode(t, X, p0, T_ode_train, time_ode_train)[0], [time_ode_train[0], time_ode_train[-1]], X0_train, method='BDF', t_eval=time_ode_train, dense_output=True)
    X_sim_train0 = sol_sim_train0.y 
    sol_sim_val0 = solve_ivp(lambda t,X: crab_model_ode(t, X, p0, T_ode_val, time_ode_val)[0], [time_ode_val[0], time_ode_val[-1]], X0_val, method='BDF', t_eval=time_ode_val, dense_output=True)
    X_sim_val0 = sol_sim_val0.y 
    
    # optimum parameters
    p_opt = pack_params(params_opt)
    sol_sim_train = solve_ivp(lambda t,X: crab_model_ode(t, X, p_opt, T_ode_train, time_ode_train)[0], [time_ode_train[0], time_ode_train[-1]], X0_train, method='BDF', t_eval=time_ode_train, dense_output=True)
    X_sim_train = sol_sim_train.y 
    sol_sim_val = solve_ivp(lambda t,X: crab_model_ode(t, X, p_opt, T_ode_val, time_ode_val)[0], [time_ode_val[0], time_ode_val[-1]], X0_val, method='BDF', t_eval=time_ode_val, dense_output=True)
    X_sim_val = sol_sim_val.y 
    
    # plots
    
    time_figure_train = np.arange(years_train[0],years_train[-1]+1,1/12)
    time_figure_val = np.arange(years_val[0],years_val[-1]+1,1/12)
    
    fig1 = plt.figure()
    fig1.WindowState = 'maximized'
    plt.scatter(years_train, juveniles_train, color = 'b',marker = 'o')
    plt.plot(time_figure_train, X_sim_train[0], 'b-',linewidth=1)
    plt.plot(time_figure_train, X_sim_train0[0], 'b--',linewidth=1)
    plt.scatter(years_train, adult_fems_train, color = 'r',marker = 'o')
    plt.plot(time_figure_train, X_sim_train[1], 'r-',linewidth=1)
    plt.plot(time_figure_train, X_sim_train0[1], 'r--',linewidth=1)
    plt.legend([r'$J_{obs}$',r'$J_{sim}, \theta_{opt}$',r'$J_{sim}, \theta_0$',r'$A_{obs}$',r'$A_{sim}, \theta_{opt}$',r'$A_{sim}, \theta_0$'])
    plt.ylabel('crabs/1000m^2')
    plt.title('Calibration');
    
    fig2 = plt.figure()
    fig2.WindowState = 'maximized'
    plt.scatter(years_val, juveniles_val, color = 'b',marker = 'o')
    plt.plot(time_figure_val, X_sim_val[0], 'b-',linewidth=1)
    plt.plot(time_figure_val, X_sim_val0[0], 'b--',linewidth=1)
    plt.scatter(years_val, adult_fems_val, color = 'r',marker = 'o')
    plt.plot(time_figure_val, X_sim_val[1], 'r-',linewidth=1)
    plt.plot(time_figure_val, X_sim_val0[1], 'r--',linewidth=1)
    plt.legend([r'$J_{obs}$',r'$J_{sim}, \theta_{opt}$',r'$J_{sim}, \theta_0$',r'$A_{obs}$',r'$A_{sim}, \theta_{opt}$',r'$A_{sim}, \theta_0$'])
    plt.ylabel('crabs/1000m^2')
    plt.title('Test');
    
    return


def barplot_results(T_ode_train, time_ode_train, years_train, X0_train, juveniles_train, adult_fems_train, T_ode_val, time_ode_val, years_val, X0_val, juveniles_val, adult_fems_val, params_opt):

    """
    Input: same as plot_results
    
    Output: none (displays matplotlib figures)
    
    Generates 4 grouped bar charts (juveniles and adults, for training and validation) comparing, year by year, the simulated values with nominal parameters, the observed data, and the simulated values with optimal parameters."""

    # initial parameters
    params0 = [20,3,30.62/12, 0.01256, 0.9/12, 1, 1/15.7, 0.294/12]
    p0 = pack_params(params0)
    sol_sim_train0 = solve_ivp(lambda t,X: crab_model_ode(t, X, p0, T_ode_train, time_ode_train)[0], [time_ode_train[0], time_ode_train[-1]], X0_train, method='BDF', t_eval=time_ode_train, dense_output=True)
    X_sim_train0 = sol_sim_train0.y 
    sol_sim_val0 = solve_ivp(lambda t,X: crab_model_ode(t, X, p0, T_ode_val, time_ode_val)[0], [time_ode_val[0], time_ode_val[-1]], X0_val, method='BDF', t_eval=time_ode_val, dense_output=True)
    X_sim_val0 = sol_sim_val0.y 
    
    # optimum parameters
    p_opt = pack_params(params_opt)
    sol_sim_train = solve_ivp(lambda t,X: crab_model_ode(t, X, p_opt, T_ode_train, time_ode_train)[0], [time_ode_train[0], time_ode_train[-1]], X0_train, method='BDF', t_eval=time_ode_train, dense_output=True)
    X_sim_train = sol_sim_train.y 
    sol_sim_val = solve_ivp(lambda t,X: crab_model_ode(t, X, p_opt, T_ode_val, time_ode_val)[0], [time_ode_val[0], time_ode_val[-1]], X0_val, method='BDF', t_eval=time_ode_val, dense_output=True)
    X_sim_val = sol_sim_val.y 
    
    # barplots
    
    X_figure_train = X_sim_train[:,0:-1:12]
    X_figure_train0 = X_sim_train0[:,0:-1:12]
    
    error_train = X_figure_train - [juveniles_train, adult_fems_train]
    error_train0 = X_figure_train0 - [juveniles_train, adult_fems_train]
    
    # colors
    c1 = "#0072BD" 
    c2 = "#EDB120"  
    c3 = "#D95319" 
    
    fig3, (s1, s2) = plt.subplots(2, 1, figsize=(12, 8))
    # juveniles
    width = 0.25
    x = np.arange(len(years_train))
    s1.bar(x - width, X_figure_train0[0, :], width=width, label='Initial params',color=c1,edgecolor=c1)
    s1.bar(x,         juveniles_train,        width=width, label='Data',color=c2,edgecolor=c2)
    s1.bar(x + width, X_figure_train[0, :],  width=width, label='Calibrated',color=c3,edgecolor=c3)
    s1.set_xticks(x)
    s1.set_xticklabels(years_train)
    s1.legend()
    s1.set_title('Juveniles train');
    # adults
    width = 0.25
    x = np.arange(len(years_train))
    s2.bar(x - width, X_figure_train0[1, :], width=width, label='Initial params',color=c1,edgecolor=c1)
    s2.bar(x,         adult_fems_train,        width=width, label='Data',color=c2,edgecolor=c2)
    s2.bar(x + width, X_figure_train[1, :],  width=width, label='Calibrated',color=c3,edgecolor=c3)
    s2.set_xticks(x)
    s2.set_xticklabels(years_train)
    s2.legend()
    s2.set_title('Adults train');
    
    X_figure_val = X_sim_val[:,0:-1:12]
    X_figure_val0 = X_sim_val0[:,0:-1:12]
    
    error_val = X_figure_val - [juveniles_val, adult_fems_val]
    error_val0 = X_figure_val0 - [juveniles_val, adult_fems_val]
    
    fig4, (s1, s2) = plt.subplots(2, 1, figsize=(12, 8))
    # juveniles
    width = 0.25
    x = np.arange(len(years_val))
    s1.bar(x - width, X_figure_val0[0, :], width=width, label='Initial params',color=c1,edgecolor=c1)
    s1.bar(x,         juveniles_val,       width=width, label='Data',color=c2,edgecolor=c2)
    s1.bar(x + width, X_figure_val[0, :],  width=width, label='Calibrated',color=c3,edgecolor=c3)
    s1.set_xticks(x)
    s1.set_xticklabels(years_val)
    s1.legend()
    s1.set_title('Juveniles test');
    # adults
    width = 0.25
    x = np.arange(len(years_val))
    s2.bar(x - width, X_figure_val0[1, :], width=width, label='Initial params',color=c1,edgecolor=c1)
    s2.bar(x,         adult_fems_val,      width=width, label='Data',color=c2,edgecolor=c2)
    s2.bar(x + width, X_figure_val[1, :],  width=width, label='Calibrated',color=c3,edgecolor=c3)
    s2.set_xticks(x)
    s2.set_xticklabels(years_val)
    s2.legend()
    s2.set_title('Adults test');
    
    return