# blue-crab-population-dynamics

A Python-based simulation and analysis framework for modelling the population dynamics of the blue crab (*Callinectes sapidus*) in response to environmental and biological drivers.

This project was developed as part of the MSc thesis *"Dynamic Modelling and Simulation of Blue Crab (Callinectes Sapidus) Populations"*, University of Padova, 2024–2025.

---

## Overview

The blue crab (*Callinectes sapidus*) is an invasive species causing significant ecological and economic impact in Mediterranean coastal ecosystems. This project implements a two-compartment ODE model describing the temporal evolution of juvenile and adult female crab densities, driven by water temperature, predation, fishing pressure, and density-dependent reproduction.

The framework includes:
- A mechanistic ODE model of juvenile and adult population dynamics
- Multi-objective parameter calibration via genetic algorithm (NSGA-II)
- Global sensitivity analysis via the Elementary Effects (Morris) method
- Visualization tools for model fit, calibration results, and sensitivity indices

---

## Repository Structure

```
blue-crab-population-dynamics/
│
├── src/
│   ├── model.py                  # ODE system: crab population dynamics
│   ├── functions.py              # ODE integration, objective function, parameter packing
│   ├── calibration.py            # NSGA-II multi-objective calibration
│   ├── sensitivity_analysis.py   # Elementary Effects sensitivity analysis
│   └── visualization.py          # Plotting functions
│
├── data_processing.ipynb         # Data loading and preprocessing
├── main.ipynb                    # Main pipeline: calibration, sensitivity analysis, visualization
│
├── raw_data/
│   ├── maryland_data.xlsx                                              # Observed crab density data (Maryland DNR)
│   ├── tos.nwa.full.hcast.monthly.regrid.r20230520.199301-201912.nc   # Sea surface temperature reanalysis (1993–2019)
│   └── tos.nwa.full.hcast.monthly.regrid.r20250715.199301-202312.nc   # Sea surface temperature reanalysis (1993–2023)
│
├── data/
│   ├── data.csv                  # Processed crab density data ready for modelling
│   └── temperature.npy           # Processed monthly temperature time series [°C]
│
├── outputs/
│   ├── calibration_out.joblib          # Saved calibration results (Pareto front, optimal parameters)
│   └── sensitivity_analysis_out.joblib # Saved sensitivity analysis results (EE indices, confidence bounds)
│
├── requirements.txt
└── README.md
```

---

## Model Description

The model tracks two state variables:
- **J** — juvenile crab density [#crabs/1000m²]
- **A** — adult female crab density [#crabs/1000m²]

The ODE system includes the following biological processes:

| Process | Description |
|---|---|
| Recruitment | Temperature-dependent Ricker model |
| Predation | Type-II functional response (juveniles) |
| Maturation | Temperature-dependent juvenile-to-adult transition |
| Fishing mortality | Linear mortality on adults |
| Natural mortality | Constant adult mortality rate |

Calibration minimizes the mean squared error between simulated and observed yearly densities (December–March window) jointly on juveniles and adults, using NSGA-II with a population of 100 individuals over 200 generations.

Sensitivity analysis identifies the most influential parameters using the Elementary Effects method with Latin Hypercube Sampling (r=1000, radial design) and bootstrapped confidence intervals (Nboot=100).

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/blue-crab-population-dynamics.git
cd blue-crab-population-dynamics
pip install -r requirements.txt
```

**Main dependencies:**

```
numpy
scipy
matplotlib
pymoo
joblib
safepython
```

---

## Usage

To get started, run `main.ipynb` directly — processed data is already available in `data/` and no preprocessing step is required.

**1. Data preprocessing** *(optional — only needed to reprocess raw data)*
```bash
jupyter notebook data_processing.ipynb
```
Loads raw input data from `raw_data/`, processes the temperature time series and observed crab densities, and saves the processed inputs to `data/`. Skip this step if you are using the data files already provided in `data/`.

**2. Main pipeline**
```bash
jupyter notebook main.ipynb
```
Covers the full modelling workflow:
1. Run sensitivity analysis to identify influential parameters
2. Run NSGA-II calibration on the selected parameters
3. Evaluate and visualize model fit on training and validation sets

Results are saved to `outputs/` as `.joblib` files and can be reloaded for visualization without re-running the full pipeline.

To run calibration and sensitivity analysis independently:

```python
from src.calibration import run_calibration, save_calibration
from src.sensitivity_analysis import run_sensitivity_analysis, save_sensitivity_analysis

# Sensitivity analysis
results_sens = run_sensitivity_analysis(time_ode, T_ode, X0, juveniles, adult_fems, thresh=0.1)
save_sensitivity_analysis(results_sens, "outputs/sensitivity_analysis_out.joblib")

# Calibration
results_cal = run_calibration(time_ode_train, X0_train, juveniles_train, adult_fems_train, T_ode_train, selected_parameters_indices)
save_calibration(results_cal, "outputs/calibration_out.joblib")
```

---

## License

This project is released for academic and research purposes.
