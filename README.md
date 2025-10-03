# Estimation of treatment effects under nonstationarity via the truncated Policy Gradient Estimator

This repository contains three Jupyter notebooks for running the simulations in the paper.

## Data Generation
### Queueing Simulation with Real-world Nonstationary Patient Arrivals.
The simulator in `nonstat_queue.ipynb` is a queueing birth-death chain with inhomogeneous, state-dependent Poisson arrival rates based on a nonstationary estimate of patient arrival rates to an emergency department (both based on day of week, and time of day), obtained using data from the [SEEStat Home Hospital database](https://see-center.iem.technion.ac.il/databases/HomeHospital/). In particular, we calculate the average arrival rates to the emergency department based on records from 2004. 

The notebook `nonstat_queue.ipynb` reads from the file `data0.csv`, which must be generated beforehand. The file should contains four columns (without column names):

| hours | arrival rate | weekday (e.g., Sun, Mon, ...) | weekday index (e.g., Sunday = 0, Monday = 1, ...) |
|-------|---------------|-------------------------------|--------------------------------------------------|
| 0.5   | xxx    | Sun                           | 0                                                |
| 1   | xxx    | Sun                           | 0                                                |
| ...   | ...    | ...                           | ...                                                |
| 24   | xxx    | Sun                           | 0                                                |
| ...   | ...    | ...                           | ...                                                |
| 24   | xxx    | Sat                           | 6                                                |

There are two ways to obtain the required patient arrival data:

1. **Download from the database**  
   The data is available at [SEEStat database](https://see-center.iem.technion.ac.il/databases/HomeHospital/), organized as monthly `.mdb` files.

2. **Access via SEEStat on the Technion SEELab server**  
   SEEStat provides tools for data aggregation (e.g., averaging across months). Access to the Technion SEELab server requires an account, which can be requested at [this registration page](https://see-center.iem.technion.ac.il/terminal-see/). Detailed tutorials on using the [SEEStat Online](https://seelab.net.technion.ac.il/seestat/) platform are also available on their website.


### NYC Ride-sharing Simulation.
The notbook `ride_sharing.ipynb` requires simulation data generated from a large-scale NYC ride-sharing simulator. Detailed instructions for running the simulator can be found in `RideSim.md`. To reproduce the plots from the paper, the following simulation outputs are required before running the notebook:

- Simulations under `pure-A` and `pure-B` configurations (used to compute the GATE).
- Simulations under  `switchback` configuration with interval lengths of 600, 1200, 1800, and 3600 seconds.

## Notebooks

### 1. `2_state_MDP.ipynb`
This notebook simulates a nonstationary 2-state Markov Decision Process (MDP). It includes:
- Simulating the state trajectories
- Evaluating the estimation under the TPG estimator

### 2. `nonstat_queue.ipynb`
This notebook simulates a non-stationary queueing system. It includes:
- Reading data from `data0.csv`
- Simulating the queueing dynamics
- Evaluating the estimation under the TPG estimator and other baseline estimators

### 3. `ride_sharing.ipynb`
This notebook analyzes the NYC ride-sharing simulation results. It includes:
- Reading data from `output/pure-A`, `output/pure-B`, `output/switchback`
- Evaluating the estimation under the TPG estimator and other baseline estimators

## Requirements

To install the necessary Python packages for the notebooks, run:

```bash
pip install -r requirements.txt
```

The additional dependencies needed to run the ride-sharing simulator are listed in `RideSim.md`.


## Acknowledgment

The NYC ride-sharing simulator is adapted from the [dn-ridesharing simulator](https://github.com/atzheng/dn-ridesharing), originally developed by Andrew Zheng (UBC Sauder) and Tianyi Peng (Columbia Business School).
We also gratefully acknowledge the SEE Research Team at the Technion—Israel Institute of Technology for providing the queueing data used in our simulations. 
