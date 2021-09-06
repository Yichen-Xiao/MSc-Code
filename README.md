# Sequential Vaccine Allocation Algorithm

This code was written for an MSc dissertation, and more details about the purpose of the code can be found within.

## File descriptions
initialise_graph.py - Functions to create and visualise the graphs for the action set of the vaccine allocation algorithm using networkX. 
weight_pushing.py - Contains functions used in weight pushing. Implements Algorithms 2, 3 and 4 from Vu et al. (2019).
observations.py - Implements function computing the observation graph. Also implements function needed to calculate the input set for Agorithm 2.
exp_oe.py - Implements a modular version of the modified Algorithm 1 from Vu et al. (2019).
wrapper.py - Implements the wrapper algorithm.
disease_model.py - Contains code for the simulation.
estimator.py - Used to compute indirect side-observations.
analysis.py - Code related to graphing results.
