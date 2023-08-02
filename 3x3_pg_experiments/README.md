# Corso 3x3 policy gradient experiments
This set of experiments originates as a project for the course *Autonomous and Adaptive Systems* held at University of Bologna.

This directory contains all the artifacts obtained from the experiments, and the specific scripts used to do so. In particular:
* `vreinforce/`, `wbaseline/`, `ac/`, `ppo/` contain the artifacts relative to their algorithm. That is:
    * The pretrained models for each different tested seed. Model directories are in the form `policy_net_<seed>` and `value_net_<seed>`.
    * All the related `tensorboard` logs (`runs/` directory).
    * A parquet dataset (`runs.parquet`) containing a tabularized summary of all the runs.
    * Figures generated for the final report (`*.svg`).
* `figure_scripts/` contains the scripts used to generate all the figures. These are rough and undocumented due to time constraints, in the end it is nothing more than some data manipulation in order to obtain clean figures. `figure_requirements.txt` contains the necessary requirements to run them (`pip install -r figure_requirements.txt`).
* `tournaments/` contains numpy arrays representing the results of tournaments played between a pool of agents and with different seeds.
* `other_figures/`, as suggested by the name, contains other figures not specifically related to an algorithm/agent.
* `seeds.txt` contains the 5 different seeds used for all experiments.
* `train3x3.py` is the driver script which runs the training pipeline with the manually tuned hyperparameters. This version of the script trains the PPO agent, however it is relatively easy to modify it or improve its generalization with a more flexible command line interface. The only requirement to run it is a correct installation of the `corso` library.
