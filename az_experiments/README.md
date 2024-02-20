# Corso AlphaZero-like experiments
This set of experiments originates as a project work for the *Deep Learning* module, held at University of Bologna. See the complete report: [Matering the game of Corso without human knowledge](Mastering_the_game_of_Corso_without_human_knowledge.pdf).

This directory contains all the artifacts obtained from the experiments, and the specific scripts used to do so. In particular:
* `5x5_64x4/` contain the artifacts relative to their algorithm. That is:
    * The pretrained model `az_network_5x5_64x4`.
    * All the related `tensorboard` logs (`runs/` directory).
    * A parquet dataset (`runs.parquet`) containing a tabularized summary of all the runs.
    * Figures generated for the final report (`*.svg`).
* `figure_scripts/` contains the scripts used to generate all the figures. These are rough and undocumented due to time constraints, in the end it is nothing more than some data manipulation in order to obtain clean figures. `figure_requirements.txt` contains the necessary requirements to run them (`pip install -r figure_requirements.txt`).
* `tournaments/` contains numpy arrays representing the results of tournaments played between a pool of agents and with different seeds.
* `other_figures/`, as suggested by the name, contains other figures not specifically related to an algorithm/agent.
* `seeds.txt` contains the only seed used during training. NOTE: reproducibility is not yet properly implemented.
* `train5x5_64x4.py` is the driver script which runs the training pipeline with the manually tuned hyperparameters. The only requirement to run it is a correct installation of the `corso` library.
