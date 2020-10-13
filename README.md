# robot-learning-replay
A survey of methods in experience replay applied to robot learning

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## Prerequisites

This project uses conda to manage packages and requirements - however pip can also be used.

+ [conda](https://docs.anaconda.com/anaconda/install/)

+ [pip](https://pip.pypa.io/en/stable/installing/)

+ [mujoco](http://www.mujoco.org/)

See `setup.sh` to get started and install requirements for the project.  
Note, conda will not have gym installed - this must be done through pip

## TODO

 - [x] Fill in Boilerplate code for training, models
 - [x] Get environments setup and random policy working
 - [x] Scope out future work for project
 - [x] Get SAC model set up
 - [ ] Get experience replay base versions (HER + PER)
 - [ ] Run some baselines
 
 ## Expiremental notes

 - Read literature, decide on sizes/hyperparameters to use, how many trails (3 - 5)
 - Random seeds: 1, 42, 169, 0, 405
 - Decide ER hyperparameters to use 
 - Look into any modifications (?)


 ## Weird windows thing: 
 PATH="~/opt/bin${PATH:+:${PATH}}"
