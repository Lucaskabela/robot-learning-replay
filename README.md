# robot-learning-replay
A survey of methods in experience replay applied to robot learning

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## Prerequisites

This project uses conda to manage packages and requirements - however pip can also be used.

+ [conda](https://docs.anaconda.com/anaconda/install/)

+ [pip](https://pip.pypa.io/en/stable/installing/)

See `setup.sh` to get started and install requirements for the project.  
Note, conda will not have gym installed - this must be done through pip

## TODO

 - [ ] Fill in Boilerplate code for training, models
 - [ ] Get environments setup and random policy working
 - [ ] Scope out future work for project

 ## Using in Colab
import os
os.environ['USER'] = <name>
os.environ['PASS'] = <password>
os.environ['REPO'] = <repo>
!git clone https://$USER:$PASS@github.com/$USER/$REPO.git