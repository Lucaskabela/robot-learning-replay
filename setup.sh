#!/bin/bash
ENV="conda"
if $ENV=="conda"; then
    conda env create -f environment.yml
    conda activate replay-survey
    pip install gym==0.17.3
    pip install robosuite
else
    pip install -r requirements.txt
fi
