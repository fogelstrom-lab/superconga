#!/bin/bash

# Note, this script is meant to be executed from the root folder of the project.
# ./examples/sweeps/swave_disc_meissner_kappa_sweep.sh

# The configuration file to use.
CONFIG="examples/swave_disc_meissner.json"

# Base path to save the data to.
DATA_PATH="data/swave_disc_meissner_kappa_sweep"

# Large penetration depth. We can be greedy, hence the large step size.
python suprcongpu.py --file $CONFIG -o $DATA_PATH/kInf --save-data --no-visualize -k -1 --opt-step-size 2
python suprcongpu.py --file $CONFIG -o $DATA_PATH/k64 --save-data --no-visualize -k 64 --opt-step-size 2
python suprcongpu.py --file $CONFIG -o $DATA_PATH/k32 --save-data --no-visualize -k 32 --opt-step-size 2

# Medium penetration depth. We need to be less greedy.
python suprcongpu.py --file $CONFIG -o $DATA_PATH/k16 --save-data --no-visualize -k 16 --opt-step-size 1
python suprcongpu.py --file $CONFIG -o $DATA_PATH/k08 --save-data --no-visualize -k 8 --opt-step-size 1

# The penetration depth is small, so we must decrease the step size.
python suprcongpu.py --file $CONFIG -o $DATA_PATH/k04 --save-data --no-visualize -k 4 --opt-step-size 0.2

# The penetration depth is very small. Lets decrease the step size even further, and increase the inertia to avoid wild oscillations. This is tricky though, because too much inertia can cause oscillations.
python suprcongpu.py --file $CONFIG -o $DATA_PATH/k02 --save-data --no-visualize -k 2 --opt-step-size 0.1 --opt-inertia 0.5
