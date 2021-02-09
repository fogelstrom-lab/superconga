#!/bin/bash

# Note, this script is meant to be executed from the root folder of the project.
# ./examples/sweeps/swave_disc_vortex_kappa_sweep.sh

# The configuration file to use.
CONFIG="examples/swave_disc_vortex.json"

# Base path to save the data to.
DATA_PATH="data/swave_disc_vortex_kappa_sweep"

# The optimizer parameters in the configuration file works pretty well for a wide range of penetration depths.
python suprcongpu.py --file $CONFIG -o $DATA_PATH/kInf --save-data --no-visualize -k -1
python suprcongpu.py --file $CONFIG -o $DATA_PATH/k64 --save-data --no-visualize -k 64
python suprcongpu.py --file $CONFIG -o $DATA_PATH/k32 --save-data --no-visualize -k 32
python suprcongpu.py --file $CONFIG -o $DATA_PATH/k16 --save-data --no-visualize -k 16
python suprcongpu.py --file $CONFIG -o $DATA_PATH/k08 --save-data --no-visualize -k 8

# The penetration depth is small, so we must decrease the step size.
python suprcongpu.py --file $CONFIG -o $DATA_PATH/k04 --save-data --no-visualize -k 4 --opt-step-size 1.5

# The penetration depth is very small. Lets decrease the step size even further.
# Also, the convergence is terribly slow. So lets increase the maximum number of iterations.
python suprcongpu.py --file $CONFIG -o $DATA_PATH/k02 --save-data --no-visualize -k 2 --opt-step-size 0.5 -M 3000
