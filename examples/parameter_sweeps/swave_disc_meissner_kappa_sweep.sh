#!/bin/bash

# Note, this script is meant to be executed from the root folder of the project.
# ./examples/parameter_sweeps/swave_disc_meissner_kappa_sweep.sh

# The configuration file to use.
CONFIG="examples/swave_disc_meissner/config.json"

# Base path to save the data to.
DATA_PATH="data/examples/swave_disc_meissner_kappa_sweep"

python superconga.py --file $CONFIG -S $DATA_PATH/kInf  --no-visualize -k -1
python superconga.py --file $CONFIG -S $DATA_PATH/k64 -L $DATA_PATH/kInf --no-visualize -k 64
python superconga.py --file $CONFIG -S $DATA_PATH/k32 -L $DATA_PATH/k64 --no-visualize -k 32
python superconga.py --file $CONFIG -S $DATA_PATH/k16 -L $DATA_PATH/k32 --no-visualize -k 16
python superconga.py --file $CONFIG -S $DATA_PATH/k08 -L $DATA_PATH/k16 --no-visualize -k 8
python superconga.py --file $CONFIG -S $DATA_PATH/k04 -L $DATA_PATH/k08 --no-visualize -k 4 
python superconga.py --file $CONFIG -S $DATA_PATH/k02 -L $DATA_PATH/k04 --no-visualize -k 2
