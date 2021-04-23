#!/bin/bash

# Note, this script is meant to be executed from the root folder of the project.
# ./examples/parameter_sweeps/swave_disc_meissner_kappa_sweep.sh

# The configuration file to use.
CONFIG="examples/swave_disc_meissner/config.json"

# Base path to save the data to.
DATA_PATH="data/swave_disc_meissner_kappa_sweep"

python suprcongpu.py --file $CONFIG -o $DATA_PATH/kInf --save-data --no-visualize -k -1
python suprcongpu.py --file $CONFIG -o $DATA_PATH/k64 --save-data --no-visualize -k 64
python suprcongpu.py --file $CONFIG -o $DATA_PATH/k32 --save-data --no-visualize -k 32
python suprcongpu.py --file $CONFIG -o $DATA_PATH/k16 --save-data --no-visualize -k 16
python suprcongpu.py --file $CONFIG -o $DATA_PATH/k08 --save-data --no-visualize -k 8
python suprcongpu.py --file $CONFIG -o $DATA_PATH/k04 --save-data --no-visualize -k 4 
python suprcongpu.py --file $CONFIG -o $DATA_PATH/k02 --save-data --no-visualize -k 2
