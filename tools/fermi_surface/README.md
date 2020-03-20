# Bulk Fermi surface properties within BdG

It assumes Python3, and requires NumPy to be installed.

## Usage
To see the help run
```
python compute_fermi_surface.py -h
```
When running the script it will produce a json file with the results organized in dicts.
To load the data in Python write
```
import json
file = open(filename, "r")
data = json.load(file)
```
