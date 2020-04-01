# Python tools for SuprConGPU
A collection of scripts

## External packages required
Python2 and Python3 should already be installed on your system, but we also need virtualenv. On Ubuntu install with
```
sudo apt install virtualenv
```
We use this to enforce the same package versions for the entire team.

## Setup the virtual environment
To create a virtual environment write in the folder you want to create it, i.e. tools:
```
virtualenv -p python3 venv
```
This will create a python3 environment in the folder venv. To activate it write
```
source venv/bin/activate
```
When activated install all our dependencies with
```
pip install -r requirements.txt
```
This only needs to be done once.

## Usage
Remember to activate it with the command above. You can then run the scripts, e.g.
```
python plot_fermi_surface.py
```

To deactivate the virtual environment write:
```
deactivate
```

## Python Libraries
The Python Libraries in `pythonLibraries` contain convenient classes for data post-processing. Please note that the library is only tested with Python2.
In particular, it contains a data IO class for reading grid data and meta data files, a data
structure class for creating conveneint data strucures of the simulation data, and finally,
a plotting class with convenient tools for displaying data and producing publication-quality
plots with matplotlib. For more in-detail information, see the [corresponding README in pythonLibraries/](tools/pythonLibraries/).
