# Python frontend

This folder contains the Python source files for the SuperConga frontend. In other words, it contains the implementation of the functionality in `superconga.py`. Each method of the frontend (e.g. `simulate` and `postprocess`) is implemented as a unique Python package. Consequently, this folder consists of the following packages:

* **`common`** - Functionality which is overlapping and used by several of the other packages, like command-line interface, plotting routines and file I/O.
* **`convergence_plotter`** - Everything related to the convergence plotter, invoked via `python superconga.py plot-convergence ...`.
* **`data_converter`** - Everything related to the data converter, used to convert data between formats, invoked via `python superconga.py convert ...`.
* **`postprocess`** - Python routines for setting up and calling the postprocessing tool, used to calculate e.g. the LDOS, which is invoked via `python superconga.py postprocess ...`
* **`postprocess_plotter`** - Implementation of the interactive spectroscopy tool (LDOS plotter), invoked via `python superconga.py plot-postprocess ...`.
* **`simulation`** - Implementation of routines responsible for setting up a new simulation, and calling the backend that carries out the simulation. Invoked via `python superconga.py simulate ...`.
* **`simulation_plotter`** - Functionality for using matplotlib to plot the results of a simulation, invoked via `python superconga.py plot-simulation ...`.

Each of the above packages is split into a number of Python modules, see the contents of the corresponding folder. Please note that a help flag has been added to each of the methods/subcommands, i.e. `python superconga.py {simulate,postprocess,plot-simulation,plot-postprocess,plot-convergence,convert} --help`. See these help commands for more information.

## Using the modules externally
Please note that the modularity of the Python packages makes it possible to import and use their functionality in your own Python scripts. This is useful for example when creating your own customized plotting scripts for publication. To do so, add something like this to your Python script:
```python
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "frontend"))
from common import io
```

The line which starts with `sys.path.insert(...)` specifies the path to the `frontend/` directory, hence the above example assumes that the Python script is in the root directory of SuperConga. The final line imports the module `io` from the package `common`. The following code will then read the configuration files and raw data from a simulation located in the folder `path/to/my/data/`: 
```python
# Specify data path.
data_path = "path/to/my/data/"

# Read simulation configuration and postprocessing configuration.
simulation_config = io.read_simulation_config(data_path)
postprocess_config = io.read_postprocess_config(data_path)

# Print the contents of the postprocessing configuration.
import pprint
pprint.pprint(postprocess_config)

# Print the temperature used in the simulation.
print(simulation_config["physics"]["temperature"])

# Read current density, vector potential and LDOS from file.
current_density = io.read_data(data_path, "current_density")
vector_potential = io.read_data(data_path, "vector_potential")
ldos = io.read_data(data_path, "ldos")
```
