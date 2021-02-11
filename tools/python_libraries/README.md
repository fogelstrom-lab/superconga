# Python Libraries
This folder contains various Python tools and libraries, which are split into the following packages:
* `file_io` - Parse data files produced by the framework backend (e.g. `.csv` and `.json` files).
* `gui` - Set up an interactive GUI for the frontend.
* `parameter_setup` - Parse and validate user arguments provided via command-line, gui and/or `.json`-configuration files.
* `physics` - Various tools used for physics and observables.
* `plotting` - Plot data produced by the framework, using matplotlib.
* `runner` - Package containing the main functionality of the frontend runner.
* `settings` - Various settings and configuration used throughout the Python library.

These packages usually contain several modules. For example, the package `parameter_setup` contains the following modules:
* `parameter_setup/argument_parsing` - Add command-line arguments to Python runner using the module `argparse`. Parse the arguments, in particular looking for a .json-file containing simulation parameters.
* `parameter_setup/argument_validation` - Validate the simulation parameters (e.g. sanity check data types and values). Another example is the package `physics`, which contains the following module:
* `physics/fermi_surface` - Compute bulk Fermi surface properties based on BdG-theory.

## Using the modules
For usage of the Python library, please see the plotter frontend `plotter.py`. Here follows a short summary of usage. Start by making sure that the calling Python script has a path to the libraries, by using the `sys` Python module:
```python
import sys
sys.path.insert(0, '../../relative/path/to/tools/')
```
where the `tools/` directory lies in the project base folder. 

The modules can then be included with simple import statements, e.g.:
```python
from python_libraries.settings import configuration
```