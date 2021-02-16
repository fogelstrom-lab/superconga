# SuprConGPU

[Niclas Wall-Wennerdal](https://gitlab.com/niclas_wall_wennerdal), [Patric Holmvall](https://gitlab.com/Holmvall), [Oleksii Shevtsov](https://gitlab.com/oshevtsov), [Mikael Håkansson](https://www.linkedin.com/in/mikael-h%C3%A5kansson-07549a1)

SuprConGPU is open-source software for simulating the physics of superconducting and condensed-matter systems. The SuprConGPU library uses C++/CUDA for the numerical computations in the backend, and the packages Numpy, SCiPy, Matplotlib and Qt for the frontend. The goal of SuprConGPU is to ...(user-friendly, efficiency, systems/physics that can be covered) . SuprConGPU is largely based on the quasiclassical theory of superconductivity and ...(eilenberger, riccati, green functions) to ...(trajectories, boundary conditions, Anderson). SuprConGPU is avaiable to use under the license agreements stated in LICENSE.txt, and it can be used on all Unix-based platforms. This makes SuprConGPU an ideal tool to simulate superconductivity, both for research but also for educational purposes.


## Download
The source can be obtained by [cloning](https://docs.gitlab.com/ee/gitlab-basics/start-using-git.html#:~:text=Clone%20a%20repository&text=You%20can%20either%20clone%20it,paste%20in%20your%20command%20line.) the repository through the above interface.


## Installation
For detailed instructions on installation and compilation, see: [https://suprcongpu.gitlab.io/suprcongpu-doc/install.html](https://suprcongpu.gitlab.io/suprcongpu-doc/install.html)


## Documentation
A user manual with tutorials, examples and additional information can be found in the official documentation: [https://suprcongpu.gitlab.io/suprcongpu-doc/](https://suprcongpu.gitlab.io/suprcongpu-doc/)

This documentation is built from the following development repository: [https://gitlab.com/suprcongpu/suprcongpu-doc](https://gitlab.com/suprcongpu/suprcongpu-doc)


## Basic usage
The code is most easily run via its Python(3) front-end, either in terminal mode
```shell
python suprcongpu.py
```
or with a GUI
```shell
python suprcongpu.py --gui
```
It is highly recommended to use the `--file` argument to specify the location of an existing configuration file which sets all simulation parameters, e.g. the s-wave Abrikosov example:
```shell
python suprcongpu.py --gui --file examples/swave_abrikosov_mesoscopic_vortex_lattice/config.json
```
These example files can easily be copied/modified with a text editor or the GUI, to create new simulations of other systems. To see further usage instructions, use the `-h` or `--help` flag
```shell
python suprcongpu.py --help
```
or refer to the [user manual](https://suprcongpu.gitlab.io/suprcongpu-doc/tutorials.html) in the online documentation.

## Data post-processing and visualization
Simulation data can be saved to file by running the program with the `--save-data` flag
```shell
python suprcongpu.py --save-data --file examples/swave_abrikosov_mesoscopic_vortex_lattice/config.json
```
This saves data in the path specified by the `"data_path"` option in the .json-file (which can also be overriden via command-line with the argument `--data-path`). Whatever data path was used, will be printed to the terminal upon successfully finishing the simulation. The data is saved in two formats: real-space observables and quantities are saved in human-readable `.csv`-files (i.e. tabular Comma-Separated Values), while parameters and and other scalar quantities are generally saved in `.json`-files. These formats are supported by most modern programming languages and post-processing/plotting tools. To facilitate post-processing and plotting, an additional Python program is available, called `plotter.py`, which demonstrates how the data can be parsed from file and plotted. To simply plot all the data in e.g. `data/swave_abrikosov_mesoscopic_vortex_lattice/T0.200000/`, use the `--data-path` argument:
```shell
python plotter.py --data-path data/swave_abrikosov_mesoscopic_vortex_lattice/T0.200000
```
To see further usage instructions, run `python plotter.py --help`, or refer to the [user manual](https://suprcongpu.gitlab.io/suprcongpu-doc/tutorials.html) in the online documentation.


## Contribute
We encourage any useful contributions to the development of SuprConGPU, by forking this repository and sending merge requests, proposing changes via our SuprConGPU discussion group, questions and discussions, or sharing bug reports via the issues page.

All contributions are acknowledged in the [contributors section](https://suprcongpu.gitlab.io/suprcongpu-doc/contributors.html) in the documentation.

Note that all contributions must adhere to the [clang-format style guide](https://suprcongpu.gitlab.io/suprcongpu-doc/contribute.html#clang-format) for C++/CUDA code, and the [PEP 8 -- Style Guide](https://suprcongpu.gitlab.io/suprcongpu-doc/contribute.html#pep8) for Python Code.

For more information, including technical advice, please see [Contributing to SuperConGPU development](https://suprcongpu.gitlab.io/suprcongpu-doc/contribute.html).


## License
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

You are free to use this software, with or without modification, provided that the conditions listed in the LICENSE.txt file are satisfied.


## Support
SuprConGPU development is supported by the Applied Quantum Physics Laboratory at the Department of Microtechnology and Nanoscience - MC2, Chalmers University of Technology. We acknowledge the Swedish Research Council (VR) for financial support of the doctoral and post-doctoral projects during which a considerable part of the development has been carried out. Computational resources have been provided by Chalmers Centre for Computational Science and Engineering provided by the Swedish National Infrastructure for Computing.


## Authors
The current lead developers are Niclas Wall-Wennerdal, Patric Holmvall and Oleksii Shevtsov. Previous developers are Mikael Håkansson. The project is supervised by Mikael Fogelström and Tomas Löfwander.
