# SuperConga

[Niclas Wall Wennerdal](https://gitlab.com/niclas_wall_wennerdal), [Patric Holmvall](https://gitlab.com/Holmvall), [Oleksii Shevtsov](https://gitlab.com/oshevtsov), [Pascal Stadler](https://gitlab.com/pascalsta), [Mikael Håkansson](https://www.linkedin.com/in/mikael-h%C3%A5kansson-07549a1)

SuperConga is an open-source framework for simulating the physics of superconducting and condensed-matter systems, specifically 2D mesoscopic grains in equilibrium, by using the quasiclassical theory of superconductivity. The core is written in C++ and CUDA, exploiting the parallel nature quasiclassics and utilizing the computational power of modern GPUs.

**Backend:** The framework self-consistently solves for the superconducting order-parameter and the magnetic vector potential, and computes the the current density, magnetic induction, free energy, as well as the local density of states. The simulations can be visualized in real-time with OpenGL.

**Frontend:** A user-friendly Python frontend is provided, enabling simulation parameters to be defined via intuitive configuration files, or via command-line interface (CLI), without requiring a deep understanding of implementation details. Furthermore, the framework ships with simple and interactive tools for visualization and data analysis, making SuperConga an ideal tool both for research but also for educational purposes. SuperConga is available to use under the license agreements stated in LICENSE.txt, and it can be used on all Unix-based platforms.

![](examples/showcase.png)

## Download
The source can be obtained by [cloning](https://docs.gitlab.com/ee/gitlab-basics/start-using-git.html#:~:text=Clone%20a%20repository&text=You%20can%20either%20clone%20it,paste%20in%20your%20command%20line.) the repository through the above interface.


## Installation
For detailed instructions on installation and compilation, see: [https://superconga.gitlab.io/superconga-doc/install.html](https://superconga.gitlab.io/superconga-doc/install.html)


## Documentation
A user manual with [tutorials](https://superconga.gitlab.io/superconga-doc/tutorials.html), [examples](https://superconga.gitlab.io/superconga-doc/guides.html) and additional information can be found in the [official documentation](https://superconga.gitlab.io/superconga-doc/). This documentation is built from the [documentation repository](https://gitlab.com/superconga/superconga-doc). API documentation can be found [here](https://superconga.gitlab.io/superconga/).


## Citing SuperConga
If you have used SuperConga in your research or found it useful, please [cite it](https://superconga.gitlab.io/superconga-doc/cite.html). Please let us know about your research in turn, as we find any application of SuperConga of great interest, and want to include your usage of it in our future outreaches.


## Basic usage
The code is most easily run via its Python frontend `superconga.py`. The subcommands available are shown by running
```shell
python superconga.py --help
```
Instructions are also available at the [tutorial](https://superconga.gitlab.io/superconga-doc/tutorials.html) in the online documentation.

In order to run a simulation the subcommand `simulate` is used. It is highly recommended to use the `--config` argument to specify the location of an existing configuration file which sets all simulation parameters, e.g. the Abrikosov-lattice example:
```shell
python superconga.py simulate --config examples/swave_abrikosov_lattice/simulation_config.json
```
These example files can easily be copied/modified with a text editor, to create new simulations of other systems. To see further usage instructions, use the `-h` or `--help` flag
```shell
python superconga.py simulate --help
```


## Data post-processing and visualization
By default the examples save data at the end of the simulation. It is written to the path specified by the `save_path` option in `simulation_config.json` (which can also be overridden by the command-line argument `--save-path`). Whatever path was used, it will be printed to the terminal upon successfully finishing the simulation. The data is saved as either CSV or hdf5 format, which can be set by the argument `--data-format`. Convergence data is always saved as CSV, and scalar quantities, e.g. averages, as JSON. To plot the simulation data, in e.g. `data/examples/swave_abrikosov_lattice/`, use the `plot-simulation` subcommand and the `--load-path` argument:
```shell
python superconga.py plot-simulation --load-path data/examples/swave_abrikosov_lattice
```
To see further usage instructions, run `python superconga.py plot-simulation --help`. How the simulation converged can be plotted using the `plot-convergence` subcommand
```shell
python superconga.py plot-convergence --view simulation --load-path data/examples/swave_abrikosov_lattice
```

After a simulation has converged the local density of states can be computed by running
```shell
python superconga.py postprocess --config examples/swave_abrikosov_lattice/postprocess_config.json
```
which will generate more data files in `data/examples/swave_abrikosov_lattice/`. The local density of states can then be visualized using the subcommand `plot-postprocess`
```shell
python superconga.py plot-postprocess --load-path data/examples/swave_abrikosov_lattice
```
Note that the plots are interactive. By clicking in either plot the data being visualized will be updated. How the postprocess converged can be plotted using the `plot-convergence` subcommand
```shell
python superconga.py plot-convergence --view postprocess --load-path data/examples/swave_abrikosov_lattice
```


## Contribute
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](CODE_OF_CONDUCT.md)

We encourage any useful contributions to the development of SuperConga, by forking this repository and sending merge requests, sharing bug reports or suggestions via the [issues page](https://gitlab.com/superconga/superconga/-/issues), or contacting us with any questions and discussions.

All contributions are acknowledged in the [contributors section](https://superconga.gitlab.io/superconga-doc/contributors.html). For more information, including technical advice, please see [Contributing to SuperConga development](https://superconga.gitlab.io/superconga-doc/contribute.html) in the online documentation. Please note that any contribution must adhere to our [code of conduct](CODE_OF_CONDUCT.md).


## License
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

You are free to use this software, with or without modification, provided that the conditions listed in the LICENSE.txt file are satisfied.


## Support
SuperConga development is supported by the Applied Quantum Physics Laboratory at the Department of Microtechnology and Nanoscience - MC2, Chalmers University of Technology. We acknowledge the Swedish Research Council (VR) for financial support of the doctoral and post-doctoral projects during which a considerable part of the development has been carried out. Computational resources have been provided by Chalmers Centre for Computational Science and Engineering provided by the Swedish National Infrastructure for Computing.


## Authors
The current lead developers are Niclas Wall Wennerdal, Patric Holmvall and Pascal Stadler. Previous developers are Mikael Håkansson and Oleksii Shevtsov. The project is supervised by Mikael Fogelström and Tomas Löfwander.
