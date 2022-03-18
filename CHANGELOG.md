# SuperConga changelog

## Latest version (unreleased)

### Improvements
- **MAJOR** Set energy cutoff instead of the number of energies.
- **MAJOR** Change units so that the energy scale is `2*pi*kB*Tc`, `xi_0 = hbar*vF/(2*pi*kB*Tc)`, `A_0 = Phi_0/(pi*xi_0)`, `B_0 = Phi_0/(pi*xi^2_0)`, and `j_d = 2*pi*kB*Tc*|e|*v_F*N_F` both internally and externally.
- **MAJOR** Group related parameters in the `config.json` file, and in `--help` message.
- **MAJOR** Use ArrayFire-Forge for visualization, support building without visualization, and without tests.
- **MAJOR** Include the residual of the boundary coherence functions in the total residual. Print residual during burn-in. Allow for burn-in until convergence by setting negative number. Allow burn-in in postprocessing.
- **MAJOR** Unified frontend via `superconga.py`. Rewritten plotters and runners.
- Update README.md.
- Speed up the coherence function computation by doing the two half steps in one go.
- Speed up the coherence function computation by computing the homogeneous solutions separately.
- Add option to continue from saved files in `--save-path` folder, and resume from `--load-path`.
- Change the area of the point DOS in `ldos_plotter.py` to a circle.
- Add argument in all Python tools for dumping the config parameters to the terminal.
- Expose discs and regular polygons in the CLI.
- Define units of plotted quantities in `--help` message.
- Update README with new CLI syntax.
- Print free energy using scientific notation.
- Set standard deviation of additive Gaussian noise via `initial_noise_stddev` parameter.
- Put `misc` parameters in the `config.json` file in alphabetical order.
- Add magnetic moment to `results.json`.
- Allow user to let CMake choose build system (via `setup.sh --no-ninja`).
- Add `compile.sh` to let CMake call the build system and compiler.
- Only build/compile Debug if running `setup.sh`/`compile.sh` with `--debug` flag.
- Add new argument `num_iterations_burnin` controlling the number of iterations to run just converging the boundary as a pre-processing step.
- Print total time to terminal, and include the visualization time.
- Compute the magnetic contribution to the free energy in the whole simulation space, not just in the grain.
- Speed up vector potential computation by switching axis of cross-correlation and unrolling for loop.
- Speed up vector potential computation by using `__restrict__` keyword in kernels.
- Support building without HDF5. CSV files will be saved instead.
- Speed up the coherence function computation, and increase the accuracy, by using the mid-point method when stepping.
- Allow multiple vortices as start guess, at user-specified locations, via command-line (`--vortices-...`) or config file (`["physics"]["vortex_lattice"]["..."]`).
- Expose the sign of the charge carrier to the user via `--charge-sign`.
- Support setting accelerator with default settings by just its name. All accelerators are different classes.
- Expose the norm used when computing residuals (L1, L2, LInf) via `--norm`.
- Set the gauge with a string instead of bool. It is set via `--gauge`.
- Make starting positions of vortices per component. Vortices added with the CLI are added to all components.
- Set radius of disc, not diameter.
- Allow setting separate norm used when computing residuals during spectroscopy computation.
- Change default colormap to YlGnBu in plotters. Reverse the colormap used for angles so that blue is always positive, and red negative. Use the same colors in live visualizer.
- Expose the min/max step-size limits in the frontend.
- Increase the maximum number of iterations in the examples.
- Add (large) penetration depth in phase-crystal example.
- Change order of regular-polygon parameters, in the CLI, to alphabetical order.
- Use deterministic initialization for the phase-crystal example.
- Update examples README with new frontend.
- Change initialization of Abrikosov-lattice example in order to get a lower free energy.
- Move installation guide to this repo, i.e. to the main README.
- Improve help messages in frontend.
- Change name of `radial` gauge to `symmetric`. Update simulation help messages.
- Change name of the Adaptive-Ball accelerator to Cognac. Do not use capital letters in the frontend for accelerators or norms.

### Bug fixes
- Fix `runIteration` doing one more iteration than required.
- Pick the intersection with the smallest difference in y-coordinate when applying boundary condition.
- Fix MatPlotLib > 3.5 bug in LDOS plotter.
- Fix different residual for the same data depending on single/double precision.
- Fix conceptual bug linear-response current test.
- Fix sign error in "chiral transformation" in Python frontend.
- Fix plotter for multi-component order parameters.
- Fix bug not including doctest when fetching it.
- Fix live visualization not working when using different GPUs for computation and visualization.
- Fix bug rejecting float parameters in `config.json` if passed as int.
- Fix bug not setting the number of energies per block to all energies if passed a negative number in postprocess. Expose the functionality in the frontend.
- Fix bug not saving convergence data for finite save frequency.
- Undo the internal scaling of the vector potential when printing the range in the live visualization, as well when saving scalar quantities.
- Fix bug in Barzilai-Borwein asserts.
- Fix bug in spatial interpolation when restarting from different resolution.
- Fix bug with finite-difference indices out of bounds.

### Developer changes
- **MAJOR** Rename project to SuperConga and the namespace to `conga`.
- **MAJOR** Add option to save data as `.h5` files.
- Make minmax a non-static member method.
- Add unit test of the current density comparing it with the linear-response expression.
- Use scientific notation in CSV files.
- Remove temperature subdirectory in data path. Save examples to `data/examples`.
- Add tool `old_data_scaler.py` for scaling old data to the new units.
- Move `crystalAxesRotation` to `OrderParameter.h`.
- Rename `Optimizer` to `Accelerator`.
- Add Doxygen comments, removed unused code, shorten method names etc in `OrderParameter.h`.
- Add test of an s-wave annulus using solenoid gauge.
- Add test of an s-wave annulus using radial gauge.
- Add test of an s-wave disc with zero phase winding.
- Remove armadillo as submodule, let CMake take care of it automatically.
- Update JsonCpp.
- Group source files in folders.
- Cleanup Green function computation (Doxygen comments, remove unnecessary code, etc).
- Reduce grid tests spatial dimensions.
- Add test of magnetic moment in an s-wave annulus using radial gauge.
- Remove more unused/broken code, e.g. read/write binary files.
- Update Python requirements.
- Use the vector potential when initializing boundary.
- Make LDOS residual more strict by computing residual of LDOS instead of DOS. Remove DOS computation.
- Use more advanced integration rules (Boole and Simpson) in the Fermi-surface averages.
- Save the temporary config file when using the frontend in `tmp/` folder. Add `private/` folder for files not meant for git.
- Simplify the coherence function computation by only using linear interpolation.
- Decrease tolerance of DOS test.
- Add explaining comments to the current linear-response test.
- Use centered finite differences everywhere.
- Increase step size in s-wave annulus tests.
- Remove order parameter initial test.
- Use JsonCpp's own styled string when writing json files.
- Rename inertia to drag in the Polyak accelerator, and change signs in accelerator expressions to conform with the paper.
- Shorten Abrikosov lattice example to `swave_abrikosov_lattice`.
- Simplify the Barzilai-Borwein accelerator.
- Improve the Adaptive-Ball acccelerator. Optimize the inertia and change default parameters.
- Use nearest-neighbour extrapolation when extending values at the boundary.
- Increase step-size limits in Adaptive-Ball accelerator.
- Add range checks for all dimensions in all CUDA kernels. Tune the number of GPU threads to be faster on larger simulations.
- Fix most debug build warnings.
- Add comments and more asserts to accelerators. Use double precision when summing an entire grid.
- Move `make_plot_row function` from common to simulation plotter.
- Create plot-type enum for strictly negative quantities in frontend.
- Add unit test of Green functions.
- Add unit test of vector-potential gauges.

---

## Version v0.7.0 (July 9, 2021)
This version is a collection of smaller fixes and cleanup.

### Improvements
- Support running Python scripts from other directories.
- Showcase a wider array of accelerator settings in the examples.
- Standardize messages to the user.
- Improve descriptions for command-line arguments for frontends (`--help` flag). 
- Update README.md.
- Add option to run with single precision via the Python frontend.
- Print the physical width in the live visualization using the external definition of the coherence length, i.e. with 2pi in the denominator.
- Use the same colormap, RdBu, for e.g. the induced magnetic field in both Python and C++.

### Bug fixes
- Fix error in `FermiSurfaceCircle.h` due to missing include.

### Developer changes
- Remove the Python frontend GUI.
- Write to convergence.csv every iteration.
- Read support for deprecated argument `bounding_disc_diameter` for `regular_polygon`.
- Hide spam from internal `doctest` dependencies.
- Improved formatting for output from `setup.sh` and CMake.

---

## Version v0.6.0 (May 29, 2021)
This version exposes more functionality in the frontend, extends the number of examples and tests, and significantly cleans up the file structure and functionality.

### Improvements
- **MAJOR** Fully expose geometry to user in config files, remove from CLI/GUI
- **MAJOR** Deprecate the GUI
- Add `CHANGELOG.md`
- Add `CONTRIBUTING.md` (contribution guide)
- Add `CODE_OF_CONDUCT.md`
- Add API documentation (generated by doxygen).
- Update release numbers
- Update framework description in README
- Add chiral d-wave example
- Add chiral-transformation option to plotter
- Specify physical size by setting polygon side-length rather than bounding circle diameter
- Various updates to examples
- Increase resolution in vortex examples, to avoid LODS artefacts caused by too low resolution (oscillations and split zero-energy peak in core).
- Deprecate `plotting.json` configuration files for plotter.
- Add showcase to README

### Bug fixes
- Fix bug in average flux density
- Fix bug in free energy related to summing multi-component order parameters
- Fix plotter for multi-component order parameters

### Developer changes
- Add test of Riccati solutions
- Add test of geometry area (disc and square)
- Add test of discretization (`gridElementSize`)
- Deprecate custom class for complex numbers (`cplx`) and switch to thrust::complex
- Add test of the induced flux-density
- Add doxygen headers to all files

---

## Version v0.5.0 (April 13, 2021)

This release significantly improves the self-consistency optimizer and adds the possibility to make it adaptive - automatically tuning the optimization parameters. This makes the code significantly faster and more robust at the same time. The release also adds a number of new tools and convenience features for the user.

### Improvements
- **MAJOR** Make self-consistency optimizer adaptive
- **MAJOR** Add Python frontend for LDOS calculation (post-processing)
- Expose gauge choice to user
- Enable starting from existing vector potential
- Update LDOS smearing in all examples
- Add labels describing what is plotted in the live visualization
- Add slider for point-contact radius in interactive LDOS plotter
- Set residual of free energy to zero in first iteration
- Enable starting simulation from data with lower resolution
- Enable masking inside/outside the geometry in Python plotter
- Save magnetic free-energy density to `results.json`
- Re-add single-letter CLI flags for geometry.
- Improve terminal output for LDOS plotter
- Expose accuracy of the Green-function approximation

### Bug fixes
- Fix crash when saving convergence status to file
- Fix bugs related to the slider for the point-contact radius in the interactive LDOS plotter
- Fix bug in detecting frontend CLI flags
- Fix CMake bug in choosing GPU architecture when multiple GPUs are present
- Fix off-center disc-geometry bug

### Developer changes
- Make circular Fermi surface default
- Only initialize context once
- Add test for the cplx class
- Improved management of unique pointers
- Cleanup format of config files
- Make DiscGeometry member disc a non-pointer
- Move scaling of lengths to functions (i.e. scaling between physical coordinate system and internal coordinate system)
- Cleanup DOS- and current-tests
- Move test executables to unique folder `tests/`

---

## Version v0.4.0 (March 04, 2021)
This release significantly improves the robustness and accuracy of the Riccati equation solver, by allowing the trajectory solver to take "half-steps". This eliminates a number of unwanted artifacts. The release makes it possible to start from previous simulations, and adds a number of post-processing tools: an LDOS calculator and interactive plotter, and a plotter for 2D data. A number of other features are added.

### Improvements
- **MAJOR** Enable taking half-steps in the Riccati equation
- **MAJOR** Add Python post-processing tool for visualizing generated data
- **MAJOR** Add C++/CUDA post-processing code for calculating the LDOS
- **MAJOR** Add interactive spectroscopy tool/LDOS plotter (Python)
- **MAJOR** Add support from starting from previous simulations
- Save simulation configuration and geometry mask at start of simulation
- Make the elementary charge a constant, for consistent sign convention
- Save area-averaged quantities to `results.json`
- Save magnetic free-energy density also separately from free energy
- Significantly cleanup of terminal output, add debug flag for additional output
- Add option for the geometry of the solenoid hole
- Various small improvements to the examples
- Update `README.md` with user instructions
- Update `requirements.txt`
- Add penetration depth sweep examples
- Expose more optimization parameters: `anderson_always`, `min_mixing`, `max_mixing` and `replace_oldest`

### Bug fixes
- Fix so that points per coherence length is exactly true
- Fix broken Python runner
- Fix `replace_worst` json bug in user-config management

### Developer changes
- Cleanup Python frontend and move functionality to Python library
- Remove legacy Python code
- Use the L2 norm in the optimizer
- Change some copy assignments to reference assignments in GeometryGroup, csv_writers and legacy_test
- Improved pointer handling in GridData (do not use a pointer to thrust vector)
- Improve save interval for output data
- Cleanup and "hide" external dependencies in `.ext/`
- Clear vector in ComputeVectorPotential in initialization

---

## Version v0.3.0 (February 09, 2021)
This release (re-)adds the ability to save data to file. This includes configuration parameters, internal parameters, area-averaged quantities, spatially resolved observables/quantities. The release also generalizes the self-consistency optimizer and makes it possible for the user to chose the optimization strategy. A change is made from CUDA 10/C++14 to CUDA 11/C++17.

### Improvements
- **MAJOR** Re-add the ability to save data to file, as human-readable `.csv` and `.json` files ()
- **MAJOR** Generalize the optimizer and expose to the user, making it possible to chose between different classes of optimizers
- **MAJOR** Change from CUDA 10/C++14 to CUDA 11/C++17
- Add a custom bulk order-parameter solver
- Update README with basic usage instructions
- Update colormaps in live visualization
- Update legacy Python code to Python 3
- Make it more intuitive to set the physical size of the system
- Various updates to examples

### Bug fixes
- Fix GUI bug for lattice sites per coherence length
- Fix bug with incorrect induced vector potential
- Fix geometry error by doing retro-reflection at corners
- Fix initialization order bug in Keldysh integrator
- Fix interpolation bug in DiscGeometry

### Developer changes
- Add current conservation test
- Update Lapack/BLAS/OpenBLAS dependencies
- Make armadillo a subrepository
- Subtract the mean of the current density before computing the vector potential
- Add venv to gitignore
- Change ComputeVectorPotential into a class
- Move computation of the induced vector potential to context
- Use template alias instead of typedef for GridData
- Use std::filesystem to handle paths and create directories, instead of outdated C-code
- Compute, and use, the zero-rotation domain labels
- Fix Cppcheck warnings in Fermi surface code
- Separate basis functions for circular, and tight-binding, Fermi surfaces
- Cleanup OrderParameterInitial
- Improve boundary interpolation
- Add data folder to gitignore

---

## Version v0.2.0 (November 12, 2020)
This point-release is a major cleanup of the legacy code, adding numerous new functionality and improvements. This release is used in the [BdG comparison repository](https://gitlab.com/superconga/research/bdg-comparison), which was used in the following paper:

- N\. Wall Wennerdal, A. Ask, P. Holmvall, T. Löfwander, and M. Fogelström, "[Breaking time-reversal and translational symmetry at edges of d-wave superconductors: microscopic theory and comparison with quasiclassical theory](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.043198)", Physical Review Research **2**, 043198 (2020).

---

## Version v0.1.0 (November 07, 2019)

- **Initial release.**
