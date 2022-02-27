# Source files
Source files used to compile binaries. Below follows a short summary
of each source file. See [the readme](../) for a usage instructions, or the
[online documentation](https://superconga.gitlab.io/superconga-doc/) for more thorough
instructions.

## main.cu
The main source to run simple examples, either by parsing the `.json`-parameter
files in `examples/`, or by parsing command-line arguments. The binary file is
built to `build/release/bin/main`.

## spectra.cu
A post-processing source to compute the local density of states. Use by running
the frontend `spectra.py` and give the path to a post-processing configuration file,
found e.g. in the folders in `examples/`. See the
[tutorials](https://superconga.gitlab.io/superconga-doc/tutorials.html)
for more instructions. The binary file is built to `build/release/bin/spectra`.
