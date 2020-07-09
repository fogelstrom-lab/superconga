# SuprConGPU

C++/CUDA library for calculation of physical properties of superconducting systems with confinement.
Below follows a summarized usage guide, see [the documentation](https://suprcongpu.gitlab.io/suprcongpu/) for more information.

## External packages required

SuprConGPU's dependencies are; `CMake` and `Ninja` to build the software,
`CUDA` for utilizing the (NVIDIA) GPU, `LAPACK` and `BLAS` are used for some linear algebra on the CPU, `boost` to manipulate files and directories,
and `FreeGLUT` is used for visualizing the results. The unit tests use `doctest`, and consistent code format is enforced by `clang-format`.

Both `Python2` and `Python3` are used throughout the project, e.g. for post-processing of the data. The Python dependencies are [Matplotlib](https://matplotlib.org/),
[NumPy](https://numpy.org/), [SciPy](https://www.scipy.org/), and [PyQt5](https://pypi.org/project/PyQt5/).

We recommend to use the included [requirements](requirements.txt) file and Python package installer ([pip](https://pip.pypa.io/en/stable/)) to satisfy the dependencies:

```
pip install -r requirements.txt
```

Please note, if you would like to install these dependencies globally (rather than in a local Python virtual environment), you need to have superuser privileges
and run the above command with `sudo -H` prefix (the `-H` flag suppresses warnings from `pip`).

### Arch/Manjaro

Using the AUR helper `yay` the dependencies can be installed with

```
sudo pacman -S blas boost cmake cuda freeglut lapack ninja yay
yay -S doctest clang-format-all-git
```

Furthermore, add the following at the end of your `.bashrc` file

```
export LD_LIBRARY_PATH=/opt/cuda/lib64/:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CPLUS_INCLUDE_PATH=/opt/cuda/include/:/usr/include/doctest/:$CPLUS_INCLUDE_PATH
```

### Ubuntu 18.04LTS

In Ubuntu it is a bit more complicated to setup all dependencies, but most can be install with

```
sudo apt install build-essential freeglut3-dev libblas-dev libboost-dev-all liblapack-dev ninja-build
```

Note that the `CMake` version must be 3.11 or higher. The version in the apt repository in Ubuntu 18.04LTS is unfortunately too old,
but [Kitware](https://apt.kitware.com/) supply the latest version

```
sudo apt-get purge --auto-remove cmake
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
sudo apt-get update
sudo apt-get install cmake
```

The following is a summary of how to install `CUDA`

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pinsudo
mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda
```

Furthermore, add the following at the end of your `.bashrc` file

```
export PATH=/usr/local/cuda-11.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

If the `CUDA` install is unsuccessful please go to the `CUDA` [download page](https://developer.nvidia.com/cuda-downloads) and follow the [official installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

Lastly, install `clang-format` version 9 and `doctest` with

```
sudo apt install clang-format-9 doctest-dev
sudo ln -s /usr/bin/clang-format-9 /usr/bin/clang-format
```

and add the following at the end of your `.bashrc` file

```
export CPLUS_INCLUDE_PATH=/usr/include/doctest:$CPLUS_INCLUDE_PATH
```

## How to compile

Set up the build with the `setup.sh` script (starting from the project root directory):

```
./setup.sh
```

Then compile by simply writing

```
ninja
```

in e.g. the `build/release/` folder.

### Build manually with custom CUDA architecture

By default, CMake tries to autodetect the appropriate CUDA architecture.
One can override this by using the command-line option `-DCUDA_ARCH`,
i.e. go to `build/release/` folder and write

```
cmake -G Ninja -DCMAKE_BUILD_TYPE=build_type -DCUDA_ARCH=XY ../..
```

where `XY` is a two-digit number (for example `30`, `61`, `75`, etc.),
and `build_type` is either `Release` or `Debug`.

## How to run the examples

The folder `examples/` contains a collection of parameter files, one per example,
demonstrating different physical aspects of superconductors that are computable through the SuprConGPU framework.

The examples can be run in two different ways; either call the C++ executable directly, or use the Python wrapper (recommended).
Simply execute the following command in the project root folder

```
python suprcongpu.py -f examples/XXX.json
```

where `XXX.json` should be substituted with any JSON file in the `examples/` folder.
All parameters in the JSON file can be overridden from the terminal, e.g.

```
python suprcongpu.py -f examples/XXX.json -T 0.5
```

will set the temperature to `T=0.5T_c` regardless of the value in the JSON file. If you try to provide invalid parameters,
the Python wrapper will spot this, print what the error is, and exit.

In order to call the C++ executable, navigate to the `build/release/` folder, and execute the following command

```
./bin/main ../../examples/XXX.json
```

Note that there is no support for overriding parameters from the command line when using the C++ executable directly,
nor is any validation of the parameters made.

## Clang-Format

We enforce consistent LLVM style with clang-format version 9. Our settings are found in the file `.clang-format`.
If we want to change style, this is done by writing e.g.

```
clang-format -style=llvm -dump-config > .clang-format
```

to get LLVM format.

## Doctest

Our goal is to test every function/method using Doctest. If `doctest` is to be installed manually (not with a package manager),
it should be in the `/usr/` prefix directory, i.e. its main header file `doctest.h` should be inside `/usr/include/doctest`.

## CTest

CMake allows to aggregate all tests into a test suite. In order to run all tests use

```
ctest [-jN]
```

inside the build folder to run all the tests (`-jN` option allows to run tests in parallel CPU threads).

In order to see the list of tests which can be run use

```
ctest -N
```

while to run a specific test ignoring all other use

```
ctest -R test_name
```
