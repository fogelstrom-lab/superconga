# SuprConGPU
C++/CUDA library for calculation of physical properties of superconducting systems with confinement. 
Below follows a summarized usage guide, see [the documentation](https://suprcongpu.gitlab.io/suprcongpu/) for more information.

## External packages required
Here goes the list of external libraries and packages required to compile and run the examples:
- Lapack
- BLAS
- gfortran
- OpenGL
- GLUT
- Ninja
- CMake

## How to compile
Set up the build with the following commands:
```
mkdir build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
```
where the `DCMAKE_BUILD_TYPE` argument is either `Debug` or `Release`. Then compile by simply writing
```
ninja
```
in the `build/` folder.

### Build with custom CUDA architecture
By default, CMake tries to autodetect the appropriate CUDA architecture. 
One can override this by using the command-line option `-DCUDA_ARCH`, i.e.
```
cmake -G Ninja -DCMAKE_BUILD_TYPE=build_type -DCUDA_ARCH=XY ..
```
where `XY` is a two-digit number (for example `30`, `61`, `75`, etc.).

## How to run the examples

The folder `examples/` contains a collection of parameter files, one per example, 
demonstrating different physical aspects of superconductors that are computable through the SuprConGPU framework.

To run an example, navigate to to the `build/` folder, and execute the following commands
```
ninja examples/main
./examples/main -f ../examples/XXX.json
```
where `XXX.json` should be substituted with any JSON file in the `examples/` folder.

## Clang-Format
We enforce consistent LLVM style with clang-format version 9. On Ubuntu install with
```
sudo apt install clang-format-9
sudo ln -s /usr/bin/clang-format-9 /usr/bin/clang-format
```
Our settings are found in the file `.clang-format`.
If we want to change style, this is done by writing e.g.
```
clang-format -style=llvm -dump-config > .clang-format
```
to get LLVM format.

## Doctest

Our goal is to test every function/method using Doctest. On Ubuntu install with
```
sudo apt install doctest-dev
```
If `doctest` is to be installed manually (not with a package manager), it should be in the `/usr/` prefix directory, 
i.e. its main header file `doctest.h` should be inside `/usr/include/doctest`.

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
