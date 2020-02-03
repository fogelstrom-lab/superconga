# SuprConGPU
C++/CUDA library for calculation of physical properties of superconducting systems with confinement.

## External packages required
Here goes the list of external libraries and packages required to compile and run the examples:
- Lapack
- BLAS
- gfortran
- boost_system
- boost_filesystem
- OpenGL
- GLUT
- Ninja
- CMake

## How to compile
In order to compile and run the simple example, use the following commands:
```
mkdir build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=build_type ..
ninja
./demo_hexagon_swave_VL
```
where `build_type` is `Debug` or `Release`.

### Build with custom CUDA architecture
By default, CMake tries to autodetect the appropriate CUDA architecture. One can override this by using the command-line option `-DCUDA_ARCH`, i.e.
```
cmake -G Ninja -DCMAKE_BUILD_TYPE=build_type -DCUDA_ARCH=XY ..
```
where `XY` is a two-digit number (for example `30`, `61`, `75`, etc.).

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
