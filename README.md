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
