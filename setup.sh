#!/bin/bash
# Get current working directory.
pwd=$PWD

# Allow user to let CMake decide build system.
if echo $* | grep -e "--no-ninja" -q
then
  echo "-- Using build system: automatic via CMake."
  GENERATOR=""
else
  echo "-- Using build system: Ninja. Run 'setup.sh --no-ninja' to let CMake decide instead."
  GENERATOR="-G Ninja"
fi

# Allow building without visualization.
if echo $* | grep -e "--no-visualize" -q
then
  echo "-- Building without visualization support."
  VISUALIZE="-DCONGA_VISUALIZE=OFF"
else
  echo "-- Building with visualization support."
  VISUALIZE="-DCONGA_VISUALIZE=ON"
fi

# Use CUDA-OpenGL interopability in visualization.
if echo $* | grep -e "--use-cuda-opengl-interop" -q
then
  echo "-- Building with CUDA-OpenGL interopability."
  CUDA_OPENGL_INTEROP="-DCONGA_USE_CUDA_OPENGL_INTEROP=ON"
else
  echo "-- Building without CUDA-OpenGL interopability."
  CUDA_OPENGL_INTEROP="-DCONGA_USE_CUDA_OPENGL_INTEROP=OFF"
fi

# Allow building without HDF5 support.
if echo $* | grep -e "--no-hdf5" -q
then
  echo "-- Building without HDF5 support."
  HDF5="-DCONGA_USE_HDF5=OFF"
else
  echo "-- Building with HDF5 support."
  HDF5="-DCONGA_USE_HDF5=ON"
fi

# Allow skip building tests.
if echo $* | grep -e "--no-tests" -q
then
  echo "-- Not building tests."
  TESTS="-DCONGA_BUILD_TESTS=OFF"
else
  echo "-- Building tests."
  TESTS="-DCONGA_BUILD_TESTS=ON"
fi

# Create folder for temporary files, e.g. config for running SuperConga.
mkdir -p $pwd/tmp

# Remove build folder.
rm -rf build/

# Build Debug, if specified by the user, otherwise Release.
if echo $* | grep -e "--debug" -q
then
  echo "================================= Setup Debug ================================="
  mkdir -p $pwd/build/debug
  cd $pwd/build/debug
  cmake $GENERATOR $VISUALIZE $CUDA_OPENGL_INTEROP $HDF5 $TESTS -DCMAKE_BUILD_TYPE=Debug $pwd
else
  echo "================================ Setup Release ================================"
  mkdir -p $pwd/build/release
  cd $pwd/build/release
  cmake $GENERATOR $VISUALIZE $CUDA_OPENGL_INTEROP $HDF5 $TESTS -DCMAKE_BUILD_TYPE=Release $pwd
fi

echo "==============================================================================="
echo "-- If build generation was successful, build and compile the code with compile.sh,"
echo "   or by manually running the build system (e.g. ninja, make, ...) in:" 
if echo $* | grep -e "--debug" -q
then
  echo "   $pwd/build/debug"
else
  echo "   $pwd/build/release"
fi
