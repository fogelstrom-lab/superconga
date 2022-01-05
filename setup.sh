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

# Remove build folder.
rm -rf build/

# Build release.
echo "================================ Setup Release ================================"
mkdir -p $pwd/build/release
cd $pwd/build/release
cmake $GENERATOR $VISUALIZE $HDF5 $TESTS -DCMAKE_BUILD_TYPE=Release $pwd

# Build debug, if specified by the user.
if echo $* | grep -e "--debug" -q
then
  echo "================================= Setup Debug ================================="
  mkdir -p $pwd/build/debug
  cd $pwd/build/debug
  cmake $GENERATOR $VISUALIZE $HDF5 $TESTS -DCMAKE_BUILD_TYPE=Debug $pwd
else
  echo "================================= Setup Debug ================================="
  echo "-- Skipping debug. Run 'setup.sh --debug' to build in debug mode."
fi

echo "==============================================================================="
echo "-- If build generation was successful, build and compile the code with compile.sh,"
echo "   or by manually running the build system (e.g. ninja, make, ...) in:" 
echo "   $pwd/build/release"

if echo $* | grep -e "--debug" -q
then
  echo "$pwd/build/debug"
fi
