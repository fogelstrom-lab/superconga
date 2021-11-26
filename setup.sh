#!/bin/bash
# Get current working directory.
pwd=$PWD

# Allow user to let CMake decide build system.
if echo $* | grep -e "--no-ninja" -q
then
  echo "-- Using build system: automatic via CMake."
  generator_flag=""
else
  echo "-- Using build system: Ninja. Run 'setup.sh --no-ninja' to let CMake decide instead."
  generator_flag="-G Ninja"
fi

# Allow building without visualization.
if echo $* | grep -e "--no-visualize" -q
then
  echo "-- Building without visualization support."
  visualize_flag="-DCONGA_VISUALIZE=OFF"
else
  echo "-- Building with visualization support."
  visualize_flag="-DCONGA_VISUALIZE=ON"
fi

# Allow skip building tests.
if echo $* | grep -e "--no-tests" -q
then
  echo "-- Not building tests."
  tests_flag="-DCONGA_BUILD_TESTS=OFF"
else
  echo "-- Building tests."
  tests_flag="-DCONGA_BUILD_TESTS=ON"
fi

# Remove build folder.
rm -rf build/

# Build release.
echo "================================ Setup Release ================================"
mkdir -p $pwd/build/release
cd $pwd/build/release
cmake $generator_flag $visualize_flag $tests_flag -DCMAKE_BUILD_TYPE=Release $pwd

# Build debug, if specified by the user.
if echo $* | grep -e "--debug" -q
then
  echo "================================= Setup Debug ================================="
  mkdir -p $pwd/build/debug
  cd $pwd/build/debug
  cmake $generator_flag $visualize_flag $tests_flag -DCMAKE_BUILD_TYPE=Debug $pwd
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
