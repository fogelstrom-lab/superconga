#!/bin/bash
# Get current working directory.
pwd=$PWD

# Update sub repos (e.g. armadillo).
echo "============================= Updating submodules ============================="
echo "-----------"
echo "Submodules:"
echo "+ Armadillo"
echo "-----------"
git submodule update --init --recursive

# Build release.
echo "================================ Setup Release ================================"
mkdir -p $pwd/build/release
cd $pwd/build/release
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release $pwd

# Build debug.
echo "================================= Setup Debug ================================="
mkdir -p $pwd/build/debug
cd $pwd/build/debug
cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug $pwd

echo "==============================================================================="
echo "If build generation was successful, build and compile the code by running ninja in:" 
echo "$pwd/build/release"
echo "$pwd/build/debug"
