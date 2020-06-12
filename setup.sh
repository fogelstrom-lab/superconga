#!/bin/bash
# Get current working directory.
pwd=$PWD

# Build release.
echo "Setup Release"
mkdir -p $pwd/build/release
cd $pwd/build/release
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release $pwd

# Build debug.
echo "Setup Debug"
mkdir -p $pwd/build/debug
cd $pwd/build/debug
cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug $pwd
