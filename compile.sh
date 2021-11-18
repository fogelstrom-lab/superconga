#!/bin/bash
# Get current working directory.
pwd=$PWD

# Get path to build.
build_dir_release=$pwd/build/release
build_dir_debug=$pwd/build/debug

# Get path to build output file from CMake.
cmake_cache_file_release=$build_dir_release/CMakeCache.txt
cmake_cache_file_debug=$build_dir_debug/CMakeCache.txt

# Check if release build directory and CMake output exists.
if [[ -d "$build_dir_release" ]] && [[ -f "$cmake_cache_file_release" ]]
then
    # Build release.
    echo "Running build system: Release"
    cd $build_dir_release
    cmake --build .
else
    # No build folder found.
    echo "Could not find build/release/CMakeCache.txt. Make sure to run CMake"
    echo "before attempting to compile (e.g. with setup.sh)."
fi

# Debug is optional, check if user is running with --debug.
if echo $* | grep -e "--debug" -q
then
    # Check if debug build directory and CMake output exists.
    if [[ -d "$build_dir_debug" ]] && [[ -f "$cmake_cache_file_debug" ]]
    then
        # Build release.
        echo "Running build system: Debug"
        cd $build_dir_debug
        cmake --build .
    else
        # No build folder found.
        echo "Could not find build/debug/CMakeCache.txt. Make sure to run CMake"
        echo "before attempting to compile (e.g. with 'setup.sh --debug')."
    fi
else
    echo "Skipping Debug. To build Debug, run 'compile.sh --debug'"
fi