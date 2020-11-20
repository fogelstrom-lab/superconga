cmake_minimum_required(VERSION 3.18)
project(
  suprcongpu
  VERSION 0.2.0
  LANGUAGES CUDA CXX)

# Output build messages
message(
  STATUS "Building ${CMAKE_PROJECT_NAME} version ${CMAKE_PROJECT_VERSION}.")

###############################################################################
# CTEST AND DOCTEST
###############################################################################
# Enable CTest.
include(CTest)

###############################################################################
# EXTERNAL LIBRARIES
###############################################################################
# So we can download libs at build time.
include(FetchContent)

# Declare the JSON lib.
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/open-source-parsers/jsoncpp.git
  GIT_TAG 1.9.4)

# Fetch the JSON lib.
FetchContent_GetProperties(json)
if(NOT json_POPULATED)
  FetchContent_Populate(json)
  # Disable building the JsonCpp tests. If true it will try to run them as well
  # as ours when running ctest.
  set(JSONCPP_WITH_TESTS
      OFF
      CACHE BOOL "Do not build tests." FORCE)
  add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

###############################################################################
# CUDA
###############################################################################
# Check the CUDA compiler
include(CheckLanguage)
check_language(CUDA)

# Check that CUDA compiler can be found.
if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR "ERROR! Could not find CUDA compiler!")
endif()

# Fetch CUDA CMake-CUDA functionality. NOTE: FindCUDA is formally deprecated,
# but no standalone alternative exists as of yet for automatic detection of
# GPU architecture, so we use it.
find_package(CUDA)

set(CUDA_ARCH
    ""
    CACHE STRING "User-specified CUDA architecture (autodetected otherwise).")
if(CUDA_ARCH)
  message(STATUS "Setting user-defined arch: ${CUDA_ARCH}")
  set(ARCH_FLAGS "-gencode;arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")
else()
  # Autodetect architecture and fetch architecture version.
  CUDA_DETECT_INSTALLED_GPUS(CUDA_ARCH)
  string(REPLACE "." "" CUDA_ARCH "${CUDA_ARCH}")
  string(REPLACE " " "" CUDA_ARCH "${CUDA_ARCH}")

  # Atuodetect compilation flags.
  cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)
endif()

# Format architecture flags for compiler.
string(REPLACE ";" " " ARCH_FLAGS_DISPLAY "${ARCH_FLAGS}")
message(STATUS "Add CUDA architecture flags: ${ARCH_FLAGS_DISPLAY}")

set(NVCC_WARN
    "-Wall,\
    -Wextra,\
    -Wnon-virtual-dtor,\
    -Wunused,\
    -Wduplicated-cond,\
    -Wduplicated-branches,\
    -Wlogical-op,\
    -Wnull-dereference,\
    -Wdouble-promotion,\
    -Wformat=2")

# Set CMAKE_CUDA_ARCHITECTURES, which is needed in CMake>=3.18.
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
  # Set CMake policies.
  cmake_policy(SET CMP0104 NEW)
  cmake_policy(SET CMP0105 NEW)

  # Set CUDA architecture flag (new in CMake>=3.18).
  set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH}")
  message(STATUS "Set CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
endif (CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")

###############################################################################
# SUBDIRECTORIES
###############################################################################
# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)

###############################################################################
# TEMPLATES
###############################################################################
# Export variables and macro defines to be synchronized throughout the project,
# in particular this is e.g. the version number, the project name and build
# paths. These variables can then be specified in a single place. Generate
# include/configure.h from include/configure.h.in.
configure_file(${CMAKE_SOURCE_DIR}/include/templates/configure.h.in
               ${CMAKE_SOURCE_DIR}/include/configure.h)
