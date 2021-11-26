cmake_minimum_required(VERSION 3.18)
project(
  superconga
  VERSION 0.7.0
  LANGUAGES CUDA CXX)

# Output build messages
message(
  STATUS "Building ${CMAKE_PROJECT_NAME} version ${CMAKE_PROJECT_VERSION}")

# So we can download libs at build time.
include(FetchContent)

# Enable CTest.
if(CONGA_BUILD_TESTS)
  include(CTest)
endif(CONGA_BUILD_TESTS)

# ##############################################################################
# FIND DEPENDENCIES
# ##############################################################################
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)

# If Armadillo is too old or not found it is fetched.
# TODO(niclas): We should set the minimum compatible version here.
find_package(Armadillo 10.7 QUIET)
if(ARMADILLO_FOUND)
  message(STATUS "Found Armadillo ${ARMADILLO_VERSION_STRING}")
else()
  # Download Armadillo, but don't build it: we want to use header-only version.
  # Note that we could let CMake check for Armadillo first, but this way we can
  # enforce a specific version we know works well with SuperConga.
  message(STATUS "Fetching Armadillo")
  # version 10.7.x
  FetchContent_Declare(
    armadillo
    GIT_REPOSITORY https://gitlab.com/conradsnicta/armadillo-code.git
    GIT_TAG d9b38626f0cb6cadb828a8d836162df089ed668b)

  # Fetch properties for armadillo target: POPULATED, SOURCE_DIR, BINARY_DIR.
  FetchContent_GetProperties(armadillo)
  # Only populate if it is not already done.
  if(NOT armadillo_POPULATED)
    FetchContent_Populate(armadillo)
    # Note: We intentionally avoid add_subdirectory. Otherwise we indicate that
    # CMake should include these to the build, and we want a header-only version
    # of armadillo to include.
  endif(NOT armadillo_POPULATED)

  # Set variables with capital names, to conform with variables from
  # find_package.
  set(ARMADILLO_INCLUDE_DIRS "${armadillo_SOURCE_DIR}/include")
  set(ARMADILLO_LIBRARIES "")
endif(ARMADILLO_FOUND)

# ##############################################################################
# CUDA
# ##############################################################################
# Check the CUDA compiler
include(CheckLanguage)
check_language(CUDA)

# Check that CUDA compiler can be found.
if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR "ERROR! Could not find CUDA compiler!")
endif(NOT CMAKE_CUDA_COMPILER)

# Fetch CUDA CMake-CUDA functionality. NOTE: FindCUDA is formally deprecated,
# but no standalone alternative exists as of yet for automatic detection of GPU
# architecture, so we use it.
find_package(CUDA REQUIRED)

set(CUDA_ARCH
    ""
    CACHE STRING "User-specified CUDA architecture (autodetected otherwise).")
if(CUDA_ARCH)
  message(STATUS "Setting user-defined arch: ${CUDA_ARCH}")
  set(ARCH_FLAGS "-gencode;arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")
else(CUDA_ARCH)
  # Autodetect architecture and fetch architecture version.
  cuda_detect_installed_gpus(DETECTED_CUDA_ARCH)
  # Convert to number valid in architecture flags: remove dots.
  string(REPLACE "." "" CUDA_ARCH "${DETECTED_CUDA_ARCH}")

  # Remove leading and trailing whitespaces.
  string(STRIP ${CUDA_ARCH} CUDA_ARCH)

  # Choose first architecture in list for one or more GPUs.
  # TODO(patric): Fix issue if GPUs have different architecture levels.
  string(REPLACE " " ";" CUDA_ARCH_LIST "${CUDA_ARCH}")
  list(GET CUDA_ARCH_LIST 0 CUDA_ARCH)

  # Atuodetect compilation flags.
  cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)

  message(STATUS "Detected CUDA architectures: ${CUDA_ARCH}")
  message(STATUS "Detected CUDA architecture flags: ${ARCH_FLAGS}")
endif(CUDA_ARCH)

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
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
  # Set CMake policies.
  cmake_policy(SET CMP0104 NEW)
  cmake_policy(SET CMP0105 NEW)

  # Set CUDA architecture flag (new in CMake>=3.18).
  set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH}")
  message(STATUS "Set CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
endif(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")

# ##############################################################################
# SUBDIRECTORIES
# ##############################################################################
# For some inexplicable reason the tests must be added first, otherwise the
# fetching of doctest somehow interferes with Forge.
if(CONGA_BUILD_TESTS)
  add_subdirectory(tests)
endif(CONGA_BUILD_TESTS)
add_subdirectory(src)

# ##############################################################################
# TEMPLATES
# ##############################################################################
# Export variables and macro defines to be synchronized throughout the project,
# in particular this is e.g. the version number, the project name and build
# paths. These variables can then be specified in a single place. Generate
# include/configure.h from include/configure.h.in.
configure_file(${CMAKE_SOURCE_DIR}/include/conga/templates/configure.h.in
               ${CMAKE_SOURCE_DIR}/include/conga/configure.h)
