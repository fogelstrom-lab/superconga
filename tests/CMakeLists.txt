cmake_minimum_required(VERSION 3.10)
project(
  suprcongpu
  VERSION 0.0.1
  LANGUAGES CUDA CXX)

find_package(doctest REQUIRED)

# Define the main set of source files
set(SRC_TEST utils_test.cu)

# Create command-line executable
add_executable(utils_test ${SRC_TEST})

# Add extra compile options and flags. Include directories.
target_include_directories(utils_test PRIVATE ../include doctest::doctest)

# Use C++14 since it is the lates standard CUDA supports
message(STATUS "Enable C++ standard: C++14")
set_target_properties(utils_test
                      PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON)

# NOTE: FindCUDA is formally deprecated, but no standalone alternative exists as
# of yet for automatic detection of GPU architecture, so we use it
set(CUDA_ARCH
    ""
    CACHE STRING "User-specified CUDA architecture (autodetected otherwise)")
if(CUDA_ARCH)
  message(STATUS "Setting user-defined arch: ${CUDA_ARCH}")
  set(ARCH_FLAGS "-gencode;arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")
else()
  find_package(CUDA)
  cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)
endif()
string(REPLACE ";" " " ARCH_FLAGS_DISPLAY "${ARCH_FLAGS}")
message(STATUS "Add CUDA architecture flags: ${ARCH_FLAGS_DISPLAY}")

target_compile_options(utils_test PUBLIC "$<$<CONFIG:RELEASE>:${ARCH_FLAGS}>")
