cmake_minimum_required(VERSION 3.15)
project(suprcongpu 
  VERSION 0.0.1
  LANGUAGES CUDA CXX
)

# Check the CUDA compiler
include(CheckLanguage)
check_language(CUDA)

# Define the main set of source files
set(SRC demo_hexagon_swave_VL.cu)

# Create command-line executable
add_executable(demo_hexagon_swave_VL ${SRC})

# Set up the binary configuration file
configure_file(demo_hexagon_swave_VL.dat demo_hexagon_swave_VL.dat COPYONLY)

# ----------------------------------- #
# Add extra compile options and flags #
# ----------------------------------- #
# Include directories
target_include_directories(demo_hexagon_swave_VL PRIVATE include)

# External libraries
target_link_libraries(demo_hexagon_swave_VL PRIVATE glut
                                                    GL
                                                    GLU
                                                    boost_system
                                                    boost_filesystem
                                                    gfortran
                                                    lapack
                                                    blas)

# Use C++14 since it is the lates standard CUDA supports
message(STATUS "Enable C++ standard: C++14")
set_target_properties(demo_hexagon_swave_VL PROPERTIES CUDA_STANDARD 14
                                            CUDA_STANDARD_REQUIRED ON)
set_target_properties(demo_hexagon_swave_VL PROPERTIES CXX_STANDARD 14
                                            CXX_STANDARD_REQUIRED ON)

# NOTE: FindCUDA is formally deprecated, but no standalone 
# alternative exists as of yet for automatic detection of 
# GPU architecture, so we use it
set(CUDA_ARCH "" CACHE STRING "User-specified CUDA architecture (autodetected otherwise)")
if (CUDA_ARCH)
  message(STATUS "Setting user-defined arch: ${CUDA_ARCH}")
  set(ARCH_FLAGS "-gencode;arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")
else()
  find_package(CUDA) 
  cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)
endif()
string(REPLACE ";" " " ARCH_FLAGS_DISPLAY "${ARCH_FLAGS}")
message(STATUS "Add CUDA architecture flags: ${ARCH_FLAGS_DISPLAY}")

set(NVCC_WARN "-Wall,\
               -Wextra,\
               -Wnon-virtual-dtor,\
               -Wunused,\
               -Wduplicated-cond,\
               -Wduplicated-branches,\
               -Wlogical-op,\
               -Wnull-dereference,\
               -Wdouble-promotion,\
               -Wformat=2")

# Set DEBUG compile flags
target_compile_options(demo_hexagon_swave_VL PUBLIC "$<$<CONFIG:DEBUG>:${ARCH_FLAGS}>"
                                             PUBLIC "$<$<CONFIG:DEBUG>:-Xcompiler>"
                                             PUBLIC "$<$<CONFIG:DEBUG>:${NVCC_WARN}>")
                                          
# Set RELEASE compile flags 
target_compile_options(demo_hexagon_swave_VL PUBLIC "$<$<CONFIG:RELEASE>:${ARCH_FLAGS}>")
