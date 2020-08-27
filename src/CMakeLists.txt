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

# Define the main set of source files
set(SRC main.cu)

# string( REPLACE ".cu" "" name ${file})
get_filename_component(name ${SRC} NAME_WE)

# Create command-line executable
add_executable(${name} ${SRC})

# Include directories
target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Build src binaries into build/bin/
set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                         "${PROJECT_BINARY_DIR}/bin")

# External libraries
target_link_libraries(
  ${name}
  PRIVATE glut
          GL
          GLU
          gfortran
          lapack
          blas
          boost_system
          boost_filesystem
          nlohmann_json::nlohmann_json)

# Use C++14 since it is the latest standard CUDA supports
message(STATUS "Enable C++ standard: C++14")
set_target_properties(${name} PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED
                                                          ON)
set_target_properties(${name} PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED
                                                         ON)

# Set DEBUG compile flags
target_compile_options(
  ${name}
  PUBLIC "$<$<CONFIG:DEBUG>:${ARCH_FLAGS}>"
  PUBLIC "$<$<CONFIG:DEBUG>:-Xcompiler>"
  PUBLIC "$<$<CONFIG:DEBUG>:${NVCC_WARN}>")

# Set RELEASE compile flags
target_compile_options(${name} PUBLIC "$<$<CONFIG:RELEASE>:${ARCH_FLAGS}>")
