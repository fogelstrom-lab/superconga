###############################################################################
# SOURCE FILES
###############################################################################
# Define the main set of source files
set(SRC main.cu)

# string( REPLACE ".cu" "" name ${file})
get_filename_component(name ${SRC} NAME_WE)

# Create command-line executable
add_executable(${name} ${SRC})

# Include directories
target_include_directories(
  ${name}
  PRIVATE ${PROJECT_SOURCE_DIR}/include
  SYSTEM ${PROJECT_SOURCE_DIR}/ext/armadillo/include)

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
          jsoncpp_lib)

# Use C++17 since it is the latest standard CUDA supports
message(STATUS "Enable C++ standard: C++17")
set_target_properties(${name} PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED
                                                          ON)
set_target_properties(${name} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED
                                                         ON)

# Set DEBUG compile flags
target_compile_options(
  ${name}
  PUBLIC "$<$<CONFIG:DEBUG>:${ARCH_FLAGS}>"
  PUBLIC "$<$<CONFIG:DEBUG>:-Xcompiler>"
  PUBLIC "$<$<CONFIG:DEBUG>:${NVCC_WARN}>")

# Set RELEASE compile flags
target_compile_options(${name} PUBLIC "$<$<CONFIG:RELEASE>:${ARCH_FLAGS}>")
