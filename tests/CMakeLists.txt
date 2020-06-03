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

# Declare the JSON lib. Note that this is a smaller mirror of the original repo:
# https://github.com/nlohmann/json
FetchContent_Declare(
  json
  GIT_REPOSITORY
    https://github.com/ArthurSonzogni/nlohmann_json_cmake_fetchcontent
  GIT_TAG v3.7.3)

# Fetch the JSON lib.
FetchContent_GetProperties(json)
if(NOT json_POPULATED)
  FetchContent_Populate(json)
  add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Define the main set of source files
set(SRC
    basis_function_normalization_test.cu
    fermi_surface_test.cu
    fermi_surface_circle_test.cu
    fermi_surface_tight_binding_test.cu
    grid_data_sum_magnitude_test.cu
    grid_data_sum_square_test.cu
    legacy_test.cu
    order_parameter_initial_test.cu
    order_parameter_bulk_test.cu
    physics/dos_test.cu
    physics/low_temp_free_energy_test.cu
    utils_test.cu)

# Enable testing functionality
enable_testing()

foreach(file ${SRC})
  # string( REPLACE ".cu" "" name ${file})
  get_filename_component(name ${file} NAME_WE)

  # Create command-line executable
  add_executable(${name} ${file})

  # Add the test to the test suite
  add_test(NAME ${name} COMMAND ${name})

  # ----------------------------------- # Add extra compile options and flags #
  # ----------------------------------- # Include directories
  target_include_directories(
    ${name}
    PRIVATE ${PROJECT_SOURCE_DIR}/include
    PRIVATE /usr/include/doctest)
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

  # Use C++14 since it is the lates standard CUDA supports
  message(STATUS "Enable C++ standard: C++14")
  set_target_properties(${name} PROPERTIES CUDA_STANDARD 14
                                           CUDA_STANDARD_REQUIRED ON)
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
endforeach()
