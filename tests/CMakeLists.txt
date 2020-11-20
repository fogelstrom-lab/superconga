###############################################################################
# EXTERNAL LIBRARIES
###############################################################################
# Declare the doctest lib.
FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/onqtam/doctest.git
  GIT_TAG 2.4.0)

# Fetch the doctest lib.
FetchContent_GetProperties(doctest)
if(NOT doctest_POPULATED)
  FetchContent_Populate(doctest)
  add_subdirectory(${doctest_SOURCE_DIR} ${doctest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

###############################################################################
# SOURCE FILES
###############################################################################
# Define the main set of source files
set(SRC
    basis_function_normalization_test.cu
    bdg_utils_test.cu
    dispersion_test.cu
    fermi_surface_test.cu
    fermi_surface_circle_test.cu
    fermi_surface_tight_binding_test.cu
    geometry_test.cu
    grid_data_sum_magnitude_test.cu
    grid_data_sum_square_test.cu
    grid_normalization_test.cu
    legacy_test.cu
    order_parameter_initial_test.cu
    order_parameter_bulk_test.cu
    parameters_test.cu
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
  target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
  # External libraries
  target_link_libraries(
    ${name}
    PRIVATE glut
            GL
            GLU
            gfortran
            lapack
            blas
            jsoncpp_lib
            doctest::doctest)

  # Use C++17 since it is the lates standard CUDA supports
  message(STATUS "Enable C++ standard: C++17")
  set_target_properties(${name} PROPERTIES CUDA_STANDARD 17
                                           CUDA_STANDARD_REQUIRED ON)
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
endforeach()
