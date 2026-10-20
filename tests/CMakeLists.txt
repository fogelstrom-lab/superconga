message(
  STATUS "####################### SuperConga Test Files #######################"
)
# If doctest is too old or not found it is fetched.
# TODO(niclas): We should set the minimum compatible version here.
find_package(doctest 2.4.1 QUIET)
if(doctest_FOUND)
  message(STATUS "Found doctest ${doctest_VERSION}")
else()
  # Declare the doctest lib.
  message(STATUS "Fetching doctest")
  FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/onqtam/doctest.git
    GIT_TAG 2.4.6)

  # Fetch the doctest lib.
  FetchContent_GetProperties(doctest)
  if(NOT doctest_POPULATED)
    FetchContent_Populate(doctest)
  endif(NOT doctest_POPULATED)

  # Set include directory. Not that this is NOT src/include because then we
  # cannot include it in C++ as <doctest/doctest.h> (which is the way it is
  # included if the package is found).
  set(DOCTEST_INCLUDE_DIRS "${doctest_SOURCE_DIR}")
endif(doctest_FOUND)

# ======= SOURCE FILES =======
# Define the main set of source files
set(SRC
    accelerators_test.cu
    basis_function_normalization_test.cu
    bdg_utils_test.cu
    dispersion_test.cu
    fermi_surface_test.cu
    fermi_surface_circle_test.cu
    fermi_surface_tight_binding_test.cu
    free_energy_test.cu
    gauge_test.cu
    geometry_test.cu
    green_elements_test.cu
    grid_data_sum_magnitude_test.cu
    grid_data_sum_square_test.cu
    grid_normalization_test.cu
    legacy_test.cu
    order_parameter_bulk_test.cu
    parameters_test.cu
    physics/current_conservation_test.cu
    physics/current_linear_response_test.cu
    physics/current_phase_gradient_test.cu
    physics/dos_test.cu
    physics/low_temp_free_energy_test.cu
    physics/penetration_depth_test.cu
    physics/swave_annulus_radial_gauge_test.cu
    physics/swave_annulus_solenoid_gauge_test.cu
    physics/swave_disc_test.cu
    physics/vector_potential_test.cu
    riccati_solutions_test.cu
    utils_test.cu)

# Enable testing functionality
enable_testing()

foreach(file ${SRC})
  message(STATUS "File: ${file}")

  get_filename_component(name ${file} NAME_WE)

  # Create command-line executable.
  add_executable(${name} ${file})

  # Add the test to the test suite.
  add_test(NAME ${name} COMMAND ${name})

  conga_configure_target(${name})
  target_include_directories(${name} SYSTEM PRIVATE ${DOCTEST_INCLUDE_DIRS})

  if(doctest_FOUND)
    # This does not seem to be completely necessary, but this is how it is done
    # in their tutorial when using find_package.
    target_link_libraries(${name} PRIVATE doctest::doctest)
  endif(doctest_FOUND)

  # Build test binaries into build/xxx/tests/ (where xxx is e.g. release).
  set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                           "${PROJECT_BINARY_DIR}/tests")

endforeach(file ${SRC})

message(
  STATUS "#####################################################################"
)
