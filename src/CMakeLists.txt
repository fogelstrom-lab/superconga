# ##############################################################################
# SOURCE FILES
# ##############################################################################
message(
  STATUS "###################### SuperConga Source Files ######################"
)

# Define the main set of source files
set(SRC superconga.cu postprocess.cu)

# Define precisions.
set(CONGA_SINGLE_PRECISION OFF ON)

# If building without HDF5 HighFive is not fetched.
if(CONGA_USE_HDF5)
  # If HighFive is too old or not found it is fetched.
  # TODO(niclas): We should set the minimum compatible version here.
  find_package(HighFive 2.3.1 QUIET)
  if(HighFive_FOUND)
    message(STATUS "Found HighFive ${HighFive_VERSION}")
  else()
    # Download HighFive.
    message(STATUS "Fetching HighFive")
    FetchContent_Declare(
      highfive
      GIT_REPOSITORY https://github.com/BlueBrain/HighFive.git
      GIT_TAG v2.3.1)

    # Fetch properties for target: POPULATED, SOURCE_DIR, BINARY_DIR.
    FetchContent_GetProperties(highfive)
    # Only populate if it is not already done.
    if(NOT highfive_POPULATED)
      FetchContent_Populate(highfive)
      set(HIGHFIVE_USE_BOOST
          OFF
          CACHE BOOL "Do not use boost." FORCE)
      # Disable building the tests.
      set(HIGHFIVE_UNIT_TESTS
          OFF
          CACHE BOOL "Do not build tests." FORCE)
      # Disable building the examples.
      set(HIGHFIVE_EXAMPLES
          OFF
          CACHE BOOL "Do not build examples." FORCE)
      # Disable building the documentation.
      set(HIGHFIVE_BUILD_DOCS
          OFF
          CACHE BOOL "Do not build documentation." FORCE)
      add_subdirectory(${highfive_SOURCE_DIR} ${highfive_BINARY_DIR}
                       EXCLUDE_FROM_ALL)
    endif(NOT highfive_POPULATED)
  endif(HighFive_FOUND)
endif(CONGA_USE_HDF5)

# If building without visualization Forge is not fetched. Note that Forge must
# be fetched last for some reason.
if(CONGA_VISUALIZE)
  # Declare the Forge lib.
  message(STATUS "Fetching Forge")
  FetchContent_Declare(
    forge
    GIT_REPOSITORY https://github.com/arrayfire/forge.git
    GIT_TAG v1.0.8)

  # Fetch the Forge lib.
  FetchContent_GetProperties(forge)
  if(NOT forge_POPULATED)
    FetchContent_Populate(forge)
    # Disable building the examples.
    set(FG_BUILD_EXAMPLES
        OFF
        CACHE BOOL "Do not build examples." FORCE)
    # Disable building the documentation.
    set(FG_BUILD_DOCS
        OFF
        CACHE BOOL "Do not build documentation." FORCE)
    add_subdirectory(${forge_SOURCE_DIR} ${forge_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif(NOT forge_POPULATED)
endif(CONGA_VISUALIZE)

foreach(file ${SRC})
  message(STATUS "File: ${file}")

  get_filename_component(name_raw ${file} NAME_WE)

  foreach(single_precision ${CONGA_SINGLE_PRECISION})
    # Append precision to name.
    if(single_precision)
      set(name ${name_raw}_f32)
    else()
      set(name ${name_raw}_f64)
    endif(single_precision)

    # Create command-line executable.
    add_executable(${name} ${file})

    # Build src binaries into build/xxx/bin/ (where xxx is e.g. release).
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                             "${PROJECT_BINARY_DIR}/bin")

    # Include directories: equivalent to -I.
    target_include_directories(
      ${name} PRIVATE ${PROJECT_SOURCE_DIR}/include SYSTEM
                      ${ARMADILLO_INCLUDE_DIRS} ${JSONCPP_INCLUDE_DIRS})

    # External libraries: equivalent to -l.
    target_link_libraries(
      ${name} PRIVATE gfortran jsoncpp ${ARMADILLO_LIBRARIES}
                      ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})

    # Use HDF5.
    if(CONGA_USE_HDF5)
      target_compile_definitions(${name} PUBLIC CONGA_USE_HDF5)
      target_link_libraries(${name} PRIVATE HighFive)
    endif(CONGA_USE_HDF5)

    # Build with visualization.
    if(CONGA_VISUALIZE)
      target_compile_definitions(${name} PUBLIC CONGA_VISUALIZE)
      target_link_libraries(${name} PRIVATE forge)
    endif(CONGA_VISUALIZE)

    # Use C++17 since it is the latest standard CUDA supports.
    set_target_properties(${name} PROPERTIES CUDA_STANDARD 17
                                             CUDA_STANDARD_REQUIRED ON)
    set_target_properties(${name} PROPERTIES CXX_STANDARD 17
                                             CXX_STANDARD_REQUIRED ON)

    # Apply compiler options.
    if(single_precision)
      target_compile_definitions(${name} PUBLIC CONGA_SINGLE_PRECISION)
    endif(single_precision)

    # Set DEBUG compile flags
    target_compile_options(
      ${name}
      PUBLIC "$<$<CONFIG:DEBUG>:${ARCH_FLAGS}>"
      PUBLIC "$<$<CONFIG:DEBUG>:-Xcompiler>"
      PUBLIC "$<$<CONFIG:DEBUG>:${NVCC_WARN}>")

    # Set RELEASE compile flags
    target_compile_options(${name} PUBLIC "$<$<CONFIG:RELEASE>:${ARCH_FLAGS}>")
  endforeach(single_precision ${CONGA_SINGLE_PRECISION})
endforeach(file ${SRC})

message(
  STATUS "#####################################################################"
)
