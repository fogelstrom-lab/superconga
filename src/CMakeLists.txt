# ##############################################################################
# SOURCE FILES
# ##############################################################################
message(
  STATUS "###################### SuperConga Source Files ######################"
)

# Define the main set of source files
set(SRC simulation.cu postprocess.cu)

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

    conga_configure_target(${name})

    # Use HDF5.
    if(CONGA_USE_HDF5)
      target_compile_definitions(${name} PUBLIC CONGA_USE_HDF5)
      target_link_libraries(${name} PRIVATE HighFive)
    endif(CONGA_USE_HDF5)

    # Build with visualization.
    if(CONGA_VISUALIZE)
      target_compile_definitions(${name} PUBLIC CONGA_VISUALIZE)
      target_link_libraries(${name} PRIVATE forge)

      # Use CUDA-OpenGl interopability.
      if(CONGA_USE_CUDA_OPENGL_INTEROP)
        target_compile_definitions(${name} PUBLIC CONGA_USE_CUDA_OPENGL_INTEROP)
      endif(CONGA_USE_CUDA_OPENGL_INTEROP)
    endif(CONGA_VISUALIZE)

    # Apply compiler options.
    if(single_precision)
      target_compile_definitions(${name} PUBLIC CONGA_SINGLE_PRECISION)
    endif(single_precision)
  endforeach(single_precision ${CONGA_SINGLE_PRECISION})
endforeach(file ${SRC})

message(
  STATUS "#####################################################################"
)
