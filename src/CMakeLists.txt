# ##############################################################################
# SOURCE FILES
# ##############################################################################
message(
  STATUS
    "################## Building SuperConga source files ##################")

# Define the main set of source files
set(SRC superconga.cu postprocess.cu)

# Define precisions.
set(CONGA_SINGLE_PRECISION OFF ON)

# If building without visualization Forge is not fetched.
if(CONGA_VISUALIZE)
  # Declare the Forge lib.
  FetchContent_Declare(
    forge
    GIT_REPOSITORY https://github.com/arrayfire/forge.git
    GIT_TAG v1.0.8)

  # Fetch the Forge lib.
  FetchContent_GetProperties(forge)
  if(NOT forge_POPULATED)
    FetchContent_Populate(forge)
    # Disable building the Forge examples.
    set(FG_BUILD_EXAMPLES
        OFF
        CACHE BOOL "Do not build examples." FORCE)
    # Disable building the Forge documentation.
    set(FG_BUILD_DOCS
        OFF
        CACHE BOOL "Do not build documentation." FORCE)
    add_subdirectory(${forge_SOURCE_DIR} ${forge_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif(NOT forge_POPULATED)
endif(CONGA_VISUALIZE)

foreach(file ${SRC})
  message(STATUS "Setting up executable: ${file}")

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
    if(CONGA_VISUALIZE)
      target_compile_definitions(${name} PUBLIC CONGA_VISUALIZE)
      target_link_libraries(
        ${name} PRIVATE gfortran forge jsoncpp ${ARMADILLO_LIBRARIES}
                        ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
    else()
      target_link_libraries(
        ${name} PRIVATE gfortran jsoncpp ${ARMADILLO_LIBRARIES}
                        ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
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
  STATUS
    "######################################################################")
