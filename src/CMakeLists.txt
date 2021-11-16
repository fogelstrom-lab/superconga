# ##############################################################################
# SOURCE FILES
# ##############################################################################
message(
  STATUS
    "################## Building SuperConga source files ##################")

# Define the main set of source files
set(SRC superconga.cu postprocess.cu)

# Define precisions.
set(SINGLE_PRECISION OFF ON)

foreach(file ${SRC})
  message(STATUS "# -- Setting up executable: ${file}.")

  get_filename_component(name_raw ${file} NAME_WE)

  foreach(single_precision ${SINGLE_PRECISION})
    # Append precision to name.
    if(single_precision)
      set(name ${name_raw}_f32)
    else()
      set(name ${name_raw}_f64)
    endif(single_precision)

    # Create command-line executable.
    add_executable(${name} ${file})

    # Include directories: equivalent to -I.
    target_include_directories(
      ${name} PRIVATE ${PROJECT_SOURCE_DIR}/include SYSTEM
                      ${ARMADILLO_INCLUDE_DIRS})

    # Build src binaries into build/xxx/bin/ (where xxx is e.g. release).
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                             "${PROJECT_BINARY_DIR}/bin")

    # External libraries: equivalent to -l.
    target_link_libraries(
      ${name}
      PRIVATE glut
              GL
              GLU
              gfortran
              jsoncpp_lib
              ${LAPACK_LIBRARIES}
              ${BLAS_LIBRARIES})

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
  endforeach(single_precision ${SINGLE_PRECISION})
endforeach(file ${SRC})

message(
  STATUS
    "######################################################################")
