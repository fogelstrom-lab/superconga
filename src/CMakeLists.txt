# ##############################################################################
# SOURCE FILES
# ##############################################################################
# Define the main set of source files
set(SRC main.cu postprocess.cu)

# Set compiler options, i.e. #defines.
option(APPROX_STEP "Use approximate steps for the coherence functions." OFF)

foreach(file ${SRC})
  get_filename_component(name ${file} NAME_WE)

  # Create command-line executable.
  add_executable(${name} ${file})

  # Include directories.
  target_include_directories(
    ${name} PRIVATE ${PROJECT_SOURCE_DIR}/include SYSTEM
                    ${PROJECT_SOURCE_DIR}/.ext/armadillo/include)

  # Build src binaries into build/xxx/bin/ (where xxx is e.g. release).
  set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                           "${PROJECT_BINARY_DIR}/bin")

  # External libraries.
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
  set_target_properties(${name} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED
                                                           ON)

  # Apply compiler options.
  if(APPROX_STEP)
    message("-- APPROX_STEP ON")
    target_compile_definitions(${name} PUBLIC APPROX_STEP)
  endif()

  # Set DEBUG compile flags
  target_compile_options(
    ${name}
    PUBLIC "$<$<CONFIG:DEBUG>:${ARCH_FLAGS}>"
    PUBLIC "$<$<CONFIG:DEBUG>:-Xcompiler>"
    PUBLIC "$<$<CONFIG:DEBUG>:${NVCC_WARN}>")

  # Set RELEASE compile flags
  target_compile_options(${name} PUBLIC "$<$<CONFIG:RELEASE>:${ARCH_FLAGS}>")
endforeach()
