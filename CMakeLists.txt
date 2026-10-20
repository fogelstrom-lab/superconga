cmake_minimum_required(VERSION 3.21)

# Dependencies fetched at configure time (HighFive 2.3, doctest, Forge) still
# declare old minimum CMake versions; let CMake >= 3.31/4.x configure them.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
if(POLICY CMP0169)
  # We populate header-only dependencies without add_subdirectory on purpose.
  cmake_policy(SET CMP0169 OLD)
endif()

# ##############################################################################
# BACKEND SELECTION
# ##############################################################################
# CONGA_BACKEND: "CUDA" (NVIDIA GPUs via nvcc) or "CPU" (any C++17 compiler;
# kernels run on the host, parallelised with OpenMP). "AUTO" picks CUDA when
# a CUDA compiler can be found, CPU otherwise.
set(CONGA_BACKEND
    "AUTO"
    CACHE STRING "Execution backend: AUTO, CUDA or CPU")
set_property(CACHE CONGA_BACKEND PROPERTY STRINGS AUTO CUDA CPU)

if(CONGA_BACKEND STREQUAL "AUTO")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    set(CONGA_BACKEND "CUDA")
  else()
    set(CONGA_BACKEND "CPU")
  endif()
endif()

if(CONGA_BACKEND STREQUAL "CUDA")
  set(CONGA_LANGUAGES CUDA CXX C)
elseif(CONGA_BACKEND STREQUAL "CPU")
  set(CONGA_LANGUAGES CXX C)
else()
  message(FATAL_ERROR "Unknown CONGA_BACKEND '${CONGA_BACKEND}'")
endif()

project(
  SuperConga
  VERSION 1.1.0
  LANGUAGES ${CONGA_LANGUAGES})

message(STATUS "Building ${CMAKE_PROJECT_NAME} version ${CMAKE_PROJECT_VERSION}")
message(STATUS "Execution backend: ${CONGA_BACKEND}")

# So we can download libs at build time.
include(FetchContent)

# Enable CTest.
if(CONGA_BUILD_TESTS)
  include(CTest)
endif(CONGA_BUILD_TESTS)

# ##############################################################################
# FIND DEPENDENCIES
# ##############################################################################
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)

# If Armadillo is too old or not found it is fetched (header-only use).
find_package(Armadillo 10.7 QUIET)
if(ARMADILLO_FOUND)
  message(STATUS "Found Armadillo ${ARMADILLO_VERSION_STRING}")
else()
  message(STATUS "Fetching Armadillo")
  FetchContent_Declare(
    armadillo
    GIT_REPOSITORY https://gitlab.com/conradsnicta/armadillo-code.git
    GIT_TAG d9b38626f0cb6cadb828a8d836162df089ed668b)
  FetchContent_GetProperties(armadillo)
  if(NOT armadillo_POPULATED)
    FetchContent_Populate(armadillo)
  endif(NOT armadillo_POPULATED)
  set(ARMADILLO_INCLUDE_DIRS "${armadillo_SOURCE_DIR}/include")
  set(ARMADILLO_LIBRARIES "")
endif(ARMADILLO_FOUND)

# Armadillo (and the LAPACK it wraps) may need gfortran's runtime when the
# reference LAPACK is used. Link it only if it is actually available.
find_library(CONGA_GFORTRAN_LIBRARY gfortran)
if(CONGA_GFORTRAN_LIBRARY)
  set(CONGA_FORTRAN_RUNTIME ${CONGA_GFORTRAN_LIBRARY})
else()
  set(CONGA_FORTRAN_RUNTIME "")
endif()

# ##############################################################################
# BACKEND: CUDA
# ##############################################################################
if(CONGA_BACKEND STREQUAL "CUDA")
  # Architecture: user override, else "native" (CMake >= 3.24), else a
  # conservative default list.
  set(CUDA_ARCH
      ""
      CACHE STRING "CUDA architecture(s), e.g. 80 or 70;80 (native otherwise)")
  if(CUDA_ARCH)
    set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH}")
  elseif(NOT CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
      set(CMAKE_CUDA_ARCHITECTURES native)
    else()
      set(CMAKE_CUDA_ARCHITECTURES 70 80)
    endif()
  endif()
  message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

  set(NVCC_WARN
      "-Wall,-Wextra,-Wnon-virtual-dtor,-Wunused,-Wduplicated-cond,\
-Wduplicated-branches,-Wlogical-op,-Wnull-dereference,-Wdouble-promotion,\
-Wformat=2")
endif()

# ##############################################################################
# BACKEND: CPU
# ##############################################################################
if(CONGA_BACKEND STREQUAL "CPU")
  find_package(OpenMP COMPONENTS CXX)
  if(APPLE AND NOT OpenMP_CXX_FOUND)
    # Apple's clang ships without OpenMP; use Homebrew's libomp.
    execute_process(
      COMMAND brew --prefix libomp
      OUTPUT_VARIABLE CONGA_LIBOMP_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    if(CONGA_LIBOMP_PREFIX AND EXISTS "${CONGA_LIBOMP_PREFIX}/lib/libomp.dylib")
      message(STATUS "Using Homebrew libomp at ${CONGA_LIBOMP_PREFIX}")
      set(OpenMP_CXX_FLAGS "-Xclang -fopenmp -I${CONGA_LIBOMP_PREFIX}/include")
      set(OpenMP_CXX_LIB_NAMES "omp")
      set(OpenMP_omp_LIBRARY "${CONGA_LIBOMP_PREFIX}/lib/libomp.dylib")
      find_package(OpenMP COMPONENTS CXX)
    endif()
  endif()
  if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: kernels will run multi-threaded")
  else()
    message(WARNING "OpenMP not found: CPU backend will run single-threaded")
  endif()

  # Thrust (header-only) from NVIDIA's CCCL. A CUDA toolkit is NOT required:
  # we use the OpenMP device system.
  find_path(CONGA_THRUST_INCLUDE_DIR thrust/version.h
            HINTS ${CONGA_THRUST_ROOT} ENV CONGA_THRUST_ROOT)
  if(CONGA_THRUST_INCLUDE_DIR)
    message(STATUS "Found Thrust at ${CONGA_THRUST_INCLUDE_DIR}")
    set(CONGA_THRUST_INCLUDE_DIRS ${CONGA_THRUST_INCLUDE_DIR})
  else()
    message(STATUS "Fetching CCCL (Thrust)")
    FetchContent_Declare(
      cccl
      GIT_REPOSITORY https://github.com/NVIDIA/cccl.git
      GIT_TAG v2.6.1
      GIT_SHALLOW TRUE)
    FetchContent_GetProperties(cccl)
    if(NOT cccl_POPULATED)
      FetchContent_Populate(cccl)
    endif()
    set(CONGA_THRUST_INCLUDE_DIRS "${cccl_SOURCE_DIR}/thrust"
                                  "${cccl_SOURCE_DIR}/libcudacxx/include"
                                  "${cccl_SOURCE_DIR}/cub")
  endif()
endif()

# ##############################################################################
# PER-TARGET CONFIGURATION
# ##############################################################################
# conga_configure_target(<target>): apply include paths, libraries, language
# standard and backend-specific flags to a SuperConga executable.
function(conga_configure_target name)
  target_include_directories(
    ${name} PRIVATE ${PROJECT_SOURCE_DIR}/include SYSTEM
                    ${ARMADILLO_INCLUDE_DIRS} ${JSONCPP_INCLUDE_DIRS})
  target_link_libraries(
    ${name} PRIVATE ${JSONCPP_LIBRARIES} ${ARMADILLO_LIBRARIES}
                    ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES}
                    ${CONGA_FORTRAN_RUNTIME})
  target_link_directories(${name} PRIVATE ${JSONCPP_LIBRARY_DIRS})
  set_target_properties(${name} PROPERTIES CXX_STANDARD 17
                                           CXX_STANDARD_REQUIRED ON)

  if(CONGA_BACKEND STREQUAL "CUDA")
    set_target_properties(${name} PROPERTIES CUDA_STANDARD 17
                                             CUDA_STANDARD_REQUIRED ON)
    target_compile_options(
      ${name}
      PRIVATE "$<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler>"
              "$<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CUDA>>:${NVCC_WARN}>")
  else()
    # Compile the .cu sources as ordinary C++.
    get_target_property(srcs ${name} SOURCES)
    set_source_files_properties(${srcs} PROPERTIES LANGUAGE CXX)
    target_compile_options(${name} PRIVATE -x c++)
    target_compile_definitions(
      ${name}
      PRIVATE CONGA_BACKEND_CPU=1
              THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
              THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP
              THRUST_IGNORE_DEPRECATED_CPP_DIALECT
              # libcu++ (CCCL) would otherwise re-declare the const-correct
              # strchr/strstr/... overloads that every C++ standard library
              # already provides; glibc announces this itself, libc++ on
              # macOS does not.
              _LIBCUDACXX_STRING_H_HAS_CONST_OVERLOADS=1)
    target_include_directories(${name} SYSTEM
                               PRIVATE ${CONGA_THRUST_INCLUDE_DIRS})
    if(OpenMP_CXX_FOUND)
      target_link_libraries(${name} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_compile_options(
      ${name} PRIVATE "$<$<CONFIG:DEBUG>:-Wall;-Wextra>"
                      "$<$<CONFIG:RELEASE>:-O3>")
  endif()
endfunction()

# ##############################################################################
# SUBDIRECTORIES
# ##############################################################################
# For some inexplicable reason the tests must be added first, otherwise the
# fetching of doctest somehow interferes with Forge.
if(CONGA_BUILD_TESTS)
  add_subdirectory(tests)
endif(CONGA_BUILD_TESTS)
add_subdirectory(src)

# ##############################################################################
# TEMPLATES
# ##############################################################################
configure_file(${CMAKE_SOURCE_DIR}/include/conga/templates/configure.h.in
               ${CMAKE_SOURCE_DIR}/include/conga/configure.h)
