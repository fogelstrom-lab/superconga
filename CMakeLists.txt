cmake_minimum_required(VERSION 3.11)
project(
  suprcongpu
  VERSION 0.0.1
  LANGUAGES CUDA CXX)

# Output build messages
message(STATUS "Project name: ${CMAKE_PROJECT_NAME}")
message(STATUS "Version: ${CMAKE_PROJECT_VERSION}")

# Enable CTest
include(CTest)

# So we can download libs at build time.
include(FetchContent)

# Check the CUDA compiler
include(CheckLanguage)
check_language(CUDA)

# Add subdirectories
add_subdirectory(include)
add_subdirectory(research)
add_subdirectory(src)
add_subdirectory(tests)
