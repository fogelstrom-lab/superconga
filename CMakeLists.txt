cmake_minimum_required(VERSION 3.11)
project(
  suprcongpu
  VERSION 0.0.1
  LANGUAGES CUDA CXX)

# Output build messages
message(
  STATUS "Building ${CMAKE_PROJECT_NAME} version ${CMAKE_PROJECT_VERSION}.")

# Enable CTest
include(CTest)

# So we can download libs at build time.
include(FetchContent)

# Declare the JSON lib. Note that this is a smaller mirror of the original repo:
# https://github.com/nlohmann/json
FetchContent_Declare(
  json
  GIT_REPOSITORY
    https://github.com/ArthurSonzogni/nlohmann_json_cmake_fetchcontent
  GIT_TAG v3.8.0)

# Fetch the JSON lib.
FetchContent_GetProperties(json)
if(NOT json_POPULATED)
  FetchContent_Populate(json)
  add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Check the CUDA compiler
include(CheckLanguage)
check_language(CUDA)

# Add subdirectories
add_subdirectory(include)
add_subdirectory(research)
add_subdirectory(src)
add_subdirectory(tests)
