cmake_minimum_required(VERSION 3.10)
project(
  suprcongpu
  VERSION 0.0.1
  LANGUAGES CUDA CXX)

# Check the CUDA compiler
include(CheckLanguage)
check_language(CUDA)

# Add subdirectories
add_subdirectory(research)
add_subdirectory(examples)
add_subdirectory(tests)
