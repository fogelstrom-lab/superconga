# Export variables and macro defines to be synchronized throughout the project,
# in particular this is e.g. the version number, the project name and build
# paths. These variables can then be specified in a single place.

# Generate include/configure.h from include/configure.h.in.
configure_file(configure.h.in ${CMAKE_SOURCE_DIR}/include/configure.h)
