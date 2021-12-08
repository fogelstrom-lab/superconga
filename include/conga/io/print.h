//===---- print.h - Util functions for printing read/write success. ----===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Util functions for printing read/write success.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_IO_PRINT_H_
#define CONGA_IO_PRINT_H_

#include <filesystem>
#include <iostream>
#include <string>

namespace conga {
namespace io {
/// \brief Utility function for printing the success of writing to a file.
///
/// \param inVerbose Verbosity. If false only print if failure.
/// \param inSuccess Whether the write was successful.
/// \param inName Name of object that was written.
/// \param inFileName File the object was written to.
///
/// \return Void.
inline void printWrite(const bool inVerbose, const bool inSuccess,
                       const std::string &inName,
                       const std::filesystem::path &inFileName) {
  if (inSuccess) {
    if (inVerbose) {
      std::cout << "-- Wrote " << inName << " to " << inFileName << std::endl;
    }
  } else {
    std::cerr << "-- ERROR! Failed to write " << inName << " to " << inFileName
              << std::endl;
  }
}

/// \brief Utility function for printing the success of reading from a file.
///
/// \param inVerbose Verbosity. If false only print if failure.
/// \param inSuccess Whether the read was successful.
/// \param inName Name of object that was read.
/// \param inFileName File the object was read from.
///
/// \return Void.
inline void printRead(const bool inVerbose, const bool inSuccess,
                      const std::string &inName,
                      const std::filesystem::path &inFileName) {
  if (inSuccess) {
    if (inVerbose) {
      std::cout << "-- Read " << inName << " from " << inFileName << std::endl;
    }
  } else {
    std::cerr << "-- ERROR! Failed to read " << inName << " from " << inFileName
              << std::endl;
  }
}
} // namespace io
} // namespace conga

#endif // CONGA_IO_PRINT_H_
