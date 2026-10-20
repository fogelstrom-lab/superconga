//===------------------- json.h - Read/write JSON files. ------------------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Functions for reading/writing JSON files.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_IO_JSON_H_
#define CONGA_IO_JSON_H_

#include <json/json.h>

#include <filesystem>
#include <iostream>

namespace conga {
namespace io {
/// \brief Write JSON to file.
///
/// \param inFileName Path to JSON file.
/// \param inJSON The JSON-object to be written.
///
/// \return Void.
void write(const std::filesystem::path &inFileName, const Json::Value &inJSON) {
  std::ofstream ofstreamJSON(inFileName);
  ofstreamJSON << inJSON.toStyledString() << std::endl;
}

/// \brief Read JSON from file.
///
/// \param inFileName Path to JSON file.
/// \param outJSON The JSON-object to be read.
///
/// \return Success.
bool read(const std::filesystem::path &inFileName, Json::Value &outJSON) {
  // Check that provided config file actually exists and can be opened.
  std::ifstream ifile(inFileName);
  if (!ifile.good()) {
    return false;
  }

  // Parse JSON.
  Json::CharReaderBuilder builder;
  JSONCPP_STRING errs;
  if (!parseFromStream(builder, ifile, &outJSON, &errs)) {
    std::cerr << errs << std::endl;
    return false;
  }

  // Success!
  return true;
}
} // namespace io
} // namespace conga

#endif // CONGA_IO_JSON_H_
