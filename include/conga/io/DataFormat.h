//===--------------- format.h - Enum determining data format. -------------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Enum class controlling data format.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_IO_FORMAT_H_
#define CONGA_IO_FORMAT_H_

namespace conga {
namespace io {
/// \brief Enum controlling data format,  { CSV, H5 }.
enum class DataFormat { CSV, H5 };

/// \brief Convert string to enum.
///
/// \param inDataFormat The data format as a string.
///
/// \return Data format enum.
DataFormat getDataFormat(const std::string &inDataFormat) {
  if (inDataFormat == "csv") {
    return DataFormat::CSV;
  }
#ifdef CONGA_USE_HDF5
  else if (inDataFormat == "h5") {
    return DataFormat::H5;
  }
#endif // CONGA_USE_HDF5
  else {
    // If not understood use CSV.
    // TODO(niclas): Print warning?
    return DataFormat::CSV;
  }
}
} // namespace io
} // namespace conga

#endif // CONGA_IO_FORMAT_H_
