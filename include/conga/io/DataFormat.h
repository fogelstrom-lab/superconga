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
/// \brief Enum controlling data format, {H5, CSV}.
enum class DataFormat { H5, CSV };

/// \brief Convert string to enum.
///
/// \param inDataFormat The data format as a string.
///
/// \return Data format enum.
DataFormat getDataFormat(const std::string &inDataFormat) {
  if (inDataFormat == "csv") {
    return DataFormat::CSV;
  } else if (inDataFormat == "h5") {
    return DataFormat::H5;
  } else {
    // If not understood use H5.
    // TODO(niclas): Print warning?
    return DataFormat::H5;
  }
}
} // namespace io
} // namespace conga

#endif // CONGA_IO_FORMAT_H_
