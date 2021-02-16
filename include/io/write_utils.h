#ifndef WRITE_UTILS_H_
#define WRITE_UTILS_H_

#include "../Type.h"
#include "csv_writers.h"
#include <memory>
#include <string>

namespace scg {
namespace io {
inline std::string getFileExtension(Type::fileFormat format) {
  switch (format) {
  case Type::binary: {
    return ".grd";
  }
  case Type::csv: {
    return ".csv";
  }
  default: {
    return "";
  }
  }
}

/// \brief Get IIterationDataSaveStrategy<T>* for concrete implementation
///
/// Return a pointer to the singleton instace of the interface implementation.
/// The implementation should be stateless in order to be used safely this way.
///
/// \param format - save format enum (see Type::fileFormat).
///
/// \return - IIterationDataSaveStrategy<T>*
template <typename T>
std::unique_ptr<IIterationDataSaveStrategy<T>>
getIterationDataSaver(const Type::fileFormat format) {
  if (format == Type::csv) {
    return std::make_unique<CSVIterationDataSaveStrategy<T>>();
  }
  return nullptr;
}

/// \brief Get ITextGridDataSaveStrategy<T>* for concrete implementation
///
/// Return a pointer to the singleton instace of the interface implementation.
/// The implementation should be stateless in order to be used safely this way.
///
/// \param format - save format enum (see Type::fileFormat).
///
/// \return - ITextGridDataSaveStrategy<T>*
template <typename T>
std::unique_ptr<ITextGridDataSaveStrategy<T>>
getTextGridDataSaver(const Type::fileFormat format) {
  if (format == Type::csv) {
    return std::make_unique<CSVGridDataSaveStrategy<T>>();
  }
  return nullptr;
}
} // namespace io
} // namespace scg

#endif // WRITE_UTILS_H_
