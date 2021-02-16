#ifndef WRITE_UTILS_H_
#define WRITE_UTILS_H_

#include "../Type.h"
#include "csv_writers.h"
#include <json/json.h>
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

/// \brief Save JSON to file.
///
/// \param inJSON JSON object with parameter values.
/// \param inFile Full path and filename of file to save.
/// \param inVerbose Whether or not to print when file is written.
///
/// \return Void.
inline void writeJSON(const Json::Value &inJSON,
                      const std::filesystem::path &inFile,
                      const bool inVerbose = false) {

  std::ofstream ofstreamJSON(inFile);
  const int indentation = 4;
  ofstreamJSON << std::setw(indentation) << inJSON << std::endl;

  if (inVerbose) {
    std::cout << "-- Wrote: " << inFile << std::endl;
  }
}
} // namespace io
} // namespace scg

#endif // WRITE_UTILS_H_
