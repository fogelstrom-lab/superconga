// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
#ifndef CONGA_CSV_POSTPROCESS_ITERATION_WRITER_H_
#define CONGA_CSV_POSTPROCESS_ITERATION_WRITER_H_

#include "../Context.h"
#include "CSVIterationWriter.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace conga {
namespace io {
template <typename T>
class CSVPostprocessIterationWriter : public CSVIterationWriter<T> {
public:
  CSVPostprocessIterationWriter(const std::filesystem::path &inFilePath,
                                const std::string &inDelimiter = ",",
                                const int inFloatPrecision = 8)
      : CSVIterationWriter<T>(), m_filePath(inFilePath),
        m_floatPrecision(inFloatPrecision), m_delimiter(inDelimiter) {}

  bool write(const Context<T> &inContext) const override {
    // Note, this class always appends data to the file. It never removes the
    // file.
    const bool fileExists = std::filesystem::exists(m_filePath);
    if (!fileExists) {
      const bool successAddingHeader = appendHeader();
      if (!successAddingHeader) {
        return false;
      }
    }
    return appendRow(inContext);
  }

private:
  const std::filesystem::path m_filePath;
  const int m_floatPrecision;
  const std::string m_delimiter;

  bool appendHeader() const {
    std::ostringstream oss;
    oss << "iteration" << m_delimiter << "boundary_coherence_residual"
        << m_delimiter << "ldos_residual";
    return appendToFile(oss.str());
  };

  bool appendRow(const Context<T> &inContext) const {
    // The number of iterations so far.
    const int numIterations = inContext.getNumIterations();

    // Boundary coherence residual.
    const T boundaryCoherenceResidual =
        inContext.getRiccatiSolver()->getResidual();

    // LDOS residual.
    const bool LDOSExist = inContext.getComputeLDOS();
    const T LDOSResidual = LDOSExist ? inContext.getComputeLDOS()->getResidual()
                                     : static_cast<T>(0);

    // Write to stream.
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(m_floatPrecision);
    oss << numIterations << m_delimiter << boundaryCoherenceResidual
        << m_delimiter << LDOSResidual;

    return appendToFile(oss.str());
  };

  bool appendToFile(const std::string &inLine) const {
    std::ofstream outFile(m_filePath, std::ios_base::app);
    if (!outFile.good()) {
      return false;
    }
    outFile << inLine << std::endl;
    return true;
  }
};
} // namespace io
} // namespace conga

#endif // CONGA_CSV_POSTPROCESS_ITERATION_WRITER_H_
