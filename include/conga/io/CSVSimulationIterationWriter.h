// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
#ifndef CONGA_CSV_SIMULATION_ITERATION_WRITER_H_
#define CONGA_CSV_SIMULATION_ITERATION_WRITER_H_

#include "../Context.h"
#include "CSVIterationWriter.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace conga {
namespace io {
template <typename T>
class CSVSimulationIterationWriter : public CSVIterationWriter<T> {
public:
  CSVSimulationIterationWriter(const std::filesystem::path &inFilePath,
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
        << m_delimiter << "order_parameter_residual" << m_delimiter
        << "current_density_residual" << m_delimiter
        << "vector_potential_residual" << m_delimiter << "free_energy_residual"
        << m_delimiter << "free_energy_per_unit_area";

    return appendToFile(oss.str());
  };

  bool appendRow(const Context<T> &inContext) const {
    // The number of iterations so far.
    const int numIterations = inContext.getNumIterations();

    // Boundary coherence residual.
    const T boundaryCoherenceResidual =
        inContext.getRiccatiSolver()->getResidual();

    // Order-parameter residual.
    const bool orderParameterExist = inContext.getComputeOrderParameter();
    const T orderParameterResidual =
        orderParameterExist
            ? inContext.getComputeOrderParameter()->getResidual()
            : static_cast<T>(0);

    // Current-density residual.
    const bool currentDensityExist = inContext.getComputeCurrent();
    const T currentDensityResidual =
        currentDensityExist ? inContext.getComputeCurrent()->getResidual()
                            : static_cast<T>(0);

    // Vector-potential residual.
    const bool vectorPotentialExist = inContext.getComputeVectorPotential();
    const T vectorPotentialResidual =
        vectorPotentialExist
            ? inContext.getComputeVectorPotential()->getResidual()
            : static_cast<T>(0);

    // Free-energy residual.
    const bool freeEnergyExist = inContext.getComputeFreeEnergy();
    const T freeEnergyResidual =
        freeEnergyExist ? inContext.getComputeFreeEnergy()->getResidual()
                        : static_cast<T>(0);
    const T freeEnergyPerUnitArea =
        freeEnergyExist
            ? inContext.getComputeFreeEnergy()->getFreeEnergyPerUnitArea()
            : static_cast<T>(0);

    // Write to stream.
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(m_floatPrecision);
    oss << numIterations << m_delimiter << boundaryCoherenceResidual
        << m_delimiter << orderParameterResidual << m_delimiter
        << currentDensityResidual << m_delimiter << vectorPotentialResidual
        << m_delimiter << freeEnergyResidual << m_delimiter
        << freeEnergyPerUnitArea;

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

#endif // CONGA_CSV_SIMULATION_ITERATION_WRITER_H_
