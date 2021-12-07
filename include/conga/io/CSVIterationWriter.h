//===----------- csv_writers.h - classes for writing csv files -----------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Functionality and classes for writing CSV files.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_CSV_ITERATION_WRITER_H_
#define CONGA_CSV_ITERATION_WRITER_H_

#include "../Context.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace conga {
namespace io {
template <typename T> class CSVIterationWriter {
public:
  CSVIterationWriter(const std::string &inDelimiter = ",",
                     const int inFloatPrecision = 8)
      : m_floatPrecision(inFloatPrecision), m_delimiter(inDelimiter) {}

  bool write(const Context<T> *const inContext,
             const std::filesystem::path &inFilePath) const {
    // Note, this class always appends data to the file. It never removes the
    // file.
    if (!std::filesystem::exists(inFilePath)) {
      if (!appendHeader(inFilePath)) {
        return false;
      };
    }
    return appendRow(inContext, inFilePath);
  }

private:
  const int m_floatPrecision;
  const std::string m_delimiter;

  bool appendHeader(const std::filesystem::path &inFilePath) const {
    std::ostringstream oss;
    oss << "iteration" << m_delimiter << "order_parameter_residual"
        << m_delimiter << "free_energy_residual" << m_delimiter
        << "current_density_residual" << m_delimiter
        << "vector_potential_residual" << m_delimiter
        << "free_energy_per_unit_area";
    return appendToFile(oss.str(), inFilePath);
  };

  bool appendRow(const Context<T> *const inContext,
                 const std::filesystem::path &inFilePath) const {
    // The number of iterations so far.
    const int numIterations = inContext->getNumIterations();

    // Order-parameter residual.
    const bool orderParameterExist = inContext->getComputeOrderParameter();
    const T orderParameterResidual =
        orderParameterExist
            ? inContext->getComputeOrderParameter()->getResidual()
            : static_cast<T>(0);

    // Free-energy residual.
    const bool freeEnergyExist = inContext->getComputeFreeEnergy();
    const T freeEnergyResidual =
        freeEnergyExist ? inContext->getComputeFreeEnergy()->getResidual()
                        : static_cast<T>(0);
    const T freeEnergyPerUnitArea =
        freeEnergyExist
            ? inContext->getComputeFreeEnergy()->getFreeEnergyPerUnitArea()
            : static_cast<T>(0);

    // Current-density residual.
    const bool currentDensityExist = inContext->getComputeCurrent();
    const T currentDensityResidual =
        currentDensityExist ? inContext->getComputeCurrent()->getResidual()
                            : static_cast<T>(0);

    // Vector-potential residual.
    const bool vectorPotentialExist = inContext->getComputeVectorPotential();
    const T vectorPotentialResidual =
        vectorPotentialExist
            ? inContext->getComputeVectorPotential()->getResidual()
            : static_cast<T>(0);

    std::ostringstream oss;
    oss << std::scientific << std::setprecision(m_floatPrecision);
    oss << numIterations << m_delimiter << orderParameterResidual << m_delimiter
        << freeEnergyResidual << m_delimiter << currentDensityResidual
        << m_delimiter << vectorPotentialResidual << m_delimiter
        << freeEnergyPerUnitArea;
    return appendToFile(oss.str(), inFilePath);
  };

  bool appendToFile(const std::string &inLine,
                    std::filesystem::path inFilePath) const {
    std::ofstream outFile(inFilePath, std::ios_base::app);
    if (!outFile.good()) {
      return false;
    }
    outFile << inLine << std::endl;
    return true;
  }
};
} // namespace io
} // namespace conga

#endif // CONGA_CSV_ITERATION_WRITER_H_
