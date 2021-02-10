#ifndef CSV_WRITERS_H_
#define CSV_WRITERS_H_

#include "../Context.h"
#include "../Grid.h"
#include "../Type.h"
#include <filesystem>
#include <iostream>
#include <string>
#include <thrust/host_vector.h>
#include <vector>

namespace scg {
namespace io {
//===------------------------------ Classes------------------------------===//
/// \brief Interface for saving per-iteration data
///
/// Interface for a class which encapsulates data saving on a per-iteration
/// basis. Should be implemented for each save strategy (e.g. new save format,
/// or different type of info to be stored).
template <typename T> class IIterationDataSaveStrategy {
public:
  virtual bool save(const Context<T> *const inContext,
                    const std::filesystem::path &inFilePath) = 0;

  virtual ~IIterationDataSaveStrategy() = default;
};

template <typename T>
class CSVIterationDataSaveStrategy : public IIterationDataSaveStrategy<T> {
public:
  CSVIterationDataSaveStrategy() = default;

  CSVIterationDataSaveStrategy(const std::string &inDelimiter,
                               const int inFloatPrecision = 10)
      : m_floatPrecision(inFloatPrecision), m_delimiter(inDelimiter) {}

  bool save(const Context<T> *const inContext,
            const std::filesystem::path &inFilePath) override {
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
  const int m_floatPrecision = 10;
  const std::string m_delimiter = ",";

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

    std::ostringstream oss;
    oss << inContext->getNumIterations() << m_delimiter
        << inContext->getComputeOrderParameter()->getResidual() << m_delimiter
        << inContext->getComputeFreeEnergy()->getResidual() << m_delimiter
        << inContext->getComputeCurrent()->getResidual() << m_delimiter
        << inContext->getComputeVectorPotential()->getResidual() << m_delimiter
        << std::setprecision(m_floatPrecision)
        << inContext->getComputeFreeEnergy()->getFreeEnergyPerUnitArea();
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

/// \brief Interface for saving GridGPU data type in textual format
///
/// Interface for a class which encapsulates saving GridGPU instances.
/// Should be implemented for each save strategy (e.g. new save format,
/// or different type of info to be stored).
template <typename T> class ITextGridDataSaveStrategy {
public:
  virtual bool save(const GridGPU<T> *const inGridGPU,
                    const std::vector<std::string> &inFieldTitles,
                    const std::filesystem::path &inFilePath) = 0;

  virtual ~ITextGridDataSaveStrategy() = default;
};

template <typename T>
class CSVGridDataSaveStrategy : public ITextGridDataSaveStrategy<T> {
public:
  CSVGridDataSaveStrategy() = default;

  CSVGridDataSaveStrategy(const std::string &inDelimiter,
                          const int inFloatPrecision = 10)
      : m_floatPrecision(inFloatPrecision), m_delimiter(inDelimiter) {}

  bool save(const GridGPU<T> *const inGridGPU,
            const std::vector<std::string> &inFieldTitles,
            const std::filesystem::path &inFilePath) override {

    if (inFieldTitles.size() != inGridGPU->numFields()) {
      return false;
    }

    // Remove the old file.
    std::filesystem::remove(inFilePath);

    if (!std::filesystem::exists(inFilePath)) {
      if (!appendHeader(inFieldTitles, inGridGPU->isComplex(), inFilePath)) {
        return false;
      };
    }
    return appendData(inGridGPU, inFilePath);
  }

private:
  const int m_floatPrecision = 10;
  const std::string m_delimiter = ",";

  bool appendHeader(const std::vector<std::string> &inColumnTitles,
                    const bool isComplex,
                    const std::filesystem::path &inFilePath) const {
    std::ostringstream oss;
    oss << "x_idx" << m_delimiter << "y_idx" << m_delimiter;
    for (const auto &columnTitle : inColumnTitles) {
      oss << columnTitle;

      if (isComplex) {
        oss << "_re" << m_delimiter << columnTitle << "_im";
      }

      if (columnTitle != inColumnTitles.back()) {
        oss << m_delimiter;
      }
    }
    return appendToFile(oss.str(), inFilePath);
  };

  bool appendData(const GridGPU<T> *const inGridGPU,
                  const std::filesystem::path &inFilePath) const {
    const thrust::host_vector<T> gridData = *(inGridGPU->getVector());
    const T *data = thrust::raw_pointer_cast(gridData.data());

    const bool isComplex = inGridGPU->isComplex();
    const int numFields = inGridGPU->numFields();
    std::vector<const T *> fieldPtrs;
    fieldPtrs.reserve(numFields * (1 + int(isComplex)));
    for (int f = 0; f < numFields; ++f) {
      fieldPtrs.emplace_back(data +
                             inGridGPU->getFieldComponentOffset(f, Type::real));
      if (isComplex) {
        fieldPtrs.emplace_back(
            data + inGridGPU->getFieldComponentOffset(f, Type::imag));
      }
    }

    const int dimX = inGridGPU->dimX();
    const int dimY = inGridGPU->dimY();
    for (int j = 0; j < dimY; ++j) {
      for (int i = 0; i < dimX; ++i) {
        const int idx = j * dimX + i;
        std::ostringstream oss;
        oss << i << m_delimiter << j << m_delimiter;
        for (auto &&fieldPtr : fieldPtrs) {
          oss << fieldPtr[idx];
          if (fieldPtr != fieldPtrs.back()) {
            oss << m_delimiter;
          }
        }
        if (!appendToFile(oss.str(), inFilePath)) {
          return false;
        }
      }
    }
    return true;
  }

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
} // namespace scg

#endif // CSV_WRITERS_H_
