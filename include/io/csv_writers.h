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
#ifndef CSV_WRITERS_H_
#define CSV_WRITERS_H_

#include "../Context.h"
#include "../Grid.h"
#include "../GridData.h"
#include "../Type.h"
#include <filesystem>
#include <iostream>
#include <string>
#include <thrust/host_vector.h>
#include <type_traits>
#include <vector>

namespace conga {
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
                    const std::filesystem::path &inFilePath) const = 0;

  virtual ~IIterationDataSaveStrategy() = default;
};

template <typename T>
class CSVIterationDataSaveStrategy : public IIterationDataSaveStrategy<T> {
public:
  CSVIterationDataSaveStrategy(const std::string &inDelimiter = ",",
                               const int inFloatPrecision = 8)
      : m_floatPrecision(inFloatPrecision), m_delimiter(inDelimiter) {}

  bool save(const Context<T> *const inContext,
            const std::filesystem::path &inFilePath) const override {
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

/// \brief Interface for saving GridGPU data type in textual format
///
/// Interface for a class which encapsulates saving GridGPU instances.
/// Should be implemented for each save strategy (e.g. new save format,
/// or different type of info to be stored).
template <typename T> class ITextGridDataSaveStrategy {
public:
  virtual bool save(const GridGPU<T> &inGrid,
                    const std::vector<std::string> &inFieldTitles,
                    const std::filesystem::path &inFilePath) const = 0;

  virtual bool save(const GridCPU<T> &inGrid,
                    const std::vector<std::string> &inFieldTitles,
                    const std::filesystem::path &inFilePath) const = 0;

  virtual ~ITextGridDataSaveStrategy() = default;
};

template <typename T>
class CSVGridDataSaveStrategy : public ITextGridDataSaveStrategy<T> {
public:
  CSVGridDataSaveStrategy(const bool inWriteIndices = true,
                          const std::string &inDelimiter = ",",
                          const int inFloatPrecision = 8)
      : m_writeIndices(inWriteIndices), m_delimiter(inDelimiter),
        m_floatPrecision(inFloatPrecision) {}

  bool save(const GridGPU<T> &inGrid,
            const std::vector<std::string> &inFieldTitles,
            const std::filesystem::path &inFilePath) const override {
    const bool success = saveInternal(inGrid, inFieldTitles, inFilePath);
    return success;
  }

  bool save(const GridCPU<T> &inGrid,
            const std::vector<std::string> &inFieldTitles,
            const std::filesystem::path &inFilePath) const override {
    const bool success = saveInternal(inGrid, inFieldTitles, inFilePath);
    return success;
  }

private:
  const std::string m_delimiter;
  const bool m_writeIndices;
  const int m_floatPrecision;

  template <class Device>
  bool saveInternal(const GridData<T, Device> &inGrid,
                    const std::vector<std::string> &inFieldTitles,
                    const std::filesystem::path &inFilePath) const {
    bool fieldsAsIndex = false;
    if (inFieldTitles.size() != inGrid.numFields()) {
      if (inFieldTitles.size() == 1) {
        // If one field title is provided, put the fields in the grid as a third
        // dimension.
        fieldsAsIndex = true;
      } else {
        // Otherwise, this does not make sense.
        return false;
      }
    }

    // Remove the old file.
    std::filesystem::remove(inFilePath);

    // If it still exists, something has gone wrong.
    if (std::filesystem::exists(inFilePath)) {
      return false;
    }

    if (!appendHeader(inFieldTitles, inGrid.isComplex(), inFilePath,
                      fieldsAsIndex)) {
      return false;
    }

    return appendData(inGrid, inFilePath, fieldsAsIndex);
  }

  bool appendHeader(const std::vector<std::string> &inColumnTitles,
                    const bool isComplex,
                    const std::filesystem::path &inFilePath,
                    const bool inFieldsAsIndex) const {
    std::ostringstream oss;
    if (m_writeIndices) {
      oss << "x_idx" << m_delimiter << "y_idx" << m_delimiter;
      if (inFieldsAsIndex) {
        // TODO(niclas): Should this be called "e_idx" for all grids?
        oss << "e_idx" << m_delimiter;
      }
    }
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

  template <class Device>
  bool appendData(const GridData<T, Device> &inGrid,
                  const std::filesystem::path &inFilePath,
                  const bool inFieldsAsIndex) const {
    const thrust::host_vector<T> &gridData = inGrid.getVector();
    const T *data = thrust::raw_pointer_cast(gridData.data());

    const bool isComplex = inGrid.isComplex();
    const int numFields = inGrid.numFields();

    const int dimX = inGrid.dimX();
    const int dimY = inGrid.dimY();

    if (inFieldsAsIndex) {
      for (int z = 0; z < numFields; ++z) {
        // Get the pointers to the real/imag parts for this field.
        const T *const dataReal =
            data + inGrid.getFieldComponentOffset(z, Type::real);
        const T *const dataImag =
            isComplex ? data + inGrid.getFieldComponentOffset(z, Type::imag)
                      : nullptr;

        // Loop over spatial coordinates.
        for (int y = 0; y < dimY; ++y) {
          for (int x = 0; x < dimX; ++x) {
            // Create stream.
            std::ostringstream oss;
            oss << std::scientific << std::setprecision(m_floatPrecision);

            // Write indices.
            if (m_writeIndices) {
              oss << x << m_delimiter << y << m_delimiter << z << m_delimiter;
            }
            // Linear index.
            const int idx = y * dimX + x;

            // Write real part to stream.
            const T real = dataReal[idx];
            if (std::is_integral<T>::value) {
              oss << static_cast<int>(real);
            } else {
              oss << real;
            }

            // Write imag part to stream.
            if (isComplex) {
              const T imag = dataImag[idx];
              if (std::is_integral<T>::value) {
                oss << m_delimiter << static_cast<int>(imag);
              } else {
                oss << m_delimiter << imag;
              }
            }

            // Write to file.
            if (!appendToFile(oss.str(), inFilePath)) {
              return false;
            }
          }
        }
      }
    } else {
      // Create a vector of pointer to the real/imag parts of each field.
      std::vector<const T *> fieldPtrs;
      fieldPtrs.reserve(numFields * (1 + int(isComplex)));
      for (int f = 0; f < numFields; ++f) {
        fieldPtrs.emplace_back(data +
                               inGrid.getFieldComponentOffset(f, Type::real));
        if (isComplex) {
          fieldPtrs.emplace_back(data +
                                 inGrid.getFieldComponentOffset(f, Type::imag));
        }
      }

      // Loop over spatial coordinates.
      for (int y = 0; y < dimY; ++y) {
        for (int x = 0; x < dimX; ++x) {
          // Create stream.
          std::ostringstream oss;
          oss << std::scientific << std::setprecision(m_floatPrecision);

          // Write indices.
          if (m_writeIndices) {
            oss << x << m_delimiter << y << m_delimiter;
          }

          // Linear index.
          const int idx = y * dimX + x;

          // Loop over all pointers, adn write to stream.
          for (auto &&fieldPtr : fieldPtrs) {
            const T val = fieldPtr[idx];
            if (std::is_integral<T>::value) {
              oss << static_cast<int>(val);
            } else {
              oss << val;
            }

            if (fieldPtr != fieldPtrs.back()) {
              oss << m_delimiter;
            }
          }

          // Write to file.
          if (!appendToFile(oss.str(), inFilePath)) {
            return false;
          }
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
} // namespace conga

#endif // CSV_WRITERS_H_
