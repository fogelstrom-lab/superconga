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
#ifndef CONGA_CSV_GRID_WRITER_H_
#define CONGA_CSV_GRID_WRITER_H_

#include "../Context.h"
#include "../Type.h"
#include "../grid.h"
#include "../grid/GridData.h"

#include <thrust/host_vector.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

namespace conga {
namespace io {
template <typename T> class CSVGridWriter {
public:
  CSVGridWriter(const bool inWriteIndices = true,
                const std::string &inDelimiter = ",",
                const int inFloatPrecision = 8)
      : m_writeIndices(inWriteIndices), m_delimiter(inDelimiter),
        m_floatPrecision(inFloatPrecision) {}

  bool write(const GridGPU<T> &inGrid,
             const std::vector<std::string> &inFieldTitles,
             const std::filesystem::path &inFilePath) const {
    const bool success = saveInternal(inGrid, inFieldTitles, inFilePath);
    return success;
  }

  bool write(const GridCPU<T> &inGrid,
             const std::vector<std::string> &inFieldTitles,
             const std::filesystem::path &inFilePath) const {
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
    if (inFieldTitles.size() != static_cast<std::size_t>(inGrid.numFields())) {
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

#endif // CONGA_CSV_GRID_WRITER_H_
