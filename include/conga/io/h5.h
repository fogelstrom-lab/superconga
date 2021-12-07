//===--------------------- h5.h - Read/write H5 files. --------------------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Functions for reading/writing H5 files.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_IO_H5_H_
#define CONGA_IO_H5_H_

#include "../Type.h"
#include "../geometry/domain_ops.h"
#include "../grid/GridData.h"
#include "../order_parameter/OrderParameter.h"

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5Easy.hpp>
#include <highfive/H5File.hpp>
#include <thrust/host_vector.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace conga {
namespace io {
// ========================== WRITE ========================== //

/// \brief Get and HighFive file for overwrite.
///
/// \param inFileName Path to H5 file.
///
/// \return The file.
H5Easy::File fileOverWrite(const std::filesystem::path &inFileName) {
  return H5Easy::File(inFileName, H5Easy::File::Overwrite);
}

/// \brief Write a grid to an H5 file.
///
/// \param inGrid The grid to be written.
/// \param inName Name of the dataset.
/// \param inOutFile The HighFive file to write to.
///
/// \return Void.
template <typename T, class Device>
void write(const GridData<T, Device> &inGrid, const std::string &inName,
           HighFive::File &inOutFile) {
  // Copy the data into an std::vector.
  // TODO(niclas): It would be nice not having to copy it twice.
  const thrust::host_vector<T> tmp = inGrid.getVector();
  const std::vector<T> data(tmp.begin(), tmp.end());

  // Dump compressed data.
  H5Easy::dump(inOutFile, inName, data,
               H5Easy::DumpOptions(H5Easy::Compression()));

  // Get info.
  const int dimX = inGrid.dimX();
  const int dimY = inGrid.dimY();
  const int dimZ = inGrid.numFields();
  const bool isComplex = inGrid.isComplex();

  // Dump attributes.
  H5Easy::dumpAttribute(inOutFile, inName, "dim_x", dimX);
  H5Easy::dumpAttribute(inOutFile, inName, "dim_y", dimY);
  H5Easy::dumpAttribute(inOutFile, inName, "dim_z", dimZ);
  H5Easy::dumpAttribute(inOutFile, inName, "is_complex", isComplex);
}

/// \brief Write an order parameter to an H5 file.
///
/// \param inOrderParameter The order parameter to be written.
/// \param inName Name of the dataset, e.g. "order_parameter".
/// \param inOutFile The HighFive file to write to.
///
/// \return Void.
template <typename T>
void write(const OrderParameter<T> &inOrderParameter, const std::string &inName,
           HighFive::File &inOutFile) {
  // Write order parameter.
  const int numComponents = inOrderParameter.getNumComponents();
  const std::vector<std::string> types = inOrderParameter.getTypes();
  for (int i = 0; i < numComponents; ++i) {
    const std::string name = inName + std::string("_") + types[i];
    const auto &grid = *(inOrderParameter.getComponentGrid(i));
    write(grid, name, inOutFile);
  }
}

// ========================== READ ========================== //

namespace internal {
/// \brief Helper function for reading a grid from a file.
///
/// \param inDefinedEverywhere Whether the grid is defined everywhere or only in
/// the grain.
/// \param inFile The HighFive file to read from.
/// \param inName Name of the dataset.
/// \param outGrid The read grid.
///
/// \return Success.
template <typename T, class Device>
bool read(const bool inDefinedEverywhere, const HighFive::File &inFile,
          const std::string &inName, GridData<T, Device> &inOutGrid) {
  // Check if the dataset exists.
  const bool exist = inFile.exist(inName);
  if (!exist) {
    return false;
  }

  // Fetch dataset.
  HighFive::DataSet dataset = inFile.getDataSet(inName);

  // Read data.
  std::vector<T> data;
  dataset.read(data);

  // Read attributes.
  int dimX;
  int dimY;
  int numFields;
  bool isComplex;
  dataset.getAttribute("dim_x").read(dimX);
  dataset.getAttribute("dim_y").read(dimY);
  dataset.getAttribute("dim_z").read(numFields);
  dataset.getAttribute("is_complex").read(isComplex);

  // Check internal dimensions.
  const bool internalDimsEqual = (numFields == inOutGrid.numFields()) &&
                                 (isComplex == inOutGrid.isComplex());
  if (!internalDimsEqual) {
    return false;
  }

  // Check spatial dimensions.
  const bool spatialDimsEqual =
      (dimX == inOutGrid.dimX()) && (dimY == inOutGrid.dimY());

  if (spatialDimsEqual) {
    // Just copy the data and we are done.
    inOutGrid.setVector(data);
  } else {
    // The resolutions are not the same, so we must interpolate.

    // Read the data into a temporary grid.
    const Type::type representation = isComplex ? Type::complex : Type::real;
    GridData<T, Device> tmp(dimX, dimY, numFields, representation);
    tmp.setVector(data);

    if (!inDefinedEverywhere) {
      // Read the domain.
      GridGPU<char> domain(dimX, dimY, 1, Type::real);
      std::vector<char> domainData;
      inFile.getDataSet("domain").read(domainData);
      domain.setVector(domainData);

      // Extend values outside of the domain (to avoid boundary interpolation
      // issues).
      geometry::extendValuesAtBoundary(domain, tmp);
    }

    // Change resolution of the grid.
    const int dimX = inOutGrid.dimX();
    const int dimY = inOutGrid.dimY();
    tmp.resample(dimX, dimY);

    // Copy.
    inOutGrid = tmp;
  }

  // Success!
  return true;
}
} // namespace internal

/// \brief Get and HighFive file for read only.
///
/// \param inFileName Path to H5 file.
///
/// \return The file.
HighFive::File fileReadOnly(const std::filesystem::path &inFileName) {
  return HighFive::File(inFileName, HighFive::File::ReadOnly);
}

/// \brief Read a grid from h5-file.
///
/// \param inFile The HighFive file to read from.
/// \param inName Name of dataset.
/// \param inOutGrid The grid to be read.
///
/// \return Success.
template <typename T, class Device>
bool read(const HighFive::File &inFile, const std::string &inName,
          GridData<T, Device> &inOutGrid) {
  // TODO(niclas): Here I just assume that it is the vector potential that is
  // being read, thus it is defined everywhere.
  const bool definedEverywhere = true;
  const bool exist =
      internal::read(definedEverywhere, inFile, inName, inOutGrid);
  if (!exist) {
    return false;
  }

  // Success!
  return true;
}

/// \brief Read an order parameter from h5-file.
///
/// \param inFile The HighFive file to read from.
/// \param inName Name of dataset.
/// \param inOutOrderParameter The order parameter to be read.
///
/// \return Success.
template <typename T>
bool read(const HighFive::File &inFile, const std::string &inName,
          OrderParameter<T> &inOutOrderParameter) {
  const int numComponents = inOutOrderParameter.getNumComponents();
  const std::vector<std::string> types = inOutOrderParameter.getTypes();

  // The order parameter is only defined in the grain.
  const bool definedEverywhere = false;

  // Read components separately.
  for (int i = 0; i < numComponents; ++i) {
    // Read.
    const std::string name = inName + std::string("_") + types[i];
    auto &grid = *(inOutOrderParameter.getComponentGrid(i));
    const bool exist = internal::read(definedEverywhere, inFile, name, grid);
    if (!exist) {
      // If this component does not exist it is left untouched.
      continue;
    }
  }

  // Success!
  return true;
}
} // namespace io
} // namespace conga

#endif // CONGA_IO_H5_H_
