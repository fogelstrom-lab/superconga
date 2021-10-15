//===---- read_utils.h - utils functions for reading data from files. ----===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Utility functions for reading data from files.
///
//===----------------------------------------------------------------------===//
#ifndef READ_UTILS_H_
#define READ_UTILS_H_

#include "../Grid.h"
#include "../OrderParameter.h"
#include "../Type.h"
#include "csv_reader.h"
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <thrust/device_vector.h>
#include <vector>

namespace conga {
namespace io {
namespace internal {
/// \brief CUDA kernel for copying two arrays, to two other arrays.
///
/// E.g. the x/y components of the vector potential, or the real/imag parts of
/// the order-parameter component.
///
/// \param inDimX The number of lattice sites along the x-axis.
/// \param inDimY The number of lattice sites along the y-axis.
/// \param inSrc1 The first array to copy from, i.e. x or real.
/// \param inSrc2 The second array to copy from, i.e. y or imag.
/// \param outDst1 The destination of the the first source array.
/// \param outDst2 The destination of the the second source array.
///
/// \return Void.
// TODO(niclas): This is essentially identical to the kernel with the same name
// in the Accelerator. Perhaps move both to a better place.
template <typename T>
__global__ void copy(const int inDimX, const int inDimY, const T *const inSrc1,
                     const T *const inSrc2, T *const outDst1,
                     T *const outDst2) {
  // Indices of spatial coordinates.
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x < inDimX) && (y < inDimY)) {
    const int idx = y * inDimX + x;
    outDst1[idx] = inSrc1[idx];
    outDst2[idx] = inSrc2[idx];
  }
}

/// \brief Copy two vectors to a grid. If the grid is complex it will be the
/// real/imag parts. Otherwise the first/second channel.
///
/// \param inSrc1 The first vector.
/// \param inSrc2 The second vector.
/// \param inOutDst The grid to copy the vectors into.
template <typename T>
void copy(const std::vector<T> inSrc1, const std::vector<T> inSrc2,
          GridGPU<T> &inOutDst) {
  // Copy to the GPU.
  const thrust::device_vector<T> src1 = inSrc1;
  const thrust::device_vector<T> src2 = inSrc2;

  // Kernel parameters.
  dim3 numBlocks;
  dim3 numThreadsPerBlock;
  inOutDst.getKernelDimensions(numBlocks, numThreadsPerBlock);

  const int dimX = inOutDst.dimX();
  const int dimY = inOutDst.dimY();
  const bool isComplex = inOutDst.isComplex();

  // Call CUDA kernel.
  if (isComplex) {
    copy<<<numBlocks, numThreadsPerBlock>>>(
        dimX, dimY, thrust::raw_pointer_cast(src1.data()),
        thrust::raw_pointer_cast(src2.data()),
        inOutDst.getDataPointer(0, Type::real),
        inOutDst.getDataPointer(0, Type::imag));
  } else {
    copy<<<numBlocks, numThreadsPerBlock>>>(
        dimX, dimY, thrust::raw_pointer_cast(src1.data()),
        thrust::raw_pointer_cast(src2.data()), inOutDst.getDataPointer(0),
        inOutDst.getDataPointer(1));
  }
}
} // namespace internal

template <typename T>
inline bool readVectorPotential(const std::filesystem::path &inFileName,
                                GridGPU<T> &inOutVectorPotential) {
  // Get the dimensions of the grid.
  const int dimX = inOutVectorPotential.dimX();
  const int dimY = inOutVectorPotential.dimY();
  const int numFields = inOutVectorPotential.numFields();
  const Type::type representation = inOutVectorPotential.representation();

  // Sanity check the number of components (fields).
  if (numFields != 2) {
    std::cerr << "-- ERROR! The vector potential must have two components."
              << std::endl;
    return false;
  }

  // Sanity check the representation.
  if (representation != Type::real) {
    std::cerr << "-- ERROR! The vector potential must be a real-valued grid."
              << std::endl;
    return false;
  }

  // Read the CSV-file to a map.
  const CSVReader<T> csvReader(inFileName);

  // Abort if something went wrong.
  if (!csvReader.success()) {
    return false;
  }

  // Get the resulting map.
  const std::map<std::string, std::vector<T>> &map = csvReader.getMap();

  // Construct the keys.
  const std::string keyX = "a_x";
  const std::string keyY = "a_y";

  // Check if the keys exist.
  const bool keyXExists = csvReader.contains(keyX);
  const bool keyYExists = csvReader.contains(keyY);

  if (!(keyXExists && keyYExists)) {
    std::cerr << "-- ERROR! The file does not contain the columns, '" << keyX
              << "' and '" << keyY << "'." << std::endl;
    return false;
  }

  // Get the vectors.
  // Note, 'map' does not have a const overload for the access operator '[]'. So
  // either the map must not be const, or we must use 'at'.
  const std::vector<T> &cpuX = map.at(keyX);
  const std::vector<T> &cpuY = map.at(keyY);

  // Sanity check the sizes.
  const std::size_t expectedSize = static_cast<std::size_t>(dimX * dimY);
  if ((cpuX.size() != expectedSize) || (cpuY.size() != expectedSize)) {
    // Get the actual lattice side-length in the read data.
    const int N = static_cast<int>(std::sqrt(static_cast<T>(cpuX.size())));

    // Check if it is a square.
    if (N * N != static_cast<int>(cpuX.size())) {
      std::cerr << "-- ERROR! The data is not square." << std::endl;
      return false;
    }

    // Resize the vector potential.
    inOutVectorPotential = GridGPU<T>(N, N, numFields, representation);

    // Copy data.
    internal::copy(cpuX, cpuY, inOutVectorPotential);

    // Resize (via interpolation), to the desired size.
    // TODO(niclas): There is an undesired effect of zeros outside of the grain
    // leaking in.
    inOutVectorPotential.resample(dimX, dimY);
  } else {
    // Copy data.
    internal::copy(cpuX, cpuY, inOutVectorPotential);
  }

  return true;
}

template <typename T>
inline bool readOrderParameter(const std::filesystem::path &inFileName,
                               OrderParameter<T> &inOutOrderParameter) {
  // Read the CSV-file to a map.
  const CSVReader<T> csvReader(inFileName);

  // Abort if something went wrong.
  if (!csvReader.success()) {
    return false;
  }

  // Get the resulting map.
  const std::map<std::string, std::vector<T>> &map = csvReader.getMap();

  // The number of order-parameter components.
  const int numComponents = inOutOrderParameter.getNumComponents();

  // The types of the order-parameter components.
  const std::vector<std::string> types = inOutOrderParameter.getTypes();

  for (int i = 0; i < numComponents; ++i) {
    // Get the component grid.
    GridGPU<T> &component = *(inOutOrderParameter.getComponentGrid(i));

    // Get the dimensions of the grid.
    const int dimX = component.dimX();
    const int dimY = component.dimY();

    // Get the type of the component, e.g. 'dx2-y2'.
    const std::string type = types[i];

    // Construct the keys.
    const std::string keyReal = type + "_re";
    const std::string keyImag = type + "_im";

    // Check if the keys exist.
    const bool keyRealExists = csvReader.contains(keyReal);
    const bool keyImagExists = csvReader.contains(keyImag);

    if (!(keyRealExists && keyImagExists)) {
      std::cerr << "-- ERROR! The file does not contain the columns, '"
                << keyReal << "' and '" << keyImag << "'." << std::endl;
      return false;
    }

    // Get the vectors.
    // Note, 'map' does not have a const overload for the access operator '[]'.
    // So either the map must not be const, or we must use 'at'.
    const std::vector<T> &cpuReal = map.at(keyReal);
    const std::vector<T> &cpuImag = map.at(keyImag);

    // Sanity check the sizes.
    const std::size_t expectedSize = static_cast<std::size_t>(dimX * dimY);
    if ((cpuReal.size() != expectedSize) || (cpuImag.size() != expectedSize)) {
      // Get the actual lattice side-length in the read data.
      const int N = static_cast<int>(std::sqrt(static_cast<T>(cpuReal.size())));

      // Check if it is a square.
      if (N * N != static_cast<int>(cpuReal.size())) {
        std::cerr << "-- ERROR! The data is not square." << std::endl;
        return false;
      }

      // Resize the component.
      const T numFields = 1;
      const Type::type representation = Type::complex;
      component = GridGPU<T>(N, N, numFields, representation);

      // Copy data.
      internal::copy(cpuReal, cpuImag, component);

      // Resize (via interpolation), to the desired size.
      // TODO(niclas): There is an undesired effect of zeros outside of the
      // grain leaking in.
      component.resample(dimX, dimY);
    } else {
      // Copy data.
      internal::copy(cpuReal, cpuImag, component);
    }
  }

  return true;
}
} // namespace io
} // namespace conga

#endif // READ_UTILS_H_
