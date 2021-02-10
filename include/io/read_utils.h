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

namespace scg {
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
// in the Optimizer. Perhaps move both to a better place.
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
  // TODO(niclas): This is not necessary. We can resample the grid.
  const std::size_t expectedSize = static_cast<std::size_t>(dimX * dimY);
  if ((cpuX.size() != expectedSize) || (cpuY.size() != expectedSize)) {
    std::cerr << "-- ERROR! The vector-potential does not have the expected "
                 "dimensions."
              << std::endl;
    return false;
  }

  // Copy to the GPU.
  const thrust::device_vector<T> gpuX = cpuX;
  const thrust::device_vector<T> gpuY = cpuY;

  // Kernel parameters.
  dim3 numBlocks;
  dim3 numThreadsPerBlock;
  inOutVectorPotential.getKernelDimensions(numBlocks, numThreadsPerBlock);

  // Call CUDA kernel.
  internal::copy<<<numBlocks, numThreadsPerBlock>>>(
      dimX, dimY, thrust::raw_pointer_cast(gpuX.data()),
      thrust::raw_pointer_cast(gpuY.data()),
      inOutVectorPotential.getDataPointer(0),
      inOutVectorPotential.getDataPointer(1));

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
  const int numComponents =
      inOutOrderParameter.getNumOrderParameterComponents();

  // The types of the order-parameter components.
  const std::vector<std::string> types = inOutOrderParameter.getTypes();

  for (int i = 0; i < numComponents; ++i) {
    // Get the component grid.
    GridGPU<T> &component =
        *(inOutOrderParameter.getOrderParameterComponentField(i));

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
      // TODO(niclas): Should this be a warning or an error? It is an error for
      // the vector potential.
      std::cerr << "-- WARNING! The file does not contain the columns, '"
                << keyReal << "' and '" << keyImag << "'." << std::endl;
      continue;
    }

    // Get the vectors.
    // Note, 'map' does not have a const overload for the access operator '[]'.
    // So either the map must not be const, or we must use 'at'.
    const std::vector<T> &cpuReal = map.at(keyReal);
    const std::vector<T> &cpuImag = map.at(keyImag);

    // Sanity check the sizes.
    // TODO(niclas): This is not necessary. We can resample the grid.
    const std::size_t expectedSize = static_cast<std::size_t>(dimX * dimY);
    if ((cpuReal.size() != expectedSize) || (cpuImag.size() != expectedSize)) {
      std::cerr << "-- ERROR! The order-parameter does not have the expected "
                   "dimensions."
                << std::endl;
      return false;
    }

    // Copy to the GPU.
    const thrust::device_vector<T> gpuReal = cpuReal;
    const thrust::device_vector<T> gpuImag = cpuImag;

    // Kernel parameters.
    dim3 numBlocks;
    dim3 numThreadsPerBlock;
    component.getKernelDimensions(numBlocks, numThreadsPerBlock);

    // Call CUDA kernel.
    internal::copy<<<numBlocks, numThreadsPerBlock>>>(
        dimX, dimY, thrust::raw_pointer_cast(gpuReal.data()),
        thrust::raw_pointer_cast(gpuImag.data()),
        component.getDataPointer(0, Type::real),
        component.getDataPointer(0, Type::imag));
  }

  return true;
}
} // namespace io
} // namespace scg

#endif // READ_UTILS_H_
