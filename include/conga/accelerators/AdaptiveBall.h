//===-------------- AdaptiveBall.h - AdaptiveBall accelerator. ------------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Adaptive-ball accelerator.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_ADAPTIVE_BALL_H_
#define CONGA_ADAPTIVE_BALL_H_

#include "../Type.h"
#include "Accelerator.h"
#include "utils.h"

#include <cuda.h>
#include <json/json.h>
#include <thrust/device_vector.h>

#include <cassert>
#include <limits>
#include <vector>

namespace conga {
namespace accelerators {
template <typename T> class AdaptiveBall : public Accelerator<T> {
public:
  /// \brief Constructor of the adaptive-ball accelerator.
  ///
  /// \param inStepSize The step size.
  /// \param inIncreaseFactor The factor to maximally increase the step size
  /// with.
  /// \param inDecreaseFactor The factor to maximally decrease the step size
  /// with.
  explicit AdaptiveBall(const T inStepSize = m_stepSizeDefault,
                        const T inIncreaseFactor = m_increaseDefault,
                        const T inDecreaseFactor = m_decreaseDefault);

  /// \brief Constructor of the adaptive-ball accelerator.
  ///
  /// \param inJSON JSON with parameters. Absent keys get default values.
  explicit AdaptiveBall(const Json::Value &inJSON);

  /// \brief Destructor of the adaptive-ball accelerator.
  ~AdaptiveBall(){};

  /// \brief Compute the next order parameter and vector potential.
  ///
  /// \param inOrderParameterG The right-hand side of the self-consistency
  /// equation for the order parameter, 'g(x)'.
  /// \param inVectorPotentialG The right-hand side of the self-consistency
  /// equation for the vector potential, 'g(x)'.
  /// \param inOutOrderParameterX The left-hand side of the self-consistency
  /// equation for the order parameter, 'x'.
  /// \param inOutVectorPotentialX The left-hand side of the self-consistency
  /// equation for the vector potential, 'x'.
  ///
  /// \return Void.
  void compute(const OrderParameter<T> &inOrderParameterG,
               const GridGPU<T> &inVectorPotentialG,
               OrderParameter<T> &inOutOrderParameterX,
               GridGPU<T> &inOutVectorPotentialX) override;

  /// \brief Compute the next order parameter.
  ///
  /// \param inOrderParameterG The right-hand side of the self-consistency
  /// equation for the order parameter, 'g(x)'.
  /// \param inOutOrderParameterX The left-hand side of the self-consistency
  /// equation for the order parameter, 'x'.
  ///
  /// \return Void.
  void compute(const OrderParameter<T> &inOrderParameterG,
               OrderParameter<T> &inOutOrderParameterX) override;

  /// \brief Compute the next vector potential.
  ///
  /// \param inVectorPotentialG The right-hand side of the self-consistency
  /// equation for the vector potential, 'g(x)'.
  /// \param inOutVectorPotentialX The left-hand side of the self-consistency
  /// equation for the vector potential, 'x'.
  ///
  /// \return Void.
  void compute(const GridGPU<T> &inVectorPotentialG,
               GridGPU<T> &inOutVectorPotentialX) override;

private:
  /// \brief Initialize the running mean, step sizes, and inertias.
  ///
  /// \param inX The left-hand side of the self-consistency equation. x.
  ///
  /// \return Void.
  void initialize(const GridGPU<T> &inX);

  /// \brief Compute the next order parameter and vector potential.
  ///
  /// \param inG The right-hand side of the self-consistency equation, 'g(x)'.
  /// \param inOutX The left-hand side of the self-consistency equation, 'x'.
  ///
  /// \return Void.
  void computeInternal(const GridGPU<T> &inG, GridGPU<T> &inOutX);

  // Defaults.
  static constexpr T m_stepSizeDefault = static_cast<T>(1);
  static constexpr T m_increaseDefault = static_cast<T>(1.2599210498948732);
  static constexpr T m_decreaseDefault = static_cast<T>(0.15915494309189535);

  // The initial step size.
  const T m_stepSizeInit;

  // Factors to increase/decrease the step size.
  const T m_increase;
  const T m_decrease;

  // If initialized.
  bool m_initialized;

  // The running-mean.
  GridGPU<T> m_mean;

  // The step size per "field".
  std::vector<T> m_stepSize;

  // The inertia per "field".
  std::vector<T> m_inertia;
};

template <typename T>
AdaptiveBall<T>::AdaptiveBall(const T inStepSize, const T inIncreaseFactor,
                              const T inDecreaseFactor)
    : Accelerator<T>(), m_stepSizeInit(inStepSize),
      m_increase(inIncreaseFactor), m_decrease(inDecreaseFactor),
      m_initialized(false) {
  assert(m_stepSizeInit > static_cast<T>(0));
  assert(m_increase > static_cast<T>(1));
  assert(m_decrease < static_cast<T>(1));
  assert(m_decrease > static_cast<T>(0));
}

template <typename T>
AdaptiveBall<T>::AdaptiveBall(const Json::Value &inJSON)
    : Accelerator<T>(),
      m_stepSizeInit(inJSON.isMember("step_size")
                         ? static_cast<T>(inJSON["step_size"].asDouble())
                         : m_stepSizeDefault),
      m_increase(inJSON.isMember("step_size_increase")
                     ? static_cast<T>(inJSON["step_size_increase"].asDouble())
                     : m_increaseDefault),
      m_decrease(inJSON.isMember("step_size_decrease")
                     ? static_cast<T>(inJSON["step_size_decrease"].asDouble())
                     : m_decreaseDefault),
      m_initialized(false) {
  assert(m_stepSizeInit > static_cast<T>(0));
  assert(m_increase > static_cast<T>(1));
  assert(m_decrease < static_cast<T>(1));
  assert(m_decrease > static_cast<T>(0));
}

template <typename T> void AdaptiveBall<T>::initialize(const GridGPU<T> &inX) {
  const int dimX = inX.dimX();
  const int dimY = inX.dimY();
  const int numFields = inX.numFields();
  const Type::type representation = inX.representation();

  m_mean = GridGPU<T>(dimX, dimY, numFields, representation);
  m_mean.setZero();

  m_stepSize.resize(numFields);
  std::fill(m_stepSize.begin(), m_stepSize.end(), m_stepSizeInit);

  m_inertia.resize(numFields);
  std::fill(m_inertia.begin(), m_inertia.end(), static_cast<T>(0));

  m_initialized = true;
}

template <typename T>
void AdaptiveBall<T>::compute(const OrderParameter<T> &inOrderParameterG,
                              const GridGPU<T> &inVectorPotentialG,
                              OrderParameter<T> &inOutOrderParameterX,
                              GridGPU<T> &inOutVectorPotentialX) {
  // We copy everything to one grid, so we need the dimensions of everything.
  const int dimX = inVectorPotentialG.dimX();
  const int dimY = inVectorPotentialG.dimY();
  const T numComponents = inOrderParameterG.getNumComponents();
  const int complexFactor = 2;
  const int numFields =
      complexFactor * numComponents + inVectorPotentialG.numFields();
  const Type::type representation = Type::real;

  // Initialize grids. The names come from the self-consistency equation:
  // x = g(x).
  GridGPU<T> X(dimX, dimY, numFields, representation);
  GridGPU<T> G(dimX, dimY, numFields, representation);

  // Concatenate the order parameter and the vector potential to real-valued
  // grids.
  concatenate(inOutOrderParameterX, inOutVectorPotentialX, X);
  concatenate(inOrderParameterG, inVectorPotentialG, G);

  // Take step.
  computeInternal(G, X);

  // Split the grid into its constituent order parameter and vector potential.
  split(X, inOutOrderParameterX, inOutVectorPotentialX);
}

template <typename T>
void AdaptiveBall<T>::compute(const OrderParameter<T> &inOrderParameterG,
                              OrderParameter<T> &inOutOrderParameterX) {
  // We copy everything to one grid, so we need the dimensions of everything.
  const int dimX = inOrderParameterG.getComponentGrid(0)->dimX();
  const int dimY = inOrderParameterG.getComponentGrid(0)->dimY();
  const T numComponents = inOrderParameterG.getNumComponents();
  const int complexFactor = 2;
  const int numFields = complexFactor * numComponents;
  const Type::type representation = Type::real;

  // Initialize grids. The names come from the self-consistency equation:
  // x = g(x).
  GridGPU<T> X(dimX, dimY, numFields, representation);
  GridGPU<T> G(dimX, dimY, numFields, representation);

  // Concatenate the order parameter and the vector potential to real-valued
  // grids.
  concatenate(inOutOrderParameterX, X);
  concatenate(inOrderParameterG, G);

  // Take step.
  computeInternal(G, X);

  // Split the grid into its constituent order parameter and vector potential.
  split(X, inOutOrderParameterX);
}

template <typename T>
void AdaptiveBall<T>::compute(const GridGPU<T> &inVectorPotentialG,
                              GridGPU<T> &inOutVectorPotentialX) {
  // Take step.
  computeInternal(inVectorPotentialG, inOutVectorPotentialX);
}

namespace adaptive_ball_internal {
template <typename T>
__global__ void computeInternal(const int inDimX, const int inDimY,
                                const int inDimZ, const T *const inStepSize,
                                const T *const inInertia, const T *const inDiff,
                                T *const inOutMean, T *const inOutX) {
  // Indices of spatial coordinates.
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Field.
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < inDimX && y < inDimY && z < inDimZ) {
    // Linear index.
    const int idx = inDimX * (z * inDimY + y) + x;

    // Update mean.
    const T inertia = inInertia[z];
    const T mean =
        inOutMean[idx] * inertia + inDiff[idx] * (static_cast<T>(1) - inertia);
    inOutMean[idx] = mean;

    // Compute the result.
    inOutX[idx] += inStepSize[z] * mean;
  }
}
} // namespace adaptive_ball_internal

template <typename T>
void AdaptiveBall<T>::computeInternal(const GridGPU<T> &inG,
                                      GridGPU<T> &inOutX) {
  // Initialize grids.
  if (!m_initialized) {
    initialize(inOutX);
  }

  // Dimensions.
  const int dimX = inOutX.dimX();
  const int dimY = inOutX.dimY();
  const int numFields = inOutX.numFields();

  // The difference.
  const GridGPU<T> diff = inG - inOutX;

  // Use for changing the step size and inertia.
  const GridGPU<T> D2 = diff * diff;
  const GridGPU<T> M2 = m_mean * m_mean;
  const GridGPU<T> DM = diff * m_mean;

  for (int i = 0; i < numFields; ++i) {
    // Compute the cosine similarity between the diff and the
    // previous step.
    const T dm = DM.sumField(i);
    const T d = std::sqrt(D2.sumField(i));
    const T m = std::sqrt(M2.sumField(i));
    const T similarity =
        dm / std::max(d * m, std::numeric_limits<T>::epsilon());

    // Adjust the the step size. Increase step size if similar, otherwise
    // decrease it.
    const T factor = similarity > static_cast<T>(0)
                         ? std::pow(m_increase, similarity)
                         : std::pow(m_decrease, -similarity);
    m_stepSize[i] *= factor;

    // Adjust the inertia. Lean towards the diff if it is smaller than the
    // previous step.
    m_inertia[i] = m > static_cast<T>(0) ? d / (d + m) : static_cast<T>(0);
  }

  // Copy to the GPU.
  const thrust::device_vector<T> stepSizeGPU(m_stepSize);
  const thrust::device_vector<T> inertiaGPU(m_inertia);

  // Kernel parameters.
  dim3 numBlocks;
  dim3 numThreadsPerBlock;
  inOutX.getKernelDimensionsFields(numBlocks, numThreadsPerBlock);

  // Take a step.
  adaptive_ball_internal::computeInternal<<<numBlocks, numThreadsPerBlock>>>(
      dimX, dimY, numFields, thrust::raw_pointer_cast(stepSizeGPU.data()),
      thrust::raw_pointer_cast(inertiaGPU.data()), diff.getDataPointer(),
      m_mean.getDataPointer(), inOutX.getDataPointer());
}
} // namespace accelerators
} // namespace conga

#endif // CONGA_ADAPTIVE_BALL_H_
