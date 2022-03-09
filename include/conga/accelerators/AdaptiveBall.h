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
  /// \param inStepSizeFactor The factor used when adjusting the step size.
  /// \param inMinCosSimilarity The mininum tolerated cosine similarity.
  /// \param inMaxStepSize The maximum step size.
  /// \param inMinStepSize The minimum step size.
  explicit AdaptiveBall(const T inStepSize = m_stepSizeDefault,
                        const T inStepSizeFactor = m_stepSizeFactorDefault,
                        const T inMinCosSimilarity = m_minCosSimilarityDefault,
                        const T inMaxStepSize = m_maxStepSizeDefault,
                        const T inMinStepSize = m_minStepSizeDefault);

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

  /// \brief Compute the new inertial value using the bisection method.
  ///
  /// \param inDiffNorm The L2-norm of the difference.
  /// \param inMeanNorm The L2-norm of the running mean.
  /// \param inDotProduct The dot product between the difference and the running
  /// mean.
  /// \param inMinCosSimilarity The target minimum tolerated cosine similarity.
  ///
  /// \return The new inertia.
  T computeInertia(const T inDiffNorm, const T inMeanNorm, const T inDotProduct,
                   const T inMinCosSimilarity);

  /// \brief Compute the next order parameter and vector potential.
  ///
  /// \param inG The right-hand side of the self-consistency equation, 'g(x)'.
  /// \param inOutX The left-hand side of the self-consistency equation, 'x'.
  ///
  /// \return Void.
  void computeInternal(const GridGPU<T> &inG, GridGPU<T> &inOutX);

  // Defaults.
  static constexpr T m_stepSizeDefault = static_cast<T>(0.5);
  static constexpr T m_stepSizeFactorDefault = static_cast<T>(1.234);
  static constexpr T m_minCosSimilarityDefault = static_cast<T>(0.7);
  static constexpr T m_maxStepSizeDefault = static_cast<T>(100);
  static constexpr T m_minStepSizeDefault = static_cast<T>(0.001);

  // The initial step size.
  const T m_stepSizeInit;

  // Parameters used for adjusting the step size and inertia.
  const T m_stepSizeFactor;
  const T m_minCosSimilarity;

  // The limits of the step size.
  const T m_maxStepSize;
  const T m_minStepSize;

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
AdaptiveBall<T>::AdaptiveBall(const T inStepSize, const T inStepSizeFactor,
                              const T inMinCosSimilarity, const T inMaxStepSize,
                              const T inMinStepSize)
    : Accelerator<T>(), m_stepSizeInit(inStepSize),
      m_stepSizeFactor(inStepSizeFactor),
      m_minCosSimilarity(inMinCosSimilarity), m_maxStepSize(inMaxStepSize),
      m_minStepSize(inMinStepSize), m_initialized(false) {
  assert(m_stepSizeInit > static_cast<T>(0));
  assert(m_stepSizeFactor > static_cast<T>(1));
  assert(m_minCosSimilarity >= static_cast<T>(0));
  assert(m_maxStepSize > static_cast<T>(0));
  assert(m_maxStepSize >= m_minStepSize);
}

template <typename T>
AdaptiveBall<T>::AdaptiveBall(const Json::Value &inJSON)
    : Accelerator<T>(),
      m_stepSizeInit(inJSON.isMember("step_size")
                         ? static_cast<T>(inJSON["step_size"].asDouble())
                         : m_stepSizeDefault),
      m_stepSizeFactor(
          inJSON.isMember("step_size_factor")
              ? static_cast<T>(inJSON["step_size_factor"].asDouble())
              : m_stepSizeFactorDefault),
      m_minCosSimilarity(
          inJSON.isMember("cos_similarity_min")
              ? static_cast<T>(inJSON["cos_similarity_min"].asDouble())
              : m_minCosSimilarityDefault),
      m_maxStepSize(inJSON.isMember("step_size_max")
                        ? static_cast<T>(inJSON["step_size_max"].asDouble())
                        : m_maxStepSizeDefault),
      m_minStepSize(inJSON.isMember("step_size_min")
                        ? static_cast<T>(inJSON["step_size_min"].asDouble())
                        : m_minStepSizeDefault),
      m_initialized(false) {
  assert(m_stepSizeInit > static_cast<T>(0));
  assert(m_stepSizeFactor > static_cast<T>(1));
  assert(m_minCosSimilarity >= static_cast<T>(0));
  assert(m_minCosSimilarity <= static_cast<T>(1));
  assert(m_maxStepSize > static_cast<T>(0));
  assert(m_maxStepSize >= m_minStepSize);
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
T AdaptiveBall<T>::computeInertia(const T inDiffNorm, const T inMeanNorm,
                                  const T inDotProduct,
                                  const T inMinCosSimilarity) {
  // In the following x = inertia.

  // The error when x = 1.
  T errorA = inDotProduct / (inDiffNorm * inMeanNorm) - inMinCosSimilarity;
  T xA = static_cast<T>(1);

  // The error when x = 0.
  T errorB = static_cast<T>(1) - inMinCosSimilarity;
  T xB = static_cast<T>(0);

  // Initial guess, in the middle.
  T x = static_cast<T>(0.5) * (xA + xB);

  // We don't have to be very strict.
  const T convergenceCriterion = static_cast<T>(0.01);

  // Loop until converged.
  bool isConverged = false;
  while (!isConverged) {
    // Compute the similarity.
    const T y = (static_cast<T>(1) - x);
    const T numer = x * inDotProduct + y * inDiffNorm * inDiffNorm;
    const T denom =
        inDiffNorm * std::sqrt(x * x * inMeanNorm * inMeanNorm +
                               static_cast<T>(2) * x * y * inDotProduct +
                               y * y * inDiffNorm * inDiffNorm);
    const T similarity = numer / denom;

    // Check convergence.
    const T error = similarity - inMinCosSimilarity;
    isConverged = std::abs(error) < convergenceCriterion;

    // Replace an end point.
    if (utils::sign(error) == utils::sign(errorA)) {
      errorA = error;
      xA = x;
    } else {
      errorB = error;
      xB = x;
    }

    // Make sure that the signs of errorA and errorB differ.
    if (utils::sign(errorA) == utils::sign(errorB)) {
#ifndef NDEBUG
      std::cerr
          << "-- WARNING! The computation of the inertia did not converge."
          << std::endl;
#endif
      break;
    }

    // New guess.
    x = static_cast<T>(0.5) * (xA + xB);
  }

  return x;
}

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
    const T d = std::sqrt(D2.sumField(i));
    const T m = std::sqrt(M2.sumField(i));
    const T denom = d * m;

    if (denom > static_cast<T>(0)) {
      // Compute the cosine ditsance.
      const T dot = DM.sumField(i);
      const T cosSimilarity = dot / denom;

      // Adjust the the step size. Increase step size if similar, otherwise
      // decrease it.
      m_stepSize[i] *= std::pow(m_stepSizeFactor, cosSimilarity);

      // New inertia.
      if (cosSimilarity < m_minCosSimilarity) {
        m_inertia[i] = computeInertia(d, m, dot, m_minCosSimilarity);
      }
    }

    // Clamp to range.
    m_stepSize[i] = std::clamp(m_stepSize[i], m_minStepSize, m_maxStepSize);
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
