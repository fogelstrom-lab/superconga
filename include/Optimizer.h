#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

// Because we are compiling with NVCC and not GCC.
#ifndef ARMA_ALLOW_FAKE_GCC
#define ARMA_ALLOW_FAKE_GCC
#endif

#include "Grid.h"
#include "Optimizer.h"
#include "Type.h"
#include "armadillo_wrapper.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cuda.h>
#include <limits>
#include <vector>

namespace scg {
template <typename T> class Optimizer {
public:
  /// \brief Constructor of the optimizer base class.
  ///
  /// It tries to find a solution to the fixed-point problem
  ///
  /// x = g(x)
  ///
  /// It does so by writing it as
  ///
  /// f(x) = g(x) - x = 0
  ///
  /// and by solving the least-squares problem
  ///
  /// min_w ||sum_i (f_i * w_i)||^2
  /// subject to: sum_i (w_i) = 1
  ///
  /// and forming a linear combination of the N previous steps
  ///
  /// x_min = sum_i (w_i * x_i)
  /// f_min = sum_i (w_i * f_i)
  /// g_min = x_min + f_min * mix
  ///
  /// where the 'mix' factor is computed as in Matthias Eschrig's note. The
  /// min/max value of the 'mix' factor is controlled by the user.
  ///
  /// The raw "gradient"
  ///
  /// d_i = x_i - (x_min + f_min)
  ///
  /// is formed, and the running mean
  ///
  /// m_i = v * m_{i-1} + (1 - v) * d_i
  ///
  /// is updated, where 'v' is the inertia, giving the final result as
  ///
  /// x_{i+1} = x_i - s * m_i
  ///
  /// where 's' is the step size.
  ///
  /// Some common settings:
  ///
  /// Anderson: (s, v, N) = (1, 0, ~10)
  /// Picard: (s, v, N) = (1, 0, 1)
  /// Rolling ball: (s, v, N) = (1.5, 0.333, 1)
  ///
  /// \param inStepSize The step size, s.
  /// \param inInertia The inertia, v.
  /// \param inRank The number of previous steps to use, N, i.e. the rank of the
  /// least-squares problem.
  /// \param inReplaceOldest Whether to replace the oldest, or the worst, of the
  /// stored steps, when the storage is full.
  /// \param inAndersonAlways Whether to take Anderson steps all the time, or
  /// only when the storage is full.
  /// \param inMinMixing The minimum allowed value of the mixing factor in the
  /// Anderson step.
  /// \param inMaxMixing The maximum allowed value of the mixing factor in the
  /// Anderson step.
  Optimizer(const T inStepSize = static_cast<T>(1),
            const T inInertia = static_cast<T>(0), const std::size_t inRank = 1,
            const bool inReplaceOldest = true,
            const bool inAndersonAlways = true,
            const T inMinMixing = static_cast<T>(1),
            const T inMaxMixing = static_cast<T>(1));

  /// \brief Destructor of the optimizer base class.
  ~Optimizer(){};

  /// \brief Compute the next order parameter and vector potential.
  ///
  /// \param inOrderParameterX The left-hand side of the self-consistency
  /// equation for the order parameter, 'x'.
  /// \param inVectorPotentialX The left-hand side of the self-consistency
  /// equation for the vector potential, 'x'.
  /// \param inOrderParameterG The right-hand side of the self-consistency
  /// equation for the order parameter, 'g(x)'.
  /// \param inVectorPotentialG The right-hand side of the self-consistency
  /// equation for the vector potential, 'g(x)'.
  ///
  /// \return Void.
  void compute(const OrderParameter<T> &inOrderParameterX,
               const GridGPU<T> &inVectorPotentialX,
               const OrderParameter<T> &inOrderParameterG,
               const GridGPU<T> &inVectorPotentialG);

  /// \brief Get the next order parameter and vector potential.
  ///
  /// \param outOrderParameter The new order parameter.
  /// \param outVectorPotential The new vector potential.
  ///
  /// \return Void.
  void update(OrderParameter<T> &outOrderParameter,
              GridGPU<T> &outVectorPotential) const;

  /// \brief Reset the optimizer.
  ///
  /// Remove all stored grids etc. The next time the optimizer is called, it
  /// will be like it is the first time.
  ///
  /// \return Void.
  void reset();

private:
  /// \brief Concatenate the order parameter and the vector potential.
  ///
  /// The real and imaginary parts are treated as separate fields.
  ///
  /// \param inOrderParameter The order parameter.
  /// \param inVectorPotential The vector potential.
  /// \param inOutDst The concatenated result.
  ///
  /// \return Void.
  void concatenate(const OrderParameter<T> &inOrderParameter,
                   const GridGPU<T> &inVectorPotential,
                   GridGPU<T> &inOutDst) const;

  /// \brief Split the concatenated order parameter and the vector potential.
  ///
  /// \param inSrc The order parameter and vector potential concatenated.
  /// \param inOutOrderParameter The order parameter.
  /// \param inOutVectorPotential The vector potential.
  ///
  /// \return Void.
  void split(const GridGPU<T> &inSrc, OrderParameter<T> &inOutOrderParameter,
             GridGPU<T> &inOutVectorPotential) const;

  /// \brief Solve the least-squares problem.
  ///
  /// I.e.
  ///
  /// min_w ||sum_i (f_i * w_i)||^2
  /// subject to: sum_i (w_i) = 1
  ///
  /// This is done by solving the normal equation
  ///
  /// (F^T F) w = 0
  ///
  /// where F = (f_0, f_1, ... f_{N-1}). The constraint is enforced using a
  /// Lagrange multiplier, L. With N = 2 we have the following equation to
  /// solve:
  ///
  /// | dot(f_0, f_0) , dot(f_0, f_1) , 1 | w_0 |   | 0 |
  /// | dot(f_1, f_0) , dot(f_1, f_1) , 1 | w_1 | = | 0 |
  /// |       1       ,       1       , 0 |  L  |   | 1 |
  ///
  /// If the system is singular no values for the weights are obtained.
  ///
  /// \param outWeights The resulting weights {w_i}.
  ///
  /// \return Whether the solve was successful or not.
  bool solve(std::vector<T> &outWeights);

  /// \brief Set the concatenated grids.
  ///
  /// \param inX The left-hand side of the self-consistency equation. x.
  /// \param inF The residual, f(x) = g(x) - x.
  ///
  /// \return Whether the latest residual was smaller than the worst stored one.
  bool set(const GridGPU<T> &inX, const GridGPU<T> &inF);

  /// \brief Compute the next order parameter and vector potential.
  ///
  /// \param inX The current order parameter and vector potential.
  /// \param inG The effective G. If rank = 1, then G = RHS, otherwise it is
  /// computed by solving the least-squares problem.
  /// \param outNewX The new order parameter and vector potential.
  ///
  /// \return Void.
  void computeNext(const GridGPU<T> &inX, const GridGPU<T> &inG,
                   GridGPU<T> &outNewX);

  /// \brief Sum all elements of a grid.
  ///
  /// \param inGrid The grid to sum.
  ///
  /// \return The sum.
  T sumGrid(const GridGPU<T> &inGrid) const;

  /// \brief Initialize the running mean and update.
  ///
  /// \param inX The left-hand side of the self-consistency equation. x.
  ///
  /// \return Void.
  void initialize(const GridGPU<T> &inX);

  /// \brief Prune the stored steps. Remove the worst half.
  ///
  /// \return Void.
  void prune();

  // The step size.
  const T m_stepSize;

  // The inertia.
  const T m_inertia;

  // The maximum number of stored steps. When we store this many, the worst (or
  // oldest) one will be replaced.
  const std::size_t m_rank;

  // Whether to replace the oldest stored step, or the worst.
  const bool m_replaceOldest;

  // Whether to take Anderson steps all the time, or only when the storage is
  // full.
  const bool m_andersonAlways;

  // As part of the Anderson step we have, minG = minX + minF * mix. These
  // parameters control the range of the mixing factor.
  const T m_minMixing;
  const T m_maxMixing;

  // The iteration counter.
  std::size_t m_iterationIdx;

  // The predicted residual, i.e. |minF|, from the Anderson step.
  T m_predictedResidual;

  // The storage of the differences f(x) = g(x) - x, where x is the order
  // parameter and the vector potential concatenated.
  std::vector<GridGPU<T>> m_F;

  // The storage of x, where x is the order parameter and the vector potential
  // concatenated.
  std::vector<GridGPU<T>> m_X;

  // The storage of the squared residuals.
  std::vector<T> m_squaredResiduals;

  // The storage of the iteration indices of when steps were stored.
  std::vector<std::size_t> m_indices;

  // The running mean update.
  GridGPU<T> m_runningMean;

  // The next order parameter and vector potential.
  GridGPU<T> m_next;
};

template <typename T>
Optimizer<T>::Optimizer(const T inStepSize, const T inInertia,
                        const std::size_t inRank, const bool inReplaceOldest,
                        const bool inAndersonAlways, const T inMinMixing,
                        const T inMaxMixing)
    : m_stepSize(inStepSize), m_inertia(inInertia), m_rank(inRank),
      m_replaceOldest(inReplaceOldest), m_andersonAlways(inAndersonAlways),
      m_minMixing(inMinMixing), m_maxMixing(inMaxMixing), m_iterationIdx(0),
      m_predictedResidual(static_cast<T>(0)) {}

template <typename T> void Optimizer<T>::initialize(const GridGPU<T> &inX) {
  const int dimX = inX.dimX();
  const int dimY = inX.dimY();
  const int numFields = inX.numFields();
  const Type::type representation = inX.representation();

  m_runningMean = GridGPU<T>(dimX, dimY, numFields, representation);
  m_runningMean.setZero();

  m_next = GridGPU<T>(dimX, dimY, numFields, representation);
  m_next.setZero();
}

template <typename T> void Optimizer<T>::prune() {
  if (m_rank > 1) {
    // Compute the median squared residual.
    std::vector<T> tmp = m_squaredResiduals;
    std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
    const T median = tmp[tmp.size() / 2];

    // Temporary storage for the stuff to keep.
    std::vector<GridGPU<T>> copyX;
    std::vector<GridGPU<T>> copyF;
    std::vector<T> copySquaredResiduals;
    std::vector<std::size_t> copyIndices;

    // Copy.
    for (std::size_t i = 0; i < m_squaredResiduals.size(); ++i) {
      const T squaredResidual = m_squaredResiduals[i];
      if (squaredResidual < median) {
        copyX.push_back(m_X[i]);
        copyF.push_back(m_F[i]);
        copySquaredResiduals.push_back(m_squaredResiduals[i]);
        copyIndices.push_back(m_indices[i]);
      }
    }

    // Replace member variables.
    m_X = copyX;
    m_F = copyF;
    m_squaredResiduals = copySquaredResiduals;
    m_indices = copyIndices;
  }
}

template <typename T>
void Optimizer<T>::compute(const OrderParameter<T> &inOrderParameterX,
                           const GridGPU<T> &inVectorPotentialX,
                           const OrderParameter<T> &inOrderParameterG,
                           const GridGPU<T> &inVectorPotentialG) {
  // We copy everything to one grid, so we need the dimensions of everything.
  const int dimX = inVectorPotentialG.dimX();
  const int dimY = inVectorPotentialG.dimY();
  const T numComponents = inOrderParameterG.getNumOrderParameterComponents();
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
  Optimizer<T>::concatenate(inOrderParameterX, inVectorPotentialX, X);
  Optimizer<T>::concatenate(inOrderParameterG, inVectorPotentialG, G);

  if (m_iterationIdx == 0) {
    initialize(X);
  }

  if (m_rank == 1) {
    // If rank = 1 we don't have to do alot of complicated stuff. We can simply
    // compute the next X directly.
    computeNext(X, G, m_next);
  } else {
    // Grid of residuals, what Matthias Eschrig calls f.
    const GridGPU<T> F = G - X;

    // Set this iteration, i.e. update the member variables. If m_replaceOldest
    // == true, then success means that the new iteration is better than the
    // worst stored. If m_replaceOldest == false, success is always true.
    bool success = set(X, F);

    // Check if we should try an Anderson step.
    const bool tryAndersonStep = m_andersonAlways || (m_X.size() == m_rank);

    if (success && tryAndersonStep) {
      // Solve the system of equations, determining the next step. Success means
      // that the system of equations was not singular.
      std::vector<T> weights;
      success &= solve(weights);

      if (success) {
        GridGPU<T> minX(dimX, dimY, numFields, representation);
        GridGPU<T> minF(dimX, dimY, numFields, representation);

        minX.setZero();
        minF.setZero();

        for (std::size_t i = 0; i < m_X.size(); ++i) {
          minX += m_X[i] * weights[i];
          minF += m_F[i] * weights[i];
        }

        // Compute the mixing factor.
        const T residual = F.L1Norm();
        const T factorP = (minX - X).L1Norm() / residual;
        const T factorQ =
            std::min(static_cast<T>(1), m_predictedResidual / residual);
        const T mix = std::clamp(factorP * factorQ, m_minMixing, m_maxMixing);

        // Compute the resulting G.
        const GridGPU<T> minG = minX + minF * mix;

        // Compute the next step.
        computeNext(X, minG, m_next);

        // Update the predicted residual.
        m_predictedResidual = minF.L1Norm();
      } else {
        // Take a step using G.
        computeNext(X, G, m_next);
      }
    } else {
      // Take a step using G.
      computeNext(X, G, m_next);
    }

    if (!success) {
      // We have failed, so lets prune.
      prune();
    }
  }

  // Increase iteration index.
  m_iterationIdx++;
}

template <typename T>
void Optimizer<T>::update(OrderParameter<T> &outOrderParameter,
                          GridGPU<T> &outVectorPotential) const {
  // Split the grid into its constituent order parameter and vector potential.
  Optimizer<T>::split(m_next, outOrderParameter, outVectorPotential);
}

template <typename T>
bool Optimizer<T>::set(const GridGPU<T> &inX, const GridGPU<T> &inF) {
  // Slight optimization, compute it here instead of every time in 'solve'.
  const T squaredResidual = inF.sumSquare();

  // Assume success.
  bool success = true;

  if (m_F.size() == m_rank) {
    std::size_t idx = 0;
    if (m_replaceOldest) {
      // Get the oldest index.
      const auto minelem = std::min_element(m_indices.begin(), m_indices.end());
      idx = std::distance(m_indices.begin(), minelem);
    } else {
      // Get the index of the maximum squared residual.
      const auto maxelem = std::max_element(m_squaredResiduals.begin(),
                                            m_squaredResiduals.end());
      idx = std::distance(m_squaredResiduals.begin(), maxelem);

      // Check if this is an improvement.
      const T maxSquaredResidual = *maxelem;
      success = (squaredResidual < maxSquaredResidual);
    }

    // Replace.
    m_X[idx] = inX;
    m_F[idx] = inF;
    m_squaredResiduals[idx] = squaredResidual;
    m_indices[idx] = m_iterationIdx;
  } else {
    m_X.push_back(inX);
    m_F.push_back(inF);
    m_squaredResiduals.push_back(squaredResidual);
    m_indices.push_back(m_iterationIdx);
  }

  return success;
}

template <typename T> bool Optimizer<T>::solve(std::vector<T> &outWeights) {
  assert(m_F.size() == m_X.size());
  assert(m_F.size() == m_squaredResiduals.size());

  // The number of equations.
  const std::size_t numEqs = m_F.size();

  outWeights.clear();

  if (numEqs == 1) {
    // There is only one equation so the weight must be one.
    outWeights.push_back(static_cast<T>(1));
    const bool success = true;
    return success;
  }

  // The number of equations plus the constraint.
  const std::size_t numTotalEqs = numEqs + 1;

  // Form the matrix A and vector b.
  // TODO(niclas): Explain why ones/zeros.
  arma::Mat<T> A(numTotalEqs, numTotalEqs, arma::fill::ones);
  arma::Col<T> b(numTotalEqs, arma::fill::zeros);

  // Find a good scaling of the matrix elements of A so that the values are
  // closer to one. This lowers the condition number.
  const T sumResidual = std::accumulate(
      m_squaredResiduals.begin(), m_squaredResiduals.end(), static_cast<T>(0));
  const T meanResidual = sumResidual / static_cast<T>(numEqs);
  const T scale = static_cast<T>(1) / meanResidual;

  // Equations.
  for (std::size_t i = 0; i < numEqs; ++i) {
    A(i, i) = m_squaredResiduals[i] * scale;
    for (std::size_t j = i + 1; j < numEqs; ++j) {
      const T val = sumGrid(m_F[i] * m_F[j]) * scale;
      A(i, j) = val;
      A(j, i) = val;
    }
  }

  // The coefficent for the Langrange multiplier must be zero.
  A(numTotalEqs - 1, numTotalEqs - 1) = static_cast<T>(0);

  // The weights should sum to one.
  b(numTotalEqs - 1) = static_cast<T>(1);

  // Solve Ax = b.
  arma::Col<T> weights;
  const bool success = arma::solve(weights, A, b, arma::solve_opts::no_approx);

  // Update weights.
  if (success) {
    // Remove Lagrange multiplier.
    weights.resize(numEqs);

    // Copy weights to member variable.
    for (std::size_t i = 0; i < numEqs; ++i) {
      const T weight = weights(i);
      outWeights.push_back(weight);
    }
  }

  return success;
}

namespace opt_internal {
template <typename T>
__global__ void copy(const int inDimX, const int inDimY, const T *const inSrc,
                     T *const outDst) {
  // Indices of spatial coordinates.
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < inDimX && y < inDimY) {
    const int idx = y * inDimX + x;
    outDst[idx] = inSrc[idx];
  }
}

template <typename T>
__global__ void
computeNext(const int inDimX, const int inDimY, const int inDimZ,
            const T inStepSize, const T inInertia, const T *const inX,
            const T *const inG, T *const inOutRunningMean, T *const outResult) {
  // Indices of spatial coordinates.
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Field.
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < inDimX && y < inDimY && z < inDimZ) {
    // Linear index.
    const int idx = inDimX * (z * inDimY + y) + x;

    // Read the old value once.
    const T X = inX[idx];

    // The "gradient", i.e. the difference between the previous X and the
    // proposed next X.
    const T gradient = X - inG[idx];

    // Update the running mean.
    const T runningMean = inInertia * inOutRunningMean[idx] +
                          (static_cast<T>(1) - inInertia) * gradient;
    inOutRunningMean[idx] = runningMean;

    // Compute the result.
    outResult[idx] = X - inStepSize * runningMean;
  }
}
} // namespace opt_internal

template <typename T>
void Optimizer<T>::computeNext(const GridGPU<T> &inX, const GridGPU<T> &inG,
                               GridGPU<T> &outNewX) {
  const int dimX = inX.dimX();
  const int dimY = inX.dimY();
  const int numFields = inX.numFields();

  // Kernel parameters.
  dim3 numBlocks;
  dim3 numThreadsPerBlock;
  inX.getKernelDimensionsFields(numBlocks, numThreadsPerBlock);

  // Take a step, and update the running mean.
  opt_internal::computeNext<<<numBlocks, numThreadsPerBlock>>>(
      dimX, dimY, numFields, m_stepSize, m_inertia, inX.getDataPointer(),
      inG.getDataPointer(), m_runningMean.getDataPointer(),
      outNewX.getDataPointer());
}

template <typename T> void Optimizer<T>::reset() {
  m_F.clear();
  m_X.clear();
  m_squaredResiduals.clear();
  m_indices.clear();
  m_iterationIdx = 0;
  m_predictedResidual = static_cast<T>(0);
}

template <typename T> T Optimizer<T>::sumGrid(const GridGPU<T> &inGrid) const {
  T sum = static_cast<T>(0);
  for (int i = 0; i < inGrid.numFields(); ++i) {
    sum += inGrid.sumField(i);
  }
  return sum;
}

template <typename T>
void Optimizer<T>::concatenate(const OrderParameter<T> &inOrderParameter,
                               const GridGPU<T> &inVectorPotential,
                               GridGPU<T> &inOutDst) const {
  // Kernel parameters.
  dim3 numBlocks;
  dim3 numThreadsPerBlock;
  inOutDst.getKernelDimensions(numBlocks, numThreadsPerBlock);

  // Dimensions.
  const int dimX = inOutDst.dimX();
  const int dimY = inOutDst.dimY();
  const int numComponents = inOrderParameter.getNumOrderParameterComponents();

  // Copy order parameter.
  for (int i = 0; i < numComponents; ++i) {
    const int realIdx = 2 * i;
    const int imagIdx = realIdx + 1;

    // Get the order parameter component.
    const GridGPU<T> *const component =
        inOrderParameter.getOrderParameterComponentField(i);

    opt_internal::copy<<<numBlocks, numThreadsPerBlock>>>(
        dimX, dimY, component->getDataPointer(0, Type::real),
        inOutDst.getDataPointer(realIdx));
    opt_internal::copy<<<numBlocks, numThreadsPerBlock>>>(
        dimX, dimY, component->getDataPointer(0, Type::imag),
        inOutDst.getDataPointer(imagIdx));
  }

  // Copy vector potential.
  const int xIdx = 2 * numComponents;
  const int yIdx = xIdx + 1;
  opt_internal::copy<<<numBlocks, numThreadsPerBlock>>>(
      dimX, dimY, inVectorPotential.getDataPointer(0),
      inOutDst.getDataPointer(xIdx));
  opt_internal::copy<<<numBlocks, numThreadsPerBlock>>>(
      dimX, dimY, inVectorPotential.getDataPointer(1),
      inOutDst.getDataPointer(yIdx));
}

template <typename T>
void Optimizer<T>::split(const GridGPU<T> &inSrc,
                         OrderParameter<T> &inOutOrderParameter,
                         GridGPU<T> &inOutVectorPotential) const {
  // Kernel parameters.
  dim3 numBlocks;
  dim3 numThreadsPerBlock;
  inSrc.getKernelDimensions(numBlocks, numThreadsPerBlock);

  // Dimensions.
  const int dimX = inSrc.dimX();
  const int dimY = inSrc.dimY();
  const int numComponents =
      inOutOrderParameter.getNumOrderParameterComponents();

  // Copy order parameter.
  for (int i = 0; i < numComponents; ++i) {
    const int realIdx = 2 * i;
    const int imagIdx = realIdx + 1;

    // Get the order parameter component.
    GridGPU<T> *const component =
        inOutOrderParameter.getOrderParameterComponentField(i);

    opt_internal::copy<<<numBlocks, numThreadsPerBlock>>>(
        dimX, dimY, inSrc.getDataPointer(realIdx),
        component->getDataPointer(0, Type::real));
    opt_internal::copy<<<numBlocks, numThreadsPerBlock>>>(
        dimX, dimY, inSrc.getDataPointer(imagIdx),
        component->getDataPointer(0, Type::imag));
  }

  // Copy vector potential.
  const int xIdx = 2 * numComponents;
  const int yIdx = xIdx + 1;
  opt_internal::copy<<<numBlocks, numThreadsPerBlock>>>(
      dimX, dimY, inSrc.getDataPointer(xIdx),
      inOutVectorPotential.getDataPointer(0));
  opt_internal::copy<<<numBlocks, numThreadsPerBlock>>>(
      dimX, dimY, inSrc.getDataPointer(yIdx),
      inOutVectorPotential.getDataPointer(1));
}
} // namespace scg

#endif // OPTIMIZER_H_
