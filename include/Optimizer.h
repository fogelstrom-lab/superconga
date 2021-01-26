#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

// Because we are compiling with NVCC and not GCC.
#ifndef ARMA_ALLOW_FAKE_GCC
#define ARMA_ALLOW_FAKE_GCC
#endif

#include "Grid.h"
#include "Optimizer.h"
#include "Type.h"
#include <algorithm>
#include <armadillo>
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
  Optimizer(const T inStepSize = static_cast<T>(1),
            const T inIneartia = static_cast<T>(0),
            const std::size_t inRank = 1);

  /// \brief Destructor of the optimizer base class.
  ~Optimizer(){};

  /// \brief Compute the next order parameter and vector potential.
  ///
  /// \param inOrderParameterRHS The right-hand side of the self-consistency
  /// equation for the order parameter.
  /// \param inVectorPotentialRHS The right-hand side of the self-consistency
  /// equation for the vector potential.
  /// \param inOrderParameterLHS The left-hand side of the self-consistency
  /// equation for the order parameter, it will be updated by the step.
  /// \param inVectorPotentialLHS The left-hand side of the self-consistency
  /// equation for the vector potential, it will be updated by the step.
  ///
  /// \return Void.
  void compute(const OrderParameter<T> &inOrderParameterRHS,
               const GridGPU<T> &inVectorPotentialRHS,
               const OrderParameter<T> &inOrderParameterLHS,
               const GridGPU<T> &inVectorPotentialLHS);

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
  /// \return Whether the solve was successful or not.
  bool solve();

  /// \brief Set the concatenated grids.
  ///
  /// \param inX The left-hand side of the self-consistency equation. x.
  /// \param inG The right-hand side of the self-consistency equation, g(x).
  ///
  /// \return Void.
  void set(const GridGPU<T> &inX, const GridGPU<T> &inG);

  /// \brief Compute the next order parameter and vector potential.
  ///
  /// \param inOldX The old order parameter and vector potential.
  /// \param outNewX The new order parameter and vector potential.
  ///
  /// \return Void.
  void computeNext(const GridGPU<T> &inOldX, GridGPU<T> &inNewX);

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

  // The step size.
  const T m_stepSize;

  // The inertia.
  const T m_inertia;

  // The maximum number of stored steps. When we store this many, the oldest one
  // will be replaced.
  const std::size_t m_rank;

  // Whether the least-squares succeeded.
  bool m_success;

  // The iteration counter.
  std::size_t m_iterationIdx;

  // The storage of the differences f(x) = g(x) - x, where x is the order
  // parameter and the vector potential concatenated.
  std::vector<GridGPU<T>> m_F;

  // The storage of x, where x is the order parameter and the vector potential
  // concatenated.
  std::vector<GridGPU<T>> m_X;

  // The storage of the squared residuals.
  std::vector<T> m_squaredResiduals;

  // How much to weigh each stored step.
  std::vector<T> m_weights;

  // The running mean update.
  GridGPU<T> m_runningMean;

  // The next order parameter and vector potential.
  GridGPU<T> m_next;
};

template <typename T>
Optimizer<T>::Optimizer(const T inStepSize, const T inIneartia,
                        const std::size_t inRank)
    : m_stepSize(inStepSize), m_inertia(inIneartia), m_rank(inRank),
      m_success(true), m_iterationIdx(0) {}

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

template <typename T>
void Optimizer<T>::compute(const OrderParameter<T> &inOrderParameterRHS,
                           const GridGPU<T> &inVectorPotentialRHS,
                           const OrderParameter<T> &inOrderParameterLHS,
                           const GridGPU<T> &inVectorPotentialLHS) {
  if (!m_success) {
    // Get rid of all old stuff. But do not do a full reset. We want the running
    // mean to remain as it is.
    m_F.clear();
    m_X.clear();
    m_squaredResiduals.clear();
  }

  // We copy everything to one grid, so we need the dimensions of everything.
  const int dimX = inVectorPotentialRHS.dimX();
  const int dimY = inVectorPotentialRHS.dimY();
  const T numComponents = inOrderParameterRHS.getNumOrderParameterComponents();
  const int complexFactor = 2;
  const int numFields =
      complexFactor * numComponents + inVectorPotentialRHS.numFields();
  const Type::type representation = Type::real;

  // Initialize grids.
  GridGPU<T> rhs(dimX, dimY, numFields, representation);
  GridGPU<T> lhs(dimX, dimY, numFields, representation);

  // Concatenate the order parameter and the vector potential to real-valued
  // grids.
  Optimizer<T>::concatenate(inOrderParameterRHS, inVectorPotentialRHS, rhs);
  Optimizer<T>::concatenate(inOrderParameterLHS, inVectorPotentialLHS, lhs);

  if (m_iterationIdx == 0) {
    initialize(lhs);
  }

  // Set this iteration, i.e. update the member variables.
  set(lhs, rhs);

  // Solve the system of equations, determining the next step.
  m_success = solve();

  // Get the next step.
  if (m_success) {
    computeNext(lhs, m_next);
  } else {
    // Take a Picard step if the system was singular.
    // TODO(niclas): Perhaps do something more clever.
    m_next = rhs;
  }

  // Increase iteration index.
  m_iterationIdx++;
}

template <typename T>
void Optimizer<T>::update(OrderParameter<T> &outOrderParameterLHS,
                          GridGPU<T> &outVectorPotentialLHS) const {
  // Split the grid into its constituent order parameter and vector potential.
  Optimizer<T>::split(m_next, outOrderParameterLHS, outVectorPotentialLHS);
}

template <typename T>
void Optimizer<T>::set(const GridGPU<T> &inX, const GridGPU<T> &inG) {
  // Grid of residuals, what Matthias Eschrig calls f.
  const GridGPU<T> F = inG - inX;

  // Slight optimization, compute it here instead of every time in 'solve'.
  const T squaredResidual = F.sumSquare();

  if (m_F.size() >= m_rank) {
    const std::size_t idx = m_iterationIdx % m_rank;
    m_X[idx] = inX;
    m_F[idx] = F;
    m_squaredResiduals[idx] = squaredResidual;
  } else {
    m_X.push_back(inX);
    m_F.push_back(F);
    m_squaredResiduals.push_back(squaredResidual);
  }
}

template <typename T> bool Optimizer<T>::solve() {
  assert(m_F.size() == m_X.size());

  // The number of equations.
  const std::size_t numEqs = m_F.size();

  if (numEqs == 1) {
    // There is only one equation so the weight must be one.
    m_weights.push_back(static_cast<T>(1));
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
  m_weights.clear();
  if (success) {
    // Remove Lagrange multiplier.
    weights.resize(numEqs);

    // Copy weights to member variable.
    for (std::size_t i = 0; i < numEqs; ++i) {
      const T weight = weights(i);
      m_weights.push_back(weight);
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
__global__ void computeNext(const int inDimX, const int inDimY,
                            const int inDimZ, const T inStepSize,
                            const T inIneartia, const T *const inMinX,
                            const T *const inMinF, const T *const inOldX,
                            T *const inOutRunningMean, T *const outNewX) {
  // Indices of spatial coordinates.
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Field.
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < inDimX && y < inDimY && z < inDimZ) {
    // Linear index.
    const int idx = z * inDimX * inDimY + y * inDimX + x;

    // Read the old value once.
    const T oldX = inOldX[idx];

    // Update the running mean.
    // TODO(niclas): Should this factor be exposed to the user? If we want some
    // stochastic behavior, it could be a random number. Perhaps with the
    // standard deviation controlled by the user.
    const T mix = static_cast<T>(1);
    const T g = oldX - (inMinX[idx] + inMinF[idx] * mix);
    const T runningMean = inIneartia * inOutRunningMean[idx] +
                          (static_cast<T>(1) - inIneartia) * g;
    inOutRunningMean[idx] = runningMean;

    // Compute the result.
    outNewX[idx] = oldX - inStepSize * runningMean;
  }
}
} // namespace opt_internal

template <typename T>
void Optimizer<T>::computeNext(const GridGPU<T> &inOldX, GridGPU<T> &outNewX) {
  const int dimX = inOldX.dimX();
  const int dimY = inOldX.dimY();
  const int numFields = inOldX.numFields();
  const Type::type representation = inOldX.representation();

  GridGPU<T> minX(dimX, dimY, numFields, representation);
  GridGPU<T> minF(dimX, dimY, numFields, representation);

  minX.setZero();
  minF.setZero();

  for (std::size_t i = 0; i < m_X.size(); ++i) {
    minX += m_X[i] * m_weights[i];
    minF += m_F[i] * m_weights[i];
  }

  // Kernel parameters.
  dim3 numBlocks;
  dim3 numThreadsPerBlock;
  inOldX.getKernelDimensionsFields(numBlocks, numThreadsPerBlock);

  // Take a step, and update the running mean.
  opt_internal::computeNext<<<numBlocks, numThreadsPerBlock>>>(
      dimX, dimY, numFields, m_stepSize, m_inertia, minX.getDataPointer(),
      minF.getDataPointer(), inOldX.getDataPointer(),
      m_runningMean.getDataPointer(), outNewX.getDataPointer());
}

template <typename T> void Optimizer<T>::reset() {
  m_F.clear();
  m_X.clear();
  m_squaredResiduals.clear();
  m_weights.clear();
  m_iterationIdx = 0;
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
    const GridGPU<T> *const component =
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
