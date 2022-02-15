//===------------------ Anderson.h - Anderson accelerator. ----------------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Anderson accelerator.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_ANDERSON_H_
#define CONGA_ANDERSON_H_

#include "../Type.h"
#include "../wrappers/armadillo.h"
#include "Accelerator.h"
#include "utils.h"

#include <cuda.h>
#include <json/json.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <vector>

namespace conga {
namespace accelerators {
template <typename T> class Anderson : public Accelerator<T> {
public:
  /// \brief Constructor of the accelerator base class.
  ///
  /// It tries to find a solution to the fixed-point problem
  ///
  /// x = g(x)
  ///
  /// It does so by writing it as
  ///
  /// f(x) = x - g(x) = 0
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
  /// x_{i+1} = x_min - f_min * s
  ///
  /// where 's' is the step size.
  ///
  /// \param inStepSize The step size, s.
  /// \param inRank The number of previous steps to use, N, i.e. the rank of the
  /// least-squares problem.
  explicit Anderson(const T inStepSize = m_stepSizeDefault,
                    const std::size_t inRank = m_rankDefault);

  /// \brief Constructor of the Anderson accelerator.
  ///
  /// \param inJSON JSON with parameters. Absent keys get default values.
  explicit Anderson(const Json::Value &inJSON);

  /// \brief Destructor of the accelerator base class.
  ~Anderson(){};

  /// \brief Compute the next order parameter and vector potential.
  ///
  /// \param inOrderParameterG The right-hand side of the self-consistency
  /// equation for the order parameter, 'g(x)'.
  /// \param inVectorPotentialG The right-hand side of the self-consistency
  /// equation for the vector potential, 'g(x)'.
  /// \param inOutOrderParameterX The left-hand side of the self-consistency
  /// equation for the order parameter, 'x'.
  /// \param inOutVectorPotentialX The left-hand side of the
  /// self-consistency equation for the vector potential, 'x'.
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
  /// If the system is singular approximate weights are returned.
  ///
  /// \return The resulting weights {w_i}.
  std::vector<T> solve();

  /// \brief Set the concatenated grids.
  ///
  /// \param inX The left-hand side of the self-consistency equation. x.
  /// \param inF The residual, f(x) = g(x) - x.
  ///
  /// \return Void.
  void set(const GridGPU<T> &inX, const GridGPU<T> &inF);

  /// \brief Compute the next order parameter and vector potential.
  ///
  /// \param inG The right-hand side of the self-consistency equation, 'g(x)'.
  /// \param inOutX The left-hand side of the self-consistency equation, 'x'.
  ///
  /// \return Void.
  void computeInternal(const GridGPU<T> &inG, GridGPU<T> &inOutX);

  // Defaults.
  static constexpr T m_stepSizeDefault = static_cast<T>(1);
  static constexpr std::size_t m_rankDefault = 4;

  // The step size.
  const T m_stepSize;

  // The maximum number of stored steps. When we store this many, the worst (or
  // oldest) one will be replaced.
  const std::size_t m_rank;

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

  // The storage of the iteration indices of when steps were stored.
  std::vector<std::size_t> m_indices;
};

template <typename T>
Anderson<T>::Anderson(const T inStepSize, const std::size_t inRank)
    : Accelerator<T>(), m_stepSize(inStepSize), m_rank(inRank),
      m_iterationIdx(0) {
  assert(m_stepSize > static_cast<T>(0));
  assert(m_rank > 0);
}

template <typename T>
Anderson<T>::Anderson(const Json::Value &inJSON)
    : Accelerator<T>(),
      m_stepSize(inJSON.isMember("step_size")
                     ? static_cast<T>(inJSON["step_size"].asDouble())
                     : m_stepSizeDefault),
      m_rank(inJSON.isMember("rank")
                 ? static_cast<std::size_t>(inJSON["rank"].asInt())
                 : m_rankDefault),
      m_iterationIdx(0) {
  assert(m_stepSize > static_cast<T>(0));
  assert(m_rank > 0);
}

template <typename T>
void Anderson<T>::compute(const OrderParameter<T> &inOrderParameterG,
                          const GridGPU<T> &inVectorPotentialG,
                          OrderParameter<T> &inOutOrderParameterX,
                          GridGPU<T> &inOutVectorPotentialX) {
  // We copy everything to one grid, so we need the dimensions of everything.
  const int dimX = inOutVectorPotentialX.dimX();
  const int dimY = inOutVectorPotentialX.dimY();
  const T numComponents = inOrderParameterG.getNumComponents();
  const int complexFactor = 2;
  const int numFields =
      complexFactor * numComponents + inOutVectorPotentialX.numFields();
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
void Anderson<T>::compute(const OrderParameter<T> &inOrderParameterG,
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
void Anderson<T>::compute(const GridGPU<T> &inVectorPotentialG,
                          GridGPU<T> &inOutVectorPotentialX) {
  // Take step.
  computeInternal(inVectorPotentialG, inOutVectorPotentialX);
}

template <typename T>
void Anderson<T>::set(const GridGPU<T> &inX, const GridGPU<T> &inF) {
  // Slight optimization, compute it here instead of every time in 'solve'.
  const T squaredResidual = inF.sumSquare();

  if (m_F.size() == m_rank) {
    // Get the oldest index.
    const auto minelem = std::min_element(m_indices.begin(), m_indices.end());
    const std::size_t idx = std::distance(m_indices.begin(), minelem);

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
}

template <typename T> std::vector<T> Anderson<T>::solve() {
  assert(m_F.size() == m_X.size());
  assert(m_F.size() == m_squaredResiduals.size());

  // The number of equations.
  const std::size_t numEqs = m_F.size();

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
  const arma::Col<T> weights = arma::solve(A, b, arma::solve_opts::allow_ugly);

  // Copy to vector.
  const T *const ptr = weights.memptr();
  return std::vector<T>(ptr, ptr + numEqs);
}

template <typename T>
void Anderson<T>::computeInternal(const GridGPU<T> &inG, GridGPU<T> &inOutX) {
  // Grid of residuals, what Matthias Eschrig calls f. What is also called
  // "gradient" in other accelerators.
  const GridGPU<T> F = inOutX - inG;

  // Set this iteration, i.e. update the member variables.
  set(inOutX, F);

  // Solve the system of equations, determining the next step.
  const std::vector<T> weights = solve();

  // Compute result.
  inOutX.setZero();
  for (std::size_t i = 0; i < m_X.size(); ++i) {
    inOutX += (m_X[i] - m_F[i] * m_stepSize) * weights[i];
  }

  // Increase iteration index.
  m_iterationIdx++;
}
} // namespace accelerators
} // namespace conga

#endif // CONGA_ANDERSON_H_
