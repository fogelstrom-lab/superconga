#ifndef COMPUTE_VECTOR_POTENTIAL_H_
#define COMPUTE_VECTOR_POTENTIAL_H_

// Because we are compiling with NVCC and not GCC.
#define ARMA_ALLOW_FAKE_GCC

#include "Grid.h"
#include "Type.h"
#include <armadillo>
#include <array>
#include <cstddef>
#include <cuda.h>
#include <limits>
#include <thrust/device_vector.h>

namespace scg {
/// \brief Class for computing the vector potential.
template <typename T> class ComputeVectorPotential {
private:
  // The eigen decomposition of a matrix is given by A = VDV^T. We store a few
  // of the eigenvectors. They are called filters as they are used for
  // cross-correlation.
  std::vector<thrust::device_vector<T>> m_filters;

  // We need to keep track of whether the eigenvalue, corresponding to a filter,
  // is positive or negative.
  std::vector<bool> m_positiveFilter;

  // Whether the filters are initialized or not.
  bool m_isInitialized;

  // Factor to scale the final result with.
  double m_scale;

  // The maximum error of the approximation.
  const double m_maxError;

  // Whether to use the analytical expression of the Green function.
  const bool m_analytical;

  /// \brief Initialize filters.
  ///
  /// \param inGridResolution The number of lattice sites per side.
  /// \param inGridCellLength Physical side length of a lattice site, in units
  /// of xi_0 = hbar <v_F> / (k_B T_c).
  ///
  /// \return void.
  void initializeFilters(const int inGridResolution, const T inGridCellLength);

  /// \brief Make the 2D filter, i.e. the Green function. We separate it into
  /// two parts, G = Q + C. C is a uniform matrix, so only a single value is
  /// returned.
  ///
  /// \param inGridResolution The number of lattice sites per side, denoted N.
  /// \param inGridCellLength Physical side length of a lattice site, in units
  /// of xi_0 = hbar <v_F> / (k_B T_c).
  /// \param inAnalytical Whether to use the analytical expression of the Green
  /// function, or a simpler approximation.
  /// \param outQ The Q matrix, a NxN matrix.
  /// \param outC The C value.
  ///
  /// \return void.
  void makeFilter2D(const int inGridResolution, const double inGridCellLength,
                    const bool inAnalytical, arma::Mat<double> &outQ,
                    double &outC) const;

  /// \brief Compute the matrix element (i,j) of the Green function.
  ///
  /// \param inI The index i.
  /// \param inJ The index j.
  /// \param inAnalytical Whether to use the analytical expression of the Green
  /// function, or a simpler approximation.
  double computeMatrixElement(const int inI, const int inJ,
                              const bool inAnalytical) const;

  // Helper methods for computing the analytical Green function.
  double helperAlpha(const int inA, const int inB) const;
  double helperBeta(const int inA, const int inB) const;

  /// \brief Add 1D filter.
  ///
  /// The 2D filter being approximated needs to be eigen decomposed beforehand.
  ///
  /// \param inEigVec The eigenvector.
  /// \param inEigVal The eigenvalue.
  /// \param inScale Value to divide the eigenvalue by.
  ///
  /// \return The 1D filter as a (2N-1) vector.
  void addFilter1D(const arma::Col<double> &inEigVec, const double inEigVal,
                   const double inScale);

public:
  /// \brief Constructor.
  ///
  /// \param inMaxError The maximum error, in terms of relative Frobenius norm,
  /// of the approximation.
  /// \param inAnalytical Whether to use the analytical expression of the Green
  /// function, or a simpler approximation.
  ComputeVectorPotential(const T inMaxError = static_cast<T>(1e-6),
                         const bool inAnalytical = true);

  /// \brief Compute the vector potential.
  ///
  /// \param inCurrentDensity The current density, in units of J_d = 4 pi |e|
  /// <v_F> N_F k_B T_c.
  /// \param inGridCellLength Physical side length of a lattice site, in units
  /// of xi_0 = hbar <v_F> / (k_B T_c).
  /// \param inKappa The Ginzburg-Landau parameter, kappa = lambda_L / xi_0,
  /// where lambda_L = (mu_0 e^2 <v_F>^2 N_F)^(- 1 / 2)
  /// \param outVectorPotential The vector potential, in units of A_0 = Phi_0 /
  /// xi_0, where Phi_0 = pi hbar / |e|.
  ///
  /// \return void.
  void compute(const GridGPU<T> &inCurrentDensity, const T inGridCellLength,
               const T inKappa, GridGPU<T> &outVectorPotential);
};

template <typename T>
ComputeVectorPotential<T>::ComputeVectorPotential(const T inMaxError,
                                                  const bool inAnalytical)
    : m_maxError(static_cast<double>(inMaxError)), m_analytical(inAnalytical),
      m_isInitialized(false), m_scale(1.0) {}

template <typename T>
void ComputeVectorPotential<T>::makeFilter2D(const int inGridResolution,
                                             const double inGridCellLength,
                                             const bool inAnalytical,
                                             arma::Mat<double> &outQ,
                                             double &outC) const {
  // Compute Q.
  outQ = arma::Mat<double>(inGridResolution, inGridResolution);
  for (int j = 0; j < inGridResolution; ++j) {
    for (int i = 0; i < inGridResolution; ++i) {
      outQ(i, j) = computeMatrixElement(i, j, inAnalytical);
    }
  }

  // Symmetrize.
  outQ = 0.5 * (outQ + outQ.t());

  // Scale with common factor.
  const double scale = 1.0 / (4.0 * M_PI);
  outQ *= scale;

  const double constant =
      inAnalytical ? (2.0 * std::log(inGridCellLength) - std::log(2.0) - 3.0)
                   : (2.0 * std::log(inGridCellLength));

  outC = constant * scale;

  // Move the mean of Q to C instead.
  // TODO(niclas): This was just a hunch of mine. Investigate this more.
  const double meanQ = arma::mean(arma::mean(outQ));
  outQ -= meanQ;
  outC += meanQ;
}

template <typename T>
double ComputeVectorPotential<T>::helperAlpha(const int inA,
                                              const int inB) const {
  const double distance =
      std::hypot(static_cast<double>(inA), static_cast<double>(inB));
  const double factor = static_cast<double>(inA * inB);
  return factor * (2.0 * std::log(distance) - std::log(2.0));
}

template <typename T>
double ComputeVectorPotential<T>::helperBeta(const int inA,
                                             const int inB) const {
  const double factor =
      0.5 * static_cast<double>((2 - inA + inB) * (inA + inB) - 2);
  const double angle = static_cast<double>(inA) / static_cast<double>(inB);
  return factor * std::atan(angle);
}

template <typename T>
double
ComputeVectorPotential<T>::computeMatrixElement(const int inI, const int inJ,
                                                const bool inAnalytical) const {
  if (inAnalytical) {
    const std::array<int, 2> iVec{-inI, inI};
    const std::array<int, 2> jVec{-inJ, inJ};

    double sum = 0.0;
    for (const int j : jVec) {
      const int b = 1 + 2 * j;
      for (const int i : iVec) {
        const int a = 1 + 2 * i;
        sum += helperAlpha(a, b) + helperBeta(a, b) + helperBeta(b, a);
      }
    }
    return sum * 0.25;
  } else {
    // Value if i = j = 0.
    const double valOrigin = 0.5 * M_PI - std::log(2.0) - 3.0;

    const double distance =
        std::hypot(static_cast<double>(inI), static_cast<double>(inJ));
    const double valGeneral = 2.0 * std::log(distance);

    const bool isOrigin = (inI == 0) && (inJ == 0);
    const double val = isOrigin ? valOrigin : valGeneral;

    return val;
  }
}

template <typename T>
void ComputeVectorPotential<T>::addFilter1D(const arma::Col<double> &inEigVec,
                                            const double inEigVal,
                                            const double inScale) {
  const bool isPositive = inEigVal > 0.0;
  const double sqrtEigVal = std::sqrt(std::abs(inEigVal) / inScale);

  // For convenience.
  const int N = static_cast<int>(inEigVec.size());

  std::vector<T> filter;
  for (int i = -N + 1; i < N; ++i) {
    const int idx = std::abs(i);
    const double elem = inEigVec(idx) * sqrtEigVal;
    filter.push_back(static_cast<T>(elem));
  }
  assert(filter.size() == static_cast<std::size_t>(2 * N - 1));

  m_filters.push_back(thrust::device_vector<T>(filter));
  m_positiveFilter.push_back(isPositive);
  assert(m_filters.size() == m_positiveFilter.size());
}

template <typename T>
void ComputeVectorPotential<T>::initializeFilters(const int inGridResolution,
                                                  const T inGridCellLength) {
  // Construct the 2D filter, i.e. the Green function.
  // It is separated into two parts, G = Q + C:
  // 1. Q - The variable part, that takes different values.
  // 2. C - The constant part, that is completely uniform.
  // The benefit of doing this is that the eigen decomposition of C is trivial,
  // and can be done analytically. This benefits the decomposition of Q.
  arma::Mat<double> Q;
  double C;
  makeFilter2D(inGridResolution, static_cast<double>(inGridCellLength),
               m_analytical, Q, C);

  // Compute eigen decomposition of Q.
  arma::Col<double> eigValQ;
  arma::Mat<double> eigVecQ;
  const bool success = arma::eig_sym(eigValQ, eigVecQ, Q);
  if (!success) {
    // TODO(niclas): Raise exception instead?
    std::cerr << "ERROR! Eigen decomposition failed." << std::endl;
  }

  // In order to scale the filters to more appropriate values. The final result
  // is then scaled as well, undoing this scaling.
  const double maxEigValQ = arma::abs(eigValQ).max();
  const double maxEigValC = std::abs(C);
  m_scale = 0.5 * (maxEigValQ + maxEigValC);

  // Clear filters.
  m_filters.clear();

  // Construct the filter for C, which is trivial.
  if (std::abs(C) > 0.0) {
    const double val = C;
    const arma::Col<double> vec(inGridResolution, arma::fill::ones);
    addFilter1D(vec, val, m_scale);
  }

  // Compute the indices of the eigenvalues sorted by absolute value.
  const arma::uvec indices = arma::sort_index(arma::abs(eigValQ), "descend");

  // The squared Frobenius norm of the Q matrix.
  const double norm = arma::norm(eigValQ);
  const double normSquared = norm * norm;

  double partialNormSquared = 0.0;
  for (const auto i : indices) {
    // Add the filter.
    const double val = eigValQ(i);
    const arma::Col<double> vec = eigVecQ.col(i);
    addFilter1D(vec, val, m_scale);

    // The squared Frobenius norm of the approximation.
    partialNormSquared += val * val;

    // The squared relative error, |Q - approxQ|^2 / |Q|^2.
    const double errorSquared = 1.0 - partialNormSquared / normSquared;
    if (errorSquared < 0.0) {
      std::cerr << "WARNING! Squared error < 0, i.e. catastrophic cancellation "
                   "occurred."
                << std::endl;
    }

    // The relative error, |Q - approxQ| / |Q|.
    const double error = std::sqrt(std::max(errorSquared, 0.0));
    if (error < m_maxError) {
      break;
    }
  }

  // Initialization complete.
  m_isInitialized = true;
}

namespace cvp_internal {
/// \brief CUDA kernel for cross-correlation of a n-dimensional source, on a
/// square lattice, with a 1D filter. The result is transposed.
///
/// The x- and y-coordinates of the kernal call correspond to spatial
/// coordinates of the source, whereas the z-coordinate corresponds to the
/// dimension. The memory layout of the source is assumed to be all elements
/// corresponding to the first dimension, followed by the second and third etc.
///
/// \param inGridResolution The number of lattice sites per side of the source,
/// denoted N.
/// \param inDimension The dimension of the source, e.g. 2 for a 2D current,
/// denoted d.
/// \param inSrc The source, an array with d * N^2 elements.
/// \param inFilter The filter to cross-correlate the source with, an array with
/// 2N - 1 elements.
/// \param outDst The result, an array with d * N^2 elements.
///
/// \return void.
template <typename T>
__global__ void
crossCorrelateTranspose(const int inGridResolution, const int inDimension,
                        const T *const inSrc, const T *const inFilter,
                        T *const outDst) {
  // Indices of spatial coordinates.
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Index of component, e.g. x ==> 0 or y ==> 1.
  const int d = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < inGridResolution && y < inGridResolution && d < inDimension) {
    // Index offsets for the filter and the source.
    // Note that the filter offset depends on the spatial coordinate x, whereas
    // the source offset depends on the spatial coordinate y.
    const int offsetFilter = inGridResolution - 1 - x;
    const int offsetSrc =
        d * inGridResolution * inGridResolution + y * inGridResolution;

    // Compute the dot product between each spatial column of the source and the
    // filter.
    T sum = static_cast<T>(0);
    for (int i = 0; i < inGridResolution; ++i) {
      const int idxFilter = i + offsetFilter;
      const T filterVal = inFilter[idxFilter];

      const int idxSrc = i + offsetSrc;
      const T srcVal = inSrc[idxSrc];

      sum += srcVal * filterVal;
    }

    // Destination index. Note the we write the transposed result, i.e. swapped
    // x- and y-index.
    const int idxDst =
        d * inGridResolution * inGridResolution + x * inGridResolution + y;

    // Write result.
    outDst[idxDst] = sum;
  }
}
} // namespace cvp_internal

template <typename T>
void ComputeVectorPotential<T>::compute(const GridGPU<T> &inCurrentDensity,
                                        const T inGridCellLength,
                                        const T inKappa,
                                        GridGPU<T> &outVectorPotential) {
  // Check that the sizes of the grids, as well as representation (i.e. real or
  // complex), are equal.
  assert(inCurrentDensity == outVectorPotential);

  // Check that the grids are real.
  assert(inCurrentDensity.representation() == Type::real);

  // Check that the grids are squares.
  assert(inCurrentDensity.dimX() == inCurrentDensity.dimY());

  // Get the size and dimension.
  const int gridResolution = inCurrentDensity.dimX();
  const int dimension = inCurrentDensity.numFields();

  // Initialize filters if needed.
  // TODO(niclas): Check dimensions and re-initialize if they do not agree.
  if (!m_isInitialized) {
    initializeFilters(gridResolution, inGridCellLength);
  }

  // Kernel parameters.
  dim3 numBlocks;
  dim3 numThreadsPerBlock;
  outVectorPotential.getKernelDimensionsFields(numBlocks, numThreadsPerBlock);

  // The number of filters.
  assert(m_filters.size() == m_positiveFilter.size());
  const std::size_t numFilters = m_filters.size();

  // Set to zero because we will add stuff to it.
  outVectorPotential.setZero();

  // Temporary grids.
  GridGPU<T> tmp0(gridResolution, gridResolution, dimension, Type::real);
  GridGPU<T> tmp1(gridResolution, gridResolution, dimension, Type::real);

  // Loop over filters.
  for (std::size_t i = 0; i < numFilters; ++i) {
    // Get filter pointer.
    const T *const filter = thrust::raw_pointer_cast(m_filters[i].data());

    // Cross-correlate the current with the filter, and transpose the result.
    cvp_internal::crossCorrelateTranspose<<<numBlocks, numThreadsPerBlock>>>(
        gridResolution, dimension, inCurrentDensity.getDataPointer(), filter,
        tmp0.getDataPointer());

    // Cross-correlate the previous result with the filter, and transpose the
    // result.
    cvp_internal::crossCorrelateTranspose<<<numBlocks, numThreadsPerBlock>>>(
        gridResolution, dimension, tmp0.getDataPointer(), filter,
        tmp1.getDataPointer());

    // Accumulate.
    const bool isPositive = m_positiveFilter[i];
    if (isPositive) {
      outVectorPotential += tmp1;
    } else {
      outVectorPotential -= tmp1;
    }
  }

  // Factor from the equation itself.
  const T equationFactor = -static_cast<T>(4.0) / (inKappa * inKappa);

  // Area per lattice site.
  const T gridCellArea = inGridCellLength * inGridCellLength;

  // Scale the final result.
  outVectorPotential *= equationFactor * gridCellArea * static_cast<T>(m_scale);
}
} // namespace scg

#endif // COMPUTE_VECTOR_POTENTIAL_H_
