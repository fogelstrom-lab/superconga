#ifndef COMPUTE_VECTOR_POTENTIAL_H_
#define COMPUTE_VECTOR_POTENTIAL_H_

// Because we are compiling with NVCC and not GCC.
#ifndef ARMA_ALLOW_FAKE_GCC
#define ARMA_ALLOW_FAKE_GCC
#endif

#include "ComputeResidual.h"
#include "Grid.h"
#include "Type.h"
#include "armadillo_wrapper.h"
#include <array>
#include <cstddef>
#include <cuda.h>
#include <limits>
#include <thrust/device_vector.h>

namespace scg {
/// \brief Class for computing the vector potential.
template <typename T> class ComputeVectorPotential {
private:
  // The vector potential.
  // TODO(niclas): Use unique_ptr.
  GridGPU<T> *m_vectorPotential;

  // The eigen decomposition of a matrix is given by A = VDV^T. We store a few
  // of the eigenvectors. They are called filters as they are used for
  // cross-correlation.
  std::vector<thrust::device_vector<T>> m_filters;

  // We need to keep track of whether the eigenvalue, corresponding to a filter,
  // is positive or negative.
  std::vector<bool> m_positiveFilter;

  // The residual.
  // TODO(niclas): I am not sure it should live here.
  T m_residual;

  // Whether the filters are initialized or not.
  bool m_isInitialized;

  // Factor to scale the final result with.
  double m_scale;

  // The physical area of a lattice site.
  T m_latticeSiteArea;

  // The number of lattice sites in the grain. It is needed for enforcing
  // current conservation.
  int m_numGrainLatticeSites;

  // The maximum error of the approximation.
  const double m_maxError;

  // Whether to use the analytical expression of the Green function.
  const bool m_analytical;

  // Whether to enforce current conservation by subtracting the mean before
  // computing the vector potential.
  const bool m_enforceCurrentConservation;

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

  /// \brief Subtract the mean of, i.e. center,a grid.
  ///
  /// \param inSrc The grid to center.
  /// \param outDst The centered grid.
  ///
  /// \return Void.
  void subtractMean(const GridGPU<T> &inSrc, GridGPU<T> &outDst);

  /// \brief The internal method for solving the Poisson equation for the vector
  /// potential.
  ///
  /// \param inCurrentDensity The current density, in units of J_d = 4 pi |e|
  /// <v_F> N_F k_B T_c.
  /// \param inKappa The Ginzburg-Landau parameter, kappa = lambda_L / xi_0,
  /// where lambda_L = (mu_0 e^2 <v_F>^2 N_F)^(- 1 / 2)
  ///
  /// \return void.
  void solveInternal(const GridGPU<T> &inCurrentDensity, const T inKappa);

public:
  /// \brief Constructor.
  ///
  /// \param inMaxError The maximum error, in terms of relative Frobenius norm,
  /// of the approximation.
  /// \param inAnalytical Whether to use the analytical expression of the Green
  /// function, or a simpler approximation.
  /// \param inEnforceCurrentConservation Whether to enforce current
  /// conservation by sutracting the mean before computing the vector potential.
  ComputeVectorPotential(const T inMaxError = static_cast<T>(1e-6),
                         const bool inAnalytical = true,
                         const bool inEnforceCurrentConservation = true);

  /// \brief Destructor.
  ~ComputeVectorPotential() {
    std::cout << "~ComputeVectorPotential<T>\n";
    delete m_vectorPotential;
  }

  /// \brief Initialize.
  ///
  /// \param inGridResolution The number of lattice sites per side.
  /// \param inGridCellLength Physical side length of a lattice site, in units
  /// of xi_0 = hbar <v_F> / (k_B T_c).
  /// \param inNumGrainLatticeSites The number of lattice sites in the grain.
  ///
  /// \return void.
  void initialize(const int inGridResolution, const T inGridCellLength,
                  const int inNumGrainLatticeSites);

  /// \brief Solve the Poisson equation for the vector potential.
  ///
  /// \param inCurrentDensity The current density, in units of J_d = 4 pi |e|
  /// <v_F> N_F k_B T_c.
  /// \param inKappa The Ginzburg-Landau parameter, kappa = lambda_L / xi_0,
  /// where lambda_L = (mu_0 e^2 <v_F>^2 N_F)^(- 1 / 2)
  ///
  /// \return void.
  void solve(const GridGPU<T> &inCurrentDensity, const T inKappa);

  /// \brief Compute the residual.
  ///
  /// Given the self-consistency equation
  ///
  /// x = g(x)
  ///
  /// it computes
  ///
  /// residual = |x - g(x)| / |x|
  ///
  /// \param inLHS The left-hand side of the self-consitency equation, 'x'.
  ///
  /// \return Void.
  void computeResidual(const GridGPU<T> &inLHS);

  /// \brief Get the resulting vector potential.
  ///
  /// \return The vector potential, in units of A_0 = Phi_0 /
  /// xi_0, where Phi_0 = pi hbar / |e|.
  GridGPU<T> *getResult() const { return m_vectorPotential; }

  /// \brief Get the residual.
  ///
  /// \return The residual.
  T getResidual() const { return m_residual; }
};

template <typename T>
ComputeVectorPotential<T>::ComputeVectorPotential(
    const T inMaxError, const bool inAnalytical,
    const bool inEnforceCurrentConservation)
    : m_vectorPotential(nullptr), m_maxError(static_cast<double>(inMaxError)),
      m_analytical(inAnalytical),
      m_enforceCurrentConservation(inEnforceCurrentConservation),
      m_residual(static_cast<T>(0)), m_isInitialized(false), m_scale(0.0),
      m_latticeSiteArea(static_cast<T>(0.0)) {}

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
  m_positiveFilter.clear();

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
    const int idxDst = inGridResolution * (d * inGridResolution + x) + y;

    // Write result.
    outDst[idxDst] = sum;
  }
}

/// \brief CUDA kernel for subtracting the mean of the current density.
///
/// \param inGridResolution The number of lattice sites per side of the source,
/// denoted N.
/// \param inDimension The dimension of the source, e.g. 2 for a 2D current,
/// \param inMean The mean of each component.
/// \param inSrc The raw current density.
/// \param inOutResultY The current density with zero mean.
///
/// \return void.
template <typename T>
__global__ void subtractMean(const int inGridResolution, const int inDimension,
                             const T *const inMean, const T *const inSrc,
                             T *const outDst) {
  // Indices of spatial coordinates.
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Index of component, e.g. x ==> 0 or y ==> 1.
  const int d = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < inGridResolution && y < inGridResolution && d < inDimension) {
    const int idx = inGridResolution * (d * inGridResolution + y) + x;
    outDst[idx] = inSrc[idx] - inMean[d];
  }
}
} // namespace cvp_internal

template <typename T>
void ComputeVectorPotential<T>::subtractMean(const GridGPU<T> &inSrc,
                                             GridGPU<T> &outDst) {
  // Check dimensions.
  assert(inSrc == outDst);

  // Get dimensions.
  const int gridResolution = inSrc.dimX();
  const int dimension = inSrc.numFields();

  // Compute the mean on the CPU.
  std::vector<T> meansCPU;
  for (int i = 0; i < dimension; ++i) {
    const T sum = inSrc.sumField(i);
    const T mean = sum / static_cast<T>(m_numGrainLatticeSites);
    meansCPU.push_back(mean);
  }

  // Copy to the GPU.
  const thrust::device_vector<T> meansGPU = meansCPU;

  // Kernel parameters.
  dim3 numBlocks;
  dim3 numThreadsPerBlock;
  inSrc.getKernelDimensionsFields(numBlocks, numThreadsPerBlock);

  // Call CUDA kernel.
  cvp_internal::subtractMean<<<numBlocks, numThreadsPerBlock>>>(
      gridResolution, dimension, thrust::raw_pointer_cast(meansGPU.data()),
      inSrc.getDataPointer(), outDst.getDataPointer());
}

template <typename T>
void ComputeVectorPotential<T>::initialize(const int inGridResolution,
                                           const T inGridCellLength,
                                           const int inNumGrainLatticeSites) {
  // Initialize the vector potential.
  const int dimension = 2;
  GridGPU<T>::initializeGrid(m_vectorPotential, inGridResolution,
                             inGridResolution, dimension, Type::real);
  m_vectorPotential->setZero();

  // Initialize the filters.
  initializeFilters(inGridResolution, inGridCellLength);

  m_latticeSiteArea = inGridCellLength * inGridCellLength;

  m_numGrainLatticeSites = inNumGrainLatticeSites;

  // Initialization complete.
  m_isInitialized = true;
}

template <typename T>
void ComputeVectorPotential<T>::solveInternal(
    const GridGPU<T> &inCurrentDensity, const T inKappa) {
  // Check that the sizes of the grids, as well as representation (i.e. real or
  // complex), are equal.
  assert(inCurrentDensity == *m_vectorPotential);

  // Check that the grids are real.
  assert(inCurrentDensity.representation() == Type::real);

  // Check that the grids are squares.
  assert(inCurrentDensity.dimX() == inCurrentDensity.dimY());

  // Get the size and dimension.
  const int gridResolution = inCurrentDensity.dimX();
  const int dimension = inCurrentDensity.numFields();

  // The number of filters.
  assert(m_filters.size() == m_positiveFilter.size());
  const std::size_t numFilters = m_filters.size();

  // Set to zero because we will add stuff to it.
  m_vectorPotential->setZero();

  // Temporary grids.
  GridGPU<T> tmp0(gridResolution, gridResolution, dimension, Type::real);
  GridGPU<T> tmp1(gridResolution, gridResolution, dimension, Type::real);

  // Kernel parameters.
  dim3 numBlocks;
  dim3 numThreadsPerBlock;
  m_vectorPotential->getKernelDimensionsFields(numBlocks, numThreadsPerBlock);

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
      *m_vectorPotential += tmp1;
    } else {
      *m_vectorPotential -= tmp1;
    }
  }

  // Factor from the equation itself.
  const T equationFactor = -static_cast<T>(4.0) / (inKappa * inKappa);

  // Scale the final result.
  *m_vectorPotential *=
      (equationFactor * m_latticeSiteArea * static_cast<T>(m_scale));
}

template <typename T>
void ComputeVectorPotential<T>::solve(const GridGPU<T> &inCurrentDensity,
                                      const T inKappa) {
  if (m_enforceCurrentConservation) {
    // Initialize a grid for the zero-mean current-density.
    GridGPU<T> currentDensityZeroMean = GridGPU<T>(
        inCurrentDensity.dimX(), inCurrentDensity.dimY(),
        inCurrentDensity.numFields(), inCurrentDensity.representation());

    // Subtract the mean.
    subtractMean(inCurrentDensity, currentDensityZeroMean);

    // Solve Poisson's equation.
    solveInternal(currentDensityZeroMean, inKappa);
  } else {
    // Solve Poisson's equation.
    solveInternal(inCurrentDensity, inKappa);
  }

  // TODO(niclas): Subtracting the mean of the current density makes the
  // simulation more stable. But subtracting the mean of the vector potential is
  // even better. But that might not be allowed.
}

template <typename T>
void ComputeVectorPotential<T>::computeResidual(const GridGPU<T> &inLHS) {
  // TODO(niclas): We need to specifiy the namespace in order to avoid
  // compilation errors due to this method having the same name as the function
  // being called.
  m_residual = scg::computeResidual(inLHS, *m_vectorPotential);
}
} // namespace scg

#endif // COMPUTE_VECTOR_POTENTIAL_H_
