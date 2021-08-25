#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "ComputeResidual.h"
#include "ComputeVectorPotential.h"
#include "Grid.h"
#include "OrderParameterBulk.h"
#include "doctest_wrapper.h"
#include "utils.h"

#include <cmath>
#include <limits>
#include <thrust/host_vector.h>

namespace helper {
namespace legacy {
// This legacy code generates a lot of warnings from promition to double. So,
// lets turn it off.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdouble-promotion"

// This is the implementation of atan2 that was present in the code previously.
template <typename T> T atan2(const T y, const T x) {
  T phase = (T)0.0;

  // Interior of quadrants
  if (x > (T)0.0 && y > (T)0.0)
    phase = atan(y / x);
  else if (x < (T)0.0 && y > (T)0.0)
    phase = M_PI + atan(y / x);
  else if (x < (T)0.0 && y < (T)0.0)
    phase = M_PI + atan(y / x);
  else if (x > (T)0.0 && y < (T)0.0)
    phase = (T)2.0 * M_PI + atan(y / x);
  // Boundaries
  else if (x == (T)0.0) {
    if (y > (T)0.0)
      phase = (T)0.5 * M_PI;
    if (y < (T)0.0)
      phase = (T)1.5 * M_PI;
  } else if (y == (T)0.0) {
    if (x > (T)0.0)
      phase = (T)0.0;
    if (x < (T)0.0)
      phase = M_PI;
  }
  return phase;
}

// This is the s-wave bulk solver that was present in the code previously.
template <typename T>
T computeOrderParameterBulkSWave(
    const T T_Tc,
    const T deltaConvergence = static_cast<T>(10.0) *
                               std::numeric_limits<T>::epsilon(),
    const int inNumIterations = 100000) {
  const T gMix = -0.1;
  const int numMatsubara = 1000.0 / T_Tc;

  // Delta, initial guess
  T delta = 1.76 * sqrt(1.0 - T_Tc);

  T matsubaraSum, eps, g, deltaDiff;

  bool isConverged = false;
  for (int iterationIdx = 0; iterationIdx < inNumIterations; ++iterationIdx) {
    matsubaraSum = 0.0;

    for (int i = 0; i <= numMatsubara; ++i) {
      eps = M_PI * T_Tc * (2.0 * i + 1.0);
      matsubaraSum += 1.0 / sqrt(eps * eps + delta * delta) - 1.0 / eps;
    }
    matsubaraSum *= delta;

    g = delta * log(T_Tc) - 2.0 * M_PI * T_Tc * matsubaraSum;
    deltaDiff = g * gMix;
    delta += deltaDiff;

    isConverged = (abs(deltaDiff / delta) < deltaConvergence);
    if (isConverged) {
      break;
    }
  }

  return delta;
}

// Turn warnings back on again.
#pragma GCC diagnostic pop

namespace internal {
// Compute vector potential by integrating greens function kernel
template <typename T>
__global__ void computeVectorPotential2d_kernel(
    T *const inOutVectorPotentialX, T *const inOutVectorPotentialY,
    const T *const inCurrentDensityX, const T *const inCurrentDensityY,
    const int px, const int py, const T inLatticeSpacing, const int Nx) {
  // Coordinates to grid
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < Nx && y < Nx) {
    // For convenience.
    const T latticeArea = inLatticeSpacing * inLatticeSpacing;

    T factor;
    if (x == px && y == py) {
      factor = static_cast<T>(0.25) * latticeArea *
               (static_cast<T>(M_PI - 6.0) +
                static_cast<T>(2.0) * log(latticeArea * static_cast<T>(0.5)));
    } else {
      const T dx = static_cast<T>(px - x);
      const T dy = static_cast<T>(py - y);
      factor = latticeArea * log(inLatticeSpacing * sqrt(dx * dx + dy * dy));
    }

    // Vector-potential index.
    const int idx = y * Nx + x;

    // Current-density index.
    const int idxp = py * Nx + px;

    inOutVectorPotentialX[idx] += factor * inCurrentDensityX[idxp];
    inOutVectorPotentialY[idx] += factor * inCurrentDensityY[idxp];
  }
}
} // namespace internal

template <typename T>
void computeVectorPotential(scg::GridGPU<T> &inOutVectorPotential,
                            const scg::GridGPU<T> &inCurrentDensity,
                            const scg::GridGPU<char> &inDomainLabels,
                            const T inLatticeSpacing, const T inKappa) {
  // Set to zero because we'll add stuff to it.
  inOutVectorPotential.setZero();

  // Get the required kernel dimensions.
  dim3 bpg, tpb;
  inOutVectorPotential.getKernelDimensions(bpg, tpb);

  const int N = inOutVectorPotential.dimX();

  // Get the thrust-vector from the domain labels, and copy to the CPU.
  const thrust::host_vector<char> &domainLabelsCPU = inDomainLabels.getVector();
  const char *const domainLabelsPtr =
      thrust::raw_pointer_cast(domainLabelsCPU.data());

  for (int y = 0; y < N; ++y) {
    for (int x = 0; x < N; ++x) {
      if (domainLabelsPtr[y * N + x] > 0) {
        internal::computeVectorPotential2d_kernel<<<bpg, tpb>>>(
            inOutVectorPotential.getDataPointer(0),
            inOutVectorPotential.getDataPointer(1),
            inCurrentDensity.getDataPointer(0),
            inCurrentDensity.getDataPointer(1), x, y, inLatticeSpacing, N);
      }
    }
  }

  // Factor from the equation itself.
  const T equationFactor = -static_cast<T>(1.0) / (inKappa * inKappa);

  // Factor from the Green function.
  const T greenFactor = static_cast<T>(1) / static_cast<T>(2.0 * M_PI);

  // Scale the result appropriately.
  inOutVectorPotential *= equationFactor * greenFactor;
}
} // namespace legacy

// A helper function to test legacy::atan2.
template <typename T>
void testLegacyAtan2(const T inTolerance, const int inNumAngles = 32,
                     const int inNumRadii = 10,
                     const T inMinRadius = static_cast<T>(1e-4),
                     const T inMaxRadius = static_cast<T>(1e4)) {
  // For convenience.
  const T angleStep = static_cast<T>(2.0 * M_PI) / static_cast<T>(inNumAngles);
  const T radiusStep =
      (inMaxRadius - inMinRadius) / static_cast<T>(inNumRadii - 1);

  for (int radiusIdx = 0; radiusIdx < inNumRadii; ++radiusIdx) {
    const T radius = inMinRadius + radiusStep * static_cast<T>(radiusIdx);
    for (int angleIdx = 0; angleIdx < inNumAngles; ++angleIdx) {
      const T angle = angleStep * static_cast<T>(angleIdx);

      const T x = radius * cos(angle);
      const T y = radius * sin(angle);

      const T angleOld = legacy::atan2(y, x);
      const T angleNew = scg::utils::wrapToRangeTwoPi(atan2(y, x));
      CHECK(angle == doctest::Approx(angleNew).epsilon(inTolerance));
      CHECK(angleOld == doctest::Approx(angleNew).epsilon(inTolerance));
    }
  }
}

// A helper function to test legacy::computeOrderParameterBulkSWave.
template <typename T>
void testComputeOrderParameterBulkSWave(const T inTolerance) {
  // The temperatures to test.
  const int numTemperatures = 2;
  const std::array<T, numTemperatures> temperatures = {static_cast<T>(0.1),
                                                       static_cast<T>(0.9)};

  // Loop over temperatures.
  // Note, the internal energy scale is changed to 2*pi*k_B*T_c.
  const T scale = static_cast<T>(1) / static_cast<T>(2.0 * M_PI);
  for (const T &temperature : temperatures) {
    const T oldOrderParameter =
        legacy::computeOrderParameterBulkSWave(temperature) * scale;

    // Compute the new value.
    const T newOrderParameter =
        scg::computeOrderParameterBulkSWave(temperature);

    // Compare results.
    CHECK(oldOrderParameter ==
          doctest::Approx(newOrderParameter).epsilon(inTolerance));
  }
}

// A helper function to test legacy::computeVectorPotential.
template <typename T> void testComputeVectorPotential(const T inTolerance) {
  // =================== ARRANGE ==================== //
  // The number of lattice sites per side.
  const int N = 32;

  // Physical side length of a lattice cell.
  const T latticeSpacing = static_cast<T>(0.1);

  // The London penetration depth.
  const T penetrationDepth = static_cast<T>(1);

  // The number of "fields" for the domain labels and the current density.
  const int domainDimension = 1;
  const int currentDimension = 2;

  // Set the domain labels to "1" everywhere.
  scg::GridGPU<char> domainLabels(N, N, domainDimension, scg::Type::real);
  domainLabels.setValue('1', scg::Type::real);

  // Construct the current density.
  scg::GridGPU<T> currenDensity(N, N, currentDimension, scg::Type::real);
  currenDensity.setValue(static_cast<T>(1), scg::Type::real);

  // =================== ACT ==================== //
  // Construct a vector-potential grid on the GPU.
  scg::GridGPU<T> vectorPotential(N, N, currentDimension, scg::Type::real);

  legacy::computeVectorPotential(vectorPotential, currenDensity, domainLabels,
                                 latticeSpacing, penetrationDepth);

  // =================== ASSERT ==================== //
  // The maximum error, in terms of relative Frobenius norm, of the
  // eigen-decomposition approximation.
  const T maxError = static_cast<T>(1e-6);

  // Whether to use the analytical expression for the matrix elements of the
  // Green function. This must be false, because we didn't have an analytical
  // implementation before.
  const T analytical = false;

  // Do not enforce current conservation.
  const bool enforceCurrentConservation = false;
  const int numGrainLatticeSites = N * N;

  // Compute the induced vector potential.
  scg::ComputeVectorPotential<T> computeVectorPotential(
      maxError, analytical, enforceCurrentConservation);
  computeVectorPotential.initialize(N, latticeSpacing, numGrainLatticeSites);
  computeVectorPotential.solve(currenDensity, penetrationDepth);

  scg::GridGPU<T> expVectorPotential = *(computeVectorPotential.getResult());

  const T residual = scg::computeResidual(vectorPotential, expVectorPotential);

  // Compare results.
  const T expResidual = static_cast<T>(0);
  CHECK(residual == doctest::Approx(expResidual).epsilon(inTolerance));
}
} // namespace helper

// =============================== Test atan2 =============================== //

TEST_CASE("Test legacy::atan2<float>") {
  // Relative tolerance.
  // Note, machine precision is too harsh.
  const float tolerance = 10.0F * std::numeric_limits<float>::epsilon();
  helper::testLegacyAtan2(tolerance);
}

TEST_CASE("Test legacy::atan2<double>") {
  // Relative tolerance.
  // Note, machine precision is too harsh.
  const double tolerance = 10.0 * std::numeric_limits<double>::epsilon();
  helper::testLegacyAtan2(tolerance);
}

// ================== Test computeOrderParameterBulkSWave ================== //

TEST_CASE("Test legacy::computeOrderParameterBulkSWave<float>") {
  // Relative tolerance.
  // Note that the two implementations differ in how they determine when
  // convergence has been achieved amongs other things. Also, that the legacy
  // implementation might not even converge. So we only check that they roughly
  // give the same answer.
  const float tolerance = 1e-4F;
  helper::testComputeOrderParameterBulkSWave(tolerance);
}

TEST_CASE("Test legacy::computeOrderParameterBulkSWave<double>") {
  // Relative tolerance.
  // Note that the two implementations differ in how they determine when
  // convergence has been achieved amongs other things. Also, that the legacy
  // implementation might not even converge. So we only check that they roughly
  // give the same answer.
  const double tolerance = 1e-4;
  helper::testComputeOrderParameterBulkSWave(tolerance);
}

// ================== Test computeVectorPotential ================== //

TEST_CASE("Test legacy::testComputeVectorPotential<float>") {
  const float tolerance = 1e-6F;
  helper::testComputeVectorPotential(tolerance);
}

TEST_CASE("Test legacy::testComputeVectorPotential<double>") {
  const double tolerance = 1e-8;
  helper::testComputeVectorPotential(tolerance);
}
