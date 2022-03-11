#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "conga/Type.h"
#include "conga/compute/ComputeVectorPotential.h"
#include "conga/grid.h"
#include "conga/wrappers/doctest.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cmath>
#include <iostream>
#include <limits>

namespace helper {
// All float/double tests use the same tolerances.
// TODO(niclas): Why must the float symmetry-tolerance be so large?
const float FLOAT_TOLERANCE = 2e-3f;
const float FLOAT_SYMMETRY_TOLERANCE = 1e-3;
const double DOUBLE_TOLERANCE = 2e-4;
const double DOUBLE_SYMMETRY_TOLERANCE = 1e-8;

// When the max error is non-zero, this is the value used.
const double MAX_ERROR = 1e-6;

/// \brief Compute the relative Frobenius norm given two grids.
///
/// \param inTest The computed vector potential.
/// \param inRef The analytical, reference, solution.
///
/// \return The relative Frobenius norm.
template <typename T>
T computeRelativeFrobeniusNorm(const conga::GridCPU<T> &inTest,
                               const conga::GridCPU<T> &inRef) {

  const T *const testPtr = inTest.getDataPointer();
  const T *const refPtr = inRef.getDataPointer();

  const int numX = inTest.dimX();
  const int numY = inRef.dimY();

  double meanA2 = 0.0;
  double meanB2 = 0.0;
  double meanAB = 0.0;
  for (int n = 0; n < numX * numY; ++n) {
    const double a = static_cast<double>(testPtr[n]);
    const double b = static_cast<double>(refPtr[n]);

    // Using Welford's algorithm for numerical precision.
    // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    const double invNumElements = 1.0 / static_cast<double>(n + 1);

    // Update A*A mean.
    meanA2 += (a * a - meanA2) * invNumElements;

    // Update B*B mean.
    meanB2 += (b * b - meanB2) * invNumElements;

    // Update A*B mean.
    meanAB += (a * b - meanAB) * invNumElements;
  }

  const double result = std::sqrt((meanA2 + meanB2 - 2.0 * meanAB) / meanB2);
  return static_cast<T>(result);
}

/// \brief Test that the vector potential is square symmetric.
///
/// \param inVectorPotential The vector potential.
/// \param inTolerance The tolerance.
///
/// \return Void.
template <typename T>
void testSymmetry(const conga::GridCPU<T> &inVectorPotential,
                  const T inTolerance) {
  // Get pointers.
  const T *const ptrX = inVectorPotential.getDataPointer(0);
  const T *const ptrY = inVectorPotential.getDataPointer(1);

  // The number of lattice sites per side.
  const int N = inVectorPotential.dimX();

  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {
      const int idx = j * N + i;
      const T x = ptrX[idx];
      const T y = ptrY[idx];

      // Rotate 90 degrees.
      const int idxRot90 = i * N + (N - 1 - j);
      REQUIRE(-y == doctest::Approx(ptrX[idxRot90]).epsilon(inTolerance));
      REQUIRE(x == doctest::Approx(ptrY[idxRot90]).epsilon(inTolerance));

      // Rotate 180 degrees.
      const int idxRot180 = (N - 1 - j) * N + (N - 1 - i);
      REQUIRE(-x == doctest::Approx(ptrX[idxRot180]).epsilon(inTolerance));
      REQUIRE(-y == doctest::Approx(ptrY[idxRot180]).epsilon(inTolerance));

      // Rotate 270 degrees.
      const int idxRot270 = (N - 1 - i) * N + j;
      REQUIRE(y == doctest::Approx(ptrX[idxRot270]).epsilon(inTolerance));
      REQUIRE(-x == doctest::Approx(ptrY[idxRot270]).epsilon(inTolerance));

      // Transpose.
      const int idxTrans = i * N + j;
      REQUIRE(std::hypot(x, y) ==
              doctest::Approx(std::hypot(ptrX[idxTrans], ptrY[idxTrans]))
                  .epsilon(inTolerance));
    }
  }
}

/// \brief Test vector-potential computations against analytical solution on a
/// disc.
///
/// \param inAnalytical Whether to use the analytical expression for the Green
/// function matrix-elements.
/// \param inMaxError Max error in Frobenious norm of the Green function.
/// \param inTolerance The tolerance for the relative
/// Frobenius norm between the analytical, and the computed, solutions.
/// \param inSymmetryTolerance The relative tolerance when testing that the
/// solution is symmetric.
///
/// \return Void.
template <typename T>
void testMagnetoStaticsDisc(const bool inAnalytical, const T inMaxError,
                            const T inTolerance, const T inSymmetryTolerance) {
  // =================== ARRANGE ==================== //
  // The number of lattice sites per side.
  const int N = 256;

  // Physical side length of a lattice cell.
  const T latticeSpacing = static_cast<T>(0.1);

  // The number of "fields" for the current density.
  const int currentDimension = 2;

  // The radius and center coordinate of a disk.
  const T R = latticeSpacing * static_cast<T>(N - 1) * static_cast<T>(0.4);
  const T center = static_cast<T>(N - 1) * static_cast<T>(0.5);

  // Construct the current density, and the expected potential, on the CPU.
  conga::GridCPU<T> currenDensityCPU(N, N, currentDimension, conga::Type::real);
  conga::GridCPU<T> expectedPotential(N, N, currentDimension,
                                      conga::Type::real);

  // Get the pointers.
  T *const srcPtrX = currenDensityCPU.getDataPointer(0);
  T *const srcPtrY = currenDensityCPU.getDataPointer(1);
  T *const potPtrX = expectedPotential.getDataPointer(0);
  T *const potPtrY = expectedPotential.getDataPointer(1);

  int numGrainLatticeSites = 0;
  for (int j = 0; j < N; ++j) {
    // Y-coordinate relative to the center of the disc.
    const T y = latticeSpacing * (static_cast<T>(j) - center);
    for (int i = 0; i < N; ++i) {
      // Y-coordinate relative to the center of the disc.
      const T x = latticeSpacing * (static_cast<T>(i) - center);

      // The distance from the center of the disc.
      const T r = std::hypot(x, y);

      // Whether a point is within the disk.
      const bool inDomain = r <= R;

      // Increase the grain lattice site counter.
      numGrainLatticeSites += inDomain ? 1 : 0;

      // The linear index.
      const int idx = j * N + i;

      // The current density is given by r \hat{\phi}, i.e. there are rotating
      // currents.
      srcPtrX[idx] = inDomain ? -y : static_cast<T>(0);
      srcPtrY[idx] = inDomain ? x : static_cast<T>(0);

      // ===== Compute the expected vector potential ===== //
      // See the vector-potential notes on Overleaf for the derivation of this
      // expression.

      // For convenience.
      const T R2 = R * R;
      const T r2 = r * r;

      // Angular part of the expected potential inside the disc.
      const T expectedAngularValueInside =
          static_cast<T>(0.125) * (static_cast<T>(2) * R2 - r2);

      // Angular part of the expected potential outside the disc.
      const T expectedAngularValueOutside =
          static_cast<T>(0.125) * R2 * R2 / r2;

      // The expected angular value in general.
      const T expectedAngularValue =
          inDomain ? expectedAngularValueInside : expectedAngularValueOutside;

      // Convert to Cartesian coordinates.
      const T expectedValueX = -expectedAngularValue * y;
      const T expectedValueY = expectedAngularValue * x;
      potPtrX[idx] = expectedValueX;
      potPtrY[idx] = expectedValueY;
    }
  }

  // Copy the current density to the GPU.
  const conga::GridGPU<T> currenDensityGPU(currenDensityCPU);

  // =================== ACT ==================== //
  // Compute the induced vector potential.
  conga::ComputeVectorPotential<T> computeVectorPotential(inMaxError,
                                                          inAnalytical);
  computeVectorPotential.initialize(N, latticeSpacing, numGrainLatticeSites);
  computeVectorPotential.solve(currenDensityGPU);

  // Copy the vector potential to the CPU.
  const conga::GridCPU<T> potentialCPU = *(computeVectorPotential.getResult());

  // =================== ASSERT ==================== //
  const T testRelativeFrobeniusNorm =
      computeRelativeFrobeniusNorm(potentialCPU, expectedPotential);
  CHECK(testRelativeFrobeniusNorm < inTolerance);

  // It is a bit ugly to test the symmetry here as well. But it saves time.
  SUBCASE("Symmetry") { testSymmetry(potentialCPU, inSymmetryTolerance); }
}
} // namespace helper

// === Test simulation - analytical, non-zero max-error === //

TEST_CASE("Test simulation - analytical, non-zero max-error (float)") {
  const bool analytical = true;
  helper::testMagnetoStaticsDisc(
      analytical, static_cast<float>(helper::MAX_ERROR),
      helper::FLOAT_TOLERANCE, helper::FLOAT_SYMMETRY_TOLERANCE);
}

TEST_CASE("Test simulation - analytical, non-zero max-error (double)") {
  const bool analytical = true;
  helper::testMagnetoStaticsDisc(analytical, helper::MAX_ERROR,
                                 helper::DOUBLE_TOLERANCE,
                                 helper::DOUBLE_SYMMETRY_TOLERANCE);
}

// === Test simulation - non-analytical, non-zero max-error === //

TEST_CASE("Test simulation - non-analytical, non-zero max-error (float)") {
  const bool analytical = false;
  helper::testMagnetoStaticsDisc(
      analytical, static_cast<float>(helper::MAX_ERROR),
      helper::FLOAT_TOLERANCE, helper::FLOAT_SYMMETRY_TOLERANCE);
}

TEST_CASE("Test simulation - non-analytical, non-zero max-error (double)") {
  const bool analytical = false;
  helper::testMagnetoStaticsDisc(analytical, helper::MAX_ERROR,
                                 helper::DOUBLE_TOLERANCE,
                                 helper::DOUBLE_SYMMETRY_TOLERANCE);
}

// === Test simulation - analytical, zero max-error === //

TEST_CASE("Test simulation - analytical, zero max-error (float)") {
  const bool analytical = true;
  const float maxError = 0.0f;
  helper::testMagnetoStaticsDisc(analytical, maxError, helper::FLOAT_TOLERANCE,
                                 helper::FLOAT_SYMMETRY_TOLERANCE);
}

TEST_CASE("Test simulation - analytical, zero max-error (double)") {
  const bool analytical = true;
  const double maxError = 0.0;
  helper::testMagnetoStaticsDisc(analytical, maxError, helper::DOUBLE_TOLERANCE,
                                 helper::DOUBLE_SYMMETRY_TOLERANCE);
}

// === Test simulation - non-analytical, zero max-error === //

TEST_CASE("Test simulation - non-analytical, zero max-error (float)") {
  const bool analytical = false;
  const float maxError = 0.0f;
  helper::testMagnetoStaticsDisc(analytical, maxError, helper::FLOAT_TOLERANCE,
                                 helper::FLOAT_SYMMETRY_TOLERANCE);
}

TEST_CASE("Test simulation - non-analytical, zero max-error (double)") {
  const bool analytical = false;
  const double maxError = 0.0;
  helper::testMagnetoStaticsDisc(analytical, maxError, helper::DOUBLE_TOLERANCE,
                                 helper::DOUBLE_SYMMETRY_TOLERANCE);
}
