#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "ComputeVectorPotential.h"
#include "Grid.h"
#include "Type.h"
#include <cmath>
#include <doctest/doctest.h>
#include <iostream>
#include <limits>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace scg {
namespace helper {
/// \brief Compute the relative Frobenius norm given two grids.
///
/// \param inTest The computed vector potential.
/// \param inRef The analytical, reference, solution.
///
/// \return The relative Frobenius norm.
template <typename T>
T computeRelativeFrobeniusNorm(const GridCPU<T> &inTest,
                               const GridCPU<T> &inRef) {

  T *const testPtr = inTest.getDataPointer();
  T *const refPtr = inRef.getDataPointer();

  const int numX = inTest.dimX();
  const int numY = inRef.dimY();

  T sumA2 = static_cast<T>(0);
  T sumB2 = static_cast<T>(0);
  T sumAB = static_cast<T>(0);
  for (int j = 0; j < numY; ++j) {
    for (int i = 0; i < numX; ++i) {
      const int idx = j * numX + i;
      const T a = testPtr[idx];
      const T b = refPtr[idx];
      sumA2 += a * a;
      sumB2 += b * b;
      sumAB += a * b;
    }
  }
  return std::sqrt(sumA2 + sumB2 - static_cast<T>(2) * sumAB) /
         std::sqrt(sumB2);
}

/// \brief Test that the vector potential is square symmetric.
///
/// \param inVectorPotential The vector potential.
/// \param inTolerance The tolerance.
///
/// \return Void.
template <typename T>
void testSymmetry(const GridCPU<T> &inVectorPotential, const T inTolerance) {
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
/// disk.
///
/// \param inTolerance The tolerance for the relative Frobenius norm between the
/// analytical, and the computed, solutions.
/// \param inSymmetryTolerance The relative tolerance when testing that the
/// solution is symmetric.
///
/// \return Void.
template <typename T>
void testMagnetoStaticsDisc(const T inTolerance, const T inSymmetryTolerance) {
  // =================== ARRANGE ==================== //
  // The number of lattice sites per side.
  const int N = 555;

  // Physical side length of a lattice cell.
  const T latticeLength = static_cast<T>(0.1);

  // The London penetration depth. It is chosen so that the factor "4/k^2" in
  // "ComputeVectorPotential" is one.
  const T penetrationDepth = static_cast<T>(2);

  // The number of "fields" for the domain labels and the current density.
  const int domainDimension = 1;
  const int currentDimension = 2;

  // The radius and center coordinate of a disk.
  const T R = latticeLength * static_cast<T>(N - 1) * static_cast<T>(0.4);
  const T center = static_cast<T>(N - 1) * static_cast<T>(0.5);

  // Set the domain labels to "1" everywhere, i.e. compute the vector potential
  // everywhere. We do this because the analytical solution extends beyond the
  // grain.
  GridData<char, thrust::device_vector<char>> domainGPU(N, N, domainDimension,
                                                        Type::real);
  domainGPU.setValue('1', scg::Type::real);

  // Construct the current density, and the expected potential, on the CPU.
  GridCPU<T> currenDensityCPU(N, N, currentDimension, Type::real);
  GridCPU<T> expectedPotential(N, N, currentDimension, Type::real);

  // Get the pointers.
  T *const srcPtrX = currenDensityCPU.getDataPointer(0);
  T *const srcPtrY = currenDensityCPU.getDataPointer(1);
  T *const potPtrX = expectedPotential.getDataPointer(0);
  T *const potPtrY = expectedPotential.getDataPointer(1);

  for (int j = 0; j < N; ++j) {
    // Y-coordinate relative to the center of the disc.
    const T y = latticeLength * (static_cast<T>(j) - center);
    for (int i = 0; i < N; ++i) {
      // Y-coordinate relative to the center of the disc.
      const T x = latticeLength * (static_cast<T>(i) - center);

      // The distance from the center of the disc.
      const T r = std::hypot(x, y);

      // Whether a point is within the disk.
      const bool inDomain = r <= R;

      // The lienar index.
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

      // Angular part of the expected potential inside the disc.
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
  const GridGPU<T> currenDensityGPU(currenDensityCPU);

  // =================== ACT ==================== //
  // Construct a vector-potential grid on the GPU.
  GridGPU<T> potentialGPU(N, N, currentDimension, Type::real);

  // Compute the induced vector potential.
  computeVectorPotential(potentialGPU, currenDensityGPU, domainGPU,
                         latticeLength, penetrationDepth);

  // Copy the vector potential to the CPU.
  const GridCPU<T> potentialCPU(potentialGPU);

  // =================== ASSERT ==================== //
  const T expectedRelativeFrobeniusNorm = static_cast<T>(0);
  const T testRelativeFrobeniusNorm =
      computeRelativeFrobeniusNorm(potentialCPU, expectedPotential);
  CHECK(testRelativeFrobeniusNorm ==
        doctest::Approx(expectedRelativeFrobeniusNorm).epsilon(inTolerance));

  // It is a bit ugly to test the symmetry here as well. But it saves time.
  testSymmetry(potentialCPU, inSymmetryTolerance);
}
} // namespace helper

// ================= Test magnetostatics ================= //

TEST_CASE("Test magnetostatics (float)") {
  const float tolerance = 1e-3f;
  // TODO(niclas): The current implementation of how to compute the vector
  // potential is abyssmal for float. It is not very symmetric. But the overall
  // solution is still fairly good. Investigate!
  const float symmetryTolerance = 1e-2f;
  helper::testMagnetoStaticsDisc(tolerance, symmetryTolerance);
}

TEST_CASE("Test magnetostatics (double)") {
  const double tolerance = 1e-4;
  const double symmetryTolerance = 1e-8;
  helper::testMagnetoStaticsDisc(tolerance, symmetryTolerance);
}
} // namespace scg
