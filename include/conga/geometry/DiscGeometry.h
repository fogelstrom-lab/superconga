//===------------- DiscGeometry.h - Disc shape primitve class. ------------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Class handling the discrete representation of a disc shape primitive.
/// Derives from the GeometryComponent class.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_DISC_GEOMETRY_H_
#define CONGA_DISC_GEOMETRY_H_

#include "../defines.h"
#include "../utils.h"
#include "Disc.h"
#include "GeometryComponent.h"

#include <cuda.h>

#include <cmath>

namespace conga {
template <typename T> class DiscGeometry : public GeometryComponent<T> {
public:
  // Initialize a disc. inRadius gives the diameter of the
  // disc relative to the available simulation space (to be padded by the grain
  // margin). The center coordinate is (inCenterX, inCenterY).
  DiscGeometry(const T inGrainFraction,
               const T inRadius = static_cast<T>(constants::MAXIMUM_COORDINATE),
               const T inCenterX = static_cast<T>(0.0),
               const T inCenterY = static_cast<T>(0.0));
  DiscGeometry(const DiscGeometry &inOther);
  ~DiscGeometry() {}

  DiscGeometry &operator=(DiscGeometry inOther);

  DiscGeometry *clone() const override { return new DiscGeometry(*this); }

  void create(const T inAngle) override;
  GridGPU<char> *createDomain(const int inLatticeSideLength) const override;

  Disc<T> &getDisc() { return m_disc; }
  const Disc<T> &getDisc() const { return m_disc; }

private:
  friend void swap(DiscGeometry &inDiscA, DiscGeometry &inDiscB) {
    swap(static_cast<GeometryComponent<T> &>(inDiscA),
         static_cast<GeometryComponent<T> &>(inDiscB));
    std::swap(inDiscA.m_disc, inDiscB.m_disc);
  }

  /// \brief Convert a (fractional) coordinate in [-MAX, MAX], to actual grid
  /// coordinate.
  ///
  /// \param inFractionalCoordinate Coordinate in [-MAX, MAX] of the simulation
  /// space.
  /// \param inSimulationLatticeSideLength The number of lattice sites
  /// per side length of the simulation space.
  /// \param inGridLatticeSideLength The number of elements per side length of
  /// the grid.
  ///
  /// \return The converted coordinate.
  T convertCoordinate(const T inFractionalCoordinate,
                      const int inSimulationLatticeSideLength,
                      const int inGridLatticeSideLength) const {
    const int diff = inGridLatticeSideLength - inSimulationLatticeSideLength;
    const T halfDiff = static_cast<T>(0.5) * static_cast<T>(diff);
    return static_cast<T>(inSimulationLatticeSideLength - 1) *
               (inFractionalCoordinate + constants::MAXIMUM_COORDINATE) +
           halfDiff;
  }

  Disc<T> m_disc;
};

namespace internal {
template <typename T>
__device__ char computeDomainLabel(const int inX, const int inY,
                                   const T inRadius, const T inOffsetX,
                                   const T inOffsetY) {
  // For convenience.
  const T radiusSquared = inRadius * inRadius;

  // In order to determine the domain label of a pixel we check, in a 3x3
  // neighborhood, whether those pixels are within the circle or not, and then
  // computing the row majority. The row majority is due to the trajectories
  // going downward along the y-axis, i.e. the columns.
  //
  // Example A:
  //
  // X 0 0     0
  // X X 0 ==> X
  // X X X     X
  //
  // where 0 = outside circle, X = inside circle, and the center pixel is the
  // one we want to label. There are four labels:
  //
  // 3: Exit    - (X, X, 0)^T
  // 2: Entry   - (0, X, X)^T
  // 1: Inside  - (X, X, X)^T
  // 0: Outside - All other
  //
  // So the center pixel in the example is an entry node.
  //
  // Example B:
  //
  // 0 0 0     0
  // 0 X 0 ==> 0
  // X X X     X
  //
  // This gives that the pixel is outside, even though the pixel is within the
  // circle. This also means that
  //
  // 0 X 0     0
  // X X X ==> X
  // X X X     X
  //
  // gives an entry node. The rational for this is that we want to lonesome
  // pixels like the top one. If we only looked at the center column in
  // Example B, we double determine the center pixel to be an entry node.
  // But consider the same case, but rotated 90 degrees.
  //
  // X 0 0     0
  // X X 0 ==> X
  // X 0 0     0
  //
  // If we only looked at the center column, this same pixel, would be
  // determined to be outside the domain. Which is an inconsistency.
  // By looking at a 3x3 neighborhood we can get rid of those pixels in all
  // directions.
  const int numExtend = 1;
  const int numPoints = 2 * numExtend + 1;
  bool neighbors[numPoints];
  for (int j = 0; j < numPoints; ++j) {
    // Neighbour y-coordinate.
    const T ny = static_cast<T>(inY - 1 + j) - inOffsetY;

    int rowSum = 0;
    for (int i = 0; i < numPoints; ++i) {
      // Neighbour x-coordinate.
      const T nx = static_cast<T>(inX - 1 + i) - inOffsetX;

      // Is the neighbor inside the circle?
      const bool isInside = nx * nx + ny * ny <= radiusSquared;

      rowSum += isInside ? 1 : 0;
    }
    // Check if majority is inside.
    neighbors[j] = rowSum > numExtend;
  }

  // Determine domain label.
  char domainLabel = 0;
  if (neighbors[0] && neighbors[1] && neighbors[2]) {
    // Inside: (X, X, X)^T
    domainLabel = 1;
  } else if (!neighbors[0] && neighbors[1] && neighbors[2]) {
    // Entry: (0, X, X)^T
    domainLabel = 2;
  } else if (neighbors[0] && neighbors[1] && !neighbors[2]) {
    // Exit: (X, X, 0)^T
    domainLabel = 3;
  } else {
    // Outside.
    domainLabel = 0;
  }

  return domainLabel;
}

template <typename T>
__global__ void
generateDisc_Kernel(const int inNumX, const int inNumY, const T inRadius,
                    const T inOffsetX, const T inOffsetY,
                    char *const outDomainLabels, T *const outNormalsX,
                    T *const outNormalsY, T *const outBoundaryDistances) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  if (tx < inNumX && ty < inNumY) {
    // Compute domain label.
    const char domainLabel =
        computeDomainLabel(tx, ty, inRadius, inOffsetX, inOffsetY);

    // Centered coordinates.
    const T x = static_cast<T>(tx) - inOffsetX;
    const T y = static_cast<T>(ty) - inOffsetY;

    // Distance along y (trajectory path direction) to exact boundary location
    T boundaryDistance = static_cast<T>(0);
    T normalX = static_cast<T>(0);
    T normalY = static_cast<T>(0);

    // If the pixel is entry or exit.
    if (domainLabel > 1) {
      // Determine boundary normal at exact boundary location
      boundaryDistance = sqrt(inRadius * inRadius - x * x) - abs(y);
      normalX = -x / inRadius;

      if (y <= static_cast<T>(0)) {
        normalY = -(y - boundaryDistance) / inRadius;
      } else {
        normalY = -(y + boundaryDistance) / inRadius;
      }
    }

    // Store computed data
    //
    // (Boundary intersection is regarded as inside.
    // Maybe implement special case for this, to avoid dividing by zero, etc.)
    const int idx = ty * inNumX + tx;
    outDomainLabels[idx] = domainLabel;
    outBoundaryDistances[idx] = boundaryDistance;
    outNormalsX[idx] = normalX;
    outNormalsY[idx] = normalY;
  }
}

template <typename T>
__global__ void generateDisc_Kernel(const int inNumX, const int inNumY,
                                    const T inRadius, const T inOffsetX,
                                    const T inOffsetY,
                                    char *const outDomainLabels) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  if (tx < inNumX && ty < inNumY) {
    // Linear index.
    const int idx = ty * inNumX + tx;

    // Compute domain label.
    const char domainLabel =
        computeDomainLabel(tx, ty, inRadius, inOffsetX, inOffsetY);

    // For polygons it is only ones and zeros.
    outDomainLabels[idx] = domainLabel > 0 ? 1 : 0;
  }
}
} // namespace internal

template <typename T>
DiscGeometry<T>::DiscGeometry(const DiscGeometry<T> &inOther)
    : GeometryComponent<T>(inOther), m_disc() {}

template <typename T>
DiscGeometry<T>::DiscGeometry(const T inGrainFraction, const T inRadius,
                              const T inCenterX, const T inCenterY)
    : GeometryComponent<T>() {

  // Apply margin/scaling.
  const T discRadiusPadded = inRadius * inGrainFraction;

  // Apply scaling and margin to center coordinates
  const T centerPaddedX = inCenterX * inGrainFraction;
  const T centerPaddedY = inCenterY * inGrainFraction;
  m_disc = Disc<T>(discRadiusPadded, centerPaddedX, centerPaddedY);
}

template <typename T>
DiscGeometry<T> &DiscGeometry<T>::operator=(DiscGeometry<T> inOther) {
  swap(*this, inOther);
  return *this;
}

template <typename T> void DiscGeometry<T>::create(const T inAngle) {
  typedef GeometryComponent<T> G;

  G::allocateMem();

  const T centerX = m_disc.getPosX();
  const T centerY = m_disc.getPosY();

  T rotatedCenterX;
  T rotatedCenterY;
  utils::rotate(inAngle, centerX, centerY, rotatedCenterX, rotatedCenterY);

  const auto [numX, numY] = G::getGridDimensions();

  const int N = G::getParameters()->getGridResolutionBase();
  const T offsetX = convertCoordinate(rotatedCenterX, N, numX);
  const T offsetY = convertCoordinate(rotatedCenterY, N, numY);
  const T radius = m_disc.getRadius() * static_cast<T>(N - 1);

  const auto [bpg, tpb] = G::getKernelDimensions();

  // Defines discrete boundary domain and properties
  internal::generateDisc_Kernel<T><<<bpg, tpb>>>(
      numX, numY, radius, offsetX, offsetY, G::getDomainLabels(),
      G::getNormals(0), G::getNormals(1), G::getBoundaryDistance());
}

template <typename T>
GridGPU<char> *
DiscGeometry<T>::createDomain(const int inLatticeSideLength) const {
  const int numFields = 1;
  GridGPU<char> *dom_component = new GridGPU<char>(
      inLatticeSideLength, inLatticeSideLength, numFields, Type::real);
  dom_component->setZero();

  // Get the center of the disc.
  const T centerX = m_disc.getPosX();
  const T centerY = m_disc.getPosY();

  const int N = GeometryComponent<T>::getParameters()->getGridResolutionBase();
  const T offsetX = convertCoordinate(centerX, N, inLatticeSideLength);
  const T offsetY = convertCoordinate(centerY, N, inLatticeSideLength);
  const T radius = m_disc.getRadius() * static_cast<T>(N - 1);

  // Get the required kernel dimensions.
  const auto [bpg, tpb] =
      GridGPU<T>::getKernelDimensions(inLatticeSideLength, inLatticeSideLength);

  // Compute domain labels.
  internal::generateDisc_Kernel<T>
      <<<bpg, tpb>>>(inLatticeSideLength, inLatticeSideLength, radius, offsetX,
                     offsetY, dom_component->getDataPointer());

  return dom_component;
}
} // namespace conga

#endif // CONGA_DISC_GEOMETRY_H_
