//
// Handles the discrete representation of a disc shape primitive.
//
#ifndef DISC_GEOMETRY_H_
#define DISC_GEOMETRY_H_

#include "Disc.h"
#include "GeometryComponent.h"
#include "defines.h"
#include <cmath>
#include <cuda.h>

namespace scg {
template <typename T> class DiscGeometry : public GeometryComponent<T> {
public:
  DiscGeometry();
  virtual ~DiscGeometry();

  DiscGeometry(const DiscGeometry &other);

  // Initialize a disc. inDiameter gives the diameter of the
  // disc relative to the available simulation space (to be padded by the grain
  // margin). The center coordinate is (inCenterX, inCenterY).
  DiscGeometry(const T inDiameter, const T inCenterX = static_cast<T>(0.0),
               const T inCenterY = static_cast<T>(0.0));

  DiscGeometry &operator=(DiscGeometry other);

  virtual DiscGeometry *clone() const { return new DiscGeometry(*this); }

  virtual void create(const T inAngle);
  virtual Grid<char>::GPU *createDomain(const int N);

  Disc<T> *getDisc() const { return m_disc; }

private:
  friend void swap(DiscGeometry &A, DiscGeometry &B) {
    swap(static_cast<GeometryComponent<T> &>(A),
         static_cast<GeometryComponent<T> &>(B));
    std::swap(A.m_disc, B.m_disc);
  }

  Disc<T> *m_disc;
};

namespace internal {
template <typename T>
__global__ void
generateDisc_Kernel(char *const outDomainLabels, T *const outNormalsX,
                    T *const outNormalsY, T *const outBoundaryDistances,
                    const int Nx, const int Ny, const T inRadius,
                    const T inRotatedCenterX, const T inRotatedCenterY) {
  int i;

  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  if (tx < Nx && ty < Ny) {
    const int idx = ty * Nx + tx;

    T coord_x, coord_y[3], distSq[3];
    const T radiusSquared = inRadius * inRadius;

    coord_x = (T)tx - (Nx - 1) * (T)0.5 - inRotatedCenterX;

    for (i = 0; i < 3; ++i) {
      coord_y[i] = (T)(ty - 1 + i) - (Ny - 1) * (T)0.5 - inRotatedCenterY;
      distSq[i] = coord_x * coord_x + coord_y[i] * coord_y[i];
    }

    // Compute domain labels
    char domainLabel = 0;

    if (distSq[1] <= radiusSquared) {
      domainLabel = 1;
      if (distSq[0] > radiusSquared)
        domainLabel = 2;
      else if (distSq[2] > radiusSquared)
        domainLabel = 3;

      if (distSq[0] > radiusSquared && distSq[2] > radiusSquared)
        domainLabel = 0;
    }

    // Distance along y (trajectory path direction) to exact boundary location
    T boundaryDistance = (T)0.0;
    T normalX = (T)0.0;
    T normalY = (T)0.0;

    if (domainLabel > 1) {
      // Determine boundary normal at exact boundary location
      boundaryDistance =
          sqrt(radiusSquared - coord_x * coord_x) - abs(coord_y[1]);
      normalX = -coord_x / inRadius;

      if (coord_y[1] <= (T)0.0)
        normalY = -(coord_y[1] - boundaryDistance) / inRadius;
      else
        normalY = -(coord_y[1] + boundaryDistance) / inRadius;
    }

    // Store computed data
    //
    // (Boundary intersection is regarded as inside.
    // Maybe implement special case for this, to avoid dividing by zero, etc.)
    outDomainLabels[idx] = domainLabel;
    outBoundaryDistances[idx] = boundaryDistance;
    outNormalsX[idx] = normalX;
    outNormalsY[idx] = normalY;
  }
}
} // namespace internal

template <typename T>
DiscGeometry<T>::DiscGeometry() : GeometryComponent<T>(), m_disc(0) {}

template <typename T>
DiscGeometry<T>::DiscGeometry(const DiscGeometry<T> &other)
    : GeometryComponent<T>(other), m_disc(0) {
  if (other.m_disc)
    m_disc = new Disc<T>(*other.m_disc);
}

template <typename T>
DiscGeometry<T>::DiscGeometry(const T inDiameter, const T inCenterX,
                              const T inCenterY)
    : GeometryComponent<T>() {

  // Scale to internal coordinate system and apply the margin (grain size frac).
  const T coordinateScaleFactor =
      static_cast<T>(constants::GRAIN_SIZE_FRACTION *
                     constants::INTERNAL_MAXIMUM_COORDINATE_DISC /
                     constants::MAXIMUM_COORDINATE);
  // Calculate radius from diameter and apply margin/scaling.
  const T discRadiusPadded =
      static_cast<T>(0.5) * inDiameter * coordinateScaleFactor;
  // Apply scaling and margin to center coordinates
  const T centerPaddedX = inCenterX * coordinateScaleFactor;
  const T centerPaddedY = inCenterY * coordinateScaleFactor;
  m_disc = new Disc<T>(discRadiusPadded, centerPaddedX, centerPaddedY);
}

template <typename T> DiscGeometry<T>::~DiscGeometry() {
  if (m_disc)
    delete m_disc;
}

template <typename T>
DiscGeometry<T> &DiscGeometry<T>::operator=(DiscGeometry<T> other) {
  swap(*this, other);
  return *this;
}

template <typename T> void DiscGeometry<T>::create(const T inAngle) {
  typedef GeometryComponent<T> G;

  G::allocateMem();

  Parameters<T> *param = G::getParameters();

  if (!param) {
    std::cout << "DiscGeometry::create(): ERROR - No parameters object!\n";
    return;
  }

  const int N = G::getParameters()->getGridResolutionBase();

  const T centerX = m_disc->getPosX();
  const T centerY = m_disc->getPosY();

  const T rotatedCenterX =
      (std::cos(inAngle) * centerX - std::sin(inAngle) * centerY) * (N - 1);
  const T rotatedCenterY =
      (std::sin(inAngle) * centerX + std::cos(inAngle) * centerY) * (N - 1);

  const T radius = m_disc->getRadius() * T(N - 1);

  int Nx, Ny;
  dim3 bpg, tpb;
  G::getGridDimensions(Nx, Ny);
  G::getKernelDimensions(bpg, tpb);

  // Defines discrete boundary domain and properties
  internal::generateDisc_Kernel<T><<<bpg, tpb>>>(
      G::getDomainLabels(), G::getNormals(0), G::getNormals(1),
      G::getBoundaryDistance(), Nx, Ny, radius, rotatedCenterX, rotatedCenterY);
}

template <typename T>
Grid<char>::GPU *DiscGeometry<T>::createDomain(const int N) {
  Grid<char>::GPU *dom_component = new Grid<char>::GPU(N, N, 1, Type::real);
  dom_component->setZero();

  dim3 bpg, tpb;
  Grid<T>::GPU::getKernelDimensions(bpg, tpb, N, N);

  return dom_component;
}
} // namespace scg

#endif // DISC_GEOMETRY_H_
