//===--------- PolygonGeometry.h - Polygon shape primitive class. ---------===//
//
// Part of the SuprConGPU Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Class handling the discrete representation of a polygon shape primitive.
/// Derives from the GeometryComponent class.
///
//===----------------------------------------------------------------------===//
#ifndef POLYGON_GEOMETRY_H_
#define POLYGON_GEOMETRY_H_

#include "GeometryComponent.h"
#include "Polygon.h"
#include "defines.h"
#include <cmath>
#include <cuda.h>

namespace scg {
template <typename T> class PolygonGeometry : public GeometryComponent<T> {
public:
  PolygonGeometry();
  ~PolygonGeometry();

  PolygonGeometry(const PolygonGeometry &other);

  // Initialize the polygon to an approximation of a disc
  // numVertices = 4 will produce a square. inDiameter gives the diameter of the
  // disc relative to the available simulation space (to be padded by the grain
  // margin). inAngle specifies the rotation of the polygon, and (inCenterX,
  // inCenterY) is the center coordinate

  /// \brief Create a regular polygon inside a circle.
  ///
  /// \param inGrainFraction The fraction of the simulation space maximally used
  /// for the grain. In order to handle interpolation a few pixels are needed as
  /// a buffer outside of the grain, thus the grain fraction must be less than
  /// one.
  /// \param inNumVertices The numer of vertices of the polygon, e.g. 4 will
  /// produce a square.
  /// \param inSideLength The side length of the polygon. It will be scaled
  /// by the grain fraction. If it is negative the polygon vertices will be
  /// places on the largest possible bounding circle.
  /// \param inAngle The rotation of the polygon.
  /// \param inCenterX The x-coordinate of the circle center. It will be scaled
  /// by the grain fraction.
  /// \param inCenterY The y-coordiante of the circle center. It will be scaled
  /// by the grain fraction.
  PolygonGeometry(const T inGrainFraction, const int inNumVertices,
                  const T inSideLength = static_cast<T>(-1.0),
                  const T inAngle = static_cast<T>(0.0),
                  const T inCenterX = static_cast<T>(0.0),
                  const T inCenterY = static_cast<T>(0.0));

  /// \brief Create a polygon from a predefined vertex set.
  ///
  /// \param inGrainFraction The fraction of the simulation space maximally used
  /// for the grain. In order to handle interpolation a few pixels are needed as
  /// a buffer outside of the grain, thus the grain fraction must be less than
  /// one.
  /// \param inNumVertices The numer of vertices of the polygon.
  /// \param inVerticesX The x-coordinates of the polygon vertices. They will be
  /// scaled by the grain fraction.
  /// \param inVerticesY The y-coordinates of the polygon vertices.  They will
  /// be scaled by the grain fraction.
  // TODO: (Patric) This should throw an exception if inNumVertices is not the
  // same length as inVerticesX and inVerticesY. Better yet, remove
  // use of inNumvertices, and throw an exception if X and Y vertices are not
  // same length.
  PolygonGeometry(const T inGrainFraction, const int inNumVertices,
                  const T *const inVerticesX, const T *const inVerticesY);

  PolygonGeometry &operator=(PolygonGeometry other);

  PolygonGeometry *clone() const override { return new PolygonGeometry(*this); }

  void create(const T inAngle) override;
  GridGPU<char> *createDomain(const int N) const override;

  Polygon<T> *getPolygon() const { return m_polygon; }

private:
  friend void swap(PolygonGeometry &A, PolygonGeometry &B) {
    swap(static_cast<GeometryComponent<T> &>(A),
         static_cast<GeometryComponent<T> &>(B));
    std::swap(A.m_polygon, B.m_polygon);
  }

  Polygon<T> *m_polygon;
};

namespace internal {

template <typename T, int NV>
__global__ void generatePolygonGeometry_Kernel(
    char *const outDomainLabels, T *const outNormalsX, T *const outNormalsY,
    T *const outBoundaryDistances, const int Nx, const int Ny, const T inAngle,
    const T inScale, const int inNumVertices, const T *const inVerticesX,
    const T *const inVerticesY) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  int i;

  T ptX_R[NV], ptY_R[NV], lineVectX, lineVectY, crossZ, tmp_normalX,
      tmp_normalY, norm_tmp;

  T boundaryDistance, normalX, normalY;

  T m, k, tmp_boundaryDistance;
  T cosAngle, sinAngle;

  if (tx < Nx && ty < Ny) {
    const int idx = ty * Nx + tx;

    const T coordX = (T)tx - (T)(Nx - 1) * (T)0.5;
    const T coordY = (T)ty - (T)(Ny - 1) * (T)0.5;

    // Rotate vertices to local frame
    cosAngle = cos(inAngle);
    sinAngle = sin(inAngle);

    for (i = 0; i < inNumVertices; ++i) {
      ptX_R[i] =
          (cosAngle * inVerticesX[i] - sinAngle * inVerticesY[i]) * inScale;
      ptY_R[i] =
          (sinAngle * inVerticesX[i] + cosAngle * inVerticesY[i]) * inScale;
    }

    // Compute stuff
    int ip, sum_cross = 0;

    boundaryDistance = (T)100000.0;

    for (i = 0; i < inNumVertices; ++i) {
      ip = i + 1;
      if (ip == inNumVertices)
        ip = 0;

      // Vector for polygon line segment
      lineVectX = ptX_R[ip] - ptX_R[i];
      lineVectY = ptY_R[ip] - ptY_R[i];

      // z-component of cross product (used to determine inside/outside)
      crossZ =
          lineVectX * (coordY - ptY_R[i]) - lineVectY * (coordX - ptX_R[i]);

      if (crossZ >= (T)0.0)
        sum_cross++;

      // Normals
      tmp_normalX = -lineVectY;
      tmp_normalY = lineVectX;

      norm_tmp =
          (T)1.0 / sqrt(tmp_normalX * tmp_normalX + tmp_normalY * tmp_normalY);

      tmp_normalX *= norm_tmp;
      tmp_normalY *= norm_tmp;

      // Compute distance to boundary
      if (abs(lineVectX) > (T)0.00001) {
        k = lineVectY / lineVectX;
        m = ptY_R[i] - k * ptX_R[i];

        tmp_boundaryDistance = abs(k * coordX + m - coordY);
      } else {
        tmp_boundaryDistance = (T)10000.0;
      }

      if (tmp_boundaryDistance < boundaryDistance) {
        boundaryDistance = tmp_boundaryDistance;
        normalX = tmp_normalX;
        normalY = tmp_normalY;
      }
    }

    // Store computed data
    outDomainLabels[idx] = (char)(int)(sum_cross == inNumVertices);
    outBoundaryDistances[idx] = boundaryDistance;
    outNormalsX[idx] = normalX;
    outNormalsY[idx] = normalY;
  }
}

template <typename T, int NV>
__global__ void generatePolygonGeometry_Kernel(char *const outDomainLabels,
                                               const int inNumVertices,
                                               const T *const inVerticesX,
                                               const T *const inVerticesY,
                                               const int inNumX,
                                               const int inNumY) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  if (tx < inNumX && ty < inNumY) {
    const int idx = ty * inNumX + tx;

    const T coordX =
        static_cast<T>(tx) / static_cast<T>(inNumX - 1) - static_cast<T>(0.5);
    const T coordY =
        static_cast<T>(ty) / static_cast<T>(inNumY - 1) - static_cast<T>(0.5);

    // Compute stuff
    int sum_cross = 0;
    for (int i = 0; i < inNumVertices; ++i) {
      const int ip = (i == inNumVertices - 1) ? 0 : i + 1;

      // Vector for polygon line segment
      const T lineVectX = inVerticesX[ip] - inVerticesX[i];
      const T lineVectY = inVerticesY[ip] - inVerticesY[i];

      // z-component of cross product (used to determine inside/outside)
      const T crossZ = lineVectX * (coordY - inVerticesY[i]) -
                       lineVectY * (coordX - inVerticesX[i]);

      if (crossZ >= static_cast<T>(0)) {
        sum_cross++;
      }
    }

    // Store computed data
    outDomainLabels[idx] = (char)(int)(sum_cross == inNumVertices);
  }
}

template <typename T>
__global__ void
maskNormals_Kernel(const char *const inDomainLabels, T *const inOutNormalsX,
                   T *const inOutNormalsY, T *const inOutBoundaryDistances,
                   const int Nx, const int Ny) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  const int idx = ty * Nx + tx;

  if (tx < Nx && ty < Ny) {
    const T mult = T(int(inDomainLabels[idx]) > 1);

    // Store computed data
    inOutBoundaryDistances[idx] *= mult;
    inOutNormalsX[idx] *= mult;
    inOutNormalsY[idx] *= mult;
  }
}
} // namespace internal

template <typename T>
PolygonGeometry<T>::PolygonGeometry() : GeometryComponent<T>(), m_polygon(0) {}

template <typename T>
PolygonGeometry<T>::PolygonGeometry(const PolygonGeometry<T> &other)
    : GeometryComponent<T>(other), m_polygon(0) {
  if (other.m_polygon)
    m_polygon = new Polygon<T>(*other.m_polygon);
}

// Create an n-gon, where the vertices are put equidistantly on a circle.
template <typename T>
PolygonGeometry<T>::PolygonGeometry(const T inGrainFraction,
                                    const int inNumVertices,
                                    const T inSideLength, const T inAngle,
                                    const T inCenterX, const T inCenterY)
    : GeometryComponent<T>() {
  // Calculate radius from the side length.
  const T angleStep =
      static_cast<T>(2.0 * M_PI) / static_cast<T>(inNumVertices);
  const T radiusFromSideLength =
      inSideLength /
      std::hypot(std::cos(angleStep) - static_cast<T>(1), std::sin(angleStep));

  // Set the radius. If it is negative, use default value.
  const bool useDefaultRadius = inSideLength < static_cast<T>(0);
  const T radius = useDefaultRadius
                       ? static_cast<T>(constants::MAXIMUM_COORDINATE)
                       : radiusFromSideLength;

  // Apply scaling and margin..
  const T radiusPadded = radius * inGrainFraction;
  const T centerPaddedX = inCenterX * inGrainFraction;
  const T centerPaddedY = inCenterY * inGrainFraction;

  m_polygon = new Polygon<T>(inNumVertices, radiusPadded, inAngle,
                             centerPaddedX, centerPaddedY);
}

// Create an n-gon by explicitly specifying the vertex coordinates.
// Todo(patric): Check that inVertices actually has the same length as
// inNumVertices, or better yet, send in a vector.
template <typename T>
PolygonGeometry<T>::PolygonGeometry(const T inGrainFraction,
                                    const int inNumVertices,
                                    const T *const inVerticesX,
                                    const T *const inVerticesY)
    : GeometryComponent<T>() {

#ifndef NDEBUG
  if (inNumVertices < 3) {
    // Todo(patric): Raise exception
    std::cerr
        << "-- ERROR! PolygonGeometry: A polygon needs at least 3 vertices. "
           "inNumVertices = "
        << inNumVertices << std::endl;
  }
#endif

  T scaledVerticesX[inNumVertices] = {static_cast<T>(0.0)};
  T scaledVerticesY[inNumVertices] = {static_cast<T>(0.0)};

  // Loop over vertex coordinates and check that all lie within allowed domain.
  for (int vertexIdx = 0; vertexIdx < inNumVertices; ++vertexIdx) {
    // For convenience.
    const T vertexX = inVerticesX[vertexIdx];
    const T vertexY = inVerticesY[vertexIdx];

#ifndef NDEBUG
    // Check that vertex coordinates are reasonable
    if ((std::abs(vertexX) > static_cast<T>(constants::MAXIMUM_COORDINATE)) ||
        (std::abs(vertexY) > static_cast<T>(constants::MAXIMUM_COORDINATE))) {
      // Todo(patric): Raise exception
      std::cerr
          << "-- ERROR! PolygonGeometry: Vertex coordinates must be within "
             "[-"
          << constants::MAXIMUM_COORDINATE << ", "
          << constants::MAXIMUM_COORDINATE << "]." << std::endl;
    }
#endif

    scaledVerticesX[vertexIdx] = vertexX * inGrainFraction;
    scaledVerticesY[vertexIdx] = vertexY * inGrainFraction;
  }

  m_polygon = new Polygon<T>(inNumVertices, scaledVerticesX, scaledVerticesY);
}

template <typename T> PolygonGeometry<T>::~PolygonGeometry() {
  if (m_polygon)
    delete m_polygon;
}

template <typename T>
PolygonGeometry<T> &PolygonGeometry<T>::operator=(PolygonGeometry<T> other) {
  swap(*this, other);
  return *this;
}

template <typename T> void PolygonGeometry<T>::create(const T inAngle) {
  typedef GeometryComponent<T> G;

  G::allocateMem();

  int Nx, Ny;
  dim3 bpg, tpb;

  G::getGridDimensions(Nx, Ny);
  G::getKernelDimensions(bpg, tpb);

  // We the size of the unrotated simulation domain, so that the size of the
  // polygon, measured in pixels, is the same.
  const int N = GeometryComponent<T>::getParameters()->getGridResolutionBase();
  const T scale = static_cast<T>(N - 1);

  // Calc inside/outside, distance, normals.
  internal::generatePolygonGeometry_Kernel<T, 256><<<bpg, tpb>>>(
      G::getDomainLabels(), G::getNormals(0), G::getNormals(1),
      G::getBoundaryDistance(), Nx, Ny, inAngle, scale,
      m_polygon->getNumVertices(), m_polygon->getVertices()->getDataPointer(0),
      m_polygon->getVertices()->getDataPointer(1));

  G::labelBoundaryNodes();

  internal::maskNormals_Kernel<T>
      <<<bpg, tpb>>>(G::getDomainLabels(), G::getNormals(0), G::getNormals(1),
                     G::getBoundaryDistance(), Nx, Ny);
}

template <typename T>
GridGPU<char> *
PolygonGeometry<T>::createDomain(const int inLatticeSideLength) const {
  const int numFields = 1;
  GridGPU<char> *dom_component = new GridGPU<char>(
      inLatticeSideLength, inLatticeSideLength, numFields, Type::real);
  dom_component->setZero();

  dim3 bpg, tpb;
  GridGPU<T>::getKernelDimensions(bpg, tpb, inLatticeSideLength,
                                  inLatticeSideLength);

  // Calc inside/outside.
  internal::generatePolygonGeometry_Kernel<T, 256><<<bpg, tpb>>>(
      dom_component->getDataPointer(), m_polygon->getNumVertices(),
      m_polygon->getVertices()->getDataPointer(0),
      m_polygon->getVertices()->getDataPointer(1), inLatticeSideLength,
      inLatticeSideLength);

  return dom_component;
}
} // namespace scg

#endif // POLYGON_GEOMETRY_H_
