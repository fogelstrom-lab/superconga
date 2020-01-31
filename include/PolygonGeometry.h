#ifndef POLYGONGEOMETRY_H
#define POLYGONGEOMETRY_H

#include "GeometryComponent.h"
#include "Polygon.h"
#include "defines.h"
#include <cuda.h>

template <typename T> class PolygonGeometry : public GeometryComponent<T> {
public:
  PolygonGeometry();
  virtual ~PolygonGeometry();

  PolygonGeometry(const PolygonGeometry &other);

  // Initialize the polygon to an approximation of a disc
  // numVertices = 4 will produce a square. inDiameter gives the diameter of the
  // disc relative to the available simulation space (to be padded by the grain
  // margin). inAngle specifies the rotation of the polygon, and (inCenterX,
  // inCenterY) is the center coordinate
  PolygonGeometry(const int inNumVertices,
                  const T inDiameter = static_cast<T>(1.0),
                  const T inAngle = static_cast<T>(0.0),
                  const T inCenterX = static_cast<T>(0.0),
                  const T inCenterY = static_cast<T>(0.0));

  // Initialize the polygon to a predefined vertex set
  // TODO: (Patric) This should throw an exception if inNumVertices is not the
  // same length as inVerticesX and inVerticesY. Better yet, remove
  // use of inNumvertices, and throw an exception if X and Y vertices are not
  // same length.
  PolygonGeometry(const int inNumVertices, const T *const inVerticesX,
                  const T *const inVerticesY);

  PolygonGeometry &operator=(PolygonGeometry other);

  virtual PolygonGeometry *clone() const { return new PolygonGeometry(*this); }

  virtual void create(const T inAngle);
  virtual Grid<char>::GPU *createDomain(const int N);

  Polygon<T> *getPolygon() const { return m_polygon; }

private:
  friend void swap(PolygonGeometry &A, PolygonGeometry &B) {
    swap(static_cast<GeometryComponent<T> &>(A),
         static_cast<GeometryComponent<T> &>(B));
    std::swap(A.m_polygon, B.m_polygon);
  }

  Polygon<T> *m_polygon;
};

namespace {

template <typename T, int NV>
__global__ void generatePolygonGeometry_Kernel(
    char *const outDomainLabels, T *const outNormalsX, T *const outNormalsY,
    T *const outBoundaryDistances, const int Nx, const int Ny, const T inAngle,
    const int inNumVertices, const T *const inVerticesX,
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
          (cosAngle * inVerticesX[i] - sinAngle * inVerticesY[i]) * (Nx - 1);
      ptY_R[i] =
          (sinAngle * inVerticesX[i] + cosAngle * inVerticesY[i]) * (Nx - 1);
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
                                               const int Nx) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  T ptX_R[NV], ptY_R[NV], lineVectX, lineVectY, crossZ;

  if (tx < Nx && ty < Nx) {
    const int idx = ty * Nx + tx;

    const T coordX = (T)tx - (T)(Nx - 1) * (T)0.5;
    const T coordY = (T)ty - (T)(Nx - 1) * (T)0.5;

    for (int i = 0; i < inNumVertices; ++i) {
      ptX_R[i] = inVerticesX[i] * (Nx - 1);
      ptY_R[i] = inVerticesY[i] * (Nx - 1);
    }

    // Compute stuff
    int ip, sum_cross = 0;

    for (int i = 0; i < inNumVertices; ++i) {
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
} // namespace

template <typename T>
PolygonGeometry<T>::PolygonGeometry() : GeometryComponent<T>(), m_polygon(0) {}

template <typename T>
PolygonGeometry<T>::PolygonGeometry(const PolygonGeometry<T> &other)
    : GeometryComponent<T>(other), m_polygon(0) {
  if (other.m_polygon)
    m_polygon = new Polygon<T>(*other.m_polygon);
}

template <typename T>
PolygonGeometry<T>::PolygonGeometry(const int inNumVertices, const T inDiameter,
                                    const T inAngle, const T inCenterX,
                                    const T inCenterY)
    : GeometryComponent<T>() {
  // Apply margin to radius
  const T polygonRadiusPadded =
      inDiameter * static_cast<T>(0.5) * qcs::GRAIN_SIZE_FRACTION;
  m_polygon = new Polygon<T>(inNumVertices, polygonRadiusPadded, inAngle,
                             inCenterX, inCenterY);
}

template <typename T>
PolygonGeometry<T>::PolygonGeometry(const int inNumVertices,
                                    const T *const inVerticesX,
                                    const T *const inVerticesY)
    : GeometryComponent<T>() {
  m_polygon = new Polygon<T>(inNumVertices, inVerticesX, inVerticesY);
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

  // Calc inside/outside, distance, normals
  generatePolygonGeometry_Kernel<T, 256><<<bpg, tpb>>>(
      G::getDomainLabels(), G::getNormals(0), G::getNormals(1),
      G::getBoundaryDistance(), Nx, Ny, inAngle, m_polygon->getNumVertices(),
      m_polygon->getVertices()->getDataPointer(0),
      m_polygon->getVertices()->getDataPointer(1));

  G::labelBoundaryNodes();

  maskNormals_Kernel<T><<<bpg, tpb>>>(G::getDomainLabels(), G::getNormals(0),
                                      G::getNormals(1),
                                      G::getBoundaryDistance(), Nx, Ny);
}

template <typename T>
Grid<char>::GPU *PolygonGeometry<T>::createDomain(const int N) {
  Grid<char>::GPU *dom_component = new Grid<char>::GPU(N, N, 1, Type::real);
  dom_component->setZero();

  dim3 bpg, tpb;
  Grid<T>::GPU::getKernelDimensions(bpg, tpb, N, N);

  // Calc inside/outside, distance, normals
  generatePolygonGeometry_Kernel<T, 256><<<bpg, tpb>>>(
      dom_component->getDataPointer(), m_polygon->getNumVertices(),
      m_polygon->getVertices()->getDataPointer(0),
      m_polygon->getVertices()->getDataPointer(1), N);

  return dom_component;
}
#endif // POLYGON_GEOMETRY_H_
