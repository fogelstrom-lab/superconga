#ifndef _POLYGONGEOMETRY_H
#define _POLYGONGEOMETRY_H

#include "GeometryComponent.h"
#include "Polygon.h"
#include <cuda.h>

template <typename T> class PolygonGeometry : public GeometryComponent<T> {
public:
  PolygonGeometry();
  virtual ~PolygonGeometry();

  PolygonGeometry(const PolygonGeometry &other);

  // Initialize the polygon to an approximation of a disc
  // numVertices = 4 will produce a square. cx, cy is the center coordinate
  PolygonGeometry(const int numVertices, T radius, T angle = 0.0, T cx = 0.0,
                  T cy = 0.0);

  // Initialize the polygon to a predefined vertex set
  PolygonGeometry(const int numVertices, T *vertX, T *vertY);

  PolygonGeometry &operator=(PolygonGeometry other);

  virtual PolygonGeometry *clone() const { return new PolygonGeometry(*this); }

  virtual void create(const T angle);
  virtual Grid<char>::GPU *createDomain(const int N);

  Polygon<T> *getPolygon() const { return _poly; }

private:
  friend void swap(PolygonGeometry &A, PolygonGeometry &B) {
    swap(static_cast<GeometryComponent<T> &>(A),
         static_cast<GeometryComponent<T> &>(B));
    std::swap(A._poly, B._poly);
  }

  Polygon<T> *_poly;
};

namespace {

template <typename T, int NV>
__global__ void
generatePolygonGeometry_Kernel(char *dom_p, T *normX_p, T *normY_p, T *dist_p,
                               const int Nx, const int Ny, const T phi,
                               const int numVertices, T *vertX, T *vertY) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  int i;

  T ptX_R[NV], ptY_R[NV], lineVectX, lineVectY, crossZ, normX, normY, norm_tmp;

  T isect_out, normX_out, normY_out;

  T m, k, isect;
  T cosPhi, sinPhi;

  if (tx < Nx && ty < Ny) {
    const int idx = ty * Nx + tx;

    const T coordX = (T)tx - (T)(Nx - 1) * (T)0.5;
    const T coordY = (T)ty - (T)(Ny - 1) * (T)0.5;

    // Rotate vertices to local frame
    cosPhi = cos(phi);
    sinPhi = sin(phi);

    for (i = 0; i < numVertices; ++i) {
      ptX_R[i] = (cosPhi * vertX[i] - sinPhi * vertY[i]) * (Nx - 1);
      ptY_R[i] = (sinPhi * vertX[i] + cosPhi * vertY[i]) * (Nx - 1);
    }

    // Compute stuff
    int ip, sum_cross = 0;

    isect_out = (T)100000.0;

    for (i = 0; i < numVertices; ++i) {
      ip = i + 1;
      if (ip == numVertices)
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
      normX = -lineVectY;
      normY = lineVectX;

      norm_tmp = (T)1.0 / sqrt(normX * normX + normY * normY);

      normX *= norm_tmp;
      normY *= norm_tmp;

      // Compute distance to boundary
      if (abs(lineVectX) > (T)0.00001) {
        k = lineVectY / lineVectX;
        m = ptY_R[i] - k * ptX_R[i];

        isect = abs(k * coordX + m - coordY);
      } else {
        isect = (T)10000.0;
      }

      if (isect < isect_out) {
        isect_out = isect;
        normX_out = normX;
        normY_out = normY;
      }
    }

    // Store computed data
    dom_p[idx] = (char)(int)(sum_cross == numVertices);
    dist_p[idx] = isect_out;
    normX_p[idx] = normX_out;
    normY_p[idx] = normY_out;
  }
}

template <typename T, int NV>
__global__ void generatePolygonGeometry_Kernel(char *dom_p,
                                               const int numVertices, T *vertX,
                                               T *vertY, const int Nx) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  T ptX_R[NV], ptY_R[NV], lineVectX, lineVectY, crossZ;

  if (tx < Nx && ty < Nx) {
    const int idx = ty * Nx + tx;

    const T coordX = (T)tx - (T)(Nx - 1) * (T)0.5;
    const T coordY = (T)ty - (T)(Nx - 1) * (T)0.5;

    for (int i = 0; i < numVertices; ++i) {
      ptX_R[i] = vertX[i] * (Nx - 1);
      ptY_R[i] = vertY[i] * (Nx - 1);
    }

    // Compute stuff
    int ip, sum_cross = 0;

    for (int i = 0; i < numVertices; ++i) {
      ip = i + 1;
      if (ip == numVertices)
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
    dom_p[idx] = (char)(int)(sum_cross == numVertices);
  }
}

template <typename T>
__global__ void maskNormals_Kernel(char *dom_p, T *normX_p, T *normY_p,
                                   T *dist_p, const int Nx, const int Ny) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  const int idx = ty * Nx + tx;

  if (tx < Nx && ty < Ny) {
    T mult = T(int(dom_p[idx]) > 1);

    // Store computed data
    dist_p[idx] *= mult;
    normX_p[idx] *= mult;
    normY_p[idx] *= mult;
  }
}
} // namespace

template <typename T>
PolygonGeometry<T>::PolygonGeometry() : GeometryComponent<T>(), _poly(0) {}

template <typename T>
PolygonGeometry<T>::PolygonGeometry(const PolygonGeometry<T> &other)
    : GeometryComponent<T>(other), _poly(0) {
  if (other._poly)
    _poly = new Polygon<T>(*other._poly);
}

template <typename T>
PolygonGeometry<T>::PolygonGeometry(const int numVertices, T radius, T angle,
                                    T cx, T cy)
    : GeometryComponent<T>() {
  _poly = new Polygon<T>(numVertices, radius, angle, cx, cy);
}

template <typename T>
PolygonGeometry<T>::PolygonGeometry(const int numVertices, T *vertX, T *vertY)
    : GeometryComponent<T>() {
  _poly = new Polygon<T>(numVertices, vertX, vertY);
}

template <typename T> PolygonGeometry<T>::~PolygonGeometry() {
  if (_poly)
    delete _poly;
}

template <typename T>
PolygonGeometry<T> &PolygonGeometry<T>::operator=(PolygonGeometry<T> other) {
  swap(*this, other);
  return *this;
}

template <typename T> void PolygonGeometry<T>::create(const T angle) {
  typedef GeometryComponent<T> G;

  G::allocateMem();

  int Nx, Ny;
  dim3 bpg, tpb;

  G::getGridDimensions(Nx, Ny);
  G::getKernelDimensions(bpg, tpb);

  // Calc inside/outside, distance, normals
  generatePolygonGeometry_Kernel<T, 256><<<bpg, tpb>>>(
      G::getDomainLabels(), G::getNormals(0), G::getNormals(1),
      G::getBoundaryDistance(), Nx, Ny, angle, _poly->getNumVertices(),
      _poly->getVertices()->getDataPointer(0),
      _poly->getVertices()->getDataPointer(1));

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
  generatePolygonGeometry_Kernel<T, 256>
      <<<bpg, tpb>>>(dom_component->getDataPointer(), _poly->getNumVertices(),
                     _poly->getVertices()->getDataPointer(0),
                     _poly->getVertices()->getDataPointer(1), N);

  return dom_component;
}
#endif
