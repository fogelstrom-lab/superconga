//
// Handles the discrete representation of a disc shape primitive.
//
#ifndef DISC_GEOMETRY_H_
#define DISC_GEOMETRY_H_

#include "GeometryComponent.h"
#include <Disc.h>
#include <cuda.h>

template <typename T> class DiscGeometry : public GeometryComponent<T> {
public:
  DiscGeometry();
  virtual ~DiscGeometry();

  DiscGeometry(const DiscGeometry &other);

  DiscGeometry(const T radius, const T pos_x = 0.0, const T pos_y = 0.0);

  DiscGeometry &operator=(DiscGeometry other);

  virtual DiscGeometry *clone() const { return new DiscGeometry(*this); }

  virtual void create(const T angle);
  virtual Grid<char>::GPU *createDomain(const int N);

  Disc<T> *getDisc() const { return _disc; }

private:
  friend void swap(DiscGeometry &A, DiscGeometry &B) {
    swap(static_cast<GeometryComponent<T> &>(A),
         static_cast<GeometryComponent<T> &>(B));
    std::swap(A._disc, B._disc);
  }

  Disc<T> *_disc;
};

namespace {
template <typename T>
__global__ void generateDisc_Kernel(char *dom_p, T *normX_p, T *normY_p,
                                    T *dist_p, const int Nx, const int Ny,
                                    const T radius, const T pos_x,
                                    const T pos_y, const T angle) {
  int i;

  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  if (tx < Nx && ty < Ny) {
    const int idx = ty * Nx + tx;

    T coord_x, coord_y[3], distSq[3];
    T radiusSq = radius * radius;

    coord_x = (T)tx - (Nx - 1) * (T)0.5 - pos_x;

    for (i = 0; i < 3; ++i) {
      coord_y[i] = (T)(ty - 1 + i) - (Ny - 1) * (T)0.5 - pos_y;
      distSq[i] = coord_x * coord_x + coord_y[i] * coord_y[i];
    }

    // Compute domain labels
    char label = 0;

    if (distSq[1] <= radiusSq) {
      label = 1;
      if (distSq[0] > radiusSq)
        label = 2;
      else if (distSq[2] > radiusSq)
        label = 3;

      if (distSq[0] > radiusSq && distSq[2] > radiusSq)
        label = 0;
    }

    // Distance along y (trajectory path direction) to exact boundary location
    T dist_intersect = (T)0.0;
    T norm_x = (T)0.0;
    T norm_y = (T)0.0;

    if (label > 1) {
      // Determine boundary normal at exact boundary location
      dist_intersect = sqrt(radiusSq - coord_x * coord_x) - abs(coord_y[1]);
      norm_x = -coord_x / radius;

      if (coord_y[1] <= (T)0.0)
        norm_y = -(coord_y[1] - dist_intersect) / radius;
      else
        norm_y = -(coord_y[1] + dist_intersect) / radius;
    }

    // Store computed data
    //
    // (Boundary intersection is regarded as inside.
    // Maybe implement special case for this, to avoid dividing by zero, etc.)
    dom_p[idx] = label;
    dist_p[idx] = dist_intersect;
    normX_p[idx] = norm_x;
    normY_p[idx] = norm_y;
  }
}
} // namespace

template <typename T>
DiscGeometry<T>::DiscGeometry() : GeometryComponent<T>(), _disc(0) {}

template <typename T>
DiscGeometry<T>::DiscGeometry(const DiscGeometry<T> &other)
    : GeometryComponent<T>(other), _disc(0) {
  if (other._disc)
    _disc = new Disc<T>(*other._disc);
}

template <typename T>
DiscGeometry<T>::DiscGeometry(const T radius, const T pos_x, const T pos_y)
    : GeometryComponent<T>() {
  _disc = new Disc<T>(radius, pos_x, pos_y);
}

template <typename T> DiscGeometry<T>::~DiscGeometry() {
  if (_disc)
    delete _disc;
}

template <typename T>
DiscGeometry<T> &DiscGeometry<T>::operator=(DiscGeometry<T> other) {
  swap(*this, other);
  return *this;
}

template <typename T> void DiscGeometry<T>::create(const T angle) {
  typedef GeometryComponent<T> G;

  G::allocateMem();

  Parameters<T> *param = G::getParameters();

  if (!param) {
    std::cout << "DiscGeometry::create(): ERROR - No parameters object!\n";
    return;
  }

  const int N = G::getParameters()->getGridResolutionBase();

  const T pos_x = _disc->getPosX();
  const T pos_y = _disc->getPosY();

  const T posRot_x = (cos(angle) * pos_x - sin(angle) * pos_y) * (N - 1);
  const T posRot_y = (sin(angle) * pos_x + cos(angle) * pos_y) * (N - 1);

  const T radius = _disc->getRadius() * T(N - 1);

  int Nx, Ny;
  dim3 bpg, tpb;
  G::getGridDimensions(Nx, Ny);
  G::getKernelDimensions(bpg, tpb);

  // Defines discrete boundary domain and properties
  generateDisc_Kernel<T><<<bpg, tpb>>>(
      G::getDomainLabels(), G::getNormals(0), G::getNormals(1),
      G::getBoundaryDistance(), Nx, Ny, radius, posRot_x, posRot_y, angle);
}

template <typename T>
Grid<char>::GPU *DiscGeometry<T>::createDomain(const int N) {
  Grid<char>::GPU *dom_component = new Grid<char>::GPU(N, N, 1, Type::real);
  dom_component->setZero();

  dim3 bpg, tpb;
  Grid<T>::GPU::getKernelDimensions(bpg, tpb, N, N);

  return dom_component;
}

#endif // DISC_GEOMETRY_H_
