//===-- GeometryComponent.h - Geometry component base class. --------------===//
//
// Part of the SuprConGPU Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Base class for the discrete representation of geometry, as used by the
/// Riccati solver.
///
//===----------------------------------------------------------------------===//

#ifndef GEOMETRY_COMPONENT_H_
#define GEOMETRY_COMPONENT_H_

#include "Grid.h"
#include "Parameters.h"
#include "Type.h"
#include <cuda.h>
#include <iostream>

namespace scg {
template <typename T> class GeometryComponent {
public:
  GeometryComponent();
  virtual ~GeometryComponent();

  GeometryComponent(const GeometryComponent<T> &other);

  virtual GeometryComponent *clone() const = 0;

  Parameters<T> *getParameters() const { return _param; }
  void setParameters(Parameters<T> *param) { _param = param; }

  // Convenience function to get discrete dimensions of geomtetry
  void getGridDimensions(int &Nx, int &Ny) const;

  // Convenience function to set cuda grid dimensions
  void getKernelDimensions(dim3 &bpg, dim3 &tpb) const;

  // Getters (pointers to data in corresponding grids)
  char *getDomainLabels() const;
  T *getNormals(const int dim) const;
  T *getBoundaryDistance() const;

  // Getters for pointers to grids containing discrete geometry representation
  GridGPU<char> *getDomainLabelsGrid() const { return _domainLabels; }
  GridGPU<T> *getNormalsGrid() const { return _normals; }
  GridGPU<T> *getBoundaryDistanceGrid() const { return _boundaryDistance; }

  // Create discrete representation for geometry component in rotational frame
  // given by angle argument Stores result in _domainLabels, _normals, and
  // _boundaryDistance
  virtual void create(const T angle) = 0;

  // Create discrete representation with resolution NxN, and return domain
  // labels only. Does not store data in any member variables.
  virtual GridGPU<char> *createDomain(const int N) const = 0;

  // Label domain nodes which are close to boundary.
  void labelBoundaryNodes();

  // Convenience functions to allocate and free memory for _domainLabels,
  // _normals, and _boundaryDistance
  void allocateMem();
  void freeMem(); // Gets called by GeometryGroup.

private:
  template <typename S>
  friend void swap(GeometryComponent<S> &A, GeometryComponent<S> &B);

  Parameters<T> *_param;

  // Below are the grids defining the augmented discrete representation of the
  // geometry. These will hold the geometry in the local frame, and are written
  // to whenever create() is called. Thus, they can never be assumed to hold the
  // geometry in reference frame (theta=0).

  // Contains labels for discrete geometry. Label codes:
  // 0=outside, 1=inside, 2=inside, near boundary entry, 3=inside, near boundary
  // exit
  GridGPU<char> *_domainLabels;

  // Field with boundary normals for discrete geometry (2 fields (x- and
  // y-component)). Only stored where nodes are labeled 2 or 3.
  GridGPU<T> *_normals;

  // Field with distances between the boundary nodes to the exact location of
  // the geometry, as measured along the positive gamma trajectory. Only stored
  // where nodes are labeled 2 or 3.
  GridGPU<T> *_boundaryDistance;
};

namespace internal {
// Label boundary interface nodes
// Codes:
// 0 - Outside domain
// 1 - Inside domain
// 2 - Trajectory entering domain
// 3 - Trajectory exiting domain

// 4 - Trajectory entering AND exiting domain at the same grid point (not
// implemented)

// Label type 2 and 3 and 4 nodes
template <typename T>
__global__ void labelEntryExit_Kernel(char *oldLabels_p, char *newLabels_p,
                                      const int Nx, const int Ny) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  if (tx < Nx && ty < Ny) {
    const int idx = ty * Nx + tx;

    char newLabel = 0;
    char oldLabel[3];

    if (ty > 0 && ty < Ny - 1) {
      // Read current and adjacent labels
      // Maybe use shared memory here for more efficient data loading and memory
      // coalescing
      for (int i = 0; i < 3; ++i)
        oldLabel[i] = oldLabels_p[(ty - 1 + i) * Nx + tx];

      // Label interface nodes
      if (oldLabel[1] == 1) {
        newLabel = 1;

        if (oldLabel[0] == 0) {
          newLabel = 2;
        } else if (oldLabel[2] == 0) {
          newLabel = 3;
        }

        if (oldLabel[0] == 0 && oldLabel[2] == 0)
          newLabel = 0;
      }
    }
    newLabels_p[idx] = newLabel;
  }
}
} // namespace internal

template <typename T>
GeometryComponent<T>::GeometryComponent()
    : _param(0), _domainLabels(0), _normals(0), _boundaryDistance(0) {
#ifndef NDEBUG
  std::cout << "GeometryComponent()\n";
#endif
}

template <typename T>
GeometryComponent<T>::GeometryComponent(const GeometryComponent<T> &other)
    : _param(other._param), _domainLabels(0), _normals(0),
      _boundaryDistance(0) {
  if (other._domainLabels)
    _domainLabels = new GridGPU<char>(*other._domainLabels);
  if (other._normals)
    _normals = new GridGPU<T>(*other._normals);
  if (other._boundaryDistance)
    _boundaryDistance = new GridGPU<T>(*other._boundaryDistance);
}

template <typename T> GeometryComponent<T>::~GeometryComponent() {
#ifndef NDEBUG
  std::cout << "~GeometryComponent()\n";
#endif
  freeMem();
}

template <typename T>
void swap(GeometryComponent<T> &A, GeometryComponent<T> &B) {
  std::swap(A._param, B._param);
  std::swap(A._domainLabels, B._domainLabels);
  std::swap(A._normals, B._normals);
  std::swap(A._boundaryDistance, B._boundaryDistance);
}

template <typename T> void GeometryComponent<T>::allocateMem() {
  int N;
  if (_param) {
    N = _param->getGridResolutionDeltaRotated();
  } else {
#ifndef NDEBUG
    std::cout << "GeometryComponent<T>::allocateMem(): Error: No Parameters<T> "
                 "object!\n";
#endif
    return;
  }

  GridGPU<T>::initializeGrid(_normals, N, N, 2, Type::real);
  GridGPU<T>::initializeGrid(_boundaryDistance, N, N, 1, Type::real);
  GridGPU<char>::initializeGrid(_domainLabels, N, N, 1, Type::real);
}

template <typename T> void GeometryComponent<T>::freeMem() {
  GridGPU<T>::deleteGrid(_normals);
  GridGPU<T>::deleteGrid(_boundaryDistance);
  GridGPU<char>::deleteGrid(_domainLabels);
}

template <typename T> void GeometryComponent<T>::labelBoundaryNodes() {
  int Nx, Ny;
  dim3 tpb, bpg;

  getGridDimensions(Nx, Ny);
  getKernelDimensions(bpg, tpb);

  // Label boundary interface nodes
  GridGPU<char> *oldLabels = _domainLabels;
  GridGPU<char> *newLabels(0);
  GridGPU<char>::initializeGrid(newLabels, oldLabels);

  internal::labelEntryExit_Kernel<T><<<bpg, tpb>>>(
      oldLabels->getDataPointer(), newLabels->getDataPointer(), Nx, Ny);

  _domainLabels = newLabels;
  delete oldLabels;
}

template <typename T>
void GeometryComponent<T>::getGridDimensions(int &Nx, int &Ny) const {
  if (_domainLabels) {
    Nx = _domainLabels->dimX();
    Ny = _domainLabels->dimY();
  } else {
#ifndef NDEBUG
    std::cout << "GeometryComponent<T>::getGridDimensions() Error: "
                 "_domainLabels not initialized.\n";
#endif
    Nx = 0;
    Ny = 0;
  }
}

template <typename T>
void GeometryComponent<T>::getKernelDimensions(dim3 &bpg, dim3 &tpb) const {
  int Nx, Ny;
  getGridDimensions(Nx, Ny);

  int tx = 64;
  int bx = Nx / tx;
  if (tx * bx != Nx)
    bx++;

  tpb = dim3(tx, 1);
  bpg = dim3(bx, Ny);
}

template <typename T> char *GeometryComponent<T>::getDomainLabels() const {
  return _domainLabels->getDataPointer();
}

template <typename T> T *GeometryComponent<T>::getNormals(const int dim) const {
  return _normals->getDataPointer(dim);
}

template <typename T> T *GeometryComponent<T>::getBoundaryDistance() const {
  return _boundaryDistance->getDataPointer();
}
} // namespace scg

#endif // GEOMETRY_COMPONENT_H_
