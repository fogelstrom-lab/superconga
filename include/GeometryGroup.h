// Container which stores and manages objects of type GeometryComponent.
// This class compounds the different components together, and produces the
// final discrete geometry output.
#ifndef GEOMETRY_GROUP_H_
#define GEOMETRY_GROUP_H_

#include "GeometryComponent.h"
#include "Grid.h"
#include "Parameters.h"
#include "Type.h"
#include "cplx.h"
#include <cuda.h>

namespace scg {
template <typename T> class GeometryGroup : public ContextModule<T> {
public:
  GeometryGroup();
  virtual ~GeometryGroup();

  GeometryGroup(const GeometryGroup<T> &other);
  GeometryGroup(Context<T> *ctx);

  GeometryGroup<T> &operator=(GeometryGroup<T> other);

public:
  // Allocates memory for member variables (grids)
  virtual int initialize();

  // Add/set geometry component to contribute to resulting compounded geometry
  int add(GeometryComponent<T> *newComponent, Type::type operation);
  void set(GeometryComponent<T> *newComponent, Type::type operation,
           int geomIdx);

  // Get a pointer to the specified individual geometry component
  GeometryComponent<T> *getGeometryComponent(const int comp) const;

  int getNumGeometryComponents() const { return _geomComponentList.size(); }

  // Create discrete representation for compounded geometry in rotational frame
  // given by angle argument. Stores result in _domainLabels, _normals, and
  // _boundaryDistance
  void create(const T angle);

  // Create discrete representation with resolution NxN, and return domain
  // labels only. Does not store data in any member variables.
  GridGPU<char> *createDomain(const int N) const;

  /// \brief Extrapolates values outwards from boundary, using the rotated
  /// geometry.
  ///
  /// This is used prior to rotating e.g. the order parameter lattice into a
  /// rotational frame in the momentum integral. The reason is that the rotation
  /// together with the linear interpolation will cause the boundary lattice
  /// sites to (wrongly) have fractional values, i.e. less than they should be.
  /// By extrapolating the boundary values, this artifact is eliminated.
  ///
  /// \param target The grid to be extended.
  void extendValuesAtBoundary(GridGPU<T> *target) const;

  /// \brief Extrapolates values outwards from boundary, using the non-rotated
  /// reference geometry.
  ///
  /// This is used prior to rotating e.g. the order parameter lattice into a
  /// rotational frame in the momentum integral. The reason is that the rotation
  /// together with the linear interpolation will cause the boundary lattice
  /// sites to (wrongly) have fractional values, i.e. less than they should be.
  /// By extrapolating the boundary values, this artifact is eliminated.
  ///
  /// \param target The grid to be extended.
  void extendValuesAtBoundaryRef(GridGPU<T> *target) const;

  /// \brief Set values outside of the domain to zero, using the non-rotated
  /// reference geometry.
  ///
  /// Because of the extrapolation in extendValuesAtBoundary(), the computed
  /// quantities will have some non-zero values slightly outside of the actual
  /// simulation domain. These do not belong to the solution, and can be removed
  /// with the zeroValuesOutsideDomainRef() function.
  ///
  /// \param target The grid to be zeroed.
  void zeroValuesOutsideDomainRef(GridGPU<T> *target) const;

  int getNumGrainLatticeSites() const;
  // Computes the physical area of the geometry.
  T area() const;

  // Computes the average of the specified field in the inpute grid, only using
  // the values inside the actual geometry definition.
  cplx<T> fieldAverage(GridGPU<T> *data, const int field = 0);

  // Getters
  void getGridDimensions(int &Nx, int &Ny) const;
  int getNumAngles() const;
  T getCurrentAngle() const { return _currentAngle; }

  // Getters (pointers to data in corresponding grids)
  char *getDomainLabels() const;
  T *getNormals(const int dim) const;
  T *getBoundaryDistance() const;

  // Getters for pointers to grids containing discrete geometry representation
  GridGPU<char> *getDomainLabelsGrid() const { return _domainLabels; }
  GridGPU<T> *getNormalsGrid() const { return _normals; }
  GridGPU<T> *getBoundaryDistanceGrid() const { return _boundaryDistance; }

  // Get domain labels grid for the geometry in reference frame.
  GridGPU<char> *getDomainGrid() const { return _domain_ref; }

  // Helper functions to find locations of geometry boundary crossings. Only
  // used by the BoundaryStorage class so far, when determining how much memory
  // is required to store all the boundary values for the coherence functions.
  void countBoundariesAlongTrajectory(int Nx, int Ny, char *dom_ptr,
                                      int *idx_in_ptr, int *idx_out_ptr,
                                      int *numBndIsect_ptr, int &sum_idx_in,
                                      int &sum_idx_out);
  void countBoundariesAlongNegativeTrajectory(int Nx, int Ny, char *dom_ptr,
                                              int *idx_in_ptr, int *idx_out_ptr,
                                              int *numBndIsect_ptr,
                                              int &sum_idx_in,
                                              int &sum_idx_out);

  void findBoundaryCrossingsAlongTrajectory(int Nx, int Ny, char *dom_p,
                                            int *&coord_ptr);
  void findBoundaryCrossingsAlongNegativeTrajectory(int Nx, int Ny, char *dom_p,
                                                    int *&coord_ptr);

private:
  template <typename S>
  friend void swap(GeometryGroup<S> &A, GeometryGroup<S> &B);

  void setGeomOpMatrix();

  void geomOpAdd(GeometryComponent<T> *newGeom);
  void geomOpRemove(GeometryComponent<T> *newGeom);
  void geomOp(GeometryComponent<T> *newGeom, GridGPU<char> *geomOpMatrix);

  void geomOpAdd(GridGPU<char> *dom, GridGPU<char> *dom_local) const;
  void geomOpRemove(GridGPU<char> *dom, GridGPU<char> *dom_local) const;

  int computeNumGrainLatticeSites(const GridGPU<char> &inDomainLabels);

  void extendValuesAtBoundary(const GridGPU<char> &inDomainLabels,
                              GridGPU<T> &inOutTarget) const;

  std::vector<GeometryComponent<T> *> _geomComponentList;

  std::vector<int> _geomOpList;
  GridGPU<char> *_geomOpAdd;
  GridGPU<char> *_geomOpRemove;

  T _currentAngle;

  // Below are the grids defining the augmented discrete representation of the
  // geometry. These will hold the geometry in the local frame, and are written
  // to whenever create() is called. Thus, they can never be assumed to hold the
  // geometry in reference frame (theta=0).

  // Specifically, the grids store domain node labels
  // (inside/outside/entry/exit), and normals at (exact) boundary location. The
  // _boundaryDistance grid stores the distance to the exact boundary location
  // (along the trajectory), for the corresponding grid point. So the normals
  // stored in grid at integer grid point (x,y) (generally not coinciding with
  // domain boundary), are actually located in real space at:
  //
  // ( x, y + abs(_boundaryDistance(x,y)) )  (when exiting)
  // ( x, y - abs(_boundaryDistance(x,y)) )  (when entering)
  //
  // which are the exact boundary domain locations.

  // Contains labels for discrete geometry. Label codes:
  // 0=outside, 1=inside, 2=inside, near boundary entry, 3=inside, near boundary
  // exit.
  const char m_maxValidLabel = 3;
  GridGPU<char> *_domainLabels;

  // Field with boundary normals for discrete geometry (2 fields (x- and
  // y-component)). Only stored where nodes are labeled 2 or 3.
  GridGPU<T> *_normals;

  // Field with distances between the boundary nodes to the exact location of
  // the geometry, as measured along the positive gamma trajectory. Only stored
  // where nodes are labeled 2 or 3.
  GridGPU<T> *_boundaryDistance;

  // Geometry domain labels grid in reference/unrotated frame (theta=0)
  GridGPU<char> *_domain_ref;

  // The number of lattice sites in the grain.
  int m_numGrainLatticeSites;
};

namespace internal {

template <typename T> __device__ void normalize(T &vec_x, T &vec_y) {
  const T norm = static_cast<T>(1.0) / sqrt(vec_x * vec_x + vec_y * vec_y);
  vec_x *= norm;
  vec_y *= norm;
}

// Adds *1 to existing *0
template <typename T>
__global__ void geomOp_kernel(char *dom0, T *norm0_x, T *norm0_y, T *dist0,
                              char *dom1, T *norm1_x, T *norm1_y, T *dist1,
                              char *geomOpMatrix, const int Nx, const int Ny) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  if (tx < Nx && ty < Ny) {
    const int idx = ty * Nx + tx;

    char dom0_val = dom0[idx];
    char dom1_val = dom1[idx];

    const int geomOpMatrixIdx = dom0_val * 4 + dom1_val;

    const T dist0_val = dist0[idx];
    const T dist1_val = dist1[idx];

    dom0_val = geomOpMatrix[geomOpMatrixIdx];

    T normal_x, normal_y;

    if (dom0_val == 2 || dom0_val == 3) {
      normal_x = norm0_x[idx] + norm1_x[idx];
      normal_y = norm0_y[idx] + norm1_y[idx];

      normalize(normal_x, normal_y);
    } else {
      normal_x = static_cast<T>(0.0);
      normal_y = static_cast<T>(0.0);
    }

    norm0_x[idx] = normal_x;
    norm0_y[idx] = normal_y;

    dom0[idx] = dom0_val;
    dist0[idx] = max(dist0_val, dist1_val);
  }
}

// Adds *1 to existing *0
template <typename T>
__global__ void geomOpAdd_kernel(char *dom0, char *dom1, const int Nx) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  if (tx < Nx && ty < Nx) {
    const int idx = ty * Nx + tx;

    if (dom1[idx]) {
      dom0[idx] = 1;
    }
  }
}

// Adds *1 to existing *0
template <typename T>
__global__ void
geomOpRemove_kernel(char *dom0, T *norm0_x, T *norm0_y, T *dist0, char *dom1,
                    T *norm1_x, T *norm1_y, T *dist1, char *domTmp,
                    char *geomOpMatrix, const int Nx, const int Ny) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  if (tx < Nx && ty > 0 && ty < Ny - 1) {
    const int idxm = (ty - 1) * Nx + tx;
    const int idx = ty * Nx + tx;
    const int idxp = (ty + 1) * Nx + tx;

    char dom0_val = dom0[idx];

    const char dom1m_val = dom1[idxm];
    const char dom1_val = dom1[idx];
    const char dom1p_val = dom1[idxp];

    T normal_x, normal_y;

    if (dom1_val > 0) {
      dom0_val = 0;
    } else if (dom1p_val == 2) {
      if (dom0_val == 1) {
        dom0_val = 3;
        normal_x = norm0_x[idx] - norm1_x[idxp];
        normal_y = norm0_y[idx] - norm1_y[idxp];
        normalize(normal_x, normal_y);
        norm0_x[idx] = normal_x;
        norm0_y[idx] = normal_y;
        dist0[idx] = static_cast<T>(1.0) - dist1[idxp];
      }
    } else if (dom1m_val == 3) {
      if (dom0_val == 1) {
        dom0_val = 2;
        normal_x = norm0_x[idx] - norm1_x[idxm];
        normal_y = norm0_y[idx] - norm1_y[idxm];
        normalize(normal_x, normal_y);
        norm0_x[idx] = normal_x;
        norm0_y[idx] = normal_y;
        dist0[idx] = static_cast<T>(1.0) - dist1[idxm];
      }
    }

    domTmp[idx] = dom0_val;
  }
}

template <typename T>
__global__ void geomOpRemove_kernel(char *dom0, char *dom1, const int Nx) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  if (tx < Nx && ty < Nx) {
    const int idx = ty * Nx + tx;

    const bool val0 = dom0[idx] > 0;
    const bool val1 = dom1[idx] > 0;

    dom0[idx] = (!val0 || val1) ? 0 : 1;
  }
}

// Slow operation, so we have two explicit kernels, one for complex and one for
// real valued.
template <typename T>
__global__ void extendValuesAtBoundary_kernel( // complex
    const char *const dom_ptr, T *grid_real, T *grid_imag, int Nx_geom,
    int Ny_geom, int Nx_target, int Ny_target, int Nx_geom_offset,
    int Ny_geom_offset, const int numExtend) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
  int tf = blockIdx.z;

  if (tx < Nx_target && ty < Ny_target) {
    int idx_geom = (ty + Ny_geom_offset) * Nx_geom + (tx + Nx_geom_offset);

    int i, j, idx_neighbor, idx_geom_neighbor;

    T interp_real = static_cast<T>(0.0);
    T interp_imag = static_cast<T>(0.0);

    int numNeighbors = 0;

    if (dom_ptr[idx_geom] == 0) {
      for (j = -numExtend; j <= numExtend; ++j) {
        for (i = -numExtend; i <= numExtend; ++i) {
          if ((tx + i) > 0 && (tx + i) < Nx_target && (ty + j) > 0 &&
              (ty + j) < Ny_target) {
            idx_geom_neighbor =
                (ty + j + Ny_geom_offset) * Nx_geom + (tx + i + Nx_geom_offset);

            if (dom_ptr[idx_geom_neighbor] > 0) {
              idx_neighbor = tf * 2 * Nx_target * Ny_target +
                             (ty + j) * Nx_target + (tx + i);

              interp_real += grid_real[idx_neighbor];
              interp_imag += grid_imag[idx_neighbor];

              numNeighbors++;
            }
          }
        }
      }
    }

    if (numNeighbors) {
      const int idx = tf * 2 * Nx_target * Ny_target + ty * Nx_target + tx;
      grid_real[idx] = interp_real / (T)numNeighbors;
      grid_imag[idx] = interp_imag / (T)numNeighbors;
    }
  }
}

template <typename T>
__global__ void extendValuesAtBoundary_kernel( // real
    const char *const dom_ptr, T *grid_real, int Nx_geom, int Ny_geom,
    int Nx_target, int Ny_target, int Nx_geom_offset, int Ny_geom_offset,
    const int numExtend) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
  int tf = blockIdx.z;

  if (tx < Nx_target && ty < Ny_target) {
    int idx_geom = (ty + Ny_geom_offset) * Nx_geom + (tx + Nx_geom_offset);

    int i, j, idx_neighbor, idx_geom_neighbor;

    T interp_real = static_cast<T>(0.0);

    int numNeighbors = 0;

    if (dom_ptr[idx_geom] == 0) {
      for (j = -numExtend; j <= numExtend; ++j) {
        for (i = -numExtend; i <= numExtend; ++i) {
          if ((tx + i) > 0 && (tx + i) < Nx_target && (ty + j) > 0 &&
              (ty + j) < Ny_target) {
            idx_geom_neighbor =
                (ty + j + Ny_geom_offset) * Nx_geom + (tx + i + Nx_geom_offset);

            if (dom_ptr[idx_geom_neighbor] > 0) {
              idx_neighbor =
                  tf * Nx_target * Ny_target + (ty + j) * Nx_target + (tx + i);

              interp_real += grid_real[idx_neighbor];

              numNeighbors++;
            }
          }
        }
      }
    }

    if (numNeighbors) {
      const int idx = tf * Nx_target * Ny_target + ty * Nx_target + tx;
      grid_real[idx] = interp_real / (T)numNeighbors;
    }
  }
}

template <typename T>
__global__ void zeroValuesOutsideDomain_kernel(
    char *dom_ptr, T *target_ptr, int Nx_geom, int Ny_geom, int Nx_target,
    int Ny_target, int Nx_geom_offset, int Ny_geom_offset, int isComplex) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;
  const int tf = blockIdx.z;

  if (tx < Nx_target && ty < Ny_target) {
    const int idx =
        tf * (Nx_target * Ny_target * (1 + isComplex)) + ty * Nx_target + tx;

    const int domainX = tx + Nx_geom_offset;
    const int domainY = ty + Ny_geom_offset;

    if ((0 <= domainX) && (domainX < Nx_geom) && (0 <= domainY) &&
        (domainY < Ny_geom)) {
      const int domainIdx = domainY * Nx_geom + domainX;
      const char domainLabel = dom_ptr[domainIdx];
      if (domainLabel == 0) {
        target_ptr[idx] = static_cast<T>(0.0);
      }
    } else {
      target_ptr[idx] = static_cast<T>(0.0);
    }
  }
}
} // namespace internal

template <typename T>
GeometryGroup<T>::GeometryGroup()
    : _domainLabels(0), _normals(0), _boundaryDistance(0), _domain_ref(0),
      m_numGrainLatticeSites(0) {
  setGeomOpMatrix();
}

template <typename T>
GeometryGroup<T>::GeometryGroup(const GeometryGroup<T> &other)
    : ContextModule<T>(other), _domainLabels(0), _normals(0),
      _boundaryDistance(0), _domain_ref(0) {
  if (other._domainLabels) {
    _domainLabels = new GridGPU<char>(*other._domainLabels);
  }
  if (other._normals) {
    _normals = new GridGPU<T>(*other._normals);
  }
  if (other._boundaryDistance) {
    _boundaryDistance = new GridGPU<T>(*other._boundaryDistance);
  }
  if (other._domain_ref) {
    _domain_ref = new GridGPU<char>(*other._domain_ref);
  }

  const int Nc = other.getNumGeometryComponents();

  if (Nc > 0) {
    for (int i = 0; i < Nc; ++i) {
      _geomComponentList.push_back(other._geomComponentList[i]->clone());
      _geomOpList.push_back(other._geomOpList[i]);
    }
  }

  setGeomOpMatrix();
}

template <typename T>
GeometryGroup<T>::GeometryGroup(Context<T> *ctx)
    : ContextModule<T>(ctx), _domainLabels(0), _normals(0),
      _boundaryDistance(0), _domain_ref(0), _geomOpList(0) {
  std::cout << "GeometryGroup<T>::GeometryGroup( Context<T>* )\n";

  ctx->set(this);

  // Set operation matrices
  setGeomOpMatrix();
}

template <typename T> GeometryGroup<T>::~GeometryGroup() {
  std::cout << "~GeometryGroup()\n";

  if (_domainLabels) {
    delete _domainLabels;
  }
  if (_normals) {
    delete _normals;
  }
  if (_boundaryDistance) {
    delete _boundaryDistance;
  }
  if (_domain_ref) {
    delete _domain_ref;
  }
  if (_geomOpAdd) {
    delete _geomOpAdd;
  }
  if (_geomOpRemove) {
    delete _geomOpRemove;
  }

  for (int i = 0; i < getNumGeometryComponents(); ++i) {
    delete _geomComponentList[i];
  }
}

template <typename T>
GeometryGroup<T> &GeometryGroup<T>::operator=(GeometryGroup<T> other) {
  swap(&this, other);
  return *this;
}

template <typename T> void swap(GeometryGroup<T> &A, GeometryGroup<T> &B) {
  std::swap(A._geomComponentList, B._geomComponentList);
  std::swap(A._geomOpList, B._geomOpList);
  std::swap(A._geomOpAdd, B._geomOpAdd);
  std::swap(A._geomOpRemove, B._geomOpRemove);
  std::swap(A._currentAngle, B._currentAngle);
  std::swap(A._domainLabels, B._domainLabels);
  std::swap(A._normals, B._normals);
  std::swap(A._boundaryDistance, B._boundaryDistance);
  std::swap(A._domain_ref, B._domain_ref);
}

template <typename T> int GeometryGroup<T>::initialize() {
  std::cout << "GeometryGroup<T>::initialize()\n";
  const int N =
      ContextModule<T>::getParameters()->getGridResolutionDeltaRotated();

  // Initialize grids with data for ONE angle
  GridGPU<T>::initializeGrid(_normals, N, N, 2, Type::real);
  GridGPU<T>::initializeGrid(_boundaryDistance, N, N, 1, Type::real);
  GridGPU<char>::initializeGrid(_domainLabels, N, N, 1, Type::real);

  if (_domain_ref) {
    delete _domain_ref;
  }

  // So that we have a geometry to work with straight away.
  const T angle = static_cast<T>(0);
  create(angle);

  // Create the "reference" domain labels. They are fixed and will never be
  // rotated.
  const int gridResolutionBase =
      ContextModule<T>::getParameters()->getGridResolutionBase();
  _domain_ref = createDomain(gridResolutionBase);

  // Compute the number of lattice sites.
  m_numGrainLatticeSites = computeNumGrainLatticeSites(*_domain_ref);

  return 0;
}

template <typename T> void GeometryGroup<T>::setGeomOpMatrix() {
  // TODO(niclas): WTF is this!?!
  char op_add[16] = {0, 1, 2, 3, 1, 1, 1, 1, 2, 1, 2, 1, 3, 1, 1, 3};
  char op_remove[16] = {0, 0, 0, 0, 1, 0, 3, 2, 2, 0, 0, 2, 3, 0, 3, 0};

  _geomOpAdd = new GridGPU<char>(16, 1, 1, Type::real);
  _geomOpRemove = new GridGPU<char>(16, 1, 1, Type::real);

  _geomOpAdd->setData(op_add, 16);
  _geomOpRemove->setData(op_remove, 16);
}

template <typename T> void GeometryGroup<T>::create(const T angle) {
  _currentAngle = angle;

  // Initialize compound geom grids
  _domainLabels->setZero();
  _normals->setZero();
  _boundaryDistance->setZero();

  GeometryComponent<T> *tmpGeom;

  // Perform operations for each element
  for (int i = 0; i < getNumGeometryComponents(); ++i) {
    tmpGeom = _geomComponentList[i];
    tmpGeom->create(angle);

    if (_geomOpList[i] == Type::add) {
      geomOpAdd(tmpGeom);
    }

    if (_geomOpList[i] == Type::remove) {
      geomOpRemove(tmpGeom);
    }
  }
}

template <typename T>
GridGPU<char> *GeometryGroup<T>::createDomain(const int N) const {
  // Initialize compound geom grids
  GridGPU<char> *dom = new GridGPU<char>(N, N, 1, Type::real);
  dom->setZero();

  // Perform operations for each element
  for (int i = 0; i < getNumGeometryComponents(); ++i) {
    GridGPU<char> *dom_local = _geomComponentList[i]->createDomain(N);

    if (_geomOpList[i] == Type::add) {
      geomOpAdd(dom, dom_local);
    }

    if (_geomOpList[i] == Type::remove) {
      geomOpRemove(dom, dom_local);
    }

    delete dom_local;
  }

  return dom;
}

template <typename T>
int GeometryGroup<T>::add(GeometryComponent<T> *newComponent,
                          Type::type operation) {
  // Add new Geometry component
  //
  newComponent->setParameters(ContextModule<T>::getParameters());

  _geomComponentList.push_back(newComponent);

  // Set geometry operation
  _geomOpList.push_back(int(operation));

  return getNumGeometryComponents();
}

template <typename T>
void GeometryGroup<T>::set(GeometryComponent<T> *newComponent,
                           Type::type operation, int geomIdx) {
  newComponent->setParameters(ContextModule<T>::getParameters());

  if (geomIdx >= 0 && geomIdx < getNumGeometryComponents()) {
    delete _geomComponentList[geomIdx];

    _geomComponentList[geomIdx] = newComponent;
    _geomOpList[geomIdx] = int(operation);
  } else {
    std::cout << "GeometryGroup<T>::setGeometryComponent(): Nothing added, "
                 "index out of range\n";
  }
}

template <typename T>
GeometryComponent<T> *
GeometryGroup<T>::getGeometryComponent(const int comp) const {
  int compLim = min(getNumGeometryComponents() - 1, comp);
  compLim = max(0, compLim);

  return _geomComponentList[compLim];
}

template <typename T>
void GeometryGroup<T>::geomOp(GeometryComponent<T> *newGeom,
                              GridGPU<char> *geomOpMatrix) {
  int Nx, Ny;
  getGridDimensions(Nx, Ny);

  dim3 bpg, tpb;
  _domainLabels->getKernelDimensions(bpg, tpb);

  internal::geomOp_kernel<T><<<bpg, tpb>>>(
      getDomainLabels(), getNormals(0), getNormals(1), getBoundaryDistance(),
      newGeom->getDomainLabels(), newGeom->getNormals(0),
      newGeom->getNormals(1), newGeom->getBoundaryDistance(),
      geomOpMatrix->getDataPointer(), Nx, Ny);
}

template <typename T>
void GeometryGroup<T>::geomOpAdd(GeometryComponent<T> *newGeom) {
  geomOp(newGeom, _geomOpAdd);
}

template <typename T>
void GeometryGroup<T>::geomOpRemove(GeometryComponent<T> *newGeom) {
  int Nx, Ny;
  getGridDimensions(Nx, Ny);

  dim3 bpg, tpb;
  _domainLabels->getKernelDimensions(bpg, tpb);

  GridGPU<char> *domResult = _domainLabels;

  GridGPU<char> *domTmp(0);
  GridGPU<char>::initializeGrid(domTmp, domResult);
  domTmp->setZero();

  internal::geomOpRemove_kernel<T><<<bpg, tpb>>>(
      getDomainLabels(), getNormals(0), getNormals(1), getBoundaryDistance(),
      newGeom->getDomainLabels(), newGeom->getNormals(0),
      newGeom->getNormals(1), newGeom->getBoundaryDistance(),
      domTmp->getDataPointer(), _geomOpRemove->getDataPointer(), Nx, Ny);

  _domainLabels = domTmp;
  delete domResult;
}

template <typename T>
void GeometryGroup<T>::geomOpAdd(GridGPU<char> *dom,
                                 GridGPU<char> *dom_local) const {
  int N = dom->dimX();

  dim3 bpg, tpb;
  GridGPU<T>::getKernelDimensions(bpg, tpb, N, N);

  internal::geomOpAdd_kernel<T>
      <<<bpg, tpb>>>(dom->getDataPointer(), dom_local->getDataPointer(), N);
}

template <typename T>
void GeometryGroup<T>::geomOpRemove(GridGPU<char> *dom,
                                    GridGPU<char> *dom_local) const {
  int N = dom->dimX();

  dim3 bpg, tpb;
  GridGPU<T>::getKernelDimensions(bpg, tpb, N, N);

  internal::geomOpRemove_kernel<T>
      <<<bpg, tpb>>>(dom->getDataPointer(), dom_local->getDataPointer(), N);
}

template <typename T>
int GeometryGroup<T>::computeNumGrainLatticeSites(
    const GridGPU<char> &inDomainLabels) {
  // Get size.
  const int numX = inDomainLabels.dimX();
  const int numY = inDomainLabels.dimY();

  // Copy the domain to the CPU.
  const thrust::host_vector<char> domainCPU = inDomainLabels.getVector();

  // Get the Pointer.
  const char *const domainPtr = thrust::raw_pointer_cast(domainCPU.data());

  // Count the number of lattice sites in the grain.
  int numGrainLatticeSites = 0;
  for (int y = 0; y < numY; ++y) {
    for (int x = 0; x < numX; ++x) {
      const int idx = y * numX + x;
      const char node = domainPtr[idx];
      if (node < 0 || node > m_maxValidLabel) {
        // TODO(niclas): Raise exception instead?
        std::cerr << "WARNING: Domain label, " << static_cast<int>(node)
                  << ", not understood!" << std::endl;
      }

      if (node > 0) {
        numGrainLatticeSites++;
      }
    }
  }

  return numGrainLatticeSites;
}

template <typename T> int GeometryGroup<T>::getNumGrainLatticeSites() const {
  return m_numGrainLatticeSites;
}

template <typename T> T GeometryGroup<T>::area() const {
  const T gridElementSize =
      ContextModule<T>::getParameters()->getGridElementSize();
  const T areaPerLatticeSite = gridElementSize * gridElementSize;
  return areaPerLatticeSite * static_cast<T>(m_numGrainLatticeSites);
}

// Compute field aveage, only accounting for nodes inside geometry
template <typename T>
cplx<T> GeometryGroup<T>::fieldAverage(GridGPU<T> *data, const int field) {
  const T angle = static_cast<T>(0);
  create(angle);

  VectorCPU<T> vec_cpu = _domainLabels->getVector();
  GridCPU<T> data_cpu = *data;

  const int complex = data->isComplex();

  // Pointers
  T *data_re = data_cpu->getDataPointer(field, Type::real);
  T *data_im = data_cpu->getDataPointer(field, Type::imag);

  char *dom = thrust::raw_pointer_cast(vec_cpu.data());

  // Dimensions
  int Nx_data = data->dimX();
  int Ny_data = data->dimY();

  int Nx_dom = _domainLabels->dimX();
  int Ny_dom = _domainLabels->dimY();

  int domOffsetX = (Nx_dom - Nx_data) / 2;
  int domOffsetY = (Ny_dom - Ny_data) / 2;

  cplx<T> sum(0.0, 0.0);

  int idx_data, idx_dom, num_discrete = 0;

  for (int j = 0; j < Ny_data; ++j) {
    for (int i = 0; i < Nx_data; ++i) {
      idx_data = j * Nx_data + i;
      idx_dom = (j + domOffsetY) * Nx_dom + (i + domOffsetX);

      if (dom[idx_dom] == 1) {
        sum.x += data_re[idx_data];
        if (complex) {
          sum.y += data_im[idx_data];
        }
        num_discrete++;
      }
    }
  }

  return sum / (T)num_discrete;
}

template <typename T>
void GeometryGroup<T>::countBoundariesAlongTrajectory(
    int Nx, int Ny, char *dom_ptr, int *idx_in_ptr, int *idx_out_ptr,
    int *numBndIsect_ptr, int &sum_idx_in, int &sum_idx_out) {
  int i, j, idx;

  int *num_in = new int[Nx];
  int *num_out = new int[Nx];

  // Initialize entry/exit counters for current angle
  for (i = 0; i < Nx; ++i) {
    num_in[i] = 0;
    num_out[i] = 0;
  }

  // Trace along trajectories to find entry/exit points
  for (i = 0; i < Nx; ++i) {
    for (j = 0; j < Ny; ++j) {
      idx = j * Nx + i;

      if (dom_ptr[idx] == 2) {
        num_in[i] += 1;
      }

      if (dom_ptr[idx] == 3) {
        num_out[i] += 1;
      }
    }

    // Number of times this trajectory terminates at a boundary (for reflection)
    numBndIsect_ptr[i] = num_out[i];
  }

  // Calculate indices to sparse storage
  for (i = 0; i < Nx; ++i) {
    if (num_in[i] == 0) {
      idx_in_ptr[i] = -1;
    } else {
      idx_in_ptr[i] = sum_idx_in;
    }

    sum_idx_in += num_in[i];

    if (num_out[i] == 0) {
      idx_out_ptr[i] = -1;
    } else {
      idx_out_ptr[i] = sum_idx_out;
    }

    sum_idx_out += num_out[i];
  }

  delete[] num_in;
  delete[] num_out;
}

template <typename T>
void GeometryGroup<T>::countBoundariesAlongNegativeTrajectory(
    int Nx, int Ny, char *dom_ptr, int *idx_in_ptr, int *idx_out_ptr,
    int *numBndIsect_ptr, int &sum_idx_in, int &sum_idx_out) {
  int i, j, idx;

  int *num_in = new int[Nx];
  int *num_out = new int[Nx];

  // Initialize entry/exit counters for current angle
  for (i = 0; i < Nx; ++i) {
    num_in[i] = 0;
    num_out[i] = 0;
  }

  // Trace along trajectories to find entry/exit points
  for (i = 0; i < Nx; ++i) {
    for (j = Ny - 1; j >= 0; --j) {
      idx = j * Nx + i;

      if (dom_ptr[idx] == 3) {
        num_in[i] += 1;
      }

      if (dom_ptr[idx] == 2) {
        num_out[i] += 1;
      }
    }

    numBndIsect_ptr[i] = num_out[i];
  }

  // Calculate indices to sparse storage
  for (i = 0; i < Nx; ++i) {
    if (num_in[i] == 0) {
      idx_in_ptr[i] = -1;
    } else {
      idx_in_ptr[i] = sum_idx_in;
    }

    sum_idx_in += num_in[i];

    if (num_out[i] == 0) {
      idx_out_ptr[i] = -1;
    } else {
      idx_out_ptr[i] = sum_idx_out;
    }

    sum_idx_out += num_out[i];
  }

  delete[] num_in;
  delete[] num_out;
}

template <typename T>
void GeometryGroup<T>::findBoundaryCrossingsAlongTrajectory(int Nx, int Ny,
                                                            char *dom_p,
                                                            int *&coord_ptr) {
  int i, j, idx;

  for (i = 0; i < Nx; ++i) {
    for (j = 0; j < Ny; ++j) {
      idx = j * Nx + i;

      if (dom_p[idx] == 3) {
        *coord_ptr = j;
        coord_ptr++;
      }
    }
  }
}

template <typename T>
void GeometryGroup<T>::findBoundaryCrossingsAlongNegativeTrajectory(
    int Nx, int Ny, char *dom_p, int *&coord_ptr) {
  int i, j, idx;

  for (i = 0; i < Nx; ++i) {
    for (j = Ny - 1; j >= 0; --j) {
      idx = j * Nx + i;

      if (dom_p[idx] == 2) {
        *coord_ptr = j;
        coord_ptr++;
      }
    }
  }
}

template <typename T>
void GeometryGroup<T>::extendValuesAtBoundary(
    const GridGPU<char> &inDomainLabels, GridGPU<T> &inOutTarget) const {
  const int Nx_geom = inDomainLabels.dimX();
  const int Ny_geom = inDomainLabels.dimY();

  const int Nx_target = inOutTarget.dimX();
  const int Ny_target = inOutTarget.dimY();

  const int Nx_geom_offset = (Nx_geom - Nx_target) / 2;
  const int Ny_geom_offset = (Ny_geom - Ny_target) / 2;

  const int Nf = inOutTarget.numFields();

  // Set kernel dimensions
  dim3 bpg, tpb;
  inOutTarget.getKernelDimensionsFields(bpg, tpb);

  // This is a sensible value. It probably does not need to be exposed to the
  // user. Note that 2 is chosen due to corner artifacts otherwise.
  // TODO(niclas): Have a separate domain label for corners, and handle them
  // separately.
  const int numExtend = 2;

  if (inOutTarget.isComplex()) {
    internal::extendValuesAtBoundary_kernel<T><<<bpg, tpb>>>(
        inDomainLabels.getDataPointer(),
        inOutTarget.getDataPointer(0, Type::real),
        inOutTarget.getDataPointer(0, Type::imag), Nx_geom, Ny_geom, Nx_target,
        Ny_target, Nx_geom_offset, Ny_geom_offset, numExtend);
  } else {
    internal::extendValuesAtBoundary_kernel<T><<<bpg, tpb>>>(
        inDomainLabels.getDataPointer(),
        inOutTarget.getDataPointer(0, Type::real), Nx_geom, Ny_geom, Nx_target,
        Ny_target, Nx_geom_offset, Ny_geom_offset, numExtend);
  }
}

template <typename T>
void GeometryGroup<T>::extendValuesAtBoundary(GridGPU<T> *grid) const {
  extendValuesAtBoundary(*_domainLabels, *grid);
}

template <typename T>
void GeometryGroup<T>::extendValuesAtBoundaryRef(GridGPU<T> *grid) const {
  extendValuesAtBoundary(*_domain_ref, *grid);
}

template <typename T>
void GeometryGroup<T>::zeroValuesOutsideDomainRef(GridGPU<T> *target) const {
  const int Nx_dom = _domain_ref->dimX();
  const int Ny_dom = _domain_ref->dimY();

  const int Nx_img = target->dimX();
  const int Ny_img = target->dimY();

  const int Nx_diff = (Nx_dom - Nx_img) / 2;
  const int Ny_diff = (Ny_dom - Ny_img) / 2;

  dim3 blocksPerGrid, threadsPerBlock;
  target->getKernelDimensionsFields(blocksPerGrid, threadsPerBlock);

  internal::zeroValuesOutsideDomain_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      _domain_ref->getDataPointer(0, Type::real),
      target->getDataPointer(0, Type::real), Nx_dom, Ny_dom, Nx_img, Ny_img,
      Nx_diff, Ny_diff, int(target->isComplex()));

  if (target->isComplex()) {
    internal::
        zeroValuesOutsideDomain_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            _domain_ref->getDataPointer(0, Type::real),
            target->getDataPointer(0, Type::imag), Nx_dom, Ny_dom, Nx_img,
            Ny_img, Nx_diff, Ny_diff, int(target->isComplex()));
  }
}

template <typename T>
void GeometryGroup<T>::getGridDimensions(int &Nx, int &Ny) const {
  Nx = _domainLabels->dimX();
  Ny = _domainLabels->dimY();
}

template <typename T> int GeometryGroup<T>::getNumAngles() const {
  return ContextModule<T>::getParameters()->getAngularResolution();
}

template <typename T> char *GeometryGroup<T>::getDomainLabels() const {
  return _domainLabels->getDataPointer();
}

template <typename T> T *GeometryGroup<T>::getNormals(const int dim) const {
  return _normals->getDataPointer(dim);
}

template <typename T> T *GeometryGroup<T>::getBoundaryDistance() const {
  return _boundaryDistance->getDataPointer();
}
} // namespace scg

#endif // GEOMETRY_GROUP_H_
