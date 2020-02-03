#ifndef BOUNDARY_STORAGE_H_
#define BOUNDARY_STORAGE_H_

template <typename T> class GeometryGroup;
template <typename T> class BoundaryStorage;
template <typename T> class IntegrationIterator;
template <typename T> class ComputeProperty;

#include "ComputeProperty.h"
#include "FermiSurface.h"
#include "GeometryGroup.h"
#include "Grid.h"
#include "IntegrationIterator.h"
#include "Type.h"
#include "utils.h"
#include <iostream>

template <typename T> class BoundaryStorage {
  typedef typename Grid<T>::GPU GridGPU;
  typedef typename Grid<T>::CPU GridCPU;

public:
  BoundaryStorage();
  virtual ~BoundaryStorage();

  BoundaryStorage(GeometryGroup<T> *geom, FermiSurface<T> *fermiSurface);

  void initialize(const int numTrajectoriesPerAngle, const int numEnergies,
                  const int numAngles);
  void setBoundaryZero();

  __device__ static int getSparseIndex(const int angleIdx, const int energyIdx,
                                       const int xIdx, const int boundaryCount,
                                       const int numEnergies, const int numX,
                                       const int *const sparseIndices);

  // Storage getters
  Grid<int>::GPU *getIndexSparseIn_gamma() const {
    return _index_sparse_in_gamma;
  }
  Grid<int>::GPU *getIndexSparseOut_gamma() const {
    return _index_sparse_out_gamma;
  }
  Grid<int>::GPU *getIndexSparseIn_gammabar() const {
    return _index_sparse_in_gammabar;
  }
  Grid<int>::GPU *getIndexSparseOut_gammabar() const {
    return _index_sparse_out_gammabar;
  }

  Grid<int>::GPU *getBoundaryIntersectCoords_gamma() const {
    return _coord_bnd_intersect_gamma;
  }
  Grid<int>::GPU *getBoundaryIntersectCoords_gammabar() const {
    return _coord_bnd_intersect_gammabar;
  }

  Grid<int>::GPU *getNumBoundaryIntersect_gamma() const {
    return _num_bnd_intersect_gamma;
  }
  Grid<int>::GPU *getNumBoundaryIntersect_gammabar() const {
    return _num_bnd_intersect_gammabar;
  }

  GridGPU *getBoundaryIn_gamma() const { return _boundary_in_gamma; }
  GridGPU *getBoundaryOut_gamma() const { return _boundary_out_gamma; }
  GridGPU *getBoundaryIn_gammabar() const { return _boundary_in_gammabar; }
  GridGPU *getBoundaryOut_gammabar() const { return _boundary_out_gammabar; }

  // Helper array getters (see description below)
  Grid<int>::CPU *getTrajectoryRangeX() const { return _trajectoryRange_x; }
  Grid<int>::CPU *getTrajectoryRangeY() const { return _trajectoryRange_y; }

  int getNumIntersectingTrajectoriesAngle(const int angleIdx) const;
  int getTrajectoryStartIndexAngle(const int angleIdx) const;

private:
  void allocMemSparseStorage(const int numElemGammaIn,
                             const int numElemGammaOut,
                             const int numElemGammaBarIn,
                             const int numElemGammaBarOut,
                             const int numEnergies);

  void computeSparseIndices(const int numEnergies, const int numAngles);
  void computeIntersectingTrajectories(const int numAngles);

  GeometryGroup<T> *_geom;
  FermiSurface<T> *_fermiSurface;

  int _numSparseIn_gamma;
  int _numSparseOut_gamma;
  int _numSparseIn_gammabar;
  int _numSparseOut_gammabar;

  int _numEnergies;
  int _numAngles;

  // These arrays are for finding indices into sparse storage for boundary
  // gammas. Arrays are of dimension (Nx,numAngles) (trajectory along y). So for
  // each column (trajectory), for each angle, an index offset is stored. This
  // index offset corresponds to the address to get the first in/out boundary
  // gammas for a particular trajectory. If there are multiple entry/exit points
  // on one trajectory, this has to be kept track of.
  //

  // MAYBE MORE HIGH LEVEL FUNCTIONS FOR API USE.

  // Indices into sparse storage arrays
  //
  Grid<int>::GPU *_index_sparse_in_gamma;
  Grid<int>::GPU *_index_sparse_out_gamma;

  Grid<int>::GPU *_index_sparse_in_gammabar;
  Grid<int>::GPU *_index_sparse_out_gammabar;

  // Arrays with information about on how to find the trajectories which
  // actually intersects the computational domain (geometry). This information
  // is used when constructing threadblocks to avoid unneccesary thread
  // launching.

  // Start and end x- and y-indices for trajectory integration
  Grid<int>::CPU *_trajectoryRange_x;
  Grid<int>::CPU *_trajectoryRange_y;

  // Per trajectory start and end y-indices for trajectory integration
  Grid<int>::GPU *_trajectoryRangeAllStart_y;
  Grid<int>::GPU *_trajectoryRangeAllEnd_y;

  // Local (to frame of current angle) y-coordinates where trajectory intersects
  // boundary. Used for geometries where one trajectory can have multiple
  // boundary intersections.
  Grid<int>::GPU *_coord_bnd_intersect_gamma;
  Grid<int>::GPU *_coord_bnd_intersect_gammabar;

  Grid<int>::GPU *_num_bnd_intersect_gamma;
  Grid<int>::GPU *_num_bnd_intersect_gammabar;

  // Storage for domain boundary gammas. Use _index_sparse_* to get values.
  GridGPU *_boundary_in_gamma;
  GridGPU *_boundary_out_gamma;

  GridGPU *_boundary_in_gammabar;
  GridGPU *_boundary_out_gammabar;
};

// CONSIDER IMPLEMENTING WITH MAJORITY: angle, energy, trajectory
// (for performance reasons (better memory coalescing, but harder implementation
// in finding sparse index))

template <typename T>
__device__ int
BoundaryStorage<T>::getSparseIndex(const int angleIdx, const int energyIdx,
                                   const int xIdx, const int boundaryCount,
                                   const int numEnergies, const int numX,
                                   const int *const sparseIndices) {
  // Find sparse index. Majority: angle, trajectory, energy
  const int sparseIdx =
      (sparseIndices[angleIdx * numX + xIdx] + boundaryCount) * numEnergies +
      energyIdx;

  return sparseIdx;
}

template <typename T>
BoundaryStorage<T>::BoundaryStorage()
    : _index_sparse_in_gamma(0), _index_sparse_out_gamma(0),
      _index_sparse_in_gammabar(0), _index_sparse_out_gammabar(0),
      _coord_bnd_intersect_gamma(0), _coord_bnd_intersect_gammabar(0),
      _num_bnd_intersect_gamma(0), _num_bnd_intersect_gammabar(0),
      _boundary_in_gamma(0), _boundary_out_gamma(0), _boundary_in_gammabar(0),
      _boundary_out_gammabar(0), _trajectoryRange_x(0), _trajectoryRange_y(0),
      _trajectoryRangeAllStart_y(0), _trajectoryRangeAllEnd_y(0),
      _numEnergies(0) {}

template <typename T> BoundaryStorage<T>::~BoundaryStorage() {
  std::cout << "BoundaryStorage::~BoundaryStorage()\n";

  if (_index_sparse_in_gamma)
    delete _index_sparse_in_gamma;
  if (_index_sparse_out_gamma)
    delete _index_sparse_out_gamma;
  if (_index_sparse_in_gammabar)
    delete _index_sparse_in_gammabar;
  if (_index_sparse_out_gammabar)
    delete _index_sparse_out_gammabar;

  if (_coord_bnd_intersect_gamma)
    delete _coord_bnd_intersect_gamma;
  if (_coord_bnd_intersect_gammabar)
    delete _coord_bnd_intersect_gammabar;
  if (_num_bnd_intersect_gamma)
    delete _num_bnd_intersect_gamma;
  if (_num_bnd_intersect_gammabar)
    delete _num_bnd_intersect_gammabar;

  if (_boundary_in_gamma)
    delete _boundary_in_gamma;
  if (_boundary_out_gamma)
    delete _boundary_out_gamma;
  if (_boundary_in_gammabar)
    delete _boundary_in_gammabar;
  if (_boundary_out_gammabar)
    delete _boundary_out_gammabar;

  if (_trajectoryRange_x)
    delete _trajectoryRange_x;
  if (_trajectoryRange_y)
    delete _trajectoryRange_y;
  if (_trajectoryRangeAllStart_y)
    delete _trajectoryRangeAllStart_y;
  if (_trajectoryRangeAllEnd_y)
    delete _trajectoryRangeAllEnd_y;
}

template <typename T>
BoundaryStorage<T>::BoundaryStorage(GeometryGroup<T> *geom,
                                    FermiSurface<T> *fermiSurface)
    : _index_sparse_in_gamma(0), _index_sparse_out_gamma(0),
      _index_sparse_in_gammabar(0), _index_sparse_out_gammabar(0),
      _coord_bnd_intersect_gamma(0), _coord_bnd_intersect_gammabar(0),
      _num_bnd_intersect_gamma(0), _num_bnd_intersect_gammabar(0),
      _boundary_in_gamma(0), _boundary_out_gamma(0), _boundary_in_gammabar(0),
      _boundary_out_gammabar(0), _trajectoryRange_x(0), _trajectoryRange_y(0),
      _trajectoryRangeAllStart_y(0), _trajectoryRangeAllEnd_y(0),
      _numEnergies(0) {
  std::cout << "BoundaryStorage<T>::BoundaryStorage(GeometryGroup<T> * geom, "
               "FermiSurface<T> *fermiSurface)\n";

  _geom = geom;
  _fermiSurface = fermiSurface;
}

template <typename T>
void BoundaryStorage<T>::initialize(const int numTrajectoriesPerAngle,
                                    const int numEnergies,
                                    const int numAngles) {
  std::cout << "BoundaryStorage<T>::initialize(): numTrajectoriesPerAngle = "
            << numTrajectoriesPerAngle << ", numEnergies = " << numEnergies
            << ", numAngles = " << numAngles << "\n";

  if (_index_sparse_in_gamma &&
      _index_sparse_in_gamma->dimX() == numTrajectoriesPerAngle &&
      _index_sparse_in_gamma->numFields() == numAngles &&
      _numEnergies == numEnergies) {
    std::cout << "Existing sparse storage identical to input parameters, not "
                 "initializing\n";
  } else {
    std::cout << "BoundaryStorage<T>::initialize(): Parameters changed, "
                 "initializing new grids.\n";

    _numEnergies = numEnergies;
    _numAngles = numAngles;

    Grid<int>::GPU::initializeGrid(_index_sparse_in_gamma,
                                   numTrajectoriesPerAngle, 1, numAngles,
                                   Type::real);
    Grid<int>::GPU::initializeGrid(_index_sparse_out_gamma,
                                   numTrajectoriesPerAngle, 1, numAngles,
                                   Type::real);
    Grid<int>::GPU::initializeGrid(_index_sparse_in_gammabar,
                                   numTrajectoriesPerAngle, 1, numAngles,
                                   Type::real);
    Grid<int>::GPU::initializeGrid(_index_sparse_out_gammabar,
                                   numTrajectoriesPerAngle, 1, numAngles,
                                   Type::real);
    Grid<int>::GPU::initializeGrid(_num_bnd_intersect_gamma,
                                   numTrajectoriesPerAngle, 1, numAngles,
                                   Type::real);
    Grid<int>::GPU::initializeGrid(_num_bnd_intersect_gammabar,
                                   numTrajectoriesPerAngle, 1, numAngles,
                                   Type::real);

    // Compute indices to access sparse storage
    computeSparseIndices(numEnergies, numAngles);

    // Find intersecting trajectories
    computeIntersectingTrajectories(numAngles);
  }
}

template <typename T>
void BoundaryStorage<T>::computeIntersectingTrajectories(const int numAngles) {
  std::cout << "BoundaryStorage<T>::computeIntersectingTrajectories()\n";

  int numPossibleTrajectories = _index_sparse_in_gamma->dimX();

  Grid<int>::CPU::initializeGrid(_trajectoryRange_x, numAngles, 1, 2,
                                 Type::real);
  Grid<int>::CPU::initializeGrid(_trajectoryRange_y, numAngles, 1, 2,
                                 Type::real);
  Grid<int>::GPU::initializeGrid(_trajectoryRangeAllStart_y,
                                 numPossibleTrajectories, 1, numAngles,
                                 Type::real);
  Grid<int>::GPU::initializeGrid(_trajectoryRangeAllEnd_y,
                                 numPossibleTrajectories, 1, numAngles,
                                 Type::real);

  Grid<int>::CPU trajectoryRangeAllStart_y(numPossibleTrajectories, 1,
                                           numAngles, Type::real);
  Grid<int>::CPU trajectoryRangeAllEnd_y(numPossibleTrajectories, 1, numAngles,
                                         Type::real);

  trajectoryRangeAllStart_y.setValue(-1);
  trajectoryRangeAllEnd_y.setValue(-1);

  int *idx_x_start = _trajectoryRange_x->getDataPointer(0);
  int *idx_x_end = _trajectoryRange_x->getDataPointer(1);

  int *idx_y_start = _trajectoryRange_y->getDataPointer(0);
  int *idx_y_end = _trajectoryRange_y->getDataPointer(1);

  Grid<int>::CPU index_sparse_in_gamma = *_index_sparse_in_gamma;

  for (int angleIdx = 0; angleIdx < numAngles; ++angleIdx) {
    idx_x_start[angleIdx] = 0;

    int *sparse_idx = index_sparse_in_gamma.getDataPointer(angleIdx);

    int *y_start = trajectoryRangeAllStart_y.getDataPointer(angleIdx);
    int *y_end = trajectoryRangeAllEnd_y.getDataPointer(angleIdx);

    // Look left->right for starting x-index
    for (int i = 0; i < numPossibleTrajectories; ++i) {
      if (sparse_idx[i] > -1) {
        // idxStart = i;
        idx_x_start[angleIdx] = i;
        break;
      }
    }

    // Look right->left for ending x-index
    idx_x_end[angleIdx] = idx_x_start[angleIdx];
    for (int i = numPossibleTrajectories - 1; i >= idx_x_start[angleIdx]; --i) {
      if (sparse_idx[i] > -1) {
        // idxEnd = i;
        idx_x_end[angleIdx] = i;
        break;
      }
    }

    // Find start and end y-indices
    const T systemAngle = _fermiSurface->getSystemAngle(angleIdx);
    _geom->create(-systemAngle);

    Grid<char>::CPU dom = *(_geom->getDomainLabelsGrid());
    char *dom_p = dom.getDataPointer();

    int Nx = dom.dimX();
    int Ny = dom.dimY();

    idx_y_start[angleIdx] = Ny - 1;
    idx_y_end[angleIdx] = 0;

    int idx;

    for (int i = idx_x_start[angleIdx]; i <= idx_x_end[angleIdx]; ++i) {
      // Find y_start by stepping forward from 0
      for (int j = 0; j < Ny; ++j) {
        idx = j * Nx + i;
        if (dom_p[idx] > 0) {
          y_start[i] = j;
          if (j < idx_y_start[angleIdx]) {
            idx_y_start[angleIdx] = j;
            break;
          }
        }
      }

      // Find y_end by stepping backward from Ny-1
      for (int j = Ny - 1; j >= 0; --j) {
        idx = j * Nx + i;
        if (dom_p[idx] > 0) {
          y_end[i] = j;
          if (j > idx_y_end[angleIdx]) {
            idx_y_end[angleIdx] = j;
            break;
          }
        }
      }
    }
  }

  *_trajectoryRangeAllStart_y = trajectoryRangeAllStart_y;
  *_trajectoryRangeAllEnd_y = trajectoryRangeAllEnd_y;
}

template <typename T>
int BoundaryStorage<T>::getNumIntersectingTrajectoriesAngle(
    const int angleIdx) const {
  int xMin = (_trajectoryRange_x->getDataPointer(0))[angleIdx];
  int xMax = (_trajectoryRange_x->getDataPointer(1))[angleIdx];
  return xMax - xMin + 1;
}

template <typename T>
int BoundaryStorage<T>::getTrajectoryStartIndexAngle(const int angleIdx) const {
  return (_trajectoryRange_x->getDataPointer(0))[angleIdx];
}

template <typename T>
void BoundaryStorage<T>::computeSparseIndices(const int numEnergies,
                                              const int numAngles) {
  int sum_idx_in_gamma = 0;
  int sum_idx_out_gamma = 0;
  int sum_idx_in_gammabar = 0;
  int sum_idx_out_gammabar = 0;

  int *idx_in_p, *idx_out_p, *numBndIsect_p;
  char *dom_p;

  int Nx, Ny;
  _geom->getGridDimensions(Nx, Ny);

  int X = _index_sparse_in_gamma->dimX();
  int Y = _index_sparse_in_gamma->dimY();
  int F = _index_sparse_in_gamma->numFields();

  Grid<int>::CPU index_sparse_in_gamma(X, Y, F, Type::real);
  Grid<int>::CPU index_sparse_out_gamma(X, Y, F, Type::real);
  Grid<int>::CPU index_sparse_in_gammabar(X, Y, F, Type::real);
  Grid<int>::CPU index_sparse_out_gammabar(X, Y, F, Type::real);
  Grid<int>::CPU num_bnd_intersect_gamma(X, Y, F, Type::real);
  Grid<int>::CPU num_bnd_intersect_gammabar(X, Y, F, Type::real);

  Grid<char>::CPU dom_cpu;

  std::cout << "BoundaryStorage<T>::computeSparseIndices(): Nx,Ny = " << Nx
            << " " << Ny << "\n";

  for (int angleIdx = 0; angleIdx < numAngles; ++angleIdx) {
    // Current angle in radians
    const T systemAngle = _fermiSurface->getSystemAngle(angleIdx);

    // Create geometry data for current angle
    _geom->create(-systemAngle);

    // Copy data to CPU
    dom_cpu = *(_geom->getDomainLabelsGrid());
    dom_p = dom_cpu.getDataPointer();

    // Get pointers, and find boundaries for gamma trajectory (positive y)
    idx_in_p = index_sparse_in_gamma.getDataPointer(angleIdx);
    idx_out_p = index_sparse_out_gamma.getDataPointer(angleIdx);
    numBndIsect_p = num_bnd_intersect_gamma.getDataPointer(angleIdx);

    _geom->countBoundariesAlongTrajectory(Nx, Ny, dom_p, idx_in_p, idx_out_p,
                                          numBndIsect_p, sum_idx_in_gamma,
                                          sum_idx_out_gamma);

    // Get pointers, and find boundaries for gammabar trajectory (negative y)
    idx_in_p = index_sparse_in_gammabar.getDataPointer(angleIdx);
    idx_out_p = index_sparse_out_gammabar.getDataPointer(angleIdx);
    numBndIsect_p = num_bnd_intersect_gammabar.getDataPointer(angleIdx);

    _geom->countBoundariesAlongNegativeTrajectory(
        Nx, Ny, dom_p, idx_in_p, idx_out_p, numBndIsect_p, sum_idx_in_gammabar,
        sum_idx_out_gammabar);
  }

  // Copy data from CPU to GPU
  *_index_sparse_in_gamma = index_sparse_in_gamma;
  *_index_sparse_out_gamma = index_sparse_out_gamma;
  *_index_sparse_in_gammabar = index_sparse_in_gammabar;
  *_index_sparse_out_gammabar = index_sparse_out_gammabar;

  *_num_bnd_intersect_gamma = num_bnd_intersect_gamma;
  *_num_bnd_intersect_gammabar = num_bnd_intersect_gammabar;

  // Initialize and allocate memory for sparse storage
  printf("%d, %d, %d, %d, %d\n\n\n\n", sum_idx_in_gamma, sum_idx_out_gamma,
         sum_idx_in_gammabar, sum_idx_out_gammabar, numEnergies);

  allocMemSparseStorage(sum_idx_in_gamma, sum_idx_out_gamma,
                        sum_idx_in_gammabar, sum_idx_out_gammabar, numEnergies);

  // Compute boundary intersection y-coordinate arrays
  X = _coord_bnd_intersect_gamma->dimX();
  Y = _coord_bnd_intersect_gamma->dimY();
  F = _coord_bnd_intersect_gamma->numFields();

  Grid<int>::CPU coord_bnd_intersect_gamma(X, Y, F, Type::real);
  Grid<int>::CPU coord_bnd_intersect_gammabar(X, Y, F, Type::real);

  int *coord_gamma_ptr = coord_bnd_intersect_gamma.getDataPointer();
  int *coord_gammabar_ptr = coord_bnd_intersect_gammabar.getDataPointer();

  for (int angleIdx = 0; angleIdx < numAngles; ++angleIdx) {
    // Current angle in radians
    const T systemAngle = _fermiSurface->getSystemAngle(angleIdx);

    // Create geometry data for current angle
    _geom->create(-systemAngle);

    // Copy data to CPU
    dom_cpu = *(_geom->getDomainLabelsGrid());
    dom_p = dom_cpu.getDataPointer();

    // Find boundary crossing coordinates
    _geom->findBoundaryCrossingsAlongTrajectory(Nx, Ny, dom_p, coord_gamma_ptr);
    _geom->findBoundaryCrossingsAlongNegativeTrajectory(Nx, Ny, dom_p,
                                                        coord_gammabar_ptr);
  }

  *_coord_bnd_intersect_gamma = coord_bnd_intersect_gamma;
  *_coord_bnd_intersect_gammabar = coord_bnd_intersect_gammabar;

  std::cout << "BoundaryStorage<T>::computeSparseIndices(): Done!\n";
}

template <typename T>
void BoundaryStorage<T>::allocMemSparseStorage(const int numElemGammaIn,
                                               const int numElemGammaOut,
                                               const int numElemGammaBarIn,
                                               const int numElemGammaBarOut,
                                               const int numEnergies) {
  std::cout << "BoundaryStorage<T>::allocMemSparseStorage():\n";

  // Initialize sparse storage
  _numSparseIn_gamma = numElemGammaIn;
  _numSparseOut_gamma = numElemGammaOut;
  _numSparseIn_gammabar = numElemGammaBarIn;
  _numSparseOut_gammabar = numElemGammaBarOut;

  int numStorage_in_gamma = numElemGammaIn * numEnergies;
  int numStorage_out_gamma = numElemGammaOut * numEnergies;
  int numStorage_in_gammabar = numElemGammaBarIn * numEnergies;
  int numStorage_out_gammabar = numElemGammaBarOut * numEnergies;

  Grid<int>::GPU::initializeGrid(_coord_bnd_intersect_gamma, numElemGammaOut, 1,
                                 1, Type::real);
  Grid<int>::GPU::initializeGrid(_coord_bnd_intersect_gammabar,
                                 numElemGammaBarOut, 1, 1, Type::real);

  GridGPU::initializeGrid(_boundary_in_gamma, numStorage_in_gamma, 1, 1,
                          Type::complex);
  GridGPU::initializeGrid(_boundary_out_gamma, numStorage_out_gamma, 1, 1,
                          Type::complex);
  GridGPU::initializeGrid(_boundary_in_gammabar, numStorage_in_gammabar, 1, 1,
                          Type::complex);
  GridGPU::initializeGrid(_boundary_out_gammabar, numStorage_out_gammabar, 1, 1,
                          Type::complex);

  // Print some info
  std::cout << "\nSparse storage num energies = " << numEnergies << "\n";
  std::cout << "numStorage_in_gamma = " << numStorage_in_gamma << "\n";
  std::cout << "numStorage_out_gamma = " << numStorage_out_gamma << "\n";
  std::cout << "numStorage_in_gammabar = " << numStorage_in_gammabar << "\n";
  std::cout << "numStorage_out_gammabar = " << numStorage_out_gammabar
            << "\n\n";

  size_t bytes = (numStorage_in_gamma + numStorage_in_gammabar +
                  numStorage_out_gamma + numStorage_out_gammabar) *
                 2 * sizeof(T);

  std::cout << "Boundary storage data: " << bytes << " bytes.\n\n";
}

template <typename T> void BoundaryStorage<T>::setBoundaryZero() {
  _boundary_in_gamma->setZero();
  _boundary_out_gamma->setZero();

  _boundary_in_gammabar->setZero();
  _boundary_out_gammabar->setZero();
}

#endif // BOUNDARY_STORAGE_H_
