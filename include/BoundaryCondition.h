#ifndef BOUNDARY_CONDITION_H_
#define BOUNDARY_CONDITION_H_

#include "Context.h"
#include "Grid.h"

template <typename T> class BoundaryCondition {
  typedef typename Grid<T>::GPU GridGPU;
  typedef typename Grid<T>::CPU GridCPU;

public:
  BoundaryCondition() : _bc_map(0) {}
  virtual ~BoundaryCondition() {
    if (_bc_map)
      delete _bc_map;
  }

  // BoundaryCondition( Context<T>* ctx ) : ContextModule<T>(ctx), _bc_map(0)
  // {ctx->set(this);}

  virtual int initialize() {
    _bc_map = 0;
    return 0;
  } // Do actual initialization when implementing!

  virtual void computeBoundaryCondition(Context<T> *ctx) = 0;

  // Not used yet
  Grid<char> *getBoundaryConditionMap() const { return _bc_map; }

  // Not implemented
  // Should set values/labels in _bc_map to specify where BC applies to.
  // E.g. if apply==true, then set values in given coordinate range to 1,
  // otherwise 0. Operation should be able to be applied multiple times, so only
  // set values in the specified range.
  void applyToCoordinateRange(const bool apply, int x_min, int x_max, int y_min,
                              int y_max) {}

private:
  // Boundary Condition map not implemented yet!
  // A 2d-grid of same dimensions as discrete geometry representation (say
  // _domainLabels). Should specify where the boundary condition will be
  // applied, so that different BC's can be mixed. Currently one BC is applied
  // to all boundary nodes.
  Grid<char> *_bc_map;
};

#endif // BOUNDARY_CONDITION_H_
