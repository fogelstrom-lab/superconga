//===-- Integrationiterator.h - Integration iterator base class. ----------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Base class for integration iterator, i.e. the class that handles the energy
/// integration.
///
//===----------------------------------------------------------------------===//
#ifndef INTEGRATION_ITERATOR_H_
#define INTEGRATION_ITERATOR_H_

#include "ComputeProperty.h"
#include "GeometryGroup.h"
#include "OrderParameter.h"
#include "ParameterIterator.h"
#include "RiccatiSolver.h"

namespace conga {
template <typename T> class IntegrationIterator : public ContextModule<T> {
public:
  IntegrationIterator();
  virtual ~IntegrationIterator();

  IntegrationIterator(Context<T> *ctx, int numEnergiesPerBlock = -1);

  // Virtual functions
  //
  virtual int initialize();

  // Call this once before starting self consistent iteration, or when changing
  // initial condition.
  virtual void setBoundaryInitialCondition() = 0;

  // Computes all properties added with addComputeProperty(). Iterate this self
  // consistently.
  virtual void compute() = 0;

  // Used by BoundaryStorage to determine size of storage
  virtual int getNumEnergiesInterval() = 0;
  virtual int getEnergyIdxOffset_boundaryStorage() = 0;

  // Energies
  const ParameterIterator<int> *getEnergyIterator() const {
    return _energy_iter;
  }
  ParameterIterator<int> *getEnergyIterator() { return _energy_iter; }
  void setEnergyIterator(ParameterIterator<int> *iter) {
    if (_energy_iter)
      delete _energy_iter;
    _energy_iter = iter;
  }

  void initIterators();
  virtual void initEnergies() = 0;

  // Getters
  GridGPU<T> *getEnergies() const { return _energies; }
  GridGPU<T> *getResidues() const { return _residues; }

  GridGPU<T> *getOrderParameterFieldCurrentAngle() const {
    return ContextModule<T>::getContext()
        ->getOrderParameter()
        ->getOrderParameterFieldRotated();
  }

  int getNumIterations() const { return _numIter; }
  int getNumEnergiesPerBlock() const { return _numEnergiesPerBlock; }
  int getBoundaryIdx_offset() const { return _boundaryIdxOffset; }

  // Setters
  void setEnergies(GridGPU<T> *en) {
    if (_energies)
      delete _energies;
    _energies = en;
  }
  void setNumEnergiesPerBlock(int num);
  void setResidues(GridGPU<T> *res) {
    if (_residues)
      delete _residues;
    _residues = res;
  }

  // Compute
  int increaseIterationCount() {
    _numIter++;
    return _numIter;
  }

  // Debug
  void setDebug(int angleIdx) { _debug_store_angle = angleIdx; }
  GridGPU<T> *getDebugGrid() const { return _debug_store_grid; }
  // void			computeDebug();

private:
  // Debug
  int _debug_store_angle;
  GridGPU<T> *_debug_store_grid;

  // Energy iterators
  GridGPU<T> *_energies;
  ParameterIterator<int> *_energy_iter;

  // Factor to multiply summand with in energy sum, i.e. different factor for
  // different energies. In the case of Ozaki integration iterator, these
  // factors are the Ozaki residues. For Matsubara and Keldysh, it will be a
  // factor of 1 for each energy.
  GridGPU<T> *_residues;

  // Max energies a compute block can have
  int _numEnergiesPerBlock;

  int _boundaryIdxOffset;

  // Angle
  int _numAngles;

  int _numIter;
};

template <typename T>
IntegrationIterator<T>::IntegrationIterator()
    : _energies(0), _energy_iter(0), _numIter(0), _debug_store_grid(0),
      _residues(0) {}

template <typename T>
IntegrationIterator<T>::IntegrationIterator(Context<T> *ctx,
                                            int numEnergiesPerBlock)
    : ContextModule<T>(ctx), _energies(0), _energy_iter(0),
      _numEnergiesPerBlock(numEnergiesPerBlock), _numIter(0),
      _debug_store_grid(0), _debug_store_angle(-1), _residues(0) {
#ifndef NDEBUG
  std::cout
      << "IntegrationIterator<T>::IntegrationIterator( Context<T>* context )\n";
#endif
}

template <typename T> IntegrationIterator<T>::~IntegrationIterator() {
#ifndef NDEBUG
  std::cout << "~IntegrationIterator()\n";
#endif

  if (_debug_store_grid)
    delete _debug_store_grid;
  if (_energies)
    delete _energies;
  if (_energy_iter)
    delete _energy_iter;
  if (_residues)
    delete _residues;
}

template <typename T> int IntegrationIterator<T>::initialize() {
#ifndef NDEBUG
  std::cout << "IntegrationIterator::initialize()\n";
#endif
  initIterators();
  return 0;
}

template <typename T> void IntegrationIterator<T>::initIterators() {
  _numAngles =
      ContextModule<T>::getContext()->getParameters()->getAngularResolution();
  _numIter = 0;
}

template <typename T>
void IntegrationIterator<T>::setNumEnergiesPerBlock(int num) {
  if (num != _numEnergiesPerBlock) {
    _numEnergiesPerBlock = num;
    initialize();
  }
}
} // namespace conga

#endif // INTEGRATION_ITERATOR_H_
