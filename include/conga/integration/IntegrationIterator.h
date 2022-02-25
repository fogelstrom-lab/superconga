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
#ifndef CONGA_INTEGRATION_ITERATOR_H_
#define CONGA_INTEGRATION_ITERATOR_H_

#include "../ParameterIterator.h"
#include "../compute/ComputeProperty.h"
#include "../geometry/GeometryGroup.h"
#include "../order_parameter/OrderParameter.h"
#include "../riccati/RiccatiSolver.h"

namespace conga {
template <typename T> class IntegrationIterator : public ContextModule<T> {
public:
  IntegrationIterator(const int inNumEnergiesPerBlock = -1);
  IntegrationIterator(Context<T> *const inOutContext,
                      const int inNumEnergiesPerBlock = -1);
  virtual ~IntegrationIterator();

  // Virtual functions
  //
  virtual int initialize();

  // Call this once before starting self consistent iteration, or when changing
  // initial condition.
  virtual void setBoundaryInitialCondition() = 0;

  // Computes all properties added with addComputeProperty(). Iterate this self
  // consistently.
  virtual void compute(const bool inIsBurnIn) = 0;

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

  const GridGPU<T> *getOrderParameterFieldCurrentAngle() const {
    return ContextModule<T>::getContext()
        ->getOrderParameter()
        ->getRotatedGrid();
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

private:
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
IntegrationIterator<T>::IntegrationIterator(const int inNumEnergiesPerBlock)
    : _energies(nullptr), _energy_iter(nullptr), _residues(nullptr),
      _numEnergiesPerBlock(inNumEnergiesPerBlock), _boundaryIdxOffset(0),
      _numAngles(0), _numIter(0) {}

template <typename T>
IntegrationIterator<T>::IntegrationIterator(Context<T> *const inOutContext,
                                            const int inNumEnergiesPerBlock)
    : ContextModule<T>(inOutContext), _energies(nullptr), _energy_iter(nullptr),
      _residues(nullptr), _numEnergiesPerBlock(inNumEnergiesPerBlock),
      _boundaryIdxOffset(0), _numAngles(0), _numIter(0) {
#ifndef NDEBUG
  std::cout
      << "IntegrationIterator<T>::IntegrationIterator( Context<T>* context )\n";
#endif
}

template <typename T> IntegrationIterator<T>::~IntegrationIterator() {
#ifndef NDEBUG
  std::cout << "~IntegrationIterator()\n";
#endif

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
} // namespace conga

#endif // CONGA_INTEGRATION_ITERATOR_H_
