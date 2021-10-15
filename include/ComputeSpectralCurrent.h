//=== ComputeSpectralCurrent.h - Class for spectral current compute object. ==//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Compute object class for spectral current.
///
//===----------------------------------------------------------------------===//
#ifndef COMPUTE_SPECTRAL_CURRENT_H_
#define COMPUTE_SPECTRAL_CURRENT_H_

#include "ComputeProperty.h"
#include "Grid.h"
#include "IntegrationIteratorKeldysh.h"
#include <cmath>
#include <limits>

namespace conga {
template <typename T> class ComputeSpectralCurrent : public ComputeProperty<T> {
public:
  // Constructors and Destructors
  ComputeSpectralCurrent();
  ~ComputeSpectralCurrent();

  ComputeSpectralCurrent(Context<T> *ctx);

  GridGPU<T> *getResult() override { return _lcurrent_all; }
  IntegrationIterator<T> *
  createIntegrationIterator(int numEnergiesPerBlock = -1) override;

  // Context initialize function
  int initialize() override;

  // This should be called once for every iteration, before any integration
  // takes place.
  void initializeIteration(int numFields = 1) override;

  // Computes greens function for current angle. This may be called multiple
  // times as we integrate over many energies. Result is in local frame and will
  // be stored in _greensCurrentAngle.
  void computeGreensAngle(ParameterIterator<int> *iter, GridGPU<T> *gamma,
                          GridGPU<T> *gammabar) override;

  // Add the result from computeGreensAngle() to total. Rotates
  // _greensCurrentAngle to global frame and adds momentum dependence.
  void integrateGreensAngle(GeometryGroup<T> *geom,
                            const int angleIdx) override;

  // Compute physical property from greens function.
  void computePropertyFromGreens() override;

  GridGPU<T> *getResultInterval(int field = 0) {
    if (field == 0)
      return _lcurrent_x;
    else
      return _lcurrent_y;
  }

  GridCPU<T> *getSpectralCurrent() const { return _current; }

  int getMultFermiFunction() const { return _multFermiFunction; }
  void setMultFermiFunction(const int mult) { _multFermiFunction = mult; }

private:
  // Computed local spectral current will be stored here at end of
  // IntegrationIterator::compute() function.
  GridGPU<T> *_lcurrent_x;
  GridGPU<T> *_lcurrent_y;

  // Total spectral current for system, for each energy level
  GridGPU<T> *_lcurrent_all;

  // Integrated current, per energy level
  GridCPU<T> *_current;

  // Temporary grid.
  GridGPU<T> *_tmp;

  // Multiply with fermi function
  int _multFermiFunction;
};

template <typename T>
ComputeSpectralCurrent<T>::ComputeSpectralCurrent()
    : ComputeProperty<T>(), _lcurrent_x(0), _lcurrent_y(0), _lcurrent_all(0),
      _current(0), _tmp(0), _multFermiFunction(0) {}

template <typename T>
ComputeSpectralCurrent<T>::ComputeSpectralCurrent(Context<T> *ctx)
    : ComputeProperty<T>(ctx), _lcurrent_x(0), _lcurrent_y(0), _lcurrent_all(0),
      _current(0), _tmp(0), _multFermiFunction(0) {
  ctx->add(this);
}

template <typename T> ComputeSpectralCurrent<T>::~ComputeSpectralCurrent() {
  if (_lcurrent_x)
    delete _lcurrent_x;
  if (_lcurrent_y)
    delete _lcurrent_y;
  if (_lcurrent_all)
    delete _lcurrent_all;
  if (_current)
    delete _current;
  if (_tmp)
    delete _tmp;
}

template <typename T>
IntegrationIterator<T> *
ComputeSpectralCurrent<T>::createIntegrationIterator(int numEnergiesPerBlock) {
  return new IntegrationIteratorKeldysh<T>(ContextModule<T>::getContext(),
                                           numEnergiesPerBlock);
}

template <typename T> int ComputeSpectralCurrent<T>::initialize() {
  typedef ContextModule<T> C;

  int N0 = C::getParameters()->getGridResolutionBase();
  int N1 = C::getParameters()->getGridResolutionCoherence();

  int numEnergies = C::getParameters()->getNumEnergies();
  int numEnergiesInterval =
      C::getIntegrationIterator()->getEnergyIterator()->getNumInterval();

  GridGPU<T>::initializeGrid(_lcurrent_all, N0, N0, 2 * numEnergies,
                             Type::real);
  GridGPU<T>::initializeGrid(_lcurrent_x, N0, N0, numEnergiesInterval,
                             Type::real);
  GridGPU<T>::initializeGrid(_lcurrent_y, N0, N0, numEnergiesInterval,
                             Type::real);
  GridGPU<T>::initializeGrid(_tmp, N0, N0, numEnergiesInterval, Type::real);
  GridCPU<T>::initializeGrid(_current, numEnergies, 1, 2, Type::real);

  _current->setZero();

  ComputeProperty<T>::initGreensAngle(N1, numEnergiesInterval);
#ifndef NDEBUG
  std::cout << "SpectralCurrent mem: "
            << (_lcurrent_all->numElements() + 3 * _lcurrent_x->numElements()) *
                   sizeof(T)
            << "\n";
#endif
  return 0;
}

template <typename T>
void ComputeSpectralCurrent<T>::initializeIteration(int numFields) {
  if (ComputeProperty<T>::doCompute()) {
    _lcurrent_x->setZero();
    _lcurrent_y->setZero();
    _tmp->setZero();
  }
}

template <typename T>
void ComputeSpectralCurrent<T>::computeGreensAngle(ParameterIterator<int> *iter,
                                                   GridGPU<T> *gamma,
                                                   GridGPU<T> *gammabar) {
  if (ComputeProperty<T>::doCompute()) {
    GreensFunctionStatic<T, Greens>::computeGreens(
        ComputeProperty<T>::getGreensCurrentAngle(), iter, gamma, gammabar);
  }
}

template <typename T>
void ComputeSpectralCurrent<T>::integrateGreensAngle(GeometryGroup<T> *geom,
                                                     const int angleIdx) {
  if (ComputeProperty<T>::doCompute()) {
    GridGPU<T> *greens = ComputeProperty<T>::getGreensCurrentAngle();

    // Compensate for numerical diffusion from interpolation during rotation
    geom->extendValuesAtBoundary(greens);

    const int numAngles =
        ContextModule<T>::getParameters()->getAngularResolution();
    const T systemAngle =
        ContextModule<T>::getFermiSurface()->getSystemAngle(angleIdx);

    T fermiVelocityX;
    T fermiVelocityY;
    ContextModule<T>::getFermiSurface()->(angleIdx, fermiVelocityX,
                                          fermiVelocityY);

    const T integrandFactor =
        ContextModule<T>::getFermiSurface()->getIntegrandFactor(angleIdx);

    _tmp->insertRotatedGrid(greens, systemAngle, 0, 0, Type::imag);
    *_lcurrent_x += *_tmp * fermiVelocityX * integrandFactor;
    *_lcurrent_y += *_tmp * fermiVelocityY * integrandFactor;
  }
}

template <typename T>
void ComputeSpectralCurrent<T>::computePropertyFromGreens() {
  if (ComputeProperty<T>::doCompute()) {
    typedef ContextModule<T> C;

    // Get some info
    const T h = C::getParameters()->getGridElementSize();
    const int numAngles = C::getParameters()->getAngularResolution();
    const T T_Tc = C::getParameters()->getTemperature();

    const ParameterIterator<int> *iter =
        C::getIntegrationIterator()->getEnergyIterator();

    int numEnergiesCurrentBlock = iter->getNumInterval();
    int energyIdx_offset = iter->getCurrentValue();

    GridGPU<T> *energies = C::getIntegrationIterator()->getEnergies();

    // Add fermifunction
    if (_multFermiFunction) {
      dim3 bpg, tpb;
      _lcurrent_x->getKernelDimensionsFields(bpg, tpb);

      compute_property_internal::addFermiFunction_kernel<<<bpg, tpb>>>(
          _lcurrent_x->getDataPointer(0, Type::real),
          energies->getDataPointer(0, Type::real) + energyIdx_offset, T_Tc,
          numEnergiesCurrentBlock, _lcurrent_x->dimX());

      compute_property_internal::addFermiFunction_kernel<<<bpg, tpb>>>(
          _lcurrent_y->getDataPointer(0, Type::real),
          energies->getDataPointer(0, Type::real) + energyIdx_offset, T_Tc,
          numEnergiesCurrentBlock, _lcurrent_y->dimX());
    }

    // TODO(niclas): Is this correct for non-circular FS???
    *_lcurrent_x *= -1.0 / (2.0 * M_PI * (T)numAngles);
    *_lcurrent_y *= -1.0 / (2.0 * M_PI * (T)numAngles);

    // Copy currents to full storage
    for (int i = 0; i < numEnergiesCurrentBlock; ++i) {
      _lcurrent_x->copyFieldTo(_lcurrent_all, i, 2 * (i + energyIdx_offset));
      _lcurrent_y->copyFieldTo(_lcurrent_all, i,
                               2 * (i + energyIdx_offset) + 1);
    }

    // Integrate local spectral current (used to compute residual)
    GridCPU<T> energies_cpu = *energies;
    T *energies_ptr =
        energies_cpu.getDataPointer(0, Type::real) + energyIdx_offset;

    T *E_ptr = _current->getDataPointer(0) + energyIdx_offset;
    T *j_ptr = _current->getDataPointer(1) + energyIdx_offset;

    T sum_x, sum_y, old, tmp, sumdiff = 0.0, sumold = 0.0;

    for (int i = 0; i < numEnergiesCurrentBlock; ++i) {
      // Sum for energy level
      sum_x = _lcurrent_x->sumField(i);
      sum_y = _lcurrent_y->sumField(i);

      old = j_ptr[i];
      j_ptr[i] = std::sqrt(sum_x * sum_x + sum_y * sum_y) * h * h;

      tmp = j_ptr[i] - old;
      sumdiff += tmp * tmp;
      sumold += old * old;

      // Store energy value
      E_ptr[i] = energies_ptr[i];
    }

    // TODO(niclas): Use computeResidual instead!
    sumdiff = std::sqrt(sumdiff /
                        std::max(sumold, std::numeric_limits<T>::epsilon()));
    ComputeProperty<T>::setResidual(sumdiff);

    const bool isPartiallyConverged =
        sumdiff < C::getParameters()->getConvergenceCriterion();
    ComputeProperty<T>::setPartiallyConverged(isPartiallyConverged);
  }
}
} // namespace conga

#endif // COMPUTE_SPECTRAL_CURRENT_H_
