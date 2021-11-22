//===---- ComputeCurrent.h - Class for current density compute object. ----===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Compute object class for current density.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_COMPUTE_CURRENT_H_
#define CONGA_COMPUTE_CURRENT_H_

#include "../defines.h"
#include "../grid.h"
#include "../integration/IntegrationIteratorOzaki.h"
#include "../utils.h"
#include "ComputeProperty.h"
#include "ComputeResidual.h"

namespace conga {
template <typename T> class ComputeCurrent : public ComputeProperty<T> {
public:
  // Constructors and Destructors
  ComputeCurrent();
  ~ComputeCurrent();

  ComputeCurrent(Context<T> *ctx);

  GridGPU<T> *getResult() override { return _current_real; }

  int initialize() override;

  IntegrationIterator<T> *
  createIntegrationIterator(int numEnergiesPerBlock = -1) override;

  // This should be called once for every iteration, before any integration
  // takes place.
  void initializeIteration(int numFields = 1) override;

  // This should be called once for every angle, before any integration takes
  // place.
  void initializeGreensAngle() override;

  // Computes greens function for current angle. This may be called multiple
  // times as we integrate over many energies. Result is in local frame and will
  // be stored in _greensCurrentAngle.
  void computeGreensAngle(const GridGPU<T> *const inGamma,
                          const GridGPU<T> *const inGammaTilde) override;

  // Add the result from computeGreensAngle() to total. Rotates
  // _greensCurrentAngle to global frame and adds momentum dependence.
  void integrateGreensAngle(GeometryGroup<T> *geom,
                            const int angleIdx) override;

  // Compute physical property from greens function.
  void computePropertyFromGreens() override;

private:
  // Computed current will be stored here at end of
  // IntegrationIterator::compute() function.
  GridGPU<T> *_current;
  GridGPU<T> *_current_real;

  // Temporary grid.
  GridGPU<T> *_tmp;
};

template <typename T>
ComputeCurrent<T>::ComputeCurrent()
    : ComputeProperty<T>(), _current(0), _current_real(0), _tmp(0) {}

template <typename T>
ComputeCurrent<T>::ComputeCurrent(Context<T> *ctx)
    : ComputeProperty<T>(ctx), _current(0), _current_real(0), _tmp(0) {
  ctx->add(this);
}

template <typename T> ComputeCurrent<T>::~ComputeCurrent() {
  if (_current)
    delete _current;
  if (_current_real)
    delete _current_real;
  if (_tmp)
    delete _tmp;
}

template <typename T> int ComputeCurrent<T>::initialize() {
#ifndef NDEBUG
  std::cout << "ComputeCurrent<T>::initialize()\n";
#endif
  typedef ContextModule<T> C;

  // Get resolution
  const int N0 = C::getParameters()->getGridResolutionBase();
  const int N1 = C::getParameters()->getGridResolutionCoherence();

  GridGPU<T>::initializeGrid(_current, N0, N0, 2);
  GridGPU<T>::initializeGrid(_current_real, N0, N0, 2, Type::real);
  GridGPU<T>::initializeGrid(_tmp, N1, N1);

  _current_real->setZero();

  ComputeProperty<T>::initGreensAngle(N1);

  return 0;
}

template <typename T>
IntegrationIterator<T> *
ComputeCurrent<T>::createIntegrationIterator(int numEnergiesPerBlock) {
  return new IntegrationIteratorOzaki<T>(ContextModule<T>::getContext(),
                                         numEnergiesPerBlock);
}

template <typename T>
void ComputeCurrent<T>::initializeIteration(int numFields) {
  // Unused.
  (void)numFields;

  typedef ContextModule<T> C;

  if (ComputeProperty<T>::doCompute())
    _current->setZero();
}

template <typename T> void ComputeCurrent<T>::initializeGreensAngle() {
  if (ComputeProperty<T>::doCompute())
    ComputeProperty<T>::initializeGreensAngle(); // Sets to zero
}

template <typename T>
void ComputeCurrent<T>::computeGreensAngle(
    const GridGPU<T> *const inGamma, const GridGPU<T> *const inGammaTilde) {

  if (ComputeProperty<T>::doCompute()) {
    typedef ContextModule<T> C;
    IntegrationIterator<T> *ii = C::getIntegrationIterator();
    const GridGPU<T> *residues = ii->getResidues();

    GreensFunctionStatic<T, Greens>::computeGreensSumEnergy(
        inGamma, inGammaTilde, residues,
        ComputeProperty<T>::getGreensCurrentAngle());
  }
}

template <typename T>
void ComputeCurrent<T>::integrateGreensAngle(GeometryGroup<T> *geom,
                                             const int angleIdx) {
  if (ComputeProperty<T>::doCompute()) {
    GridGPU<T> *greens = ComputeProperty<T>::getGreensCurrentAngle();

    // Compensate for numerical diffusion from interpolation during rotation.
    geom->extendValuesAtBoundary(greens);

    const T systemAngle =
        ComputeProperty<T>::getFermiSurface()->getSystemAngle(angleIdx);

    T fermiVelocityX;
    T fermiVelocityY;
    ComputeProperty<T>::getFermiSurface()->getFermiVelocity(
        angleIdx, fermiVelocityX, fermiVelocityY);

    const T integrandFactor =
        ComputeProperty<T>::getFermiSurface()->getIntegrandFactor(angleIdx);

    // Add momentum dependence and rotate back to reference frame:
    // x-component
    *_tmp = *greens;
    *_tmp *= fermiVelocityX * integrandFactor;
    _current->insertRotatedGridAdditive(_tmp, systemAngle, 0);

    // y-component
    *_tmp = *greens;
    *_tmp *= fermiVelocityY * integrandFactor;
    _current->insertRotatedGridAdditive(_tmp, systemAngle, 1);
  }
}

template <typename T> void ComputeCurrent<T>::computePropertyFromGreens() {
  typedef ComputeProperty<T> C;

  if (C::doCompute()) {
    GridGPU<T> current_old = *_current_real;

    _current_real->insertGrid(_current);

    const T T_Tc = C::getParameters()->getTemperature();
    // The factor of 2 comes from defining j_d = 2*pi*kB*Tc*|e|*vF*NF.
    const T factor = static_cast<T>(2.0 * constants::SIGN_E) * T_Tc;
    *_current_real *= factor;

    // Set the current to zero outside of the grain.
    ContextModule<T>::getContext()->getGeometry()->zeroValuesOutsideDomainRef(
        _current_real);

    const T residual = computeResidual(current_old, *_current_real);
    C::setResidual(residual);

    const bool isConverged =
        residual < C::getParameters()->getConvergenceCriterion();
    C::setConverged(isConverged);
  } else {
    C::setConverged(false);
  }
}
} // namespace conga

#endif // CONGA_COMPUTE_CURRENT_H_
