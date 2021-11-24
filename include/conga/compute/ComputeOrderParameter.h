//===---- ComputeCurrent.h - Class for order parameter compute object. ----===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Compute object class for order parameter.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_COMPUTE_ORDER_PARAMETER_H_
#define CONGA_COMPUTE_ORDER_PARAMETER_H_

#include "../grid.h"
#include "ComputeProperty.h"
#include "ComputeResidual.h"

namespace conga {
template <typename T> class ComputeOrderParameter : public ComputeProperty<T> {
public:
  // Constructors and Destructors
  ComputeOrderParameter();
  ~ComputeOrderParameter();

  ComputeOrderParameter(Context<T> *ctx);

  int initialize() override;

  OrderParameter<T> *getOrderParameter() { return _delta; }
  void setOrderParameter(const OrderParameter<T> *delta_in);

  GridGPU<T> *getResult() override { return _delta_fields; }

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
  // Computed order parameter will be stored here at end of
  // IntegrationIterator::compute() function.
  OrderParameter<T> *_delta;

  // Each component stored in a separate field.
  GridGPU<T> *_delta_fields;

  // Temp storage for computation of order parameter
  GridGPU<T> *_deltaTmp;
};

template <typename T>
ComputeOrderParameter<T>::ComputeOrderParameter()
    : ComputeProperty<T>(), _delta(0), _delta_fields(0), _deltaTmp(0) {}

template <typename T>
ComputeOrderParameter<T>::ComputeOrderParameter(Context<T> *ctx)
    : ComputeProperty<T>(ctx), _delta(0), _delta_fields(0), _deltaTmp(0) {
  ctx->add(this);
}

template <typename T> ComputeOrderParameter<T>::~ComputeOrderParameter() {
#ifndef NDEBUG
  std::cout << "~ComputeOrderParameter<T>\n";
#endif
  if (_delta) {
    delete _delta;
  }
  if (_deltaTmp) {
    delete _deltaTmp;
  }
  if (_delta_fields) {
    delete _delta_fields;
  }
}

template <typename T> int ComputeOrderParameter<T>::initialize() {
#ifndef NDEBUG
  std::cout << "ComputeOrderParameter<T>::initialize()\n";
#endif
  typedef ContextModule<T> C;

  // Make a local copy of order parameter, store new order parameter in this.
  setOrderParameter(C::getOrderParameter());

  const int N0 = C::getParameters()->getGridResolutionBase();
  const int N1 = C::getParameters()->getGridResolutionCoherence();

  const int Nc = _delta->getNumComponents();

  GridGPU<T>::initializeGrid(_deltaTmp, N0, N0);
  GridGPU<T>::initializeGrid(_delta_fields, N0, N0, Nc, Type::complex);

  ComputeProperty<T>::initGreensAngle(N1);

  ComputeProperty<T>::setConverged(false);

  GridCPU<T> energies_cpu = *(C::getIntegrationIterator()->getEnergies());
  GridCPU<T> residues_cpu = *(C::getIntegrationIterator()->getResidues());

  // Initialize coupling constant
  for (int i = 0; i < Nc; ++i) {
    OrderParameterComponent<T> *component = _delta->getComponent(i);
    component->initCouplingConstant(energies_cpu.getDataPointer(0, Type::imag),
                                    residues_cpu.getDataPointer(0, Type::real));
  }

  return 0;
}

template <typename T>
void ComputeOrderParameter<T>::setOrderParameter(
    const OrderParameter<T> *delta_in) {
  if (delta_in) {
    if (_delta) {
      delete _delta;
    }
    _delta = new OrderParameter<T>(*delta_in);
  }
}

template <typename T>
void ComputeOrderParameter<T>::initializeIteration(int numFields) {
  // Unused.
  (void)numFields;

  typedef ContextModule<T> C;

  if (ComputeProperty<T>::doCompute()) {
    // Temporary grids for new order parameter components
    const int numDelta = _delta->getNumComponents();

    for (int i = 0; i < numDelta; ++i) {
      _delta->getComponent(i)->getGrid()->setZero();
    }
  }
}

template <typename T> void ComputeOrderParameter<T>::initializeGreensAngle() {
  if (ComputeProperty<T>::doCompute())
    ComputeProperty<T>::initializeGreensAngle(); // Sets to zero
}

template <typename T>
void ComputeOrderParameter<T>::computeGreensAngle(
    const GridGPU<T> *const inGamma, const GridGPU<T> *const inGammaTilde) {
  if (ComputeProperty<T>::doCompute()) {
    typedef ComputeProperty<T> C;
    IntegrationIterator<T> *ii = ContextModule<T>::getIntegrationIterator();
    const GridGPU<T> *residues = ii->getResidues();

    GreensFunctionStatic<T, Greens_Delta>::computeGreensSumEnergy(
        inGamma, inGammaTilde, residues, C::getGreensCurrentAngle());
  }
}

template <typename T>
void ComputeOrderParameter<T>::integrateGreensAngle(GeometryGroup<T> *geom,
                                                    const int angleIdx) {
  typedef ComputeProperty<T> C;

  if (ComputeProperty<T>::doCompute()) {
    GridGPU<T> *greens = C::getGreensCurrentAngle();

    // Compensate for numerical diffusion from interpolation during rotation
    geom->extendValuesAtBoundary(greens);

    FermiSurface<T> *fermiSurface = C::getFermiSurface();
    const T systemAngle = fermiSurface->getSystemAngle(angleIdx);

    T momentumX;
    T momentumY;
    fermiSurface->getFermiMomentum(angleIdx, momentumX, momentumY);

    const T integrandFactor = fermiSurface->getIntegrandFactor(angleIdx);
    const T crystalAxesRotation = _delta->getCrystalAxesRotation();
    for (int i = 0; i < _delta->getNumComponents(); ++i) {
      _deltaTmp->setZero();

      // Rotate back to reference frame and add momentum dependence.
      _deltaTmp->insertRotatedGrid(greens, systemAngle);

      // It should be the complex conjugate of the basis function!
      const bool conjugateBasisFunction = true;
      _delta->getComponent(i)->computeMomentumDependence(
          _deltaTmp, momentumX, momentumY, crystalAxesRotation,
          conjugateBasisFunction);

      // Add angle contribution (angle integration)
      *(_delta->getComponent(i)->getGrid()) += *_deltaTmp * integrandFactor;
    }
  }
}

template <typename T>
void ComputeOrderParameter<T>::computePropertyFromGreens() {
  typedef ComputeProperty<T> C;

  if (C::doCompute()) {
    Parameters<T> *param = C::getParameters();
    const int Nc = _delta->getNumComponents();

    for (int i = 0; i < Nc; ++i) {
      // Compute new delta from anomalous greens function
      OrderParameterComponent<T> *component = _delta->getComponent(i);
      component->applyCouplingConstant();
    }

    // Set the order parameter to zero outside of the grain, so that we only
    // compute the residual with values that actually matter.
    const Context<T> *const ctx = ContextModule<T>::getContext();
    for (int i = 0; i < Nc; ++i) {
      GridGPU<T> *const component = _delta->getComponent(i)->getGrid();
      ctx->getGeometry()->zeroValuesOutsideDomainRef(component);
    }

    // Fetch the previous order parameter so we can compute the residual.
    OrderParameter<T> *deltaOld = ContextModule<T>::getOrderParameter();

    // Compute the residual component-wise and pick the maximum one.
    T residual = static_cast<T>(0.0);
    for (int i = 0; i < Nc; ++i) {
      const GridGPU<T> *const componentOld =
          deltaOld->getComponent(i)->getGrid();
      const GridGPU<T> *const componentNew = _delta->getComponent(i)->getGrid();

      const T residualComponent = computeResidual(*componentOld, *componentNew);
      residual = std::max(residual, residualComponent);
    }
    C::setResidual(residual);

    // Check convergence.
    const bool isConverged = residual < param->getConvergenceCriterion();
    C::setConverged(isConverged);

    // Copy results to grid with one field per order parameter component.
    for (int i = 0; i < Nc; ++i) {
      _delta->getComponentGrid(i)->copyFieldTo(_delta_fields, 0, i);
    }
  } else {
    C::setConverged(false);
  }
}
} // namespace conga

#endif // CONGA_COMPUTE_ORDER_PARAMETER_H_
