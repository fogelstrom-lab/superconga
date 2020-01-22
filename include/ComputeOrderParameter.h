#ifndef COMPUTEORDERPARAMETER_H
#define COMPUTEORDERPARAMETER_H

#include "ComputeProperty.h"
#include "IntegrationIteratorOzaki.h"
#include "Optimizer.h"

template <typename T> class ComputeOrderParameter : public ComputeProperty<T> {
  typedef typename Grid<T>::GPU GridGPU;
  typedef typename Grid<T>::CPU GridCPU;

public:
  // Constructors and Destructors
  ComputeOrderParameter();
  virtual ~ComputeOrderParameter();

  ComputeOrderParameter(Context<T> *ctx);

  virtual int initialize();
  virtual IntegrationIterator<T> *
  createIntegrationIterator(int numEnergiesPerBlock = -1);

  OrderParameter<T> *getOrderParameter() { return _delta; }
  void setOrderParameter(const OrderParameter<T> *delta_in);

  // Virtual functions
  virtual GridGPU *getResult() { return _delta_fields; }

  // This should be called once for every iteration, before any integration
  // takes place.
  virtual void initializeIteration(int numFields = 1);

  // This should be called once for every angle, before any integration takes
  // place.
  virtual void initializeGreensAngle();

  // Computes greens function for current angle. This may be called multiple
  // times as we integrate over many energies. Result is in local frame and will
  // be stored in _greensCurrentAngle.
  virtual void computeGreensAngle(ParameterIterator<int> *iter, GridGPU *gamma,
                                  GridGPU *gammabar, GeometryGroup<T> *geom);

  // Add the result from computeGreensAngle() to total. Rotates
  // _greensCurrentAngle to global frame and adds momentum dependence.
  virtual void integrateGreensAngle(GeometryGroup<T> *geom, const int angleIdx);

  // Compute physical property from greens function.
  virtual void computePropertyFromGreens();

  // Set optimizer, e.g. OptimizerFixedPoint.
  void setOptimizer(Optimizer<T> *opt) { _opt = opt; }

private:
  // Computed order parameter will be stored here at end of
  // IntegrationIterator::compute() function.
  OrderParameter<T> *_delta;

  // Optimizer.
  Optimizer<T> *_opt;

  // Each component stored in a separate field.
  GridGPU *_delta_fields;

  // Temp storage for computation of order parameter
  GridGPU *_deltaTmp;
};

template <typename T>
ComputeOrderParameter<T>::ComputeOrderParameter()
    : ComputeProperty<T>(), _delta(0), _delta_fields(0), _deltaTmp(0), _opt() {}

template <typename T>
ComputeOrderParameter<T>::ComputeOrderParameter(Context<T> *ctx)
    : ComputeProperty<T>(ctx), _delta(0), _delta_fields(0), _deltaTmp(0),
      _opt() {
  ctx->add(this);
}

template <typename T> ComputeOrderParameter<T>::~ComputeOrderParameter() {
  std::cout << "~ComputeOrderParameter<T>\n";
  if (_delta)
    delete _delta;
  if (_deltaTmp)
    delete _deltaTmp;
  if (_delta_fields)
    delete _delta_fields;
  if (_opt)
    delete _opt;
}

template <typename T> int ComputeOrderParameter<T>::initialize() {
  std::cout << "ComputeOrderParameter<T>::initialize()\n";
  typedef ContextModule<T> C;

  // Make a local copy of order parameter, store new order parameter in this.
  setOrderParameter(C::getOrderParameter());

  const int N0 = C::getParameters()->getGridResolutionBase();
  const int N1 = C::getParameters()->getGridResolutionCoherence();

  const int Nc = _delta->getNumOrderParameterComponents();

  GridGPU::initializeGrid(_deltaTmp, N0, N0);
  GridGPU::initializeGrid(_delta_fields, N0, N0, Nc, Type::complex);

  ComputeProperty<T>::initGreensAngle(N1);

  ComputeProperty<T>::setConverged(false);

  GridCPU energies_cpu = *(C::getIntegrationIterator()->getEnergies());
  GridCPU residues_cpu = *(C::getIntegrationIterator()->getResidues());

  // Initialize coupling constant
  for (int i = 0; i < Nc; ++i) {
    OrderParameterComponent<T> *component =
        _delta->getOrderParameterComponent(i);
    component->initCouplingConstant(energies_cpu.getDataPointer(0, Type::imag),
                                    residues_cpu.getDataPointer(0, Type::real));
  }

  return 0;
}

template <typename T>
IntegrationIterator<T> *
ComputeOrderParameter<T>::createIntegrationIterator(int numEnergiesPerBlock) {
  return new IntegrationIteratorOzaki<T>(ContextModule<T>::getContext(),
                                         numEnergiesPerBlock);
}

template <typename T>
void ComputeOrderParameter<T>::setOrderParameter(
    const OrderParameter<T> *delta_in) {
  if (delta_in) {
    if (_delta)
      delete _delta;
    _delta = new OrderParameter<T>(*delta_in);
  }
}

template <typename T>
void ComputeOrderParameter<T>::initializeIteration(int numFields) {
  typedef ContextModule<T> C;

  if (ComputeProperty<T>::doCompute()) {
    // Temporary grids for new order parameter components
    const int numDelta = _delta->getNumOrderParameterComponents();

    for (int i = 0; i < numDelta; ++i)
      _delta->getOrderParameterComponent(i)
          ->getOrderParameterField()
          ->setZero();
  }
}

template <typename T> void ComputeOrderParameter<T>::initializeGreensAngle() {
  if (ComputeProperty<T>::doCompute())
    ComputeProperty<T>::initializeGreensAngle(); // Sets to zero
}

template <typename T>
void ComputeOrderParameter<T>::computeGreensAngle(ParameterIterator<int> *iter,
                                                  GridGPU *gamma,
                                                  GridGPU *gammabar,
                                                  GeometryGroup<T> *geom) {
  if (ComputeProperty<T>::doCompute()) {
    typedef ComputeProperty<T> C;
    IntegrationIterator<T> *ii = ContextModule<T>::getIntegrationIterator();
    GridGPU *residues = ii->getResidues();

    GreensFunctionStatic<T, Greens_Delta>::computeGreensSumEnergy(
        C::getGreensCurrentAngle(), iter, residues, gamma, gammabar, geom);
  }
}

template <typename T>
void ComputeOrderParameter<T>::integrateGreensAngle(GeometryGroup<T> *geom,
                                                    const int angleIdx) {
  typedef ComputeProperty<T> C;

  if (ComputeProperty<T>::doCompute()) {
    GridGPU *greens = C::getGreensCurrentAngle();

    FermiSurface<T> *fermiSurface = C::getFermiSurface();
    const T systemAngle = fermiSurface->getSystemAngle(angleIdx);

    T momentumX;
    T momentumY;
    fermiSurface->getFermiMomentum(angleIdx, momentumX, momentumY);

    const T integrandFactor = fermiSurface->getIntegrandFactor(angleIdx);

    for (int i = 0; i < _delta->getNumOrderParameterComponents(); ++i) {
      _deltaTmp->setZero();

      // Compensate for numerical diffusion from interpolation during rotation
      geom->extendValuesAtBoundary(greens, 1);

      // Rotate back to reference frame and add momentum dependence.
      _deltaTmp->insertRotatedGrid(greens, systemAngle);

      // It should be the complex conjugate of the basis function!
      const bool conjugateBasis = true;
      _delta->getOrderParameterComponent(i)->computeMomentumDependence(
          _deltaTmp, momentumX, momentumY, conjugateBasis);

      // Add angle contribution (angle integration)
      *(_delta->getOrderParameterComponent(i)->getOrderParameterField()) +=
          *_deltaTmp * integrandFactor;
    }
  }
}

template <typename T>
void ComputeOrderParameter<T>::computePropertyFromGreens() {
  typedef ComputeProperty<T> C;

  if (C::doCompute()) {
    Parameters<T> *param = C::getParameters();
    const int Na = param->getAngularResolution();
    int Nc = _delta->getNumOrderParameterComponents();
    int numEnergies = param->getNumMatsubaraEnergies();

    for (int i = 0; i < Nc; ++i) {
      // Compute new delta from anomalous greens function
      OrderParameterComponent<T> *component =
          _delta->getOrderParameterComponent(i);
      component->applyCouplingConstant();
    }

    // Compute residual
    OrderParameter<T> *deltaOld = ContextModule<T>::getOrderParameter();
    _opt->step(*_delta, *deltaOld);
    const T residual = _opt->getResidual();
    C::setResidual(residual);

    // TODO(niclas): Not sure if this is needed.
    *_delta = *deltaOld;

    if (residual < param->getConvergenceCriterion())
      C::setConverged();

    // Copy results to grid with one field per order parameter component
    for (int i = 0; i < Nc; ++i)
      _delta->getOrderParameterComponentField(i)->copyFieldTo(_delta_fields, 0,
                                                              i);
  }
}

#endif // COMPUTEORDERPARAMETER_H
