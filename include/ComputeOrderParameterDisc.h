#ifndef _COMPUTEORDERPARAMETERDISC_H
#define _COMPUTEORDERPARAMETERDISC_H

#include <ComputeProperty.h>
#include <IntegrationIteratorMatsubaraDisc.h>

template <typename T>
class ComputeOrderParameterDisc : public ComputeProperty<T> {
  typedef typename Grid<T>::GPU GridGPU;

public:
  // Constructors and Destructors
  ComputeOrderParameterDisc();
  virtual ~ComputeOrderParameterDisc();

  ComputeOrderParameterDisc(Context<T> *ctx);

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
  virtual void initializeGreensAngle() {}

  // Computes greens function for current angle. This may be called multiple
  // times as we integrate over many energies. Result is in local frame and will
  // be stored in _greensCurrentAngle.
  virtual void computeGreensAngle(ParameterIterator<int> *iter, GridGPU *gamma,
                                  GridGPU *gammabar, GeometryGroup<T> *geom) {}

  // Add the result from computeGreensAngle() to total. Rotates
  // _greensCurrentAngle to global frame and adds momentum dependence.
  virtual void integrateGreensAngle(GeometryGroup<T> *geom, const T angle) {}

  // Compute physical property from greens function.
  virtual void computePropertyFromGreens();

  // Helper function to compute residual
  T computeResidual(OrderParameter<T> *old) const;

private:
  // Computed order parameter will be stored here at end of
  // IntegrationIterator::compute() function.
  OrderParameter<T> *_delta;
  GridGPU *_delta_fields; // Each component stored in a separate field.

  // Temp storage for computation of order parameter
  GridGPU *_deltaTmp;
};

template <typename T>
ComputeOrderParameterDisc<T>::ComputeOrderParameterDisc()
    : ComputeProperty<T>(), _delta(0), _deltaTmp(0), _delta_fields(0) {}

template <typename T>
ComputeOrderParameterDisc<T>::ComputeOrderParameterDisc(Context<T> *ctx)
    : ComputeProperty<T>(ctx), _delta(0), _deltaTmp(0), _delta_fields(0) {
  ctx->add(this);
}

template <typename T>
ComputeOrderParameterDisc<T>::~ComputeOrderParameterDisc() {
  std::cout << "~ComputeOrderParameterDisc<T>\n";
  if (_delta)
    delete _delta;
  if (_deltaTmp)
    delete _deltaTmp;
  if (_delta_fields)
    delete _delta_fields;
}

template <typename T> int ComputeOrderParameterDisc<T>::initialize() {
  std::cout << "ComputeOrderParameterDisc<T>::initialize()\n";

  typedef ContextModule<T> C;

  // Make a local copy of order parameter, store new order parameter in this.
  setOrderParameter(C::getOrderParameter());

  const int N0 = C::getParameters()->getGridResolutionBase();
  const int N1 = C::getParameters()->getGridResolutionCoherence();

  const int Nc = _delta->getNumOrderParameterComponents();

  GridGPU::initializeGrid(_deltaTmp, N0, N0);
  GridGPU::initializeGrid(_delta_fields, N0, N0, Nc, Type::complex);

  ComputeProperty<T>::initGreensAngle(N1);

  return 0;
}

template <typename T>
IntegrationIterator<T> *ComputeOrderParameterDisc<T>::createIntegrationIterator(
    int numEnergiesPerBlock) {
  return new IntegrationIteratorMatsubaraDisc<T>(ContextModule<T>::getContext(),
                                                 numEnergiesPerBlock);
}

template <typename T>
void ComputeOrderParameterDisc<T>::setOrderParameter(
    const OrderParameter<T> *delta_in) {
  if (delta_in) {
    if (_delta)
      delete _delta;
    _delta = new OrderParameter<T>(*delta_in);
  }
}

template <typename T>
void ComputeOrderParameterDisc<T>::initializeIteration(int numFields) {
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

template <typename T>
void ComputeOrderParameterDisc<T>::computePropertyFromGreens() {
  typedef ComputeProperty<T> C;

  if (C::doCompute()) {
    Parameters<T> *param = C::getParameters();
    const int Na = param->getAngularResolution();
    int Nc = _delta->getNumOrderParameterComponents();

    for (int i = 0; i < Nc; ++i) {
      // Compute new delta from anomalous greens function
      OrderParameterComponent<T> *component =
          _delta->getOrderParameterComponent(i);

      component->computeCouplingConstant();
      *(component->getOrderParameterField()) *= 1.0 / (T)Na;
    }

    // Compute residual
    OrderParameter<T> *deltaOld = ContextModule<T>::getOrderParameter();
    T residual = computeResidual(deltaOld);
    C::setResidual(residual);

    if (residual < param->getConvergenceCriterion())
      C::setConverged();

    // Update main order parameter
    *deltaOld = *_delta;

    // Copy results to grid with one field per order parameter component
    for (int i = 0; i < Nc; ++i)
      _delta->getOrderParameterComponentField(i)->copyFieldTo(_delta_fields, 0,
                                                              i);
  }
}

template <typename T>
T ComputeOrderParameterDisc<T>::computeResidual(OrderParameter<T> *old) const {
  T residual = 0.0;
  const int numDelta = _delta->getNumOrderParameterComponents();

  for (int i = 0; i < numDelta; ++i) {
    // Compute residual for each component
    GridGPU *deltaNew =
        _delta->getOrderParameterComponent(i)->getOrderParameterField();
    GridGPU *deltaOld =
        old->getOrderParameterComponent(i)->getOrderParameterField();

    residual +=
        GridGPU::differenceMagnitude(*deltaOld, *deltaNew) / sqrt(numDelta);
  }
  return residual;
}
#endif
