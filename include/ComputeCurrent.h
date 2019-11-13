#ifndef COMPUTECURRENT_H
#define COMPUTECURRENT_H

#include <ComputeProperty.h>
#include <IntegrationIteratorMatsubara.h>
#include <IntegrationIteratorOzaki.h>

template <typename T> class ComputeCurrent : public ComputeProperty<T> {
  typedef typename Grid<T>::GPU GridGPU;

public:
  // Constructors and Destructors
  ComputeCurrent();
  virtual ~ComputeCurrent();

  ComputeCurrent(Context<T> *ctx);

  virtual GridGPU *getResult() { return _current_real; }

  virtual int initialize();

  virtual IntegrationIterator<T> *
  createIntegrationIterator(int numEnergiesPerBlock = -1);

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
  virtual void integrateGreensAngle(GeometryGroup<T> *geom, const T angle);

  // Compute physical property from greens function.
  virtual void computePropertyFromGreens();

private:
  // Computed current will be stored here at end of
  // IntegrationIterator::compute() function.
  GridGPU *_current;
  GridGPU *_current_real;

  // Temporary grid.
  GridGPU *_tmp;
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
  std::cout << "ComputeCurrent<T>::initialize()\n";
  typedef ContextModule<T> C;

  // Get resolution
  const int N0 = C::getParameters()->getGridResolutionBase();
  const int N1 = C::getParameters()->getGridResolutionCoherence();

  GridGPU::initializeGrid(_current, N0, N0, 2);
  GridGPU::initializeGrid(_current_real, N0, N0, 2, Type::real);
  GridGPU::initializeGrid(_tmp, N1, N1);

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
  typedef ContextModule<T> C;

  if (ComputeProperty<T>::doCompute())
    _current->setZero();
}

template <typename T> void ComputeCurrent<T>::initializeGreensAngle() {
  if (ComputeProperty<T>::doCompute())
    ComputeProperty<T>::initializeGreensAngle(); // Sets to zero
}

template <typename T>
void ComputeCurrent<T>::computeGreensAngle(ParameterIterator<int> *iter,
                                           GridGPU *gamma, GridGPU *gammabar,
                                           GeometryGroup<T> *geom) {

  if (ComputeProperty<T>::doCompute()) {
    typedef ContextModule<T> C;
    IntegrationIterator<T> *ii = C::getIntegrationIterator();
    GridGPU *residues = ii->getResidues();

    GreensFunctionStatic<T, Greens_Current>::computeGreensSumEnergy(
        ComputeProperty<T>::getGreensCurrentAngle(), iter, residues, gamma,
        gammabar, geom);
  }
}

template <typename T>
void ComputeCurrent<T>::integrateGreensAngle(GeometryGroup<T> *geom,
                                             const T angle) {
  if (ComputeProperty<T>::doCompute()) {
    GridGPU *greens = ComputeProperty<T>::getGreensCurrentAngle();

    // Compensate for numerical diffusion from interpolation during rotation
    geom->extendValuesAtBoundary(greens, 1);

    // Add momentum dependence and rotate back to reference frame:
    // x-component
    *_tmp = *greens;
    *_tmp *= cos(angle + 0.5 * M_PI);

    _current->insertRotatedGridAdditive(_tmp, angle, 0);

    // y-component
    *_tmp = *greens;
    *_tmp *= sin(angle + 0.5 * M_PI);

    _current->insertRotatedGridAdditive(_tmp, angle, 1);
  }
}

template <typename T> void ComputeCurrent<T>::computePropertyFromGreens() {
  typedef ComputeProperty<T> C;

  if (C::doCompute()) {
    int numAngles = C::getParameters()->getAngularResolution();
    T T_Tc = C::getParameters()->getTemperature();

    GridGPU current_old = *_current_real;

    _current_real->insertGrid(_current);
    *_current_real *= T_Tc / (T)numAngles;

    T residual = GridGPU::differenceMagnitude(current_old, *_current_real);

    if (residual < C::getParameters()->getConvergenceCriterion())
      C::setConverged();
  }
}

#endif // COMPUTECURRENT_H
