#ifndef CONTEXT_H
#define CONTEXT_H

#include <ContextModule.h>
#include <util/csecond.c>

template <typename> class CoherenceFunctions;
template <typename> class ComputeCurrent;
template <typename> class ComputeCurrentDensity;
template <typename> class ComputeLDOS;
template <typename> class ComputeOrderParameter;
template <typename> class ComputeProperty;
template <typename> class Context;
template <typename> class GeometryGroup;
template <typename> class GradientAccelerator;
template <typename> class IntegrationIterator;
template <typename> class OrderParameter;
template <typename> class OrderParameterComponent;
template <typename> class Parameters;
template <typename> class RiccatiSolver;

// CONSIDER MAKING ENERGY ITERATOR A CONTEXT MODULE

template <typename T> class Context {
public:
  Context();
  ~Context();

  // Setters
  void set(Parameters<T> *param) { _param = param; }
  void set(GeometryGroup<T> *geom) { _geom = geom; }
  void set(OrderParameter<T> *delta) { _delta = delta; }
  void set(RiccatiSolver<T> *solver) { _solver = solver; }
  void set(IntegrationIterator<T> *iter) { _iter = iter; }
  void set(GradientAccelerator<T> *acc) { _acc = acc; }
  int add(ComputeOrderParameter<T> *compute) {
    _compute_delta = compute;
    return add(static_cast<ComputeProperty<T> *>(compute));
  }
  int add(ComputeCurrent<T> *compute) {
    _compute_current = compute;
    return add(static_cast<ComputeProperty<T> *>(compute));
  }
  int add(ComputeCurrentDensity<T> *compute) {
    _compute_current_density = compute;
    return add(static_cast<ComputeProperty<T> *>(compute));
  }
  int add(ComputeLDOS<T> *compute) {
    _compute_ldos = compute;
    return add(static_cast<ComputeProperty<T> *>(compute));
  }
  int add(ComputeProperty<T> *compute);

  // Functions
  int initialize(int numEnergiesPerBlock = -1);
  void compute();

  // Getters
  Parameters<T> *getParameters() const { return _param; }
  GeometryGroup<T> *getGeometry() const { return _geom; }
  OrderParameter<T> *getOrderParameter() const { return _delta; }
  OrderParameterComponent<T> *getOrderParameterComponent(int idx) const {
    return _delta->getOrderParameterComponent(idx);
  }
  RiccatiSolver<T> *getRiccatiSolver() const { return _solver; }
  IntegrationIterator<T> *getIntegrationIterator() const { return _iter; }
  GradientAccelerator<T> *getGradientAccelerator() const { return _acc; }

  int getNumComputeObjects() const { return _computeObjectList.size(); }
  ComputeProperty<T> *getComputeObject(int idx) const {
    return _computeObjectList[idx];
  }

  ComputeOrderParameter<T> *getComputeOrderParameter() const {
    return _compute_delta;
  }
  ComputeCurrent<T> *getComputeCurrent() const { return _compute_current; }
  ComputeCurrentDensity<T> *getComputeCurrentDensity() const {
    return _compute_current_density;
  }
  ComputeLDOS<T> *getComputeLDOS() const { return _compute_ldos; }

  T getResidual(int idx = 0) const;

  double getComputeTime() const { return _compute_time; }
  int getNumIterations() const { return _numIterations; }

  // Helpers
  int isConverged() const;

private:
  Parameters<T> *_param;
  GeometryGroup<T> *_geom;
  OrderParameter<T> *_delta;
  RiccatiSolver<T> *_solver;
  IntegrationIterator<T> *_iter;
  GradientAccelerator<T> *_acc;

  std::vector<ComputeProperty<T> *> _computeObjectList;

  double _compute_time;
  int _numIterations;

  // Helper pointers
  ComputeOrderParameter<T> *_compute_delta;
  ComputeCurrent<T> *_compute_current;
  ComputeCurrentDensity<T> *_compute_current_density;
  ComputeLDOS<T> *_compute_ldos;
};

template <typename T>
Context<T>::Context()
    : _param(0), _geom(0), _delta(0), _solver(0), _iter(0), _acc(0),
      _compute_current(0), _compute_delta(0), _compute_time(0.0),
      _numIterations(0) {}

template <typename T> Context<T>::~Context() {
  using namespace std;
  cout << "~Context()\n";

  if (_param) {
    delete _param;
    cout << "~Context(): Deleted Parameters\n";
  }

  if (_geom) {
    delete _geom;
    cout << "~Context(): Deleted Geometry\n";
  }
  if (_delta) {
    delete _delta;
    cout << "~Context(): Deleted OrderParameter\n";
  }
  if (_solver) {
    delete _solver;
    cout << "~Context(): Deleted RiccatiSolver\n";
  }
  if (_iter) {
    delete _iter;
    cout << "~Context(): Deleted IntegrationIterator\n";
  }
  if (_acc) {
    delete _acc;
    cout << "~Context(): Deleted GradientAccelerator\n";
  }

  for (int i = 0; i < getNumComputeObjects(); ++i) {
    delete _computeObjectList[i];
    cout << "~Context(): Deleted ComputeObject " << i << "\n";
  }

  Grid<float>::CPU::printAllocatedMemTotal();
  Grid<double>::CPU::printAllocatedMemTotal();
  Grid<int>::CPU::printAllocatedMemTotal();
  Grid<char>::CPU::printAllocatedMemTotal();

  Grid<float>::GPU::printAllocatedMemTotal();
  Grid<double>::GPU::printAllocatedMemTotal();
  Grid<int>::GPU::printAllocatedMemTotal();
  Grid<char>::GPU::printAllocatedMemTotal();
}

template <typename T> int Context<T>::add(ComputeProperty<T> *property) {
  _computeObjectList.push_back(property);

  return getNumComputeObjects();
}

template <typename T> int Context<T>::initialize(int numEnergiesPerBlock) {
  using namespace std;

  if (_iter == 0 && getNumComputeObjects() > 0) {
    cout << "Context::initialize() --> Creating IntegrationIterator\n";
    _computeObjectList[0]->createIntegrationIterator(
        numEnergiesPerBlock); // sets _iter member
    ContextModule<T>::checkCudaError(
        "Context<T>::initialize(): createIntegrationIterator");
  }

  if (_geom) {
    cout << "Context::initialize() --> Initializing Geometry\n";
    _geom->initialize();
    ContextModule<T>::checkCudaError("Context<T>::initialize(): GeometryGroup");
  }

  if (_delta) {
    cout << "Context::initialize() --> Initializing OrderParameter\n";
    _delta->initialize();
    ContextModule<T>::checkCudaError(
        "Context<T>::initialize(): OrderParameter");
  }

  if (_iter) {
    cout << "Context::initialize() --> Initializing IntegrationIterator\n";
    _iter->initialize();
    ContextModule<T>::checkCudaError(
        "Context<T>::initialize(): IntegrationIterator");
  }

  // RiccatiSolver requires IntegrationIterator to be initialized first
  if (_solver) {
    cout << "Context::initialize() --> Initializing RiccatiSolver\n";
    _solver->initialize();
    ContextModule<T>::checkCudaError("Context<T>::initialize(): RiccatiSolver");
  }

  if (_acc) {
    cout << "Context::initialize() --> Initializing GradientAccelerator\n";
    _acc->initialize();
    ContextModule<T>::checkCudaError(
        "Context<T>::initialize(): GradientAccelerator");
  }

  for (int i = 0; i < getNumComputeObjects(); ++i) {
    cout << "Context::initialize() --> Initializing ComputeObject " << i
         << ":\n";
    _computeObjectList[i]->initialize();
    ContextModule<T>::checkCudaError(
        "Context<T>::initialize(): ComputeProperty");
  }

  _numIterations = 0;

  cout << "\n\nContext::initialize() --> Initialization complete!\n";

  Grid<float>::CPU::printAllocatedMemTotal();
  Grid<double>::CPU::printAllocatedMemTotal();
  Grid<int>::CPU::printAllocatedMemTotal();
  Grid<char>::CPU::printAllocatedMemTotal();

  Grid<float>::GPU::printAllocatedMemTotal();
  Grid<double>::GPU::printAllocatedMemTotal();
  Grid<int>::GPU::printAllocatedMemTotal();
  Grid<char>::GPU::printAllocatedMemTotal();

  return 0;
}

template <typename T> void Context<T>::compute() {
  _compute_time = csecond();

  _iter->compute();

  // Only use accelerator if we are computing order parameter
  ComputeOrderParameter<T> *compute_delta = getComputeOrderParameter();
  if (_acc && compute_delta) {
    _acc->step(compute_delta->getResidual(),
               _delta->getOrderParameterComponent(0)->getOrderParameterField());
  }

  _compute_time = csecond() - _compute_time;
  _numIterations++;
}

template <typename T> T Context<T>::getResidual(int idx) const {
  if (idx <= getNumComputeObjects())
    return _computeObjectList[idx]->getResidual();
  else
    return 1.0;
}

template <typename T> int Context<T>::isConverged() const {
  // If we are computing order parameter self-consistently, use it to
  // determine convergence.
  if (getComputeOrderParameter()) {
    if (_acc)
      return _acc->isConverged();
    else
      return getComputeOrderParameter()->isConverged();
  } else
  // If no order parameter, use first compute object to determine convergence
  {
    return getComputeObject(0)->isConverged();
  }
}

#endif // CONTEXT_H
