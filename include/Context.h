#ifndef CONTEXT_H_
#define CONTEXT_H_

#include "ContextModule.h"
#include "Grid.h"
#include <chrono>
#include <iostream>

template <typename> class CoherenceFunctions;
template <typename> class ComputeCurrent;
template <typename> class ComputeCurrentDensity;
template <typename> class ComputeFreeEnergy;
template <typename> class ComputeLDOS;
template <typename> class ComputeOrderParameter;
template <typename> class ComputeProperty;
template <typename> class Context;
template <typename> class FermiSurface;
template <typename> class GeometryGroup;
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

  // Setters non-compute objects.
  void set(FermiSurface<T> *inFermiSurface) { m_fermiSurface = inFermiSurface; }
  void set(GeometryGroup<T> *inGeometry) { m_geometry = inGeometry; }
  void set(IntegrationIterator<T> *inIntegrationIterator) {
    m_integrationIterator = inIntegrationIterator;
  }
  void set(OrderParameter<T> *inOrderParameter) {
    m_orderParameter = inOrderParameter;
  }
  void set(Parameters<T> *inParameters) { m_parameters = inParameters; }
  void set(RiccatiSolver<T> *inRiccatiSolver) {
    m_riccatiSolver = inRiccatiSolver;
  }

  // Adders of compute objects.
  void add(ComputeCurrent<T> *inComputeCurrent);
  void add(ComputeCurrentDensity<T> *inComputeCurrentDensity);
  void add(ComputeFreeEnergy<T> *inComputeFreeEnergy);
  void add(ComputeLDOS<T> *inComputeLDOS);
  void add(ComputeOrderParameter<T> *inComputeOrderParameter);
  void deleteComputeObjects();

  // Functions
  int initialize(int numEnergiesPerBlock = -1);
  void compute();

  // Getters of non-compute objects.
  FermiSurface<T> *getFermiSurface() const { return m_fermiSurface; }
  GeometryGroup<T> *getGeometry() const { return m_geometry; }
  IntegrationIterator<T> *getIntegrationIterator() const {
    return m_integrationIterator;
  }
  OrderParameter<T> *getOrderParameter() const { return m_orderParameter; }
  OrderParameterComponent<T> *getOrderParameterComponent(int idx) const {
    return m_orderParameter->getOrderParameterComponent(idx);
  }
  Parameters<T> *getParameters() const { return m_parameters; }
  RiccatiSolver<T> *getRiccatiSolver() const { return m_riccatiSolver; }

  // Getters of compute objects.
  ComputeCurrent<T> *getComputeCurrent() const { return m_computeCurrent; }
  ComputeCurrentDensity<T> *getComputeCurrentDensity() const {
    return m_computeCurrentDensity;
  }
  ComputeFreeEnergy<T> *getComputeFreeEnergy() const {
    return m_computeFreeEnergy;
  }
  ComputeLDOS<T> *getComputeLDOS() const { return m_computeLDOS; }
  ComputeOrderParameter<T> *getComputeOrderParameter() const {
    return m_computeOrderParameter;
  }
  ComputeProperty<T> *getComputeObject(int idx) const {
    return m_computeObjectList[idx];
  }

  T getResidual(const int inIdx) const;

  T getComputeTime() const { return m_computeTime; }
  int getNumIterations() const { return m_numIterations; }
  int getNumComputeObjects() const { return m_computeObjectList.size(); }

  // Helpers
  bool isConverged() const;

private:
  void add(ComputeProperty<T> *inComputeProperty);

  Parameters<T> *m_parameters;
  GeometryGroup<T> *m_geometry;
  OrderParameter<T> *m_orderParameter;
  RiccatiSolver<T> *m_riccatiSolver;
  IntegrationIterator<T> *m_integrationIterator;
  FermiSurface<T> *m_fermiSurface;

  std::vector<ComputeProperty<T> *> m_computeObjectList;

  T m_computeTime;
  int m_numIterations;

  // Helper pointers
  ComputeCurrent<T> *m_computeCurrent;
  ComputeCurrentDensity<T> *m_computeCurrentDensity;
  ComputeFreeEnergy<T> *m_computeFreeEnergy;
  ComputeLDOS<T> *m_computeLDOS;
  ComputeOrderParameter<T> *m_computeOrderParameter;
};

template <typename T>
Context<T>::Context()
    : m_parameters(0), m_geometry(0), m_orderParameter(0), m_riccatiSolver(0),
      m_integrationIterator(0), m_computeCurrent(0), m_computeOrderParameter(0),
      m_computeTime(0.0), m_numIterations(0) {}

template <typename T> Context<T>::~Context() {
  std::cout << "~Context()\n";

  if (m_parameters) {
    delete m_parameters;
    std::cout << "~Context(): Deleted Parameters\n";
  }

  if (m_geometry) {
    delete m_geometry;
    std::cout << "~Context(): Deleted Geometry\n";
  }

  if (m_orderParameter) {
    delete m_orderParameter;
    std::cout << "~Context(): Deleted OrderParameter\n";
  }

  if (m_riccatiSolver) {
    delete m_riccatiSolver;
    std::cout << "~Context(): Deleted RiccatiSolver\n";
  }

  if (m_integrationIterator) {
    delete m_integrationIterator;
    std::cout << "~Context(): Deleted IntegrationIterator\n";
  }

  if (m_fermiSurface) {
    delete m_fermiSurface;
    std::cout << "~Context(): Deleted FermiSurface\n";
  }

  deleteComputeObjects();

  Grid<float>::CPU::printAllocatedMemTotal();
  Grid<double>::CPU::printAllocatedMemTotal();
  Grid<int>::CPU::printAllocatedMemTotal();
  Grid<char>::CPU::printAllocatedMemTotal();

  Grid<float>::GPU::printAllocatedMemTotal();
  Grid<double>::GPU::printAllocatedMemTotal();
  Grid<int>::GPU::printAllocatedMemTotal();
  Grid<char>::GPU::printAllocatedMemTotal();
}

template <typename T> void Context<T>::deleteComputeObjects() {
  for (int i = 0; i < getNumComputeObjects(); ++i) {
    delete m_computeObjectList[i];
    std::cout << "~Context(): Deleted ComputeObject " << i << "\n";
  }
  m_computeObjectList.clear();
}

template <typename T>
void Context<T>::add(ComputeProperty<T> *inComputeProperty) {
  m_computeObjectList.push_back(inComputeProperty);
}

template <typename T>
void Context<T>::add(ComputeOrderParameter<T> *inComputeOrderParameter) {
  m_computeOrderParameter = inComputeOrderParameter;
  add(static_cast<ComputeProperty<T> *>(inComputeOrderParameter));
}

template <typename T>
void Context<T>::add(ComputeCurrent<T> *inComputeCurrent) {
  m_computeCurrent = inComputeCurrent;
  add(static_cast<ComputeProperty<T> *>(inComputeCurrent));
}

template <typename T>
void Context<T>::add(ComputeCurrentDensity<T> *inComputeCurrentDensity) {
  m_computeCurrentDensity = inComputeCurrentDensity;
  add(static_cast<ComputeProperty<T> *>(inComputeCurrentDensity));
}

template <typename T>
void Context<T>::add(ComputeFreeEnergy<T> *inComputeFreeEnergy) {
  m_computeFreeEnergy = inComputeFreeEnergy;
  add(static_cast<ComputeProperty<T> *>(inComputeFreeEnergy));
}

template <typename T> void Context<T>::add(ComputeLDOS<T> *inComputeLDOS) {
  m_computeLDOS = inComputeLDOS;
  add(static_cast<ComputeProperty<T> *>(inComputeLDOS));
}

template <typename T> int Context<T>::initialize(int numEnergiesPerBlock) {
  if (m_integrationIterator == 0 && getNumComputeObjects() > 0) {
    std::cout << "Context::initialize() --> Creating IntegrationIterator\n";
    // Note, this sets m_integrationIterator member.
    m_computeObjectList[0]->createIntegrationIterator(numEnergiesPerBlock);
    ContextModule<T>::checkCudaError(
        "Context<T>::initialize(): createIntegrationIterator");
  }

  if (m_geometry) {
    std::cout << "Context::initialize() --> Initializing Geometry\n";
    m_geometry->initialize();
    ContextModule<T>::checkCudaError("Context<T>::initialize(): GeometryGroup");
  }

  if (m_fermiSurface) {
    std::cout << "Context::initialize() --> Initializing FermiSurface\n";
    const int numAngles = m_parameters->getAngularResolution();
    m_fermiSurface->initialize(numAngles);
    ContextModule<T>::checkCudaError("Context<T>::initialize(): FermiSurface");
  }

  if (m_orderParameter) {
    std::cout << "Context::initialize() --> Initializing OrderParameter\n";
    m_orderParameter->initialize();
    ContextModule<T>::checkCudaError(
        "Context<T>::initialize(): OrderParameter");
  }

  if (m_integrationIterator) {
    std::cout << "Context::initialize() --> Initializing IntegrationIterator\n";
    m_integrationIterator->initialize();
    ContextModule<T>::checkCudaError(
        "Context<T>::initialize(): IntegrationIterator");
  }

  // RiccatiSolver requires IntegrationIterator to be initialized first
  if (m_riccatiSolver) {
    std::cout << "Context::initialize() --> Initializing RiccatiSolver\n";
    m_riccatiSolver->initialize();
    ContextModule<T>::checkCudaError("Context<T>::initialize(): RiccatiSolver");
  }

  for (int i = 0; i < getNumComputeObjects(); ++i) {
    std::cout << "Context::initialize() --> Initializing ComputeObject " << i
              << ":\n";
    m_computeObjectList[i]->initialize();
    ContextModule<T>::checkCudaError(
        "Context<T>::initialize(): ComputeProperty");
  }

  m_numIterations = 0;

  std::cout << "\n\nContext::initialize() --> Initialization complete!\n";

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
  const auto startTime = std::chrono::high_resolution_clock::now();
  m_integrationIterator->compute();
  const auto stopTime = std::chrono::high_resolution_clock::now();

  // Count the number of milliseconds.
  const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                            stopTime - startTime)
                            .count();

  // Convert to seconds.
  m_computeTime = static_cast<T>(duration) / static_cast<T>(1e6);

  m_numIterations++;
}

template <typename T> T Context<T>::getResidual(int idx) const {
  if (idx < getNumComputeObjects()) {
    return m_computeObjectList[idx]->getResidual();
  } else {
    // TODO(niclas): Is this behavior really what the user expects?!
    return static_cast<T>(1.0);
  }
}

template <typename T> bool Context<T>::isConverged() const {
  if (getComputeOrderParameter()) {
    // If we are computing order parameter self-consistently, use it to
    // determine convergence.
    return getComputeOrderParameter()->isConverged();
  } else {
    // If no order parameter, use first compute object to determine convergence
    return getComputeObject(0)->isConverged();
  }
}

#endif // CONTEXT_H_
