#ifndef CONTEXT_H_
#define CONTEXT_H_

#include "ComputeVectorPotential.h"
#include "ContextModule.h"
#include "Grid.h"
#include "Optimizer.h"
#include <chrono>
#include <iostream>
#include <memory>

namespace scg {
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

  /// \brief Set the optimizer.
  ///
  /// \param inOptimizer The optimizer as a unique pointer.
  ///
  /// \return Void.
  void set(std::unique_ptr<Optimizer<T>> &&inOptimizer) {
    m_optimizer = std::move(inOptimizer);
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

  T getResidual() const;

  T getComputeTime() const { return m_computeTime; }
  int getNumIterations() const { return m_numIterations; }
  int getNumComputeObjects() const { return m_computeObjectList.size(); }

  GridGPU<T> *getInducedVectorPotential() const {
    return m_inducedVectorPotential;
  }

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

  GridGPU<T> *m_inducedVectorPotential;
  ComputeVectorPotential<T> m_computeVectorPotential;

  // The optimizer. It is only used when computing the order parameter and/or
  // the vector potential.
  std::unique_ptr<Optimizer<T>> m_optimizer;

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
    : m_parameters(nullptr), m_geometry(nullptr), m_orderParameter(nullptr),
      m_riccatiSolver(nullptr), m_integrationIterator(nullptr),
      m_fermiSurface(nullptr), m_inducedVectorPotential(nullptr),
      m_computeCurrent(nullptr), m_computeCurrentDensity(nullptr),
      m_computeFreeEnergy(nullptr), m_computeOrderParameter(nullptr),
      m_computeLDOS(nullptr), m_computeTime(static_cast<T>(0.0)),
      m_numIterations(0), m_optimizer(std::make_unique<Optimizer<T>>()) {}

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

  if (m_fermiSurface) {
    delete m_fermiSurface;
    std::cout << "~Context(): Deleted FermiSurface\n";
  }

  if (m_inducedVectorPotential) {
    delete m_inducedVectorPotential;
    std::cout << "~Context(): Deleted vector potential\n";
  }

  deleteComputeObjects();

  GridCPU<float>::printAllocatedMemTotal();
  GridCPU<double>::printAllocatedMemTotal();
  GridCPU<int>::printAllocatedMemTotal();
  GridCPU<char>::printAllocatedMemTotal();

  GridGPU<float>::printAllocatedMemTotal();
  GridGPU<double>::printAllocatedMemTotal();
  GridGPU<int>::printAllocatedMemTotal();
  GridGPU<char>::printAllocatedMemTotal();
}

template <typename T> void Context<T>::deleteComputeObjects() {
  for (int i = 0; i < getNumComputeObjects(); ++i) {
    delete m_computeObjectList[i];
    std::cout << "~Context(): Deleted ComputeObject " << i << "\n";
  }
  m_computeObjectList.clear();

  if (m_integrationIterator) {
    delete m_integrationIterator;
    std::cout << "~Context(): Deleted IntegrationIterator\n";
  }
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
  if (getNumComputeObjects() > 0) {
    std::cout << "Context::initialize() --> Creating IntegrationIterator\n";
    // Note, this sets m_integrationIterator member.
    m_integrationIterator =
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

  // The induced vector potential.
  {
    std::cout << "Context::initialize() --> Initializing vector potential\n";
    // Initialize the stored vector potential.
    const int gridResolution = m_parameters->getGridResolutionBase();
    const int numSpatialDims = 2;
    GridGPU<T>::initializeGrid(m_inducedVectorPotential, gridResolution,
                               gridResolution, numSpatialDims, Type::real);
    m_inducedVectorPotential->setZero();

    // The physical side length of a lattice site.
    const T gridElementSize = m_parameters->getGridElementSize();

    // Initialize the vector-potential solver.
    m_computeVectorPotential.initialize(gridResolution, gridElementSize);

    ContextModule<T>::checkCudaError(
        "Context<T>::initialize(): vector potential");
  }

  m_numIterations = 0;

  std::cout << "\n\nContext::initialize() --> Initialization complete!\n";

  GridCPU<float>::printAllocatedMemTotal();
  GridCPU<double>::printAllocatedMemTotal();
  GridCPU<int>::printAllocatedMemTotal();
  GridCPU<char>::printAllocatedMemTotal();

  GridGPU<float>::printAllocatedMemTotal();
  GridGPU<double>::printAllocatedMemTotal();
  GridGPU<int>::printAllocatedMemTotal();
  GridGPU<char>::printAllocatedMemTotal();

  return 0;
}

template <typename T> void Context<T>::compute() {
  // Start timer.
  const auto startTime = std::chrono::high_resolution_clock::now();

  // Compute e.g. the order parameter.
  m_integrationIterator->compute();

  if (m_computeOrderParameter) {
    // The London penetration depth.
    const T penetrationDepth = m_parameters->getPenetrationDepth();

    // Compute the induced vector potential. It can only be done if the current
    // density is computed, and the penetration depth is finite.
    if (m_computeCurrent && (penetrationDepth > static_cast<T>(0))) {
      // Note that it is assumed that all values outside of the grain is already
      // set to zero.
      const GridGPU<T> currentDensity = *(m_computeCurrent->getResult());

      // Solve Maxwell's equation relating the vector potential to the current.
      m_computeVectorPotential.solve(currentDensity, penetrationDepth);

      // Compute the residual.
      m_computeVectorPotential.computeResidual(*m_inducedVectorPotential);
    }

    // Compute the next order parameter and vector potential, but do not update.
    m_optimizer->compute(*(m_computeOrderParameter->getOrderParameter()),
                         *(m_computeVectorPotential.getResult()),
                         *m_orderParameter, *m_inducedVectorPotential);

    // Update the order parameter and vector potential.
    // TODO(niclas): Should we really update when we have reached convergence?
    // The update might screw it up (slightly).
    m_optimizer->update(*m_orderParameter, *m_inducedVectorPotential);
  }

  // Stop timer.
  const auto stopTime = std::chrono::high_resolution_clock::now();

  // Count the number of microseconds.
  const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                            stopTime - startTime)
                            .count();

  // Convert to seconds.
  m_computeTime = static_cast<T>(duration) * static_cast<T>(1e-6);

  // Increase iteration counter.
  m_numIterations++;
}

template <typename T> T Context<T>::getResidual() const {
  // Get the convergence of the vector potential first. If the vector
  // potential is not computed, it will be zero.
  T maxResidual = m_computeVectorPotential.getResidual();

  // Get the convergence for all compute objects.
  const int numComputeObjects = getNumComputeObjects();
  for (int i = 0; i < numComputeObjects; ++i) {
    const T residual = m_computeObjectList[i]->getResidual();
    maxResidual = std::max(residual, maxResidual);
  }
  return maxResidual;
}

template <typename T> bool Context<T>::isConverged() const {
  // Check the convergence of the vector potential first. If the vector
  // potential is not computed, it will be converged.
  const T convergenceCriterion = m_parameters->getConvergenceCriterion();
  const T residual = m_computeVectorPotential.getResidual();
  bool success = residual < convergenceCriterion;

  // Check the convergence for all compute objects.
  const int numComputeObjects = getNumComputeObjects();
  for (int i = 0; i < numComputeObjects; ++i) {
    success &= m_computeObjectList[i]->isConverged();
  }
  return success;
}
} // namespace scg

#endif // CONTEXT_H_
