//===--------------------- Context.h - Context class. ---------------------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Class for the context object, containing pointers and storage for most
/// objects during a simulation.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_CONTEXT_H_
#define CONGA_CONTEXT_H_

#include "Accelerator.h"
#include "ContextModule.h"
#include "compute/ComputeVectorPotential.h"
#include "fermi_surface/FermiSurface.h"
#include "fermi_surface/FermiSurfaceCircle.h"
#include "grid.h"

#include <chrono>
#include <iostream>
#include <memory>

namespace conga {
template <typename> class ComputeCurrent;
template <typename> class ComputeSpectralCurrent;
template <typename> class ComputeFreeEnergy;
template <typename> class ComputeLDOS;
template <typename> class ComputeOrderParameter;
template <typename> class ComputeProperty;
template <typename> class Geometry;
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
  void set(std::unique_ptr<FermiSurface<T>> inFermiSurface) {
    m_fermiSurface = std::move(inFermiSurface);
  }
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

  /// \brief Set the induced vector-potential.
  ///
  /// \param inInducedVectorPotential The induced vector-potential as a unique
  /// pointer.
  ///
  /// \return Void.
  // TODO(niclas): This is just grid. Perhaps too generic.
  void set(std::unique_ptr<GridGPU<T>> inInducedVectorPotential) {
    m_inducedVectorPotential = std::move(inInducedVectorPotential);
  }

  /// \brief Set the accelerator.
  ///
  /// \param inAccelerator The accelerator as a unique pointer.
  ///
  /// \return Void.
  void set(std::unique_ptr<Accelerator<T>> inAccelerator) {
    m_accelerator = std::move(inAccelerator);
  }

  // Adders of compute objects.
  void add(ComputeCurrent<T> *inComputeCurrent);
  void add(ComputeSpectralCurrent<T> *inComputeSpectralCurrent);
  void add(ComputeFreeEnergy<T> *inComputeFreeEnergy);
  void add(ComputeLDOS<T> *inComputeLDOS);
  void add(ComputeOrderParameter<T> *inComputeOrderParameter);
  void
  add(std::unique_ptr<ComputeVectorPotential<T>> inComputeVectorPotential) {
    m_computeVectorPotential = std::move(inComputeVectorPotential);
  }
  void deleteComputeObjects();

  // Functions
  int initialize();
  void compute();

  // Getters of non-compute objects.
  FermiSurface<T> *getFermiSurface() const { return m_fermiSurface.get(); }
  GeometryGroup<T> *getGeometry() const { return m_geometry; }
  IntegrationIterator<T> *getIntegrationIterator() const {
    return m_integrationIterator;
  }
  OrderParameter<T> *getOrderParameter() const { return m_orderParameter; }
  OrderParameterComponent<T> *getOrderParameterComponent(int idx) const {
    return m_orderParameter->getComponent(idx);
  }
  Parameters<T> *getParameters() const { return m_parameters; }
  RiccatiSolver<T> *getRiccatiSolver() const { return m_riccatiSolver; }

  // Getters of compute objects.
  ComputeCurrent<T> *getComputeCurrent() const { return m_computeCurrent; }
  ComputeSpectralCurrent<T> *getComputeSpectralCurrent() const {
    return m_ComputeSpectralCurrent;
  }
  ComputeFreeEnergy<T> *getComputeFreeEnergy() const {
    return m_computeFreeEnergy;
  }
  ComputeLDOS<T> *getComputeLDOS() const { return m_computeLDOS; }
  ComputeOrderParameter<T> *getComputeOrderParameter() const {
    return m_computeOrderParameter;
  }
  ComputeVectorPotential<T> *getComputeVectorPotential() const {
    return m_computeVectorPotential.get();
  }
  ComputeProperty<T> *getComputeObject(int idx) const {
    return m_computeObjectList[idx];
  }

  T getResidual() const;

  T getComputeTime() const { return m_computeTime; }
  int getNumIterations() const { return m_numIterations; }
  int getNumComputeObjects() const { return m_computeObjectList.size(); }

  GridGPU<T> *getInducedVectorPotential() const {
    return m_inducedVectorPotential.get();
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
  std::unique_ptr<FermiSurface<T>> m_fermiSurface;

  std::unique_ptr<GridGPU<T>> m_inducedVectorPotential;
  std::unique_ptr<ComputeVectorPotential<T>> m_computeVectorPotential;

  // The accelerator. It is only used when computing the order parameter and/or
  // the vector potential.
  std::unique_ptr<Accelerator<T>> m_accelerator;

  std::vector<ComputeProperty<T> *> m_computeObjectList;

  T m_computeTime;
  int m_numIterations;

  // Helper pointers
  ComputeCurrent<T> *m_computeCurrent;
  ComputeSpectralCurrent<T> *m_ComputeSpectralCurrent;
  ComputeFreeEnergy<T> *m_computeFreeEnergy;
  ComputeLDOS<T> *m_computeLDOS;
  ComputeOrderParameter<T> *m_computeOrderParameter;
};

template <typename T>
Context<T>::Context()
    : m_parameters(nullptr), m_geometry(nullptr), m_orderParameter(nullptr),
      m_riccatiSolver(nullptr), m_integrationIterator(nullptr),
      m_fermiSurface(std::make_unique<FermiSurfaceCircle<T>>()),
      m_inducedVectorPotential(nullptr), m_computeCurrent(nullptr),
      m_ComputeSpectralCurrent(nullptr), m_computeFreeEnergy(nullptr),
      m_computeOrderParameter(nullptr), m_computeLDOS(nullptr),
      m_computeVectorPotential(nullptr), m_computeTime(static_cast<T>(0.0)),
      m_numIterations(0), m_accelerator(std::make_unique<Accelerator<T>>()) {}

template <typename T> Context<T>::~Context() {
#ifndef NDEBUG
  std::cout << "~Context()\n";
#endif

  if (m_parameters) {
    delete m_parameters;
#ifndef NDEBUG
    std::cout << "~Context(): Deleted Parameters\n";
#endif
  }

  if (m_geometry) {
    delete m_geometry;
#ifndef NDEBUG
    std::cout << "~Context(): Deleted Geometry\n";
#endif
  }

  if (m_orderParameter) {
    delete m_orderParameter;
#ifndef NDEBUG
    std::cout << "~Context(): Deleted OrderParameter\n";
#endif
  }

  if (m_riccatiSolver) {
    delete m_riccatiSolver;
#ifndef NDEBUG
    std::cout << "~Context(): Deleted RiccatiSolver\n";
#endif
  }

  deleteComputeObjects();
}

template <typename T> void Context<T>::deleteComputeObjects() {
  for (int i = 0; i < getNumComputeObjects(); ++i) {
    delete m_computeObjectList[i];
#ifndef NDEBUG
    std::cout << "~Context(): Deleted ComputeObject " << i << "\n";
#endif
  }
  m_computeObjectList.clear();

  if (m_integrationIterator) {
    delete m_integrationIterator;
#ifndef NDEBUG
    std::cout << "~Context(): Deleted IntegrationIterator\n";
#endif
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
void Context<T>::add(ComputeSpectralCurrent<T> *inComputeSpectralCurrent) {
  m_ComputeSpectralCurrent = inComputeSpectralCurrent;
  add(static_cast<ComputeProperty<T> *>(inComputeSpectralCurrent));
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

template <typename T> int Context<T>::initialize() {
  if (m_geometry) {
#ifndef NDEBUG
    std::cout << "Context::initialize() --> Initializing Geometry\n";
#endif
    m_geometry->initialize();
    ContextModule<T>::checkCudaError("Context<T>::initialize(): Geometry");
  }

  if (m_fermiSurface) {
#ifndef NDEBUG
    std::cout << "Context::initialize() --> Initializing FermiSurface\n";
#endif
    const int numAngles = m_parameters->getAngularResolution();
    m_fermiSurface->initialize(numAngles);
    ContextModule<T>::checkCudaError("Context<T>::initialize(): FermiSurface");
  }

  if (m_orderParameter) {
#ifndef NDEBUG
    std::cout << "Context::initialize() --> Initializing OrderParameter\n";
#endif
    m_orderParameter->initialize();
    ContextModule<T>::checkCudaError(
        "Context<T>::initialize(): OrderParameter");
  }

  if (m_integrationIterator) {
#ifndef NDEBUG
    std::cout << "Context::initialize() --> Initializing IntegrationIterator\n";
#endif
    m_integrationIterator->initialize();
    ContextModule<T>::checkCudaError(
        "Context<T>::initialize(): IntegrationIterator");
  }

  // RiccatiSolver requires IntegrationIterator to be initialized first
  if (m_riccatiSolver) {
#ifndef NDEBUG
    std::cout << "Context::initialize() --> Initializing RiccatiSolver\n";
#endif
    m_riccatiSolver->initialize();
    ContextModule<T>::checkCudaError("Context<T>::initialize(): RiccatiSolver");
  }

  for (int i = 0; i < getNumComputeObjects(); ++i) {
#ifndef NDEBUG
    std::cout << "Context::initialize() --> Initializing ComputeObject " << i
              << ":\n";
#endif
    m_computeObjectList[i]->initialize();
    ContextModule<T>::checkCudaError(
        "Context<T>::initialize(): ComputeProperty");
  }

  // The induced vector potential.
  {
#ifndef NDEBUG
    std::cout << "Context::initialize() --> Initializing vector potential\n";
#endif
    const int gridResolution = m_parameters->getGridResolutionBase();

    if (!m_inducedVectorPotential.get()) {
      // Initialize the stored vector potential.
      const int numSpatialDims = 2;
      m_inducedVectorPotential = std::make_unique<GridGPU<T>>(
          gridResolution, gridResolution, numSpatialDims, Type::real);
      m_inducedVectorPotential->setZero();
    }

    if (m_computeVectorPotential) {
      // The physical side length of a lattice site.
      const T gridElementSize = m_parameters->getGridElementSize();

      // The number of lattice site in the grain is needed for enforcing current
      // conservation before computing the vector potential.
      const int numGrainLatticeSites = m_geometry->getNumGrainLatticeSites();

      // Initialize the vector-potential solver.
      m_computeVectorPotential->initialize(gridResolution, gridElementSize,
                                           numGrainLatticeSites);
    }

    ContextModule<T>::checkCudaError(
        "Context<T>::initialize(): vector potential");
  }

  m_numIterations = 0;

#ifndef NDEBUG
  std::cout << "\n\nContext::initialize() --> Initialization complete!\n";

  GridCPU<float>::printAllocatedMemTotal();
  GridCPU<double>::printAllocatedMemTotal();
  GridCPU<int>::printAllocatedMemTotal();
  GridCPU<char>::printAllocatedMemTotal();

  GridGPU<float>::printAllocatedMemTotal();
  GridGPU<double>::printAllocatedMemTotal();
  GridGPU<int>::printAllocatedMemTotal();
  GridGPU<char>::printAllocatedMemTotal();
#endif

  return 0;
}

template <typename T> void Context<T>::compute() {
  // Start timer.
  const auto startTime = std::chrono::high_resolution_clock::now();

  // Compute e.g. the order parameter.
  m_integrationIterator->compute();

  // The London penetration depth.
  const T penetrationDepth = m_parameters->getPenetrationDepth();

  // Compute the induced vector potential. It can only be done if the current
  // density is computed, and the penetration depth is finite.
  const bool computeVectorPotential = m_computeVectorPotential &&
                                      m_computeCurrent &&
                                      (penetrationDepth > static_cast<T>(0));
  if (computeVectorPotential) {
    // Note that it is assumed that all values outside of the grain is already
    // set to zero.
    const GridGPU<T> currentDensity = *(m_computeCurrent->getResult());

    // Solve Maxwell's equation relating the vector potential to the current.
    m_computeVectorPotential->solve(currentDensity, penetrationDepth);

    // Compute the residual.
    m_computeVectorPotential->computeResidual(*m_inducedVectorPotential);
  }

  if (m_computeOrderParameter || computeVectorPotential) {
    // Compute the next order parameter and vector potential, but do not
    // update.
    if (m_computeOrderParameter && computeVectorPotential) {
      m_accelerator->compute(*m_orderParameter, *m_inducedVectorPotential,
                             *(m_computeOrderParameter->getOrderParameter()),
                             *(m_computeVectorPotential->getResult()));
    } else if (m_computeOrderParameter && !computeVectorPotential) {
      m_accelerator->compute(*m_orderParameter, *m_inducedVectorPotential,
                             *(m_computeOrderParameter->getOrderParameter()),
                             *m_inducedVectorPotential);
    } else if (!m_computeOrderParameter && computeVectorPotential) {
      m_accelerator->compute(*m_orderParameter, *m_inducedVectorPotential,
                             *m_orderParameter,
                             *(m_computeVectorPotential->getResult()));
    }

    // Update the order parameter and vector potential.
    // TODO(niclas): Should we really update when we have reached convergence?
    // The update might screw it up (slightly).
    m_accelerator->update(*m_orderParameter, *m_inducedVectorPotential);
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
  T maxResidual = static_cast<T>(0);

  // Get the convergence of the vector potential first.
  if (m_computeVectorPotential) {
    maxResidual = m_computeVectorPotential->getResidual();
  }

  // Get the convergence for all compute objects.
  const int numComputeObjects = getNumComputeObjects();
  for (int i = 0; i < numComputeObjects; ++i) {
    const T residual = m_computeObjectList[i]->getResidual();
    maxResidual = std::max(residual, maxResidual);
  }
  return maxResidual;
}

template <typename T> bool Context<T>::isConverged() const {
  bool success = true;

  // Check the convergence of the vector potential first.
  if (m_computeVectorPotential) {
    const T convergenceCriterion = m_parameters->getConvergenceCriterion();
    const T residual = m_computeVectorPotential->getResidual();
    success = residual < convergenceCriterion;
  }

  // Check the convergence for all compute objects.
  const int numComputeObjects = getNumComputeObjects();
  for (int i = 0; i < numComputeObjects; ++i) {
    success &= m_computeObjectList[i]->isConverged();
  }
  return success;
}
} // namespace conga

#endif // CONGA_CONTEXT_H_
