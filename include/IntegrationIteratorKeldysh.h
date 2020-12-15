#ifndef INTEGRATION_ITERATOR_KELDYSH_H_
#define INTEGRATION_ITERATOR_KELDYSH_H_

#include "ComputeProperty.h"
#include "Context.h"
#include "GeometryGroup.h"
#include "OrderParameter.h"
#include "ParameterIterator.h"
#include "RiccatiSolver.h"
#include "utils.h"

namespace scg {
template <typename T>
class IntegrationIteratorKeldysh : public IntegrationIterator<T> {
  typedef typename Grid<T>::GPU GridGPU;
  typedef typename Grid<T>::CPU GridCPU;

public:
  IntegrationIteratorKeldysh();
  ~IntegrationIteratorKeldysh();

  IntegrationIteratorKeldysh(Context<T> *ctx, int numEnergiesPerBlock = -1);

  int initialize();

  // Configures energy iterators
  void initIterators(int numEnergiesPerBlock);
  void initEnergies();

  int getNumEnergiesInterval();
  int getEnergyIdxOffset_boundaryStorage();

  // Call this once before starting self consistent iteration, or when changing
  // initial condition.
  void setBoundaryInitialCondition();

  // Computes all properties added with addComputeProperty(). Iterate this self
  // consistently.
  void compute();

private:
  int m_lastEnergyIndex;
};

template <typename T>
IntegrationIteratorKeldysh<T>::IntegrationIteratorKeldysh()
    : IntegrationIterator<T>() {}

template <typename T>
IntegrationIteratorKeldysh<T>::IntegrationIteratorKeldysh(
    Context<T> *ctx, int numEnergiesPerBlock)
    : IntegrationIterator<T>(ctx, numEnergiesPerBlock) {
  std::cout << "IntegrationIteratorKeldysh<T>()\n";
  ctx->set(this);

  m_lastEnergyIndex = -1;
}

template <typename T>
IntegrationIteratorKeldysh<T>::~IntegrationIteratorKeldysh() {
  std::cout << "~IntegrationIteratorKeldysh()\n";
}

template <typename T> int IntegrationIteratorKeldysh<T>::initialize() {
  std::cout << "IntegrationIteratorKeldysh::initialize()\n";
  IntegrationIterator<T>::initialize();

  initIterators(IntegrationIterator<T>::getNumEnergiesPerBlock());
  return 0;
}

template <typename T> void IntegrationIteratorKeldysh<T>::initEnergies() {
  std::cout << "IntegrationIteratorKeldysh<T>::initEnergies()\n";

  const Parameters<T> *const param = ContextModule<T>::getParameters();

  const T eMin = param->getEnergyMin();
  const T eMax = param->getEnergyMax();
  const T smearingFactor = param->getSmearingFactor();

  const int numEnergies = param->getNumEnergies();

  std::cout << "eMin = " << eMin << ", eMax = " << eMax
            << ", numE = " << numEnergies << "\n";

  // Generate energy array
  GridCPU energies_cpu(numEnergies, 1);

  T *const energies_re = energies_cpu.getDataPointer(0, Type::real);
  T *const energies_im = energies_cpu.getDataPointer(0, Type::imag);

  const bool multipleEnergies = numEnergies > 1;
  const T step = multipleEnergies
                     ? (eMax - eMin) / static_cast<T>(numEnergies - 1)
                     : static_cast<T>(0.0);
  for (int i = 0; i < numEnergies; ++i) {
    energies_re[i] = eMin + step * static_cast<T>(i);
    energies_im[i] = smearingFactor;
  }

  GridGPU *energies = new GridGPU(energies_cpu);

  IntegrationIterator<T>::setEnergies(energies);
}

template <typename T>
void IntegrationIteratorKeldysh<T>::initIterators(int numEnergiesPerBlock) {
  std::cout << "IntegrationIteratorKeldysh<T>::initIterators()\n";

  typedef IntegrationIterator<T> I;

  initEnergies();

  int numEnergies = I::getEnergies()->dimX();

  ParameterIterator<int> *iter = new ParameterIterator<int>(0, numEnergies);
  I::setEnergyIterator(iter);

  std::cout << "Initializing keldysh iterator with Ne = " << numEnergies
            << ", NeBlock = " << numEnergiesPerBlock << "\n";

  if (numEnergiesPerBlock == -1) {
    // TODO(niclas): This should be default parameter?
    iter->setIncrement(64);
  } else {
    iter->setIncrement(numEnergiesPerBlock);
  }

  for (iter->begin(); !iter->isDone(); iter->next()) {
    std::cout << "min = " << iter->getMin() << "\n";
    std::cout << "max = " << iter->getMax() << "\n";
    std::cout << "inc = " << iter->getIncrement() << "\n";
    std::cout << "num = " << iter->getNumInterval() << "\n\n";
  }

  iter->begin();
}

template <typename T>
int IntegrationIteratorKeldysh<T>::getNumEnergiesInterval() {
  return IntegrationIterator<T>::getEnergyIterator()->getNumInterval();
}

template <typename T>
int IntegrationIteratorKeldysh<T>::getEnergyIdxOffset_boundaryStorage() {
  return 0;
}

template <typename T>
void IntegrationIteratorKeldysh<T>::setBoundaryInitialCondition() {
  std::cout << "IntegrationIteratorKeldysh<T>::setBoundaryInitialCondition()\n";

  typedef IntegrationIterator<T> I;
  typedef ContextModule<T> C;

  ParameterIterator<int> *energyIterator = I::getEnergyIterator();
  GridGPU *energies = I::getEnergies();

  RiccatiSolver<T> *riccatiSolver = C::getRiccatiSolver();
  GeometryGroup<T> *geometry = C::getGeometry();
  OrderParameter<T> *orderParameter = C::getOrderParameter();

  if (!energyIterator->isDone()) {
    // Do angle loop (integrate over momentum angles)
    const int numAngles = C::getParameters()->getAngularResolution();
    for (int angleIdx = 0; angleIdx < numAngles; ++angleIdx) {
      // Rotate order parameter field to current angle
      GridGPU *orderParameterRotated =
          orderParameter->getOrderParameterFieldRotated(-angleIdx);

      // Angle (in radians) of system in local frame
      const T systemAngle = C::getFermiSurface()->getSystemAngle(angleIdx);

      // Create discrete geometry data for current angle
      geometry->create(-systemAngle);

      // Compute coherence functions and store on boundary as initial condition
      riccatiSolver->computeInitialCondition(*energyIterator, energies,
                                             orderParameterRotated, angleIdx);
    }
  }

  ContextModule<T>::checkCudaError(
      "IntegrationIteratorKeldysh<T>::setBoundaryInitialCondition()");
}

template <typename T> void IntegrationIteratorKeldysh<T>::compute() {
  typedef IntegrationIterator<T> I;
  typedef ContextModule<T> C;

  // Get the energy iterator.
  ParameterIterator<int> *energyIterator = I::getEnergyIterator();

  // Check if there are any energies left. If not, then just return without
  // doing anything.
  if (energyIterator->isDone()) {
    std::cout << "IntegrationIteratorKeldysh<T>::compute(): Iterator done"
              << std::endl;
    return;
  }

  // Get the current energy index, the number of energies in this block, and the
  // total number of energies.
  const int currentEnergyIndex = energyIterator->getCurrentValue();
  const int numEnergiesCurrentBlock = energyIterator->getNumInterval();
  const int numEnergies = energyIterator->getRange();

  // Check that the energy is sane. If not, the return without doing anything.
  if (currentEnergyIndex >= numEnergies) {
    // TODO(niclas): This should never happen. Raise exception instead!?
    std::cout << "ERROR! Energy index out of bounds." << std::endl;
    return;
  }

  // Get the Riccati solver.
  RiccatiSolver<T> *riccatiSolver = C::getRiccatiSolver();

  // For convenience, the number of compute objects.
  const int numComputeObjects = C::getNumComputeObjects();

  // Check if we are on a new energy interval.
  if (m_lastEnergyIndex != currentEnergyIndex) {
    // This is a new energy interval, so everything needs to be initialized.

    // Intialize the Riccati solver.
    riccatiSolver->initialize();

    // Initialize boundary.
    setBoundaryInitialCondition();

    // Set all compute objects to not converged (for this energy interval).
    for (int i = 0; i < numComputeObjects; ++i) {
      C::getComputeObject(i)->setPartiallyConverged(false);
    }

    // Update the internal energy index.
    m_lastEnergyIndex = currentEnergyIndex;
  }

  // Initialize compute objects (usually just make sure memory is allocated and
  // zeroed).
  for (int i = 0; i < numComputeObjects; ++i) {
    C::getComputeObject(i)->initializeIteration(numEnergiesCurrentBlock);
  }

  // Get everything needed for the angle loop.
  GridGPU *energies = I::getEnergies();
  GeometryGroup<T> *geometry = C::getGeometry();
  OrderParameter<T> *orderParameter = C::getOrderParameter();

  // Apply the boundary condition.
  riccatiSolver->computeBoundaryConditions();

  // Do angle loop (integrate over momentum angles).
  const int numAngles = C::getParameters()->getAngularResolution();
  for (int angleIdx = 0; angleIdx < numAngles; ++angleIdx) {
    // Rotate order parameter.
    GridGPU *orderParameterRotated =
        orderParameter->getOrderParameterFieldRotated(-angleIdx);

    const T systemAngle = C::getFermiSurface()->getSystemAngle(angleIdx);

    // Create rotated geometry.
    geometry->create(-systemAngle);

    // Initialize the compute objects for the current angle.
    for (int i = 0; i < numComputeObjects; ++i) {
      C::getComputeObject(i)->initializeGreensAngle();
    }

    // Solve the Riccati equations.
    riccatiSolver->computeCoherenceFunctions(*energyIterator, energies,
                                             orderParameterRotated, angleIdx);

    // Compute the integrand for the current angle for every compute object.
    for (int i = 0; i < numComputeObjects; ++i) {
      C::getComputeObject(i)->computeGreensAngle(energyIterator,
                                                 riccatiSolver->getGamma(),
                                                 riccatiSolver->getGammabar());
    }

    // Add integration contribution of greens function from current angle.
    // TODO(niclas): This can probably be done in the loop above.
    for (int i = 0; i < numComputeObjects; ++i) {
      C::getComputeObject(i)->integrateGreensAngle(geometry, angleIdx);
    }
  }

  // Update compute objects and check (partial) convergence.
  int numConvergedComputeObjects = 0;
  for (int i = 0; i < numComputeObjects; ++i) {
    C::getComputeObject(i)->computePropertyFromGreens();
    const bool isPartiallyConverged =
        C::getComputeObject(i)->isPartiallyConverged();
    if (isPartiallyConverged) {
      numConvergedComputeObjects++;
    }
  }

  // Check if all compute objects are (partially) converged.
  const bool areAllPartiallyConverged =
      numConvergedComputeObjects == numComputeObjects;
  if (areAllPartiallyConverged) {
    // Yes! So lets take the next energy interval.
    energyIterator->next();

    if (energyIterator->isDone()) {
      // If this is the end of the energy iterator, then every interval has
      // converged. We can thus set every compute object to converged!
      for (int i = 0; i < numComputeObjects; ++i) {
        C::getComputeObject(i)->setConverged(true);
      }

      std::cout << "IntegrationIteratorKeldysh<T>::compute(): All intervals "
                   "converged"
                << std::endl;
    }
  }

  // Print CUDA error, if any.
  ContextModule<T>::checkCudaError("IntegrationIteratorKeldysh<T>::compute()");

  // Update iteration counter.
  I::increaseIterationCount();
}
} // namespace scg

#endif // INTEGRATION_ITERATOR_KELDYSH_H_
