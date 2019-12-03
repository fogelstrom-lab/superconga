#ifndef INTEGRATIONITERATORKELDYSH_H
#define INTEGRATIONITERATORKELDYSH_H

#include "utils.h"
#include <ComputeProperty.h>
#include <Context.h>
#include <GeometryGroup.h>
#include <OrderParameter.h>
#include <ParameterIterator.h>
#include <RiccatiSolver.h>

template <typename T>
class IntegrationIteratorKeldysh : public IntegrationIterator<T> {
  typedef typename Grid<T>::GPU GridGPU;
  typedef typename Grid<T>::CPU GridCPU;

public:
  IntegrationIteratorKeldysh();
  virtual ~IntegrationIteratorKeldysh();

  IntegrationIteratorKeldysh(Context<T> *ctx, int numEnergiesPerBlock = -1);

  virtual int initialize();

  // Configures energy iterators
  void initIterators(int numEnergiesPerBlock);
  virtual void initEnergies();

  virtual int getNumEnergiesInterval();
  virtual int getEnergyIdxOffset_boundaryStorage();

  // Call this once before starting self consistent iteration, or when changing
  // initial condition.
  virtual void setBoundaryInitialCondition();

  // Computes all properties added with addComputeProperty(). Iterate this self
  // consistently.
  virtual void compute();

private:
  int _lastEnergyIndex;
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

  _lastEnergyIndex = -1;
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

  const Parameters<T> *param = ContextModule<T>::getParameters();

  T eMin = param->getEnergyMin();
  T eMax = param->getEnergyMax();
  T im = param->getSmearingFactor();

  int numEnergies = param->getNumEnergies();

  std::cout << "eMin = " << eMin << ", eMax = " << eMax
            << ", numE = " << numEnergies << "\n";

  // Generate energy array
  GridCPU energies_cpu(numEnergies, 1);

  T *energies_re = energies_cpu.getDataPointer(0, Type::real);
  T *energies_im = energies_cpu.getDataPointer(0, Type::imag);

  for (int i = 0; i < numEnergies; ++i) {
    energies_re[i] = eMin + (eMax - eMin) * (T)i / (T)(numEnergies - 1);
    energies_im[i] = im;
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

  if (numEnergiesPerBlock == -1)
    iter->setIncrement(64);
  else
    iter->setIncrement(numEnergiesPerBlock);

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

  // Get some parameters and variables
  const int numAngles = C::getParameters()->getAngularResolution();

  GridGPU *deltaRotated; // Pointer to temporary grid for order parameter in
                         // local frame

  ParameterIterator<int> *energy_iter = I::getEnergyIterator();
  GridGPU *energies = I::getEnergies();

  RiccatiSolver<T> *riccati = C::getRiccatiSolver();
  GeometryGroup<T> *geom = C::getGeometry();
  OrderParameter<T> *delta = C::getOrderParameter();

  if (!energy_iter->isDone()) {
    // Do angle loop (integrate over momentum angles)
    //
    for (int angleIdx = 0; angleIdx < numAngles; ++angleIdx) {
      // Rotate order parameter field to current angle
      deltaRotated = delta->getOrderParameterFieldRotated(-angleIdx);

      // Angle (in radians) of system in local frame
      const T systemAngle = C::getFermiSurface()->getSystemAngle(angleIdx);

      // Create discrete geometry data for current angle
      geom->create(-systemAngle);

      // Compute coherence functions and store on boundary as initial condition
      riccati->computeInitialCondition(*energy_iter, energies, deltaRotated,
                                       angleIdx);
    }
  }

  ContextModule<T>::checkCudaError(
      "IntegrationIteratorKeldysh<T>::setBoundaryInitialCondition()");
}

template <typename T> void IntegrationIteratorKeldysh<T>::compute() {
  typedef IntegrationIterator<T> I;
  typedef ContextModule<T> C;

  // Get some parameters and variables
  const int numAngles = C::getParameters()->getAngularResolution();

  GridGPU *deltaRotated; // Pointer to temporary grid for order parameter in
                         // local frame

  ParameterIterator<int> *energy_iter = I::getEnergyIterator();
  GridGPU *energies = I::getEnergies();

  RiccatiSolver<T> *riccati = C::getRiccatiSolver();
  GeometryGroup<T> *geom = C::getGeometry();
  OrderParameter<T> *delta = C::getOrderParameter();

  const int NC = C::getNumComputeObjects();

  std::cout << "IntegrationIteratorKeldysh<T>::compute()\n";
  std::cout << "numEnergies = " << energy_iter->getRange()
            << ", numEnergiesCurrentBlock = " << energy_iter->getNumInterval()
            << ", energy idx = " << energy_iter->getCurrentValue() << "\n";

  // Do when energy iterator switches interval
  int currentEnergyIndex = energy_iter->getCurrentValue();

  if (_lastEnergyIndex != currentEnergyIndex) {
    setBoundaryInitialCondition();
    _lastEnergyIndex = currentEnergyIndex;
  } else if (!I::getNumIterations())
    setBoundaryInitialCondition();

  // Initialize compute objects (usually just make sure mem is allocated and
  // zeroed)
  for (int i = 0; i < NC; ++i)
    C::getComputeObject(i)->initializeIteration(energy_iter->getNumInterval());

  // Boundary condition
  riccati->computeBoundaryConditions();

  // Angle loop
  if (!energy_iter->isDone()) {
    // Do angle loop (integrate over momentum angles)
    //
    for (int angleIdx = 0; angleIdx < numAngles; ++angleIdx) {
      // Rotate order parameter field to current angle
      deltaRotated = delta->getOrderParameterFieldRotated(-angleIdx);

      // Angle (in radians) of system in local frame
      const T systemAngle = C::getFermiSurface()->getSystemAngle(angleIdx);

      // Create discrete geometry data for current angle
      geom->create(-systemAngle);

      // Solve riccati equations for current angle and compute appropriate
      // greens function
      for (int i = 0; i < NC; ++i)
        C::getComputeObject(i)->initializeGreensAngle();

      riccati->computeCoherenceFunctions(*energy_iter, energies, deltaRotated,
                                         angleIdx);

      for (int i = 0; i < NC; ++i)
        C::getComputeObject(i)->computeGreensAngle(
            energy_iter, riccati->getGamma(), riccati->getGammabar(), geom);

      // Add integration contribution of greens function from current angle
      for (int i = 0; i < NC; ++i)
        C::getComputeObject(i)->integrateGreensAngle(geom, angleIdx);
    }

    // Compute updated order parameter (components)
    // Add coupling constant and angle integration normalization factor here
    for (int i = 0; i < NC; ++i)
      C::getComputeObject(i)->computePropertyFromGreens();
  } else
    std::cout << "IntegrationIteratorKeldysh<T>::compute(): Iterator done\n";

  // Iterate over energy intervals
  int conv = 0;

  for (int i = 0; i < NC; ++i) {
    if (C::getComputeObject(i)->isConverged())
      conv++;
  }

  if (conv == NC) {
    energy_iter->next();

    if (!energy_iter->isDone()) {
      std::cout << "IntegrationIteratorKeldysh<T>::compute(): New energy "
                   "iterator interval\n";

      for (int i = 0; i < NC; ++i)
        C::getComputeObject(i)->setConverged(0);

      riccati->initialize();
    } else {
      std::cout << "IntegrationIteratorKeldysh<T>::compute(): All intervals "
                   "converged\n";
    }
  }

  // Print CUDA error, if any
  ContextModule<T>::checkCudaError("IntegrationIteratorKeldysh<T>::compute()");

  I::increaseIterationCount();
}

#endif // INTEGRATIONITERATORKELDYSH_H
