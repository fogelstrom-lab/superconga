#ifndef INTEGRATIONITERATOROZAKI_H
#define INTEGRATIONITERATOROZAKI_H

#include "FermiSurface.h"
#include "utils.h"
#include <ComputeProperty.h>
#include <Context.h>
#include <GeometryGroup.h>
#include <OrderParameter.h>
#include <ParameterIterator.h>
#include <RiccatiSolver.h>
#include <thrust/host_vector.h>

/* Wrapper of dsygv from FORTRAN LAPACK
 * Documentation: http://www.netlib.org/lapack/explore-html/d5/d2e/dsygv_8f.html
 * Requires compilation flags:
 *     GCC:       -lgfortran -llapack -lblas
 *     Intel/MKL: -lgfortran -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core
 * -lpthread -lm -ldl -DMKL_ILP64 -m64
 */
extern "C" {
void dsygv_(int64_t *ITYPE, char *JOBZ, char *UPLO, int64_t *N, double *A,
            int64_t *LDA, double *B, int64_t *LDB, double *W, double *WORK,
            int64_t *LWORK, int64_t *INFO);
}

template <typename T>
class IntegrationIteratorOzaki : public IntegrationIterator<T> {
  typedef typename Grid<T>::GPU GridGPU;
  typedef typename Grid<T>::CPU GridCPU;

public:
  IntegrationIteratorOzaki();
  virtual ~IntegrationIteratorOzaki();

  IntegrationIteratorOzaki(Context<T> *ctx, int numEnergiesPerBlock = -1);

  virtual int initialize();

  // Compute Ozaki poles and residues
  void computeOzakiPoles(const int numPoles, T *poles, T *residues);

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
};

template <typename T>
IntegrationIteratorOzaki<T>::IntegrationIteratorOzaki()
    : IntegrationIterator<T>() {}

template <typename T>
IntegrationIteratorOzaki<T>::IntegrationIteratorOzaki(Context<T> *ctx,
                                                      int numEnergiesPerBlock)
    : IntegrationIterator<T>(ctx, numEnergiesPerBlock) {
  std::cout << "IntegrationIteratorOzaki<T>()\n";
  ctx->set(this);
}

template <typename T> IntegrationIteratorOzaki<T>::~IntegrationIteratorOzaki() {
  std::cout << "~IntegrationIteratorOzaki()\n";
}

template <typename T> int IntegrationIteratorOzaki<T>::initialize() {
  std::cout << "IntegrationIteratorOzaki::initialize()\n";
  IntegrationIterator<T>::initialize();

  int numEnergies = ContextModule<T>::getParameters()->getNumFrequencies();

  initIterators(IntegrationIterator<T>::getNumEnergiesPerBlock());

  return 0;
}

template <typename T>
void IntegrationIteratorOzaki<T>::computeOzakiPoles(const int numPoles,
                                                    T *poles, T *residues) {
  std::cout << "IntegrationIteratorOzaki<T>::computeOzakiPoles( numPoles: "
            << numPoles << ", ... )\n";
  // Dimension of matrices is 2*numPoles due to positive and negative poles
  int64_t N = 2 * numPoles;

  // Ozaki matrices from PRB 75, 035123, (2007)
  // Note that A and B have switched place in relation to the paper.
  // This is a trick so dsygv can actually find a solution.
  // Paper (i start at 1):
  // Diagonal matrix A, A_ii = -(2*i-1)
  // Off-diagonal matrix B, B_i,i+1 = B_i,i-1 = -1/2
  // This implementation (i starts at 0):
  // Diagonal matrix B, B_ii = (2*i+1)
  // Off-diagonal matrix A, A_i,i+1 = A_i,i-1 = -1/2
  thrust::host_vector<double> A(N * N, 0.0);
  thrust::host_vector<double> B(N * N, 0.0);

  for (int i = 0; i < N; ++i) {
    B[i * N + i] = i * 2.0 + 1.0;

    if (i < N - 1) {
      A[N * i + (i + 1)] = -0.5;
      A[N * (i + 1) + i] = -0.5;
    }
  }

  int64_t LDA = N;       // Dimension of A: N*N
  int64_t LDB = N;       // Dimension of B: N*N
  int64_t LWORK = 3 * N; // Working dimension: (3*N)*(3*N)

  int64_t ITYPE =
      1;            // Generalized eigenvalue problem of type A*x = (lambda)*B*x
  char JOBZ = 'V';  // Compute eigenvalues and eigenvectors
  char UPLO = 'U';  // Upper triangles of A and B are stored
  int64_t INFO = 0; // Storage for info returned from dsygv

  thrust::host_vector<double> W(N, 0.0);        // Eigenvalue storage
  thrust::host_vector<double> WORK(LWORK, 0.0); // Work storage

  dsygv_(&ITYPE, &JOBZ, &UPLO, &N, A.data(), &LDA, B.data(), &LDB, W.data(),
         WORK.data(), &LWORK, &INFO);

  for (int i = 0; i < numPoles; ++i) {
    const double pole = -1.0 / W[i];
    const double residueSqrt = 0.5 * A[N * i] * pole;
    poles[i] = static_cast<T>(pole);
    residues[i] = static_cast<T>(residueSqrt * residueSqrt);
  }
}

template <typename T> void IntegrationIteratorOzaki<T>::initEnergies() {
  const Parameters<T> *param = ContextModule<T>::getParameters();

  T T_Tc = param->getTemperature();
  int numEnergies = param->getNumFrequencies();

  // Set ozaki energies
  GridCPU *ozakiEnergies_cpu = new GridCPU(numEnergies, 1);

  T *ozakiEnergies_re = ozakiEnergies_cpu->getDataPointer(0, Type::real);
  T *ozakiEnergies_im = ozakiEnergies_cpu->getDataPointer(0, Type::imag);

  GridCPU *ozakiPoles = new GridCPU(numEnergies, 1, 1, Type::real);
  GridCPU *ozakiResidues = new GridCPU(numEnergies, 1, 1, Type::real);

  T *ozakiPoles_ptr = ozakiPoles->getDataPointer(0, Type::real);
  T *ozakiResidues_ptr = ozakiResidues->getDataPointer(0, Type::real);

  // Calculate Ozaki poles and residues
  computeOzakiPoles(numEnergies, ozakiPoles_ptr, ozakiResidues_ptr);

  for (int i = 0; i < numEnergies; ++i) {
    ozakiEnergies_re[i] = 0.0;
    ozakiEnergies_im[i] = ozakiPoles_ptr[i] * T_Tc;
  }

  GridGPU *energies = new GridGPU;
  GridGPU *residues = new GridGPU;
  *energies = *ozakiEnergies_cpu;
  *residues = *ozakiResidues;

  IntegrationIterator<T>::setEnergies(energies);
  IntegrationIterator<T>::setResidues(residues);
}

template <typename T>
void IntegrationIteratorOzaki<T>::initIterators(int numEnergiesPerBlock) {
  typedef IntegrationIterator<T> I;

  initEnergies();

  int numEnergies = I::getEnergies()->dimX();

  ParameterIterator<int> *iter = new ParameterIterator<int>(0, numEnergies);
  I::setEnergyIterator(iter);

  std::cout << "Initializing ozaki iterator with Ne = " << numEnergies
            << ", NeBlock = " << numEnergiesPerBlock << "\n";

  if (numEnergiesPerBlock < 1)
    iter->setIncrement(numEnergies);
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
int IntegrationIteratorOzaki<T>::getNumEnergiesInterval() {
  return IntegrationIterator<T>::getEnergies()->dimX();
}

template <typename T>
int IntegrationIteratorOzaki<T>::getEnergyIdxOffset_boundaryStorage() {
  return IntegrationIterator<T>::getEnergyIterator()->getCurrentValue();
}

template <typename T>
void IntegrationIteratorOzaki<T>::setBoundaryInitialCondition() {
  std::cout << "IntegrationIteratorOzaki<T>::setBoundaryInitialCondition(): "
               "Begin...\n";

  typedef IntegrationIterator<T> I;

  Context<T> *ctx = ContextModule<T>::getContext();

  // Get some parameters and variables
  const int numAngles = ctx->getParameters()->getAngularResolution();

  GridGPU *deltaRotated; // Pointer to temporary grid for order parameter in
                         // local frame

  ParameterIterator<int> *energy_iter = I::getEnergyIterator();
  GridGPU *energies = I::getEnergies();

  RiccatiSolver<T> *riccati = ctx->getRiccatiSolver();
  GeometryGroup<T> *geom = ctx->getGeometry();
  OrderParameter<T> *delta = ctx->getOrderParameter();
  FermiSurface<T> *const fermiSurface = ctx->getFermiSurface();

  // Do angle loop (integrate over momentum angles)
  //
  for (int angleIdx = 0; angleIdx < numAngles; ++angleIdx) {
    // Angle (in radians) of system in local frame.
    const T systemAngle = fermiSurface->getSystemAngle(angleIdx);

    // Rotate order parameter field to current angle
    deltaRotated = delta->getOrderParameterFieldRotated(-angleIdx);

    // Create discrete geometry data for current angle
    geom->create(-systemAngle);

    // Compute coherence functions and store on boundary as initial condition
    for (energy_iter->begin(); !energy_iter->isDone(); energy_iter->next()) {
      riccati->computeInitialCondition(*energy_iter, energies, deltaRotated,
                                       angleIdx);
    }
  }

  std::cout
      << "IntegrationIteratorOzaki<T>::setBoundaryInitialCondition(): Done!\n";

  // Print CUDA error, if any
  ContextModule<T>::checkCudaError(
      "IntegrationIteratorOzaki<T>::setBoundaryInitialCondition()");
}

template <typename T> void IntegrationIteratorOzaki<T>::compute() {
  typedef IntegrationIterator<T> I;

  Context<T> *ctx = ContextModule<T>::getContext();

  // Get some parameters and variables
  const int numAngles = ctx->getParameters()->getAngularResolution();
  FermiSurface<T> *const fermiSurface = ctx->getFermiSurface();

  GridGPU *deltaRotated; // Pointer to temporary grid for order parameter in
                         // local frame
  GridGPU deltaRotatedMultFactor; // This is different to one when computing
                                  // free energy according to Sauls et al.

  ParameterIterator<int> *energy_iter = I::getEnergyIterator();
  GridGPU *energies = I::getEnergies();

  RiccatiSolver<T> *riccati = ctx->getRiccatiSolver();
  GeometryGroup<T> *geom = ctx->getGeometry();
  OrderParameter<T> *delta = ctx->getOrderParameter();

  const int NC = ctx->getNumComputeObjects();

  // Set initial condition on boundary, if
  if (!I::getNumIterations())
    setBoundaryInitialCondition();

  // Initialize compute objects (usually just make sure mem is allocated and
  // zeroed)
  for (int i = 0; i < NC; ++i)
    ctx->getComputeObject(i)->initializeIteration(
        energy_iter->getNumInterval());

  // Apply boundary condition(s)
  riccati->computeBoundaryConditions();

  // Include magnetic field induced by currents
  riccati->computeInducedMagneticField();

  // Do angle loop (=momentum integral, and sum over energies)
  for (int angleIdx = 0; angleIdx < numAngles; ++angleIdx) {
    // Rotate order parameter field to current angle.
    // A negative index is equivalent to a negative angle.
    deltaRotated = delta->getOrderParameterFieldRotated(-angleIdx);

    // If we are computing free energy with Luttinger-Ward functional, the order
    // parameter is multiplied by a constant (which is
    // orderParameterMultFactor). In all other cases the constant is 1.
    deltaRotatedMultFactor =
        (*deltaRotated) * riccati->getOrderParameterMultFactor();

    // Create discrete geometry data for current angle.
    // Angle (in radians) of system in local frame.
    const T systemAngle = fermiSurface->getSystemAngle(angleIdx);
    geom->create(-systemAngle);

    // Solve riccati equations for current angle and compute appropriate greens
    // function
    for (int i = 0; i < NC; ++i)
      ctx->getComputeObject(i)->initializeGreensAngle();

    // Iterate over (possibly) several blocks of energies.
    for (energy_iter->begin(); !energy_iter->isDone(); energy_iter->next()) {
      riccati->computeCoherenceFunctions(*energy_iter, energies,
                                         &deltaRotatedMultFactor, angleIdx);

      for (int i = 0; i < NC; ++i)
        ctx->getComputeObject(i)->computeGreensAngle(
            energy_iter, riccati->getGamma(), riccati->getGammabar(), geom);
    }

    // Add integration contribution of greens function from current angle
    for (int i = 0; i < NC; ++i)
      ctx->getComputeObject(i)->integrateGreensAngle(geom, angleIdx);
  }

  // Compute updated order parameter (components)
  // Add coupling constant and angle integration normalization factor here
  for (int i = 0; i < NC; ++i)
    ctx->getComputeObject(i)->computePropertyFromGreens();

  // Print CUDA error, if any
  ContextModule<T>::checkCudaError("IntegrationIteratorOzaki<T>::compute()");

  I::increaseIterationCount();
}

#endif // INTEGRATIONITERATOROZAKI_H
