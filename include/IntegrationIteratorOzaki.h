//===-- IntegrationiteratorOzaki.h - Ozaki integration summation. ---------===//
//
// Part of the SuprConGPU Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Ozaki implementation of the integration iterator, i.e. the class that
/// handles summation over Ozaki poles.
///
//===----------------------------------------------------------------------===//
#ifndef INTEGRATION_ITERATOR_OZAKI_H_
#define INTEGRATION_ITERATOR_OZAKI_H_

#include "ComputeProperty.h"
#include "Context.h"
#include "FermiSurface.h"
#include "GeometryGroup.h"
#include "OrderParameter.h"
#include "ParameterIterator.h"
#include "RiccatiSolver.h"
#include "utils.h"
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
namespace scg {
template <typename T>
class IntegrationIteratorOzaki : public IntegrationIterator<T> {
public:
  IntegrationIteratorOzaki();
  ~IntegrationIteratorOzaki();

  IntegrationIteratorOzaki(Context<T> *const inOutContext,
                           const int inNumEnergiesPerBlock = -1);

  int initialize();

  // Compute Ozaki poles and residues
  void computeOzakiPoles(const int numPoles, T *const outPoles,
                         T *const outResidues);

  // Configures energy iterators
  void initIterators(const int inNumEnergiesPerBlock);
  void initEnergies();

  int getNumEnergiesInterval();
  int getEnergyIdxOffset_boundaryStorage();

  // Call this once before starting self consistent iteration, or when changing
  // initial condition.
  void setBoundaryInitialCondition();

  // Computes all properties added with addComputeProperty(). Iterate this self
  // consistently.
  void compute();
};

template <typename T>
IntegrationIteratorOzaki<T>::IntegrationIteratorOzaki()
    : IntegrationIterator<T>() {}

template <typename T>
IntegrationIteratorOzaki<T>::IntegrationIteratorOzaki(
    Context<T> *const inOutContext, int inNumEnergiesPerBlock)
    : IntegrationIterator<T>(inOutContext, inNumEnergiesPerBlock) {
#ifndef NDEBUG
  std::cout << "IntegrationIteratorOzaki<T>()\n";
#endif
  inOutContext->set(this);
}

template <typename T> IntegrationIteratorOzaki<T>::~IntegrationIteratorOzaki() {
#ifndef NDEBUG
  std::cout << "~IntegrationIteratorOzaki()\n";
#endif
}

template <typename T> int IntegrationIteratorOzaki<T>::initialize() {
#ifndef NDEBUG
  std::cout << "IntegrationIteratorOzaki::initialize()\n";
#endif
  IntegrationIterator<T>::initialize();

  initIterators(IntegrationIterator<T>::getNumEnergiesPerBlock());

  return 0;
}

template <typename T>
void IntegrationIteratorOzaki<T>::computeOzakiPoles(const int inNumOzakiPoles,
                                                    T *const outPoles,
                                                    T *const outResidues) {
#ifndef NDEBUG
  std::cout
      << "IntegrationIteratorOzaki<T>::computeOzakiPoles( inNumOzakiPoles: "
      << inNumOzakiPoles << ", ... )\n";
#endif
  // Dimension of matrices is 2*poles due to positive and negative poles
  int64_t N = 2 * inNumOzakiPoles;

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

  // ITYPE 1: Generalized eigenvalue problem of type A*x = (lambda)*B*x
  int64_t ITYPE = 1;
  char JOBZ = 'V';  // Compute eigenvalues and eigenvectors
  char UPLO = 'U';  // Upper triangles of A and B are stored
  int64_t INFO = 0; // Storage for info returned from dsygv

  thrust::host_vector<double> W(N, 0.0);        // Eigenvalue storage
  thrust::host_vector<double> WORK(LWORK, 0.0); // Work storage

  dsygv_(&ITYPE, &JOBZ, &UPLO, &N, A.data(), &LDA, B.data(), &LDB, W.data(),
         WORK.data(), &LWORK, &INFO);

  for (int i = 0; i < inNumOzakiPoles; ++i) {
    const double pole = -1.0 / W[i];
    const double residueSqrt = 0.5 * A[N * i] * pole;
    outPoles[i] = static_cast<T>(pole);
    outResidues[i] = static_cast<T>(residueSqrt * residueSqrt);
  }
}

template <typename T> void IntegrationIteratorOzaki<T>::initEnergies() {
  const Parameters<T> *param = ContextModule<T>::getParameters();

  const T temperature = param->getTemperature();
  const int numEnergies = param->getNumOzakiEnergies();

  // Set ozaki energies
  GridCPU<T> *ozakiEnergiesCPU = new GridCPU<T>(numEnergies, 1);

  T *ozakiEnergiesReal = ozakiEnergiesCPU->getDataPointer(0, Type::real);
  T *ozakiEnergiesImag = ozakiEnergiesCPU->getDataPointer(0, Type::imag);

  GridCPU<T> *ozakiPoles = new GridCPU<T>(numEnergies, 1, 1, Type::real);
  GridCPU<T> *ozakiResidues = new GridCPU<T>(numEnergies, 1, 1, Type::real);

  T *ozakiPolesPtr = ozakiPoles->getDataPointer(0, Type::real);
  T *ozakiResiduesPtr = ozakiResidues->getDataPointer(0, Type::real);

  // Calculate Ozaki poles and residues
  computeOzakiPoles(numEnergies, ozakiPolesPtr, ozakiResiduesPtr);

  for (int i = 0; i < numEnergies; ++i) {
    ozakiEnergiesReal[i] = static_cast<T>(0.0);
    ozakiEnergiesImag[i] = ozakiPolesPtr[i] * temperature;
  }

  GridGPU<T> *energies = new GridGPU<T>;
  GridGPU<T> *residues = new GridGPU<T>;
  *energies = *ozakiEnergiesCPU;
  *residues = *ozakiResidues;

  IntegrationIterator<T>::setEnergies(energies);
  IntegrationIterator<T>::setResidues(residues);
}

template <typename T>
void IntegrationIteratorOzaki<T>::initIterators(
    const int inNumEnergiesPerBlock) {
  typedef IntegrationIterator<T> I;

  initEnergies();

  const int numEnergies = I::getEnergies()->dimX();

  ParameterIterator<int> *iter = new ParameterIterator<int>(0, numEnergies);
  I::setEnergyIterator(iter);
#ifndef NDEBUG
  std::cout << "Initializing ozaki iterator with Ne = " << numEnergies
            << ", NeBlock = " << inNumEnergiesPerBlock << "\n";
#endif

  if (inNumEnergiesPerBlock < 1) {
    iter->setIncrement(numEnergies);
  } else {
    iter->setIncrement(inNumEnergiesPerBlock);
  }

#ifndef NDEBUG
  for (iter->begin(); !iter->isDone(); iter->next()) {
    std::cout << "min = " << iter->getMin() << "\n";
    std::cout << "max = " << iter->getMax() << "\n";
    std::cout << "inc = " << iter->getIncrement() << "\n";
    std::cout << "num = " << iter->getNumInterval() << "\n\n";
  }
#endif

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
#ifndef NDEBUG
  std::cout << "IntegrationIteratorOzaki<T>::setBoundaryInitialCondition(): "
               "Begin...\n";
#endif

  typedef IntegrationIterator<T> I;

  Context<T> *context = ContextModule<T>::getContext();

  // Get some parameters and variables
  const int numAngles = context->getParameters()->getAngularResolution();

  // Pointer to temporary grid for order parameter in local frame
  GridGPU<T> *rotatedOrderParameter;

  ParameterIterator<int> *energyIterator = I::getEnergyIterator();
  GridGPU<T> *energies = I::getEnergies();

  RiccatiSolver<T> *riccatiSolver = context->getRiccatiSolver();
  GeometryGroup<T> *geometryGroup = context->getGeometry();
  OrderParameter<T> *orderParameter = context->getOrderParameter();
  FermiSurface<T> *const fermiSurface = context->getFermiSurface();

  // Do angle loop (integrate over momentum angles)
  //
  for (int angleIdx = 0; angleIdx < numAngles; ++angleIdx) {
    // Angle (in radians) of system in local frame.
    const T systemAngle = fermiSurface->getSystemAngle(angleIdx);

    // Rotate order parameter field to current angle
    rotatedOrderParameter =
        orderParameter->getOrderParameterFieldRotated(-angleIdx);

    // Create discrete geometry data for current angle
    geometryGroup->create(-systemAngle);

    // Compute coherence functions and store on boundary as initial condition
    for (energyIterator->begin(); !energyIterator->isDone();
         energyIterator->next()) {
      riccatiSolver->computeInitialCondition(*energyIterator, energies,
                                             rotatedOrderParameter, angleIdx);
    }
  }

#ifndef NDEBUG
  std::cout
      << "IntegrationIteratorOzaki<T>::setBoundaryInitialCondition(): Done!\n";
#endif

  // Print CUDA error, if any
  ContextModule<T>::checkCudaError(
      "IntegrationIteratorOzaki<T>::setBoundaryInitialCondition()");
}

template <typename T> void IntegrationIteratorOzaki<T>::compute() {
  typedef IntegrationIterator<T> I;

  Context<T> *context = ContextModule<T>::getContext();

  // Get some parameters and variables
  const int numAngles = context->getParameters()->getAngularResolution();
  FermiSurface<T> *const fermiSurface = context->getFermiSurface();

  ParameterIterator<int> *energyIterator = I::getEnergyIterator();
  GridGPU<T> *energies = I::getEnergies();

  RiccatiSolver<T> *riccatiSolver = context->getRiccatiSolver();
  GeometryGroup<T> *geometryGroup = context->getGeometry();
  OrderParameter<T> *orderParameter = context->getOrderParameter();

  const int numComputeObjects = context->getNumComputeObjects();
  const int numIterations = I::getNumIterations();

  // Set initial condition on boundary.
  if (numIterations == 0) {
    setBoundaryInitialCondition();
  }

  // Initialize compute objects (usually just make sure mem is allocated and
  // zeroed).
  for (int i = 0; i < numComputeObjects; ++i) {
    context->getComputeObject(i)->initializeIteration(
        energyIterator->getNumInterval());
  }

  // Apply boundary condition(s).
  riccatiSolver->computeBoundaryConditions();

  // Do angle loop (=momentum integral, and sum over energies).
  for (int angleIdx = 0; angleIdx < numAngles; ++angleIdx) {
    // Create discrete geometry data for current angle.
    // Angle (in radians) of system in local frame.
    const T systemAngle = fermiSurface->getSystemAngle(angleIdx);
    geometryGroup->create(-systemAngle);

    // If we are computing free energy with Luttinger-Ward functional, the
    // order parameter is multiplied by a constant (which is
    // orderParameterMultFactor). In all other cases the constant is 1.

    // This is != 1 when computing free energy according to Sauls et al.
    const T orderParameterFactor = riccatiSolver->getOrderParameterMultFactor();

    // Rotate order parameter field to current angle.
    // A negative index is equivalent to a negative angle.
    // TODO(niclas): Slight optimization. Don't do the multiplication with
    // orderParameterFactor if the factor is 1.
    const GridGPU<T> rotatedOrderParameter =
        *(orderParameter->getOrderParameterFieldRotated(-angleIdx)) *
        orderParameterFactor;

    // Solve Riccati equations for current angle and compute propagators.
    for (int i = 0; i < numComputeObjects; ++i) {
      context->getComputeObject(i)->initializeGreensAngle();
    }

    // Iterate over (possibly) several blocks of energies.
    for (energyIterator->begin(); !energyIterator->isDone();
         energyIterator->next()) {
      riccatiSolver->computeCoherenceFunctions(
          *energyIterator, energies, &rotatedOrderParameter, angleIdx);

      for (int i = 0; i < numComputeObjects; ++i) {
        context->getComputeObject(i)->computeGreensAngle(
            energyIterator, riccatiSolver->getGamma(),
            riccatiSolver->getGammabar());
      }
    }

    // Add integration contribution of greens function from current angle.
    for (int i = 0; i < numComputeObjects; ++i) {
      context->getComputeObject(i)->integrateGreensAngle(geometryGroup,
                                                         angleIdx);
    }
  }

  // Compute updated order parameter (components).
  // Add coupling constant and angle integration normalization factor here.
  for (int i = 0; i < numComputeObjects; ++i) {
    context->getComputeObject(i)->computePropertyFromGreens();
  }

  // Print CUDA error, if any.
  ContextModule<T>::checkCudaError("IntegrationIteratorOzaki<T>::compute()");

  I::increaseIterationCount();
}
} // namespace scg

#endif // INTEGRATION_ITERATOR_OZAKI_H_
