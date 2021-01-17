//
// Base class for the Riccati equation solver.
//
#ifndef RICCATI_SOLVER_H_
#define RICCATI_SOLVER_H_

#include "BoundaryCondition.h"
#include "BoundaryStorage.h"
#include "ComputeCurrent.h"
#include "ComputeVectorPotential.h"
#include "GeometryGroup.h"
#include "Grid.h"
#include "IntegrationIterator.h"
#include "LocalGamma.h"
#include "LocalVectorPotential.h"
#include "ParameterIterator.h"
#include "Parameters.h"
#include "VectorPotential_Kernel.h"
#include "cplx.h"
#include "utils.h"
#include <limits>

/////////////////////////////////////////////////////////
// MAYBE PUT POINTER TO ENERGIES AND ENERGYITERATOR HERE
/////////////////////////////////////////////////////////

namespace scg {
template <typename T> class BoundaryStorage;

template <typename T> class RiccatiSolver : public ContextModule<T> {
  typedef typename Grid<T>::GPU GridGPU;
  typedef typename Grid<T>::CPU GridCPU;

public:
  // Constructors and Destructors

  RiccatiSolver();
  virtual ~RiccatiSolver();

  RiccatiSolver(Context<T> *ctx);

  // Allocate memory
  virtual int initialize();

  // Get pointers to storage of Riccati solutions (gamma/coherence functions).
  // These grids have dimensions (x,y,energy)
  GridGPU *getGamma() const { return _gamma; }
  GridGPU *getGammabar() const { return _gammabar; }

  // Functions related to Boundary condition:

  // Get pointer to sparse array where boundary values (for coherence functions)
  // are stored.
  BoundaryStorage<T> *getBoundaryStorage() const { return _storage; }

  // Add boundary condition.
  void add(BoundaryCondition<T> *bc) { _bc_list.push_back(bc); }

  int getNumBoundaryConditions() const { return _bc_list.size(); }
  BoundaryCondition<T> *getBoundaryCondition(const int idx) {
    return _bc_list[idx];
  }

  // Vector potential
  GridGPU *getInducedVectorPotential() const { return m_vectorPotential; }
  GridGPU *getEnergyShift() const { return _eShift; }

  bool hasComputedInducedVectorPotential() const {
    return m_hasComputedInducedVectorPotential;
  }

  void computeInducedVectorPotential();

  // Functions
  void initializeCoherenceFunctions();
  void initializeBoundaryStorage();

  virtual void
  computeInitialCondition(const ParameterIterator<int> &energy_iter,
                          const GridGPU *energies, const GridGPU *delta,
                          const int angleIdx) = 0;

  virtual void computeBoundaryConditions();

  virtual void
  computeCoherenceFunctions(const ParameterIterator<int> &energy_iter,
                            const GridGPU *energies, const GridGPU *deltaRot,
                            const int angleIdx) = 0;

  // For Computation of Free energy (as in ComputeFreeEnergy2 class)
  T getOrderParameterMultFactor() const { return _orderParameterMult; }
  void setOrderParameterMultFactor(const T mult) { _orderParameterMult = mult; }

  // Helper functions
  void getKernelDimensions(dim3 &blocksPerGrid, dim3 &threadsPerBlock,
                           const int numEnergies);

  void getKernelDimensionsOptim(dim3 &blocksPerGrid, dim3 &threadsPerBlock,
                                const int numEnergies, const int angleIdx);

  virtual std::string getVectorPotentialType() const = 0;

private:
  // START ELIMINATING GAMMA AS CLASS MEMBERS!!! USE CoherenceFunctions CLASS
  // INSTEAD!

  // Storage for riccati trajectories
  GridGPU *_gamma;
  GridGPU *_gammabar;

  // Holds (and reflects) gammas on boundary between interations
  BoundaryStorage<T> *_storage;

  // Boundary conditions
  std::vector<BoundaryCondition<T> *> _bc_list;

  // Functor object for energy shift by the vector potential e -> e + vF.pS
  // (inclusion of magnetic field)
  // POTENTIAL<T>	_vpFunctor;

  // Switch to compute and include internally induced magnetic field
  bool m_hasComputedInducedVectorPotential;

  // Vector potential field
  GridGPU *m_vectorPotential;

  // Precomputed field for energy shift
  GridGPU *_eShift;

  // A flag to indicate that delta should be multiplied with a constant,
  // as required by the free energy computation by J. Sauls et al.
  // For all other computations, this is not used.
  T _orderParameterMult;
};

// This function could not be put into RiccatiSolver class, because of template
// parameter issues
template <typename T, template <typename> class VP>
void computeEnergyShift(ContextModule<T> *inContextModule,
                        typename Grid<T>::GPU *inVectorPotential,
                        typename Grid<T>::GPU *inEnergyShift,
                        const bool inCoupleInducedVectorPotential,
                        const int angleIdx) {
  typedef typename Grid<T>::GPU GridGPU;
  typedef typename Grid<T>::CPU GridCPU;

  Parameters<T> *parameters = inContextModule->getParameters();

  // Add contribution from external magnetic field
  const int N0 = parameters->getGridResolutionBase();
  const int N1 = parameters->getGridResolutionCoherence();
  const T B = parameters->getMagneticFieldStrength();
  const T h = parameters->getGridElementSize();

  const T systemAngle =
      inContextModule->getFermiSurface()->getSystemAngle(angleIdx);
  T fermiVelocityX;
  T fermiVelocityY;
  inContextModule->getFermiSurface()->getFermiVelocity(angleIdx, fermiVelocityX,
                                                       fermiVelocityY);

  dim3 bpg, tpb;
  inEnergyShift->getKernelDimensions(bpg, tpb);

  if (inCoupleInducedVectorPotential) {
    GridGPU vectorPotentialRotated(N1, N1, 2, Type::real);
    vectorPotentialRotated.setZero();
    vectorPotentialRotated.insertRotatedGrid(inVectorPotential, -systemAngle);

    computeEnergyShiftKernel<T, VP><<<bpg, tpb>>>(
        vectorPotentialRotated.getDataPointer(0),
        vectorPotentialRotated.getDataPointer(1), fermiVelocityX,
        fermiVelocityY, systemAngle, B, N1, h, inEnergyShift->getDataPointer());
  } else {
    computeEnergyShiftOnlyExternalKernel<T, VP>
        <<<bpg, tpb>>>(fermiVelocityX, fermiVelocityY, systemAngle, B, N1, h,
                       inEnergyShift->getDataPointer());
  }
}

template <typename T>
RiccatiSolver<T>::RiccatiSolver()
    : ContextModule<T>(), _gamma(0), _gammabar(0), _storage(0),
      m_vectorPotential(0), _eShift(0),
      m_hasComputedInducedVectorPotential(false), _orderParameterMult(1.0) {}

template <typename T>
RiccatiSolver<T>::RiccatiSolver(Context<T> *ctx)
    : ContextModule<T>(ctx), _gamma(0), _gammabar(0), _storage(0),
      m_vectorPotential(0), _eShift(0),
      m_hasComputedInducedVectorPotential(false), _orderParameterMult(1.0) {}

template <typename T> RiccatiSolver<T>::~RiccatiSolver() {
  std::cout << "~RiccatiSolver()\n";
  if (_gamma)
    delete _gamma;
  if (_gammabar)
    delete _gammabar;
  if (_storage)
    delete _storage;
  if (m_vectorPotential)
    delete m_vectorPotential;
  if (_eShift)
    delete _eShift;

  for (int i = 0; i < getNumBoundaryConditions(); ++i)
    delete _bc_list[i];
}

template <typename T> int RiccatiSolver<T>::initialize() {
  std::cout << "RiccatiSolver<T>::initialize()\n";
  initializeCoherenceFunctions();

  const int N0 = ContextModule<T>::getParameters()->getGridResolutionBase();
  const int N1 =
      ContextModule<T>::getParameters()->getGridResolutionCoherence();

  GridGPU::initializeGrid(m_vectorPotential, N0, N0, 2, Type::real);
  GridGPU::initializeGrid(_eShift, N1, N1, 1, Type::real);

  m_vectorPotential->setZero();

  return 0;
}

template <typename T> void RiccatiSolver<T>::initializeBoundaryStorage() {
  typedef ContextModule<T> C;

  if (_storage != 0)
    delete _storage;

  _storage = new BoundaryStorage<T>(C::getGeometry(), C::getFermiSurface());

  int numTrajectories = C::getParameters()->getGridResolutionCoherence();
  int numAngles = C::getParameters()->getAngularResolution();
  int numEnergies = C::getIntegrationIterator()->getNumEnergiesInterval();

  _storage->initialize(numTrajectories, numEnergies, numAngles);
}

template <typename T> void RiccatiSolver<T>::initializeCoherenceFunctions() {
  typedef ContextModule<T> C;

  int N = C::getParameters()->getGridResolutionCoherence();

  int numFields =
      C::getIntegrationIterator()->getEnergyIterator()->getNumInterval();

  GridGPU::initializeGrid(_gamma, N, N, numFields);
  GridGPU::initializeGrid(_gammabar, N, N, numFields);

  _gamma->setZero();
  _gammabar->setZero();

  long long bytes = N * N * numFields * 2 * sizeof(T) * 2;
  std::cout << "Gamma storage data: " << bytes << " bytes.\n";
  std::cout << "numFields = " << numFields << "\n\n";
}

template <typename T> void RiccatiSolver<T>::computeBoundaryConditions() {
  typedef RiccatiSolver<T> R;

  // Apply boundary condition(s)
  int Nbc = getNumBoundaryConditions();

  if (Nbc > 0) {
    for (int i = 0; i < Nbc; ++i)
      getBoundaryCondition(i)->computeBoundaryCondition(
          ContextModule<T>::getContext());
  } else {
    std::cout << "WARNING: No boundary conditions specified!\n";
  }
}

template <typename T> void RiccatiSolver<T>::computeInducedVectorPotential() {
  m_hasComputedInducedVectorPotential = false;

  const Parameters<T> *const parameters = ContextModule<T>::getParameters();
  const T penetrationDepth = parameters->getPenetrationDepth();

  if (penetrationDepth > static_cast<T>(0)) {
    ComputeCurrent<T> *j_compute =
        ContextModule<T>::getContext()->getComputeCurrent();

    if (j_compute) {
      Parameters<T> *parameters = ContextModule<T>::getParameters();

      const int N0 = parameters->getGridResolutionBase();
      const T h = parameters->getGridElementSize();

      // Note, we are not setting the current to zero outside of the grain. This
      // is handled in `computeVectorPotential` by checking the domain labels.
      const GridGPU currentDensity = *(j_compute->getResult());

      const Grid<char>::GPU domainLabels =
          *(ContextModule<T>::getContext()->getGeometry()->getDomainGrid());

      // Solve Maxwell's equation relating the vector potential to the current.
      computeVectorPotential(*m_vectorPotential, currentDensity, domainLabels,
                             h, penetrationDepth);

      m_hasComputedInducedVectorPotential = true;
    } else {
      // TODO(niclas): Raise exception instead?
      std::cout << "WARNING! There is no current computed." << std::endl;
    }
  }
}

template <typename T>
void RiccatiSolver<T>::getKernelDimensions(dim3 &blocksPerGrid,
                                           dim3 &threadsPerBlock,
                                           const int numEnergies) {
  const int N0 = ContextModule<T>::getParameters()->getGridResolutionBase();
  const int N1 =
      ContextModule<T>::getParameters()->getGridResolutionCoherence();

  int tpb_x, tpb_y, bpg_x, bpg_y;

  tpb_x = N1;
  tpb_y = 1;

  bpg_x = 1;

  if (N0 <= 151)
    tpb_y = 2; // This is somewhat faster for small grid sizes

  if (tpb_x > 512) {
    tpb_x = N1 / 2 + 1;
    bpg_x = 2;
  }

  bpg_y = numEnergies / tpb_y;

  if (bpg_y * tpb_y != numEnergies)
    bpg_y++;

  threadsPerBlock = dim3(tpb_x, tpb_y);
  blocksPerGrid = dim3(bpg_x, bpg_y);
}

template <typename T>
void RiccatiSolver<T>::getKernelDimensionsOptim(dim3 &blocksPerGrid,
                                                dim3 &threadsPerBlock,
                                                const int numEnergies,
                                                const int angleIdx) {
  int numTrajectories = _storage->getNumIntersectingTrajectoriesAngle(angleIdx);

  int tpb_x, tpb_y, bpg_x, bpg_y;

  tpb_x = numTrajectories;
  tpb_y = 1;

  bpg_x = 1;

  if (tpb_x > 512) {
    tpb_x = tpb_x / 2 + 1;
    bpg_x = 2;
  }

  bpg_y = numEnergies / tpb_y;

  if (bpg_y * tpb_y != numEnergies)
    bpg_y++;

  threadsPerBlock = dim3(tpb_x, tpb_y);
  blocksPerGrid = dim3(bpg_x, bpg_y);
}
} // namespace scg

#endif // RICCATI_SOLVER_H_
