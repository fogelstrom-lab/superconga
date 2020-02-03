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

/////////////////////////////////////////////////////////
// MAYBE PUT POINTER TO ENERGIES AND ENERGYITERATOR HERE
/////////////////////////////////////////////////////////

// This function could not be put into RiccatiSolver class, because of template
// parameter issues
template <typename T, template <typename> class VP>
void computeEnergyShift(ContextModule<T> *mod, typename Grid<T>::GPU *eShift,
                        const int angleIdx);

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
  GridGPU *getInducedVectorPotential() const { return _ps; }
  GridGPU *getEnergyShift() const { return _eShift; }

  // Magnetic field
  int doComputeInducedMagneticField() const { return _computeInduced; }
  void setComputeInducedMagneticField(int induced) {
    _computeInduced = induced;
  }
  void computeInducedMagneticField();

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
  int _computeInduced;

  // Vector potential field
  GridGPU *_ps;

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
void computeEnergyShift(ContextModule<T> *mod, typename Grid<T>::GPU *ps,
                        typename Grid<T>::GPU *eShift, const int computeInduced,
                        const int angleIdx) {
  typedef typename Grid<T>::GPU GridGPU;
  typedef typename Grid<T>::CPU GridCPU;

  Parameters<T> *param = mod->getParameters();

  // Add contribution from external magnetic field
  const int N0 = param->getGridResolutionBase();
  const int N1 = param->getGridResolutionCoherence();
  const T B = param->getMagneticFieldStrength();
  const T h = param->getGridElementSize();
  const int Na = param->getAngularResolution();

  const T systemAngle = mod->getFermiSurface()->getSystemAngle(angleIdx);

  dim3 bpg, tpb;
  eShift->getKernelDimensions(bpg, tpb);

  computeEnergyShift_external_kernel<T, VP>
      <<<bpg, tpb>>>(eShift->getDataPointer(), systemAngle, B, N1, h);

  // Add contribution from internal magnetic field induced by currents
  ComputeCurrent<T> *j_compute = mod->getContext()->getComputeCurrent();

  if (j_compute && computeInduced) {
    GridGPU ps_rot(N1, N1, 2, Type::real);
    ps_rot.setZero();
    ps_rot.insertRotatedGrid(ps, -systemAngle);

    computeEnergyShift_internal_kernel<T>
        <<<bpg, tpb>>>(eShift->getDataPointer(), ps_rot.getDataPointer(0),
                       ps_rot.getDataPointer(1), systemAngle, N1);
  }
}

template <typename T>
RiccatiSolver<T>::RiccatiSolver()
    : ContextModule<T>(), _gamma(0), _gammabar(0), _storage(0), _ps(0),
      _eShift(0), _computeInduced(0), _orderParameterMult(1.0) {}

template <typename T>
RiccatiSolver<T>::RiccatiSolver(Context<T> *ctx)
    : ContextModule<T>(ctx), _gamma(0), _gammabar(0), _storage(0), _ps(0),
      _eShift(0), _computeInduced(0), _orderParameterMult(1.0) {}

template <typename T> RiccatiSolver<T>::~RiccatiSolver() {
  std::cout << "~RiccatiSolver()\n";
  if (_gamma)
    delete _gamma;
  if (_gammabar)
    delete _gammabar;
  if (_storage)
    delete _storage;
  if (_ps)
    delete _ps;
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

  GridGPU::initializeGrid(_ps, N0, N0, 2, Type::real);
  GridGPU::initializeGrid(_eShift, N1, N1, 1, Type::real);

  _ps->setZero();

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

template <typename T> void RiccatiSolver<T>::computeInducedMagneticField() {
  typedef ContextModule<T> C;

  Parameters<T> *param = C::getParameters();

  const int N0 = param->getGridResolutionBase();
  const T h = param->getGridElementSize();

  ComputeCurrent<T> *j_compute = C::getContext()->getComputeCurrent();

  if (j_compute && doComputeInducedMagneticField()) {
    std::cout << "Solving for vector potential...\n";

    if (param->getPenetrationDepth() < 0.0000000001) {
      std::cout << "Error, penetration depth is zero or negative!\n";
      std::cout << "Error, penetration depth is zero or negative!\n";
      std::cout << "Error, penetration depth is zero or negative!\n";
    }

    const T lambda = param->getPenetrationDepth();

    std::cout << "Lambda = " << lambda << "\n";

    const T twopi = 2.0 * M_PI;
    const T mult = -twopi * twopi * twopi / (lambda * lambda);

    GridGPU *j = j_compute->getResult();

    Grid<char>::GPU *dom = C::getContext()->getGeometry()->getDomainGrid();

    Grid<char>::GPU dom_crop(N0, N0, 1, Type::real);
    dom_crop.insertGrid(dom);

    // Solve ps
    computeVectorPotential(_ps, j, &dom_crop, h, mult);
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

#endif // RICCATI_SOLVER_H_
