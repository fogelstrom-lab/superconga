#ifndef RICCATI_SOLVER_CONFINED_H_
#define RICCATI_SOLVER_CONFINED_H_

#include "RiccatiSolver.h"
#include "cplx.h"
#include <cmath>
#include <string>

template <typename T, template <typename> class GAMMA,
          template <typename> class POTENTIAL>
class RiccatiSolverConfined : public RiccatiSolver<T> {
  typedef typename Grid<T>::GPU GridGPU;

public:
  // Constructors and Destructors

  RiccatiSolverConfined();
  virtual ~RiccatiSolverConfined();

  RiccatiSolverConfined(Context<T> *ctx);

  virtual int initialize();

  // Functions
  virtual void
  computeInitialCondition(const ParameterIterator<int> &energy_iter,
                          const GridGPU *energies, const GridGPU *delta,
                          const int angleIdx);

  virtual void
  computeCoherenceFunctions(const ParameterIterator<int> &energy_iter,
                            const GridGPU *energies, const GridGPU *delta,
                            const int angleIdx);

  void gammaSymmetryTransform();

  // Get a string describing the vector potential type, e.g. solenoid.
  std::string getVectorPotentialType() const { return POTENTIAL<T>::type(); }
};

namespace internal {
template <typename T, typename GAMMA, int ENTRY, int EXIT, int DIR>
__global__ void computeGammaFunctionTemplate_new_kernel(
    const T h, const T fermiSpeed, const int N, const int Ne,
    const int energyIdx_offset, const int idx_x_min, const int idx_x_max,
    const int idx_y_min, const int idx_y_max, const int angleIdx, const T B,
    T *energies_re, T *energies_im, T *energy_shift, T *gamma_real,
    T *gamma_imag, T *delta_real, T *delta_imag, char *domainLabel,
    T *domainNormals_x, T *domainNormals_y, T *boundaryDistance,
    int *idx_in_gamma, int *idx_out_gamma, T *bnd_in_gamma_real,
    T *bnd_in_gamma_imag, T *bnd_out_gamma_real, T *bnd_out_gamma_imag) {
  const int x =
      blockDim.x * blockIdx.x + threadIdx.x + idx_x_min;   // x-coordinate
  const int e_idx = blockDim.y * blockIdx.y + threadIdx.y; // energy coordinate

  cplx<T> g, g0((T)0.0, (T)0.0), g_bnd, energy;
  cplx<T> delta, D2, D_bnd;

  int idx, gamma_idx, idx_sparse;
  char nodeType;

  const T step = h / fermiSpeed;

  if (x <= idx_x_max) {
    energy.x = energies_re[e_idx];
    energy.y = energies_im[e_idx];

    const T energy_zeroPotential = energy.x;

    int boundary_entry_count = 0;
    int boundary_exit_count = 0;

    const int y_start =
        idx_y_min * (int)(DIR == 1) + idx_y_max * (int)(DIR == -1);
    int y;

    // OPTIMIZATION: PRECOMPUTE Y_MIN AND Y_MAX FOR EACH ROTATIONAL FRAME
    // Compute gamma along trajectory
    for (int i = 0; i <= (idx_y_max - idx_y_min); ++i) {
      y = y_start + DIR * i;

      // Index to 2d-grids (Delta)
      idx = y * N + x;

      // Add energy shift from vector potential
      energy.x = energy_zeroPotential + energy_shift[idx];

      // Index to gamma grid (for all energies)
      gamma_idx = e_idx * 2 * N * N + y * N + x;

      // What type of node are we on?
      nodeType = domainLabel[idx];

      // Read order parameter
      delta.x = delta_real[idx];
      delta.y = delta_imag[idx];

      g.x = (T)0.0;
      g.y = (T)0.0;

      if (nodeType == 1) // Inside domain:
      {
        GAMMA::computeGamma_local(delta, g0, energy, step, 1, g);
      } else if (nodeType == ENTRY) // Domain boundary entry:
      {
        // Read gamma from boundary storage, and calc first gamma to grid,
        // using fractional step.
        idx_sparse = BoundaryStorage<T>::getSparseIndex(
            angleIdx, e_idx + energyIdx_offset, x, boundary_entry_count, Ne, N,
            idx_in_gamma);

        g0.x = bnd_in_gamma_real[idx_sparse];
        g0.y = bnd_in_gamma_imag[idx_sparse];

        // Distance to exact boundary location
        const T distance = abs(boundaryDistance[idx]);
        const T fractionalStep = distance * step;

        // Extrapolate delta at boundary from domain values
        D2.x = delta_real[(y + DIR) * N + x];
        D2.y = delta_imag[(y + DIR) * N + x];

        D_bnd = delta - (D2 - delta) * distance;

        // Compute gamma at current grid point, using exact distance to boundary
        GAMMA::computeGamma_local(D_bnd, g0, energy, fractionalStep, 1, g);

        boundary_entry_count++;
      } else if (nodeType == EXIT) // Domain boundary exit:
      {
        // Calc gamma to current grid node, then calc gamma with fractional
        // step at exact boundary location, and save in boundary storage.
        GAMMA::computeGamma_local(delta, g0, energy, step, 1, g);

        // Distance to exact boundary location
        const T fractionalStep = abs(boundaryDistance[idx]) * step;

        // Calc at exact boundary location
        GAMMA::computeGamma_local(delta, g0, energy, step + fractionalStep, 1,
                                  g_bnd);

        // Store boundary gammas
        idx_sparse = BoundaryStorage<T>::getSparseIndex(
            angleIdx, e_idx + energyIdx_offset, x, boundary_exit_count, Ne, N,
            idx_out_gamma);

        bnd_out_gamma_real[idx_sparse] = g_bnd.x;
        bnd_out_gamma_imag[idx_sparse] = g_bnd.y;

        boundary_exit_count++;
      }

      // Store gammas
      gamma_real[gamma_idx] = g.x;
      gamma_imag[gamma_idx] = g.y;

      g0 = g;
    }
  }
}

/////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
template <typename T, typename GAMMA, int ENTRY, int EXIT, int DIR>
__global__ void computeGammaFunctionInitialBoundaryTemplate_kernel(
    const T h, const int N, const int Ne, const int energyIdx_offset,
    const int idx_x_offset, const int angleIdx, const T B, T *energies_re,
    T *energies_im, T *energy_shift, T *gamma_real, T *gamma_imag,
    T *delta_real, T *delta_imag, char *domainLabel, T *domainNormals_x,
    T *domainNormals_y, T *boundaryDistance, int *idx_in_gamma,
    int *idx_out_gamma, T *bnd_in_gamma_real, T *bnd_in_gamma_imag,
    T *bnd_out_gamma_real, T *bnd_out_gamma_imag) {
  const int x =
      blockDim.x * blockIdx.x + threadIdx.x + idx_x_offset; // x-coordinate
  const int e_idx = blockDim.y * blockIdx.y + threadIdx.y;  // energy coordinate

  cplx<T> delta, g_bnd, energy;

  int idx, idx_sparse;
  char nodeType;

  if (x < N) {
    energy.x = energies_re[e_idx];
    energy.y = energies_im[e_idx];

    const T energy_zeroPotential = energy.x;

    int boundary_exit_count = 0;

    const int yStart = (N - 1) * (int)(DIR == -1);
    int y;

    // Compute gamma along trajectory
    for (int i = 0; i < N; ++i) {
      y = yStart + DIR * i;

      // Index to 2d-grids (Delta)
      idx = y * N + x;

      // Add energy shift from vector potential
      energy.x = energy_zeroPotential + energy_shift[idx];

      // What type of node are we on?
      nodeType = domainLabel[idx];

      if (nodeType == EXIT) {
        // Read order parameter
        delta.x = delta_real[idx];
        delta.y = delta_imag[idx];

        // Compute gammas and store
        idx_sparse = BoundaryStorage<T>::getSparseIndex(
            angleIdx, e_idx + energyIdx_offset, x, boundary_exit_count, Ne, N,
            idx_out_gamma);

        // Compute gamma at current grid point
        GAMMA::computeGamma_local(delta, cplx<T>(0.0, 0.0), energy, (T)0.0, 0,
                                  g_bnd);

        bnd_out_gamma_real[idx_sparse] = g_bnd.x;
        bnd_out_gamma_imag[idx_sparse] = g_bnd.y;

        boundary_exit_count++;
      }
    }
  }
}
} // namespace internal

template <typename T, template <typename> class GAMMA,
          template <typename> class POTENTIAL>
RiccatiSolverConfined<T, GAMMA, POTENTIAL>::RiccatiSolverConfined()
    : RiccatiSolver<T>() {}

template <typename T, template <typename> class GAMMA,
          template <typename> class POTENTIAL>
RiccatiSolverConfined<T, GAMMA, POTENTIAL>::~RiccatiSolverConfined() {}

template <typename T, template <typename> class GAMMA,
          template <typename> class POTENTIAL>
RiccatiSolverConfined<T, GAMMA, POTENTIAL>::RiccatiSolverConfined(
    Context<T> *ctx)
    : RiccatiSolver<T>(ctx) {
  std::cout << "RiccatiSolverConfined::RiccatiSolverConfined( Context<T>* )\n";
  ctx->set(this);
}

template <typename T, template <typename> class GAMMA,
          template <typename> class POTENTIAL>
int RiccatiSolverConfined<T, GAMMA, POTENTIAL>::initialize() {
  RiccatiSolver<T>::initialize();
  RiccatiSolver<T>::initializeBoundaryStorage();

  return 0;
}

template <typename T, template <typename> class GAMMA,
          template <typename> class POTENTIAL>
void RiccatiSolverConfined<T, GAMMA, POTENTIAL>::computeCoherenceFunctions(
    const ParameterIterator<int> &energy_iter, const GridGPU *energies,
    const GridGPU *deltaRot, const int current_angle_idx) {
  typedef RiccatiSolver<T> R;
  typedef ContextModule<T> C;

  // Get model parameters
  Parameters<T> *param = C::getParameters();

  // Determine some scalars
  const T h = param->getGridElementSize();
  const T B = param->getMagneticFieldStrength();
  const int N = param->getGridResolutionCoherence();
  const int numAngles = param->getAngularResolution();

  // Get pointers to grids
  GridGPU *gamma = R::getGamma();
  GridGPU *gammabar = R::getGammabar();

  GeometryGroup<T> *geom = R::getGeometry();
  BoundaryStorage<T> *storage = R::getBoundaryStorage();

  dim3 blocksPerGrid, threadsPerBlock;

  int numEnergiesCurrentBlock = energy_iter.getNumInterval();
  int energyIdxOffset = energy_iter.getCurrentValue();

  int numEnergies_storage =
      C::getIntegrationIterator()->getNumEnergiesInterval();
  int energyIdxOffset_storage =
      C::getIntegrationIterator()->getEnergyIdxOffset_boundaryStorage();

  R::getKernelDimensionsOptim(blocksPerGrid, threadsPerBlock,
                              numEnergiesCurrentBlock, current_angle_idx);

  int idx_x_min =
      (storage->getTrajectoryRangeX()->getDataPointer(0))[current_angle_idx];
  int idx_x_max =
      (storage->getTrajectoryRangeX()->getDataPointer(1))[current_angle_idx];

  int idx_y_min =
      (storage->getTrajectoryRangeY()->getDataPointer(0))[current_angle_idx];
  int idx_y_max =
      (storage->getTrajectoryRangeY()->getDataPointer(1))[current_angle_idx];

  // Gives 10-15% speedup with streams
  cudaStream_t s1, s2;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);

  // Make riccati kernel read vector potential via precomputed field.
  // To allow for efficient computation with no vector potential, it is accessed
  // via a templated function object.

  // Compute energy shift from magnetic field
  GridGPU *eShift = R::getEnergyShift();
  GridGPU *ps = R::getInducedVectorPotential();

  int doInducedBfield = R::doComputeInducedMagneticField();

  computeEnergyShift<T, POTENTIAL>(this, ps, eShift, doInducedBfield,
                                   current_angle_idx);

  const T fermiSpeed = C::getFermiSurface()->getFermiSpeed(current_angle_idx);

  // 	Solve gamma by propagating riccati trajectories along positive y
  internal::computeGammaFunctionTemplate_new_kernel<T, typename GAMMA<T>::Gamma,
                                                    2, 3, 1>
      <<<blocksPerGrid, threadsPerBlock, 0, s1>>>(
          // Model parameters and energy values
          h, fermiSpeed, N, numEnergies_storage, energyIdxOffset_storage,
          idx_x_min, idx_x_max, idx_y_min, idx_y_max, current_angle_idx, B,
          energies->getDataPointer(0, Type::real) + energyIdxOffset,
          energies->getDataPointer(0, Type::imag) + energyIdxOffset,
          eShift->getDataPointer(),
          // Grids for gamma and Delta
          gamma->getDataPointer(0, Type::real),
          gamma->getDataPointer(0, Type::imag),
          deltaRot->getDataPointer(0, Type::real),
          deltaRot->getDataPointer(0, Type::imag),
          // Grids for domain info/data
          geom->getDomainLabelsGrid()->getDataPointer(0, Type::real),
          geom->getNormalsGrid()->getDataPointer(0, Type::real),
          geom->getNormalsGrid()->getDataPointer(1, Type::real),
          geom->getBoundaryDistanceGrid()->getDataPointer(0, Type::real),
          // Data relating to sparse storage for boundary gammas
          storage->getIndexSparseIn_gamma()->getDataPointer(0, Type::real),
          storage->getIndexSparseOut_gamma()->getDataPointer(0, Type::real),
          storage->getBoundaryIn_gamma()->getDataPointer(0, Type::real),
          storage->getBoundaryIn_gamma()->getDataPointer(0, Type::imag),
          storage->getBoundaryOut_gamma()->getDataPointer(0, Type::real),
          storage->getBoundaryOut_gamma()->getDataPointer(0, Type::imag));

  // 	Solve gammabar by propagating riccati trajectories along negative y
  internal::computeGammaFunctionTemplate_new_kernel<
      T, typename GAMMA<T>::Gammabar, 3, 2, -1>
      <<<blocksPerGrid, threadsPerBlock, 0, s2>>>(
          // Model parameters and energy values
          h, fermiSpeed, N, numEnergies_storage, energyIdxOffset_storage,
          idx_x_min, idx_x_max, idx_y_min, idx_y_max, current_angle_idx, B,
          energies->getDataPointer(0, Type::real) + energyIdxOffset,
          energies->getDataPointer(0, Type::imag) + energyIdxOffset,
          eShift->getDataPointer(),
          // Grids for gamma and Delta
          gammabar->getDataPointer(0, Type::real),
          gammabar->getDataPointer(0, Type::imag),
          deltaRot->getDataPointer(0, Type::real),
          deltaRot->getDataPointer(0, Type::imag),
          // Grids for domain info/data
          geom->getDomainLabelsGrid()->getDataPointer(0, Type::real),
          geom->getNormalsGrid()->getDataPointer(0, Type::real),
          geom->getNormalsGrid()->getDataPointer(1, Type::real),
          geom->getBoundaryDistanceGrid()->getDataPointer(0, Type::real),
          // Data relating to sparse storage for boundary gammas
          storage->getIndexSparseIn_gammabar()->getDataPointer(0, Type::real),
          storage->getIndexSparseOut_gammabar()->getDataPointer(0, Type::real),
          storage->getBoundaryIn_gammabar()->getDataPointer(0, Type::real),
          storage->getBoundaryIn_gammabar()->getDataPointer(0, Type::imag),
          storage->getBoundaryOut_gammabar()->getDataPointer(0, Type::real),
          storage->getBoundaryOut_gammabar()->getDataPointer(0, Type::imag));

  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);
}

template <typename T, template <typename> class GAMMA,
          template <typename> class POTENTIAL>
void RiccatiSolverConfined<T, GAMMA, POTENTIAL>::gammaSymmetryTransform() {
  typedef RiccatiSolver<T> R;

  GridGPU *gamma = R::getGamma();
  GridGPU *gammabar = R::getGammabar();

  // Do this ONLY if testing negative vF
  GridGPU gammaTmp(gamma);
  gammaTmp = *gamma;

  *gamma = *gammabar;
  gamma->conjugate();

  *gammabar = gammaTmp;
  gammabar->conjugate();
}

template <typename T, template <typename> class GAMMA,
          template <typename> class POTENTIAL>
void RiccatiSolverConfined<T, GAMMA, POTENTIAL>::computeInitialCondition(
    const ParameterIterator<int> &energy_iter, const GridGPU *energies,
    const GridGPU *deltaRot, const int current_angle_idx) {
  typedef RiccatiSolver<T> R;
  typedef ContextModule<T> C;

  // Get model parameters
  Parameters<T> *param = C::getParameters();

  // Determine some scalars
  const T h = param->getGridElementSize();
  const T B = param->getMagneticFieldStrength();
  const int N = param->getGridResolutionCoherence();
  const int numAngles = param->getAngularResolution();

  // Get pointers to grids
  GridGPU *gamma = R::getGamma();
  GridGPU *gammabar = R::getGammabar();

  GeometryGroup<T> *geom = R::getGeometry();
  BoundaryStorage<T> *storage = R::getBoundaryStorage();

  dim3 blocksPerGrid, threadsPerBlock;

  int numEnergiesCurrentBlock = energy_iter.getNumInterval();
  int energyIdxOffset = energy_iter.getCurrentValue();

  int numEnergies_storage =
      C::getIntegrationIterator()->getNumEnergiesInterval();
  int energyIdxOffset_storage =
      C::getIntegrationIterator()->getEnergyIdxOffset_boundaryStorage();

  // Get kernel dimensions
  int trajectoryIdx_start =
      storage->getTrajectoryStartIndexAngle(current_angle_idx);
  R::getKernelDimensionsOptim(blocksPerGrid, threadsPerBlock,
                              numEnergiesCurrentBlock, current_angle_idx);

  // Get energy shift from magnetic field
  GridGPU *eShift = R::getEnergyShift();

  // 	Solve gamma by propagating riccati trajectories along positive y
  internal::computeGammaFunctionInitialBoundaryTemplate_kernel<
      T, typename GAMMA<T>::Gamma, 2, 3, 1><<<blocksPerGrid, threadsPerBlock>>>(
      // Model parameters and energy values
      h, N, numEnergies_storage, energyIdxOffset_storage, trajectoryIdx_start,
      current_angle_idx, B,
      energies->getDataPointer(0, Type::real) + energyIdxOffset,
      energies->getDataPointer(0, Type::imag) + energyIdxOffset,
      eShift->getDataPointer(),
      // Grids for gamma and Delta
      gamma->getDataPointer(0, Type::real),
      gamma->getDataPointer(0, Type::imag),
      deltaRot->getDataPointer(0, Type::real),
      deltaRot->getDataPointer(0, Type::imag),
      // Grids for domain info/data
      geom->getDomainLabelsGrid()->getDataPointer(0, Type::real),
      geom->getNormalsGrid()->getDataPointer(0, Type::real),
      geom->getNormalsGrid()->getDataPointer(1, Type::real),
      geom->getBoundaryDistanceGrid()->getDataPointer(0, Type::real),
      // Data relating to sparse storage for boundary gammas
      storage->getIndexSparseIn_gamma()->getDataPointer(0, Type::real),
      storage->getIndexSparseOut_gamma()->getDataPointer(0, Type::real),
      storage->getBoundaryIn_gamma()->getDataPointer(0, Type::real),
      storage->getBoundaryIn_gamma()->getDataPointer(0, Type::imag),
      storage->getBoundaryOut_gamma()->getDataPointer(0, Type::real),
      storage->getBoundaryOut_gamma()->getDataPointer(0, Type::imag));

  // 	Solve gammabar by propagating riccati trajectories along negative y
  internal::computeGammaFunctionInitialBoundaryTemplate_kernel<
      T, typename GAMMA<T>::Gammabar, 3, 2, -1>
      <<<blocksPerGrid, threadsPerBlock>>>(
          // Model parameters and energy values
          h, N, numEnergies_storage, energyIdxOffset_storage,
          trajectoryIdx_start, current_angle_idx, B,
          energies->getDataPointer(0, Type::real) + energyIdxOffset,
          energies->getDataPointer(0, Type::imag) + energyIdxOffset,
          eShift->getDataPointer(),
          // Grids for gammabar and Delta
          gammabar->getDataPointer(0, Type::real),
          gammabar->getDataPointer(0, Type::imag),
          deltaRot->getDataPointer(0, Type::real),
          deltaRot->getDataPointer(0, Type::imag),
          // Grids for domain info/data
          geom->getDomainLabelsGrid()->getDataPointer(0, Type::real),
          geom->getNormalsGrid()->getDataPointer(0, Type::real),
          geom->getNormalsGrid()->getDataPointer(1, Type::real),
          geom->getBoundaryDistanceGrid()->getDataPointer(0, Type::real),
          // Data relating to sparse storage for boundary gammas
          storage->getIndexSparseIn_gammabar()->getDataPointer(0, Type::real),
          storage->getIndexSparseOut_gammabar()->getDataPointer(0, Type::real),
          storage->getBoundaryIn_gammabar()->getDataPointer(0, Type::real),
          storage->getBoundaryIn_gammabar()->getDataPointer(0, Type::imag),
          storage->getBoundaryOut_gammabar()->getDataPointer(0, Type::real),
          storage->getBoundaryOut_gammabar()->getDataPointer(0, Type::imag));
}

#endif // RICCATI_SOLVER_CONFINED_H_
