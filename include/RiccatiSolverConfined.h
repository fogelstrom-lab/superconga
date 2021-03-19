#ifndef RICCATI_SOLVER_CONFINED_H_
#define RICCATI_SOLVER_CONFINED_H_

#include "Grid.h"
#include "RiccatiSolver.h"
#include "cplx.h"
#include <cmath>
#include <string>

namespace scg {
template <typename T, template <typename> class GAMMA,
          template <typename> class POTENTIAL>
class RiccatiSolverConfined : public RiccatiSolver<T> {
public:
  // Constructors and Destructors

  RiccatiSolverConfined();
  virtual ~RiccatiSolverConfined();

  RiccatiSolverConfined(Context<T> *ctx);

  virtual int initialize();

  // Functions
  virtual void
  computeInitialCondition(const ParameterIterator<int> &energy_iter,
                          const GridGPU<T> *energies, const GridGPU<T> *delta,
                          const int angleIdx);

  virtual void
  computeCoherenceFunctions(const ParameterIterator<int> &energy_iter,
                            const GridGPU<T> *energies, const GridGPU<T> *delta,
                            const int angleIdx);

  void gammaSymmetryTransform();

  // Get a string describing the vector potential type, e.g. solenoid.
  std::string getVectorPotentialType() const { return POTENTIAL<T>::type(); }
};

namespace internal {
// TODO(niclas): Put parameters in order (in, inout, out).
template <typename T, typename GAMMA, int DOMAIN_ENTRY, int DOMAIN_EXIT,
          int DIRECTION>
__global__ void computeGammaFunctionTemplate_new_kernel(
    const T h, const T fermiSpeed, const int N, const int Ne,
    const int energyIdx_offset, const int idx_x_min, const int idx_x_max,
    const int idx_y_min, const int idx_y_max, const int angleIdx,
    const T *const energies_re, const T *const energies_im,
    const T *const energy_shift, T *gamma_real, T *gamma_imag,
    const T *const delta_real, const T *const delta_imag,
    const char *const domainLabel, const T *const domainNormals_x,
    const T *const domainNormals_y, const T *const boundaryDistance,
    const int *const idx_in_gamma, const int *const idx_out_gamma,
    const T *const bnd_in_gamma_real, const T *const bnd_in_gamma_imag,
    T *bnd_out_gamma_real, T *bnd_out_gamma_imag) {
  // x-coordinate
  const int x = blockDim.x * blockIdx.x + threadIdx.x + idx_x_min;
  // energy coordinate
  const int e_idx = blockDim.y * blockIdx.y + threadIdx.y;

  // TODO(niclas): Shouldn't we check that energy coordinate is within bounds as
  // well?
  if (x <= idx_x_max) {
    // Read the energy.
    cplx<T> energy(energies_re[e_idx], energies_im[e_idx]);
    const T energy_zeroPotential = energy.x;

    // In order to read/write values from/to the boundary coherence functions,
    // we need to keep track of how many boundaries we have encountered along
    // the trajectory.
    int boundary_entry_count = 0;
    int boundary_exit_count = 0;

    // Which y-coordinate to start from is determined by whether we are solving
    // for gamma or gammatilde (which is codified by DIRECTION.
    const int y_start =
        idx_y_min * (int)(DIRECTION == 1) + idx_y_max * (int)(DIRECTION == -1);

    // The value of the coherence function at the entry edge of the current
    // lattice site. It is updated as we traverse the trajectory.
    cplx<T> gEntry(static_cast<T>(0), static_cast<T>(0));

    // The step from the entry edge to the exit edge of the lattice site.
    const T step = h / fermiSpeed;

    // The step from the center to and edge (entry or exit) of the lattice site.
    const T halfStep = static_cast<T>(0.5) * step;

    // OPTIMIZATION: PRECOMPUTE Y_MIN AND Y_MAX FOR EACH ROTATIONAL FRAME
    // Compute gamma along trajectory
    for (int i = 0; i <= (idx_y_max - idx_y_min); ++i) {
      const int y = y_start + DIRECTION * i;

      // Index to 2d-grids (Delta)
      const int idx = y * N + x;

      // Add energy shift from vector potential
      energy.x = energy_zeroPotential + energy_shift[idx];

      // Index to gamma grid (for all energies)
      const int gamma_idx = e_idx * 2 * N * N + y * N + x;

      // What type of node are we on?
      const char nodeType = domainLabel[idx];

      // Read order parameter
      const cplx<T> delta(delta_real[idx], delta_imag[idx]);

      if (nodeType == 1) {
        // Inside domain:

#ifdef APPROX_STEP
        // Take a step from the entry edge to the exit edge of the lattice
        // site.
        const cplx<T> gExit = GAMMA::computeStep(delta, energy, gEntry, step);

        // Approximate the value of the coherence function at the center of the
        // lattice site by the mean of the entry and exit values.
        const cplx<T> gCenter = static_cast<T>(0.5) * (gEntry + gExit);
#else
        // Take a step from the entry edge to the center of the lattice site.
        const cplx<T> gCenter =
            GAMMA::computeStep(delta, energy, gEntry, halfStep);

        // Take a step from the center to the exit edge of the lattice site.
        const cplx<T> gExit =
            GAMMA::computeStep(delta, energy, gCenter, halfStep);
#endif

        // Store coherence function.
        gamma_real[gamma_idx] = gCenter.x;
        gamma_imag[gamma_idx] = gCenter.y;

        // For the next lattice site, the current exit value is the entry.
        gEntry = gExit;
      } else if (nodeType == DOMAIN_ENTRY) {
        // Domain boundary entry:

        // Read boundary coherence function value.
        const int idx_sparse = BoundaryStorage<T>::getSparseIndex(
            angleIdx, e_idx + energyIdx_offset, x, boundary_entry_count, Ne, N,
            idx_in_gamma);
        const cplx<T> gBoundary(bnd_in_gamma_real[idx_sparse],
                                bnd_in_gamma_imag[idx_sparse]);

        // Distance to exact boundary location from the center of the lattice
        // site.
        const T distance = abs(boundaryDistance[idx]);
        const T boundaryToCenterStep = distance * step;

#ifdef APPROX_STEP
        // We're taking a step from the grain boundary to the exit edge of the
        // lattice site, so we must add half a lattice site distance.
        const T boundaryToEdgeStep = boundaryToCenterStep + halfStep;

        // Take a step from the grain boundary to the exit edge of the lattice
        // site.
        const cplx<T> gExit =
            GAMMA::computeStep(delta, energy, gBoundary, boundaryToEdgeStep);

        // Approximate the value of the coherence function at the center of the
        // lattice site by the weighted mean of the boundary and exit edge
        // values.
        const T weightBoundary = halfStep / boundaryToEdgeStep;
        const T weightExit = static_cast<T>(1) - weightBoundary;
        const cplx<T> gCenter =
            (gBoundary * weightBoundary + gExit * weightExit);
#else
        // Take a step from the grain boundary to the center of the lattice
        // site.
        const cplx<T> gCenter =
            GAMMA::computeStep(delta, energy, gBoundary, boundaryToCenterStep);

        // Take a step from the center to the exit edge of the lattice site.
        const cplx<T> gExit =
            GAMMA::computeStep(delta, energy, gCenter, halfStep);
#endif

        // Store coherence function.
        gamma_real[gamma_idx] = gCenter.x;
        gamma_imag[gamma_idx] = gCenter.y;

        // For the next lattice site, the current exit value is the entry.
        gEntry = gExit;

        // Increase entry counter.
        boundary_entry_count++;
      } else if (nodeType == DOMAIN_EXIT) {
        // Domain boundary exit:

        // Distance to exact boundary location from the center of the lattice
        // site.
        const T distance = abs(boundaryDistance[idx]);
        const T centerToBoundaryStep = distance * step;

#ifdef APPROX_STEP
        // We're taking a step from the entry edge of thelattice site to the
        // grain boundary, so we must add half a lattice site distance.
        const T edgeToBoundaryStep = centerToBoundaryStep + halfStep;

        // Take a step from the entry edge of the lattice site to the grain
        // boundary.
        const cplx<T> gBoundary =
            GAMMA::computeStep(delta, energy, gEntry, edgeToBoundaryStep);

        // Approximate the center value by the weighted mean.
        const T weightEntry = centerToBoundaryStep / edgeToBoundaryStep;
        const T weightBoundary = static_cast<T>(1) - weightEntry;
        const cplx<T> gCenter =
            (gEntry * weightEntry + gBoundary * weightBoundary);
#else
        // Take a step from the entry edge to the center of the lattice site.
        const cplx<T> gCenter =
            GAMMA::computeStep(delta, energy, gEntry, halfStep);

        // Take a step from the center of the lattice site to the grain
        // boundary.
        const cplx<T> gBoundary =
            GAMMA::computeStep(delta, energy, gCenter, centerToBoundaryStep);
#endif

        // Store coherence function.
        gamma_real[gamma_idx] = gCenter.x;
        gamma_imag[gamma_idx] = gCenter.y;

        // There is no exit edge value to set it to, because we hit a
        // boundary. So the next lattice site is outside of the domain.
        gEntry = cplx<T>(static_cast<T>(0), static_cast<T>(0));

        // Store boundary gamma.
        const int idx_sparse = BoundaryStorage<T>::getSparseIndex(
            angleIdx, e_idx + energyIdx_offset, x, boundary_exit_count, Ne, N,
            idx_out_gamma);
        bnd_out_gamma_real[idx_sparse] = gBoundary.x;
        bnd_out_gamma_imag[idx_sparse] = gBoundary.y;

        // Increase exit counter.
        boundary_exit_count++;
      } else {
        // Outside the domain.
        gamma_real[gamma_idx] = static_cast<T>(0);
        gamma_imag[gamma_idx] = static_cast<T>(0);
      }
    }
  }
}

/////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
// TODO(niclas): This function does not need all of these input parameters!
template <typename T, typename GAMMA, int DOMAIN_ENTRY, int DOMAIN_EXIT,
          int DIRECTION>
__global__ void computeGammaFunctionInitialBoundaryTemplate_kernel(
    const T h, const int N, const int Ne, const int energyIdx_offset,
    const int idx_x_offset, const int angleIdx, const T *const energies_re,
    const T *const energies_im, const T *const energy_shift,
    const T *const gamma_real, const T *const gamma_imag,
    const T *const delta_real, const T *const delta_imag,
    const char *const domainLabel, const T *const domainNormals_x,
    const T *const domainNormals_y, const T *const boundaryDistance,
    const int *const idx_in_gamma, const int *const idx_out_gamma,
    const T *const bnd_in_gamma_real, const T *const bnd_in_gamma_imag,
    T *bnd_out_gamma_real, T *bnd_out_gamma_imag) {
  // x-coordinate
  const int x = blockDim.x * blockIdx.x + threadIdx.x + idx_x_offset;
  // energy coordinate
  const int e_idx = blockDim.y * blockIdx.y + threadIdx.y;

  if (x < N) {
    cplx<T> energy(energies_re[e_idx], energies_im[e_idx]);
    const T energy_zeroPotential = energy.x;

    int boundary_exit_count = 0;

    const int yStart = (N - 1) * (int)(DIRECTION == -1);

    // Compute gamma along trajectory
    for (int i = 0; i < N; ++i) {
      const int y = yStart + DIRECTION * i;

      // Index to 2d-grids (Delta)
      const int idx = y * N + x;

      // Add energy shift from vector potential
      energy.x = energy_zeroPotential + energy_shift[idx];

      // What type of node are we on?
      const char nodeType = domainLabel[idx];

      if (nodeType == DOMAIN_EXIT) {
        // Read order parameter
        const cplx<T> delta(delta_real[idx], delta_imag[idx]);

        // Compute gammas and store
        const int idx_sparse = BoundaryStorage<T>::getSparseIndex(
            angleIdx, e_idx + energyIdx_offset, x, boundary_exit_count, Ne, N,
            idx_out_gamma);

        // Compute gamma at current grid point
        const cplx<T> gBoundary = GAMMA::computeHomogeneous(delta, energy);

        bnd_out_gamma_real[idx_sparse] = gBoundary.x;
        bnd_out_gamma_imag[idx_sparse] = gBoundary.y;

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
#ifndef NDEBUG
  std::cout << "RiccatiSolverConfined::RiccatiSolverConfined( Context<T>* )\n";
#endif
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
    const ParameterIterator<int> &energy_iter, const GridGPU<T> *energies,
    const GridGPU<T> *deltaRot, const int current_angle_idx) {
  typedef RiccatiSolver<T> R;
  typedef ContextModule<T> C;

  // Get model parameters
  Parameters<T> *param = C::getParameters();

  // Determine some scalars
  const T h = param->getGridElementSize();
  const int N = param->getGridResolutionCoherence();
  const int numAngles = param->getAngularResolution();

  // Get pointers to grids
  GridGPU<T> *gamma = R::getGamma();
  GridGPU<T> *gammabar = R::getGammabar();

  GeometryGroup<T> *geom = R::getGeometry();
  BoundaryStorage<T> *storage = R::getBoundaryStorage();

  dim3 blocksPerGrid, threadsPerBlock;

  const int numEnergiesCurrentBlock = energy_iter.getNumInterval();
  const int energyIdxOffset = energy_iter.getCurrentValue();

  const int numEnergies_storage =
      C::getIntegrationIterator()->getNumEnergiesInterval();
  const int energyIdxOffset_storage =
      C::getIntegrationIterator()->getEnergyIdxOffset_boundaryStorage();

  R::getKernelDimensionsOptim(blocksPerGrid, threadsPerBlock,
                              numEnergiesCurrentBlock, current_angle_idx);

  const int idx_x_min =
      (storage->getTrajectoryRangeX()->getDataPointer(0))[current_angle_idx];
  const int idx_x_max =
      (storage->getTrajectoryRangeX()->getDataPointer(1))[current_angle_idx];

  const int idx_y_min =
      (storage->getTrajectoryRangeY()->getDataPointer(0))[current_angle_idx];
  const int idx_y_max =
      (storage->getTrajectoryRangeY()->getDataPointer(1))[current_angle_idx];

  // Gives 10-15% speedup with streams
  cudaStream_t s1, s2;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);

  // Make riccati kernel read vector potential via precomputed field.
  // To allow for efficient computation with no vector potential, it is accessed
  // via a templated function object.

  // Compute energy shift from magnetic field
  GridGPU<T> *eShift = R::getEnergyShift();

  // Fetch the vector potential.
  GridGPU<T> *inducedVectorPotential =
      C::getContext()->getInducedVectorPotential();

  computeEnergyShift<T, POTENTIAL>(this, inducedVectorPotential, eShift,
                                   current_angle_idx);

  const T fermiSpeed = C::getFermiSurface()->getFermiSpeed(current_angle_idx);

  // 	Solve gamma by propagating riccati trajectories along positive y
  internal::computeGammaFunctionTemplate_new_kernel<T, typename GAMMA<T>::Gamma,
                                                    2, 3, 1>
      <<<blocksPerGrid, threadsPerBlock, 0, s1>>>(
          // Model parameters and energy values
          h, fermiSpeed, N, numEnergies_storage, energyIdxOffset_storage,
          idx_x_min, idx_x_max, idx_y_min, idx_y_max, current_angle_idx,
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
          idx_x_min, idx_x_max, idx_y_min, idx_y_max, current_angle_idx,
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

  GridGPU<T> *gamma = R::getGamma();
  GridGPU<T> *gammabar = R::getGammabar();

  // Do this ONLY if testing negative vF
  GridGPU<T> gammaTmp(gamma);
  gammaTmp = *gamma;

  *gamma = *gammabar;
  gamma->conjugate();

  *gammabar = gammaTmp;
  gammabar->conjugate();
}

template <typename T, template <typename> class GAMMA,
          template <typename> class POTENTIAL>
void RiccatiSolverConfined<T, GAMMA, POTENTIAL>::computeInitialCondition(
    const ParameterIterator<int> &energy_iter, const GridGPU<T> *energies,
    const GridGPU<T> *deltaRot, const int current_angle_idx) {
  typedef RiccatiSolver<T> R;
  typedef ContextModule<T> C;

  // Get model parameters
  Parameters<T> *param = C::getParameters();

  // Determine some scalars
  const T h = param->getGridElementSize();
  const int N = param->getGridResolutionCoherence();
  const int numAngles = param->getAngularResolution();

  // Get pointers to grids
  GridGPU<T> *gamma = R::getGamma();
  GridGPU<T> *gammabar = R::getGammabar();

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
  GridGPU<T> *eShift = R::getEnergyShift();

  // 	Solve gamma by propagating riccati trajectories along positive y
  internal::computeGammaFunctionInitialBoundaryTemplate_kernel<
      T, typename GAMMA<T>::Gamma, 2, 3, 1><<<blocksPerGrid, threadsPerBlock>>>(
      // Model parameters and energy values
      h, N, numEnergies_storage, energyIdxOffset_storage, trajectoryIdx_start,
      current_angle_idx,
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
          trajectoryIdx_start, current_angle_idx,
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
} // namespace scg

#endif // RICCATI_SOLVER_CONFINED_H_
