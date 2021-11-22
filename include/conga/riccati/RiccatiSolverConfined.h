//= RiccatiSolverConfined.h - Solve Riccati equations in confined geometry. =//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Riccati solver implementation for confined geometries, i.e. finite system
/// confined by well-defined boundaries. Derives from the virtual class
/// RiccatiSolver.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_RICCATI_SOLVER_CONFINED_H_
#define CONGA_RICCATI_SOLVER_CONFINED_H_

#include "../grid.h"
#include "RiccatiSolver.h"

#include <thrust/complex.h>

#include <cmath>
#include <string>

namespace conga {
template <typename T, template <typename> class POTENTIAL>
class RiccatiSolverConfined : public RiccatiSolver<T> {
public:
  // Constructors and Destructors
  RiccatiSolverConfined();
  ~RiccatiSolverConfined();

  RiccatiSolverConfined(Context<T> *ctx);

  int initialize() override;

  // Functions
  void computeInitialCondition(const ParameterIterator<int> &energy_iter,
                               const GridGPU<T> *energies,
                               const GridGPU<T> *delta,
                               const int angleIdx) override;

  void computeCoherenceFunctions(const ParameterIterator<int> &energy_iter,
                                 const GridGPU<T> *energies,
                                 const GridGPU<T> *delta,
                                 const int angleIdx) override;

  // Get a string describing the vector potential type, e.g. solenoid.
  std::string getVectorPotentialType() const { return POTENTIAL<T>::type(); }
};

namespace internal {
// TODO(niclas): Put parameters in order (in, inout, out).
template <typename T, typename Coherence, int DOMAIN_ENTRY, int DOMAIN_EXIT,
          int DIRECTION>
__global__ void computeGammaFunctionTemplate_new_kernel(
    const T inStep, const int N, const int Ne, const int energyIdx_offset,
    const int idx_x_min, const int idx_x_max, const int idx_y_min,
    const int idx_y_max, const int angleIdx, const T *const energies_re,
    const T *const energies_im, const T *const energy_shift, T *gamma_real,
    T *gamma_imag, const T *const delta_real, const T *const delta_imag,
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
    const thrust::complex<T> energyZeroPotential(energies_re[e_idx],
                                                 energies_im[e_idx]);

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
    thrust::complex<T> gEntry(static_cast<T>(0), static_cast<T>(0));

    // The step from the center to and edge (entry or exit) of the lattice site.
    const T halfStep = static_cast<T>(0.5) * inStep;

    // OPTIMIZATION: PRECOMPUTE Y_MIN AND Y_MAX FOR EACH ROTATIONAL FRAME
    // Compute gamma along trajectory
    for (int i = 0; i <= (idx_y_max - idx_y_min); ++i) {
      const int y = y_start + DIRECTION * i;

      // Index to 2d-grids (Delta)
      const int idx = y * N + x;

      // Add energy shift from vector potential
      const thrust::complex<T> energy = energyZeroPotential + energy_shift[idx];

      // Index to gamma grid (for all energies)
      const int gamma_idx = e_idx * 2 * N * N + y * N + x;

      // What type of node are we on?
      const char nodeType = domainLabel[idx];

      // Read order parameter
      const thrust::complex<T> delta(delta_real[idx], delta_imag[idx]);

      // Read homogeneous solution.
      const thrust::complex<T> gHomo(gamma_real[gamma_idx],
                                     gamma_imag[gamma_idx]);

      if (nodeType == 1) {
        // Inside domain:

        // Take a step from the entry edge to the center of the lattice site,
        // and from the center to the exit edge of the lattice site.
        thrust::complex<T> gCenter;
        thrust::complex<T> gExit;
        Coherence::computeHalfSteps(gHomo, delta, energy, gEntry, inStep,
                                    gCenter, gExit);

        // Store coherence function.
        gamma_real[gamma_idx] = gCenter.real();
        gamma_imag[gamma_idx] = gCenter.imag();

        // For the next lattice site, the current exit value is the entry.
        gEntry = gExit;
      } else if (nodeType == DOMAIN_ENTRY) {
        // Domain boundary entry:

        // Read boundary coherence function value.
        const int idx_sparse = BoundaryStorage<T>::getSparseIndex(
            angleIdx, e_idx + energyIdx_offset, x, boundary_entry_count, Ne, N,
            idx_in_gamma);
        const thrust::complex<T> gBoundary(bnd_in_gamma_real[idx_sparse],
                                           bnd_in_gamma_imag[idx_sparse]);

        // Distance to exact boundary location from the center of the lattice
        // site.
        const T distance = abs(boundaryDistance[idx]);
        const T boundaryToCenterStep = distance * inStep;

        // Take a step from the grain boundary to the center of the lattice
        // site.
        const thrust::complex<T> gCenter = Coherence::computeStep(
            gHomo, delta, energy, gBoundary, boundaryToCenterStep);

        // Take a step from the center to the exit edge of the lattice site.
        const thrust::complex<T> gExit =
            Coherence::computeStep(gHomo, delta, energy, gCenter, halfStep);

        // Store coherence function.
        gamma_real[gamma_idx] = gCenter.real();
        gamma_imag[gamma_idx] = gCenter.imag();

        // For the next lattice site, the current exit value is the entry.
        gEntry = gExit;

        // Increase entry counter.
        boundary_entry_count++;
      } else if (nodeType == DOMAIN_EXIT) {
        // Domain boundary exit:

        // Distance to exact boundary location from the center of the lattice
        // site.
        const T distance = abs(boundaryDistance[idx]);
        const T centerToBoundaryStep = distance * inStep;

        // Take a step from the entry edge to the center of the lattice site.
        const thrust::complex<T> gCenter =
            Coherence::computeStep(gHomo, delta, energy, gEntry, halfStep);

        // Take a step from the center of the lattice site to the grain
        // boundary.
        const thrust::complex<T> gBoundary = Coherence::computeStep(
            gHomo, delta, energy, gCenter, centerToBoundaryStep);

        // Store coherence function.
        gamma_real[gamma_idx] = gCenter.real();
        gamma_imag[gamma_idx] = gCenter.imag();

        // There is no exit edge value to set it to, because we hit a
        // boundary. So the next lattice site is outside of the domain.
        gEntry = thrust::complex<T>(static_cast<T>(0), static_cast<T>(0));

        // Store boundary gamma.
        const int idx_sparse = BoundaryStorage<T>::getSparseIndex(
            angleIdx, e_idx + energyIdx_offset, x, boundary_exit_count, Ne, N,
            idx_out_gamma);
        bnd_out_gamma_real[idx_sparse] = gBoundary.real();
        bnd_out_gamma_imag[idx_sparse] = gBoundary.imag();

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

/// \brief Compute the homogenous solutions to the Riccati equations.
///
/// \param inDimX The number of lattice sites along the x-axis.
/// \param inDimY The number of lattice sites along the y-axis.
/// \param inNumEnergies The number of energies.
/// \param inEnergyReal The real part of the energies.
/// \param inEnergyImag The imag part of the energies.
/// \param inEnergyShift The spatial energy shift due to the vector potential.
/// \param inOrderParameterReal The real part of the order parameter.
/// \param inOrderParameterImag The imag part of the order parameter.
/// \param outGammaReal The real part of the coherence function gamma.
/// \param outGammaImag The imag part of the coherence function gamma.
/// \param outGammaTildeReal The real part of the coherence function gammaTilde.
/// \param outGammaTildeImag The imag part of the coherence function gammaTilde.
template <typename T>
__global__ void
computeHomogeneous(const int inDimX, const int inDimY, const int inNumEnergies,
                   const T *const inEnergyReal, const T *const inEnergyImag,
                   const T *const inEnergyShift,
                   const T *const inOrderParameterReal,
                   const T *const inOrderParameterImag, T *const outGammaReal,
                   T *const outGammaImag, T *const outGammaTildeReal,
                   T *const outGammaTildeImag) {
  // Spatial coordinates.
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;

  // Energy coordinate.
  const int z = blockDim.z * blockIdx.z + threadIdx.z;

  // Check if within range.
  if (x < inDimX && y < inDimY && z < inNumEnergies) {
    // Indices.
    const int gamma_idx = z * 2 * inDimX * inDimY + y * inDimX + x;
    const int idx = y * inDimX + x;

    // Construct complex energy and order parameter.
    const thrust::complex<T> energy(inEnergyReal[z] + inEnergyShift[idx],
                                    inEnergyImag[z]);
    const thrust::complex<T> orderParameter(inOrderParameterReal[idx],
                                            inOrderParameterImag[idx]);

    // Construct the homogeneous solutions.
    thrust::complex<T> gammaHomogeneous;
    thrust::complex<T> gammaTildeHomogeneous;
    riccati::computeHomogeneous(orderParameter, energy, gammaHomogeneous,
                                gammaTildeHomogeneous);

    // Write.
    outGammaReal[gamma_idx] = gammaHomogeneous.real();
    outGammaImag[gamma_idx] = gammaHomogeneous.imag();
    outGammaTildeReal[gamma_idx] = gammaTildeHomogeneous.real();
    outGammaTildeImag[gamma_idx] = gammaTildeHomogeneous.imag();
  }
}

/////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
// TODO(niclas): This function does not need all of these input parameters!
template <typename T, int DOMAIN_ENTRY, int DOMAIN_EXIT, int DIRECTION>
__global__ void computeGammaFunctionInitialBoundaryTemplate_kernel(
    const int N, const int Ne, const int energyIdx_offset,
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
    // Read the energy.
    const thrust::complex<T> energyZeroPotential(energies_re[e_idx],
                                                 energies_im[e_idx]);

    int boundary_exit_count = 0;

    const int yStart = (N - 1) * (int)(DIRECTION == -1);

    // Compute gamma along trajectory
    for (int i = 0; i < N; ++i) {
      const int y = yStart + DIRECTION * i;

      // Index to 2d-grids (Delta)
      const int idx = y * N + x;

      // Add energy shift from vector potential
      const thrust::complex<T> energy = energyZeroPotential + energy_shift[idx];

      // What type of node are we on?
      const char nodeType = domainLabel[idx];

      if (nodeType == DOMAIN_EXIT) {
        // Read order parameter
        const thrust::complex<T> delta(delta_real[idx], delta_imag[idx]);

        // Compute gammas and store
        const int idx_sparse = BoundaryStorage<T>::getSparseIndex(
            angleIdx, e_idx + energyIdx_offset, x, boundary_exit_count, Ne, N,
            idx_out_gamma);

        // Compute gamma at current grid point
        thrust::complex<T> gamma;
        thrust::complex<T> gammaTilde;
        riccati::computeHomogeneous(delta, energy, gamma, gammaTilde);
        const thrust::complex<T> gBoundary = DIRECTION > 0 ? gamma : gammaTilde;

        bnd_out_gamma_real[idx_sparse] = gBoundary.real();
        bnd_out_gamma_imag[idx_sparse] = gBoundary.imag();

        boundary_exit_count++;
      }
    }
  }
}
} // namespace internal

template <typename T, template <typename> class POTENTIAL>
RiccatiSolverConfined<T, POTENTIAL>::RiccatiSolverConfined()
    : RiccatiSolver<T>() {}

template <typename T, template <typename> class POTENTIAL>
RiccatiSolverConfined<T, POTENTIAL>::~RiccatiSolverConfined() {}

template <typename T, template <typename> class POTENTIAL>
RiccatiSolverConfined<T, POTENTIAL>::RiccatiSolverConfined(Context<T> *ctx)
    : RiccatiSolver<T>(ctx) {
#ifndef NDEBUG
  std::cout << "RiccatiSolverConfined::RiccatiSolverConfined( Context<T>* )\n";
#endif
  ctx->set(this);
}

template <typename T, template <typename> class POTENTIAL>
int RiccatiSolverConfined<T, POTENTIAL>::initialize() {
  RiccatiSolver<T>::initialize();
  RiccatiSolver<T>::initializeBoundaryStorage();

  return 0;
}

template <typename T, template <typename> class POTENTIAL>
void RiccatiSolverConfined<T, POTENTIAL>::computeCoherenceFunctions(
    const ParameterIterator<int> &energy_iter, const GridGPU<T> *energies,
    const GridGPU<T> *deltaRot, const int current_angle_idx) {
  typedef RiccatiSolver<T> R;
  typedef ContextModule<T> C;

  // Get model parameters
  Parameters<T> *param = C::getParameters();

  // Determine some scalars
  const T h = param->getGridElementSize();
  const T fermiSpeed = C::getFermiSurface()->getFermiSpeed(current_angle_idx);
  const T stepSize = h / fermiSpeed;
  const int N = param->getGridResolutionCoherence();
  const int numAngles = param->getAngularResolution();

  const int numEnergiesCurrentBlock = energy_iter.getNumInterval();
  const int energyIdxOffset = energy_iter.getCurrentValue();

  const int numEnergies_storage =
      C::getIntegrationIterator()->getNumEnergiesInterval();
  const int energyIdxOffset_storage =
      C::getIntegrationIterator()->getEnergyIdxOffset_boundaryStorage();

  // Compute energy shift from magnetic field
  GridGPU<T> *eShift = R::getEnergyShift();

  // Fetch the vector potential.
  GridGPU<T> *inducedVectorPotential =
      C::getContext()->getInducedVectorPotential();

  computeEnergyShift<T, POTENTIAL>(this, inducedVectorPotential, eShift,
                                   current_angle_idx);

  // Get pointers to grids
  GridGPU<T> *gamma = R::getGamma();
  GridGPU<T> *gammabar = R::getGammabar();

  // Set the coherence functions to the homogeneous solutions.
  {
    const int dimX = gamma->dimX();
    const int dimY = gamma->dimY();

    dim3 numBlocks;
    dim3 numThreadsPerBlock;
    gamma->getKernelDimensionsFields(numBlocks, numThreadsPerBlock);

    internal::computeHomogeneous<<<numBlocks, numThreadsPerBlock>>>(
        dimX, dimY, numEnergies_storage,
        energies->getDataPointer(0, Type::real) + energyIdxOffset,
        energies->getDataPointer(0, Type::imag) + energyIdxOffset,
        eShift->getDataPointer(), deltaRot->getDataPointer(0, Type::real),
        deltaRot->getDataPointer(0, Type::imag),
        gamma->getDataPointer(0, Type::real),
        gamma->getDataPointer(0, Type::imag),
        gammabar->getDataPointer(0, Type::real),
        gammabar->getDataPointer(0, Type::imag));
  }

  GeometryGroup<T> *geom = R::getGeometry();
  BoundaryStorage<T> *storage = R::getBoundaryStorage();

  dim3 blocksPerGrid, threadsPerBlock;

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

  // 	Solve gamma by propagating riccati trajectories along positive y
  internal::computeGammaFunctionTemplate_new_kernel<
      T, typename riccati::Gamma<T>, 2, 3, 1>
      <<<blocksPerGrid, threadsPerBlock, 0, s1>>>(
          // Model parameters and energy values
          stepSize, N, numEnergies_storage, energyIdxOffset_storage, idx_x_min,
          idx_x_max, idx_y_min, idx_y_max, current_angle_idx,
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
      T, typename riccati::GammaTilde<T>, 3, 2, -1>
      <<<blocksPerGrid, threadsPerBlock, 0, s2>>>(
          // Model parameters and energy values
          stepSize, N, numEnergies_storage, energyIdxOffset_storage, idx_x_min,
          idx_x_max, idx_y_min, idx_y_max, current_angle_idx,
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

template <typename T, template <typename> class POTENTIAL>
void RiccatiSolverConfined<T, POTENTIAL>::computeInitialCondition(
    const ParameterIterator<int> &energy_iter, const GridGPU<T> *energies,
    const GridGPU<T> *deltaRot, const int current_angle_idx) {
  typedef RiccatiSolver<T> R;
  typedef ContextModule<T> C;

  // Get model parameters
  Parameters<T> *param = C::getParameters();

  // Determine some scalars
  const int N = param->getGridResolutionCoherence();
  const int numAngles = param->getAngularResolution();

  // Get pointers to grids
  GridGPU<T> *gamma = R::getGamma();
  GridGPU<T> *gammabar = R::getGammabar();

  GeometryGroup<T> *geom = R::getGeometry();
  BoundaryStorage<T> *storage = R::getBoundaryStorage();

  const int numEnergiesCurrentBlock = energy_iter.getNumInterval();
  const int energyIdxOffset = energy_iter.getCurrentValue();

  const int numEnergies_storage =
      C::getIntegrationIterator()->getNumEnergiesInterval();
  const int energyIdxOffset_storage =
      C::getIntegrationIterator()->getEnergyIdxOffset_boundaryStorage();

  // Get kernel dimensions
  const int trajectoryIdx_start =
      storage->getTrajectoryStartIndexAngle(current_angle_idx);

  dim3 blocksPerGrid;
  dim3 threadsPerBlock;
  R::getKernelDimensionsOptim(blocksPerGrid, threadsPerBlock,
                              numEnergiesCurrentBlock, current_angle_idx);

  // Get energy shift from magnetic field
  GridGPU<T> *eShift = R::getEnergyShift();

  // 	Solve gamma by propagating riccati trajectories along positive y
  internal::computeGammaFunctionInitialBoundaryTemplate_kernel<T, 2, 3, 1>
      <<<blocksPerGrid, threadsPerBlock>>>(
          // Model parameters and energy values
          N, numEnergies_storage, energyIdxOffset_storage, trajectoryIdx_start,
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
  internal::computeGammaFunctionInitialBoundaryTemplate_kernel<T, 3, 2, -1>
      <<<blocksPerGrid, threadsPerBlock>>>(
          // Model parameters and energy values
          N, numEnergies_storage, energyIdxOffset_storage, trajectoryIdx_start,
          current_angle_idx,
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
} // namespace conga

#endif // CONGA_RICCATI_SOLVER_CONFINED_H_
