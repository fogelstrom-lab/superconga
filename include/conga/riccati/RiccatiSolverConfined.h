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

  void computeCoherenceFunctionsBurnIn(
      const ParameterIterator<int> &energy_iter, const GridGPU<T> *energies,
      const GridGPU<T> *delta, const int angleIdx) override;

  // Get a string describing the vector potential type, e.g. solenoid.
  std::string getVectorPotentialType() const { return POTENTIAL<T>::type(); }

private:
  void computeCoherenceFunctionsInternal(
      const ParameterIterator<int> &energy_iter, const GridGPU<T> *energies,
      const GridGPU<T> *delta, const int angleIdx, const bool isBurnIn);
};

namespace internal {
/// \brief Make order parameter from real/imag parts. The stepping formulas are
/// identical for gamma and gammaTilde except gammaTilde has -\Delta^* instead
/// of \Delta. So by applying the negative conjugate during construction we only
/// need to implement one general stepping function.
///
/// \param inReal The real part of the order parameter.
/// \param inImag The imaginary part of the order parameter.
/// \param inIsGamma Whether the order parameter will be used for computing
/// gamma or gammaTilde.
///
/// \return The order parameter (if for gamma) or the negative conjugate of the
/// order parameter (if for gammaTilde).
template <typename T>
inline __device__ thrust::complex<T>
makeOrderParameter(const T inReal, const T inImag, const bool inIsGamma) {
  return inIsGamma ? thrust::complex<T>(inReal, inImag)
                   : thrust::complex<T>(-inReal, inImag);
}

/// \brief Interpolate linearly taking a half unit step.
///
/// \param inVal0 The value at zero step.
/// \param inVal1 The value at unit step.
///
/// \return The interpolated value.
template <typename T>
inline __device__ thrust::complex<T>
interpolateLinear(const thrust::complex<T> &inVal0,
                  const thrust::complex<T> &inVal1) {
  return (inVal1 + inVal0) * static_cast<T>(0.5);
}

/// \brief Take a step along a trajectory and read/write boundary values if
/// encountered.
///
/// \param inStep The step length (h/vF).
/// \param inXIdx The x-index.
/// \param inYIdx The y-index.
/// \param inAngleIdx The angle index.
/// \param inEnergyIdx The energy index.
/// \param inEnergyIdxOffset The energy offset (needed for computing sparse
/// indices).
/// \param inNumEnergies The number of energies.
/// \param inLatticeSideLength The number of lattice sites per side length
/// (assuming a square lattice).
/// \param inEnergy The energy at the current lattice site.
/// \param inEnergyPrevious The energy at the previous lattice site.
/// \param inOrderParameter The order parameter at the current lattice site.
/// \param inOrderParameterPrevious The order parameter at the previous lattice
/// site.
/// \param inBoundaryDistances Array of distances to the boundary.
/// \param inCoherenceIdxInward Some array of indices needed for computing
/// sparse indices (inward).
/// \param inCoherenceIdxOutward Some array of indices needed for computing
/// sparse indices (outward).
/// \param inBoundaryCoherenceInwardReal The inward, real boundary coherence
/// value.
/// \param inBoundaryCoherenceInwardImag The inward, imag boundary coherence
/// value.
/// \param inOutBoundaryCoherenceOutwardReal The outward, real boundary
/// coherence value.
/// \param inOutBoundaryCoherenceOutwardImag The outward, imag boundary
/// coherence value.
/// \param inOutBoundaryEntryCount The number of encountered entry sites.
/// \param inOutBoundaryExitCount The number of encountered exit sites.
/// \param inOutCoherence The previous coherence value, will be overwritten by
/// the current value.
template <typename T, int DOMAIN_ENTRY, int DOMAIN_EXIT>
inline __device__ void computeStep(
    const T inStep, const int inXIdx, const int inYIdx, const int inAngleIdx,
    const int inEnergyIdx, const int inEnergyIdxOffset, const int inNumEnergies,
    const int inLatticeSideLength, const thrust::complex<T> &inEnergy,
    const thrust::complex<T> &inEnergyPrevious,
    const thrust::complex<T> &inOrderParameter,
    const thrust::complex<T> &inOrderParameterPrevious,
    const char inDomainLabel, const T *const __restrict__ inBoundaryDistances,
    const int *const __restrict__ inCoherenceIdxInward,
    const int *const __restrict__ inCoherenceIdxOutward,
    const T *const __restrict__ inBoundaryCoherenceInwardReal,
    const T *const __restrict__ inBoundaryCoherenceInwardImag,
    T *const __restrict__ inOutBoundaryCoherenceOutwardReal,
    T *const __restrict__ inOutBoundaryCoherenceOutwardImag,
    int &inOutBoundaryEntryCount, int &inOutBoundaryExitCount,
    thrust::complex<T> &inOutCoherence) {
  // Spatial index for boundary distance.
  const int spatialIdx = inYIdx * inLatticeSideLength + inXIdx;

  switch (inDomainLabel) {
  case 1: {
    // =========== Inside domain =========== //
    const thrust::complex<T> energyMiddle =
        interpolateLinear(inEnergyPrevious, inEnergy);
    const thrust::complex<T> deltaMiddle =
        interpolateLinear(inOrderParameterPrevious, inOrderParameter);
    // Take a step to the center of the lattice site.
    inOutCoherence =
        riccati::computeStep(deltaMiddle, energyMiddle, inOutCoherence, inStep);
    break;
  }
  case DOMAIN_ENTRY: {
    // =========== Boundary entry =========== //

    // Read boundary coherence function value.
    const int sparseIdx = BoundaryStorage<T>::getSparseIndex(
        inAngleIdx, inEnergyIdx + inEnergyIdxOffset, inXIdx,
        inOutBoundaryEntryCount, inNumEnergies, inLatticeSideLength,
        inCoherenceIdxInward);
    const thrust::complex<T> boundaryCoherence(
        inBoundaryCoherenceInwardReal[sparseIdx],
        inBoundaryCoherenceInwardImag[sparseIdx]);

    // Distance to exact boundary location from the center of the lattice
    // site.
    const T distance = abs(inBoundaryDistances[spatialIdx]);
    const T boundaryToCenterStep = distance * inStep;

    // Take a step from the grain boundary to the center of the lattice
    // site.
    inOutCoherence = riccati::computeStep(
        inOrderParameter, inEnergy, boundaryCoherence, boundaryToCenterStep);

    // Increase entry counter.
    inOutBoundaryEntryCount++;
    break;
  }
  case DOMAIN_EXIT: {
    // =========== Boundary exit =========== //

    // Take a step to the center of the lattice site.
    const thrust::complex<T> energyMiddle =
        interpolateLinear(inEnergyPrevious, inEnergy);
    const thrust::complex<T> deltaMiddle =
        interpolateLinear(inOrderParameterPrevious, inOrderParameter);
    inOutCoherence =
        riccati::computeStep(deltaMiddle, energyMiddle, inOutCoherence, inStep);

    // Distance to exact boundary location from the center of the lattice
    // site.
    const T distance = abs(inBoundaryDistances[spatialIdx]);
    const T centerToBoundaryStep = distance * inStep;

    // Take a step from the center of the lattice site to the grain
    // boundary.
    const thrust::complex<T> boundaryCoherence = riccati::computeStep(
        inOrderParameter, inEnergy, inOutCoherence, centerToBoundaryStep);

    // Store boundary gamma.
    const int sparseIdx = BoundaryStorage<T>::getSparseIndex(
        inAngleIdx, inEnergyIdx + inEnergyIdxOffset, inXIdx,
        inOutBoundaryExitCount, inNumEnergies, inLatticeSideLength,
        inCoherenceIdxOutward);
    inOutBoundaryCoherenceOutwardReal[sparseIdx] = boundaryCoherence.real();
    inOutBoundaryCoherenceOutwardImag[sparseIdx] = boundaryCoherence.imag();

    // Increase exit counter.
    inOutBoundaryExitCount++;
    break;
  }
  default: {
    // =========== Outside domain =========== //
    inOutCoherence = thrust::complex<T>(static_cast<T>(0), static_cast<T>(0));
  }
  }
}

// TODO(niclas): Put parameters in order (in, inout, out).
template <typename T, int DOMAIN_ENTRY, int DOMAIN_EXIT, int DIRECTION>
__global__ void computeCoherenceFunction(
    const T inStep, const int N, const int Ne, const int energyIdx_offset,
    const int idx_x_min, const int idx_x_max, const int idx_y_min,
    const int idx_y_max, const int angleIdx,
    const T *const __restrict__ energies_re,
    const T *const __restrict__ energies_im,
    const T *const __restrict__ energy_shift, T *gamma_real, T *gamma_imag,
    const T *const __restrict__ delta_real,
    const T *const __restrict__ delta_imag,
    const char *const __restrict__ domainLabels,
    const T *const __restrict__ boundaryDistance,
    const int *const __restrict__ idx_in_gamma,
    const int *const __restrict__ idx_out_gamma,
    const T *const __restrict__ bnd_in_gamma_real,
    const T *const __restrict__ bnd_in_gamma_imag,
    T *const __restrict__ bnd_out_gamma_real,
    T *const __restrict__ bnd_out_gamma_imag) {
  // x-coordinate
  const int x = blockDim.x * blockIdx.x + threadIdx.x + idx_x_min;
  // energy coordinate
  const int e_idx = blockDim.y * blockIdx.y + threadIdx.y;

  if (x <= idx_x_max && e_idx < Ne) {
    // Read the energy.
    const thrust::complex<T> energyZeroPotential(energies_re[e_idx],
                                                 energies_im[e_idx]);

    // In order to read/write values from/to the boundary coherence functions,
    // we need to keep track of how many boundaries we have encountered along
    // the trajectory.
    int boundary_entry_count = 0;
    int boundary_exit_count = 0;

    const bool isGamma = DIRECTION > 0;

    // Which y-coordinate to start from is determined by whether we are solving
    // for gamma or gammatilde (which is codified by DIRECTION.
    const int y_start = isGamma ? idx_y_min : idx_y_max;

    // The previous energy and order parameter. They are updated as we traverse
    // the trajectory.
    thrust::complex<T> energyPrev(static_cast<T>(0), static_cast<T>(0));
    thrust::complex<T> deltaPrev(static_cast<T>(0), static_cast<T>(0));

    // The value of the coherence function at the center of the current
    // lattice site. It is updated as we traverse the trajectory.
    thrust::complex<T> coherence(static_cast<T>(0), static_cast<T>(0));

    // OPTIMIZATION: PRECOMPUTE Y_MIN AND Y_MAX FOR EACH ROTATIONAL FRAME
    // Compute gamma along trajectory
    for (int i = 0; i <= (idx_y_max - idx_y_min); ++i) {
      const int y = y_start + DIRECTION * i;

      // Index to 2d-grids (Delta)
      const int idx = y * N + x;

      // The energy and order parameter.
      const thrust::complex<T> energy = energyZeroPotential + energy_shift[idx];
      const thrust::complex<T> delta =
          makeOrderParameter(delta_real[idx], delta_imag[idx], isGamma);

      // What type of site are we on?
      const char domainLabel = domainLabels[idx];

      // Take step and read/write boundary values.
      computeStep<T, DOMAIN_ENTRY, DOMAIN_EXIT>(
          inStep, x, y, angleIdx, e_idx, energyIdx_offset, Ne, N, energy,
          energyPrev, delta, deltaPrev, domainLabel, boundaryDistance,
          idx_in_gamma, idx_out_gamma, bnd_in_gamma_real, bnd_in_gamma_imag,
          bnd_out_gamma_real, bnd_out_gamma_imag, boundary_entry_count,
          boundary_exit_count, coherence);

      // Store coherence function.
      // Index to gamma grid (for all energies)
      const int gamma_idx = e_idx * 2 * N * N + y * N + x;
      gamma_real[gamma_idx] = coherence.real();
      gamma_imag[gamma_idx] = coherence.imag();

      // Update previous energy and order parameter.
      energyPrev = energy;
      deltaPrev = delta;
    }
  }
}

// TODO(niclas): Put parameters in order (in, inout, out).
template <typename T, int DOMAIN_ENTRY, int DOMAIN_EXIT, int DIRECTION>
__global__ void
computeCoherenceFunctionBurnIn(const T inStep, const int N, const int Ne,
                               const int energyIdx_offset, const int idx_x_min,
                               const int idx_x_max, const int idx_y_min,
                               const int idx_y_max, const int angleIdx,
                               const T *const __restrict__ energies_re,
                               const T *const __restrict__ energies_im,
                               const T *const __restrict__ energy_shift,
                               const T *const __restrict__ delta_real,
                               const T *const __restrict__ delta_imag,
                               const char *const __restrict__ domainLabels,
                               const T *const __restrict__ boundaryDistance,
                               const int *const __restrict__ idx_in_gamma,
                               const int *const __restrict__ idx_out_gamma,
                               const T *const __restrict__ bnd_in_gamma_real,
                               const T *const __restrict__ bnd_in_gamma_imag,
                               T *const __restrict__ bnd_out_gamma_real,
                               T *const __restrict__ bnd_out_gamma_imag) {
  // x-coordinate
  const int x = blockDim.x * blockIdx.x + threadIdx.x + idx_x_min;
  // energy coordinate
  const int e_idx = blockDim.y * blockIdx.y + threadIdx.y;

  if (x <= idx_x_max && e_idx < Ne) {
    // Read the energy.
    const thrust::complex<T> energyZeroPotential(energies_re[e_idx],
                                                 energies_im[e_idx]);

    // In order to read/write values from/to the boundary coherence functions,
    // we need to keep track of how many boundaries we have encountered along
    // the trajectory.
    int boundary_entry_count = 0;
    int boundary_exit_count = 0;

    const bool isGamma = DIRECTION > 0;

    // Which y-coordinate to start from is determined by whether we are solving
    // for gamma or gammatilde (which is codified by DIRECTION.
    const int y_start = isGamma ? idx_y_min : idx_y_max;

    // The previous energy and order parameter. They are updated as we traverse
    // the trajectory.
    thrust::complex<T> energyPrev(static_cast<T>(0), static_cast<T>(0));
    thrust::complex<T> deltaPrev(static_cast<T>(0), static_cast<T>(0));

    // The value of the coherence function at the center of the current
    // lattice site. It is updated as we traverse the trajectory.
    thrust::complex<T> coherence(static_cast<T>(0), static_cast<T>(0));

    // OPTIMIZATION: PRECOMPUTE Y_MIN AND Y_MAX FOR EACH ROTATIONAL FRAME
    // Compute gamma along trajectory
    for (int i = 0; i <= (idx_y_max - idx_y_min); ++i) {
      const int y = y_start + DIRECTION * i;

      // Index to 2d-grids (Delta)
      const int idx = y * N + x;

      // The energy and order parameter.
      const thrust::complex<T> energy = energyZeroPotential + energy_shift[idx];
      const thrust::complex<T> delta =
          makeOrderParameter(delta_real[idx], delta_imag[idx], isGamma);

      // What type of node are we on?
      const char domainLabel = domainLabels[idx];

      // Take step and read/write boundary values.
      computeStep<T, DOMAIN_ENTRY, DOMAIN_EXIT>(
          inStep, x, y, angleIdx, e_idx, energyIdx_offset, Ne, N, energy,
          energyPrev, delta, deltaPrev, domainLabel, boundaryDistance,
          idx_in_gamma, idx_out_gamma, bnd_in_gamma_real, bnd_in_gamma_imag,
          bnd_out_gamma_real, bnd_out_gamma_imag, boundary_entry_count,
          boundary_exit_count, coherence);

      // Update previous energy and order parameter.
      energyPrev = energy;
      deltaPrev = delta;
    }
  }
}

/////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
// TODO(niclas): This function does not need all of these input parameters!
template <typename T, int DOMAIN_ENTRY, int DOMAIN_EXIT, int DIRECTION>
__global__ void computeGammaFunctionInitialBoundaryTemplate_kernel(
    const int N, const int Ne, const int energyIdx_offset,
    const int idx_x_offset, const int angleIdx,
    const T *const __restrict__ energies_re,
    const T *const __restrict__ energies_im,
    const T *const __restrict__ energy_shift,
    const T *const __restrict__ gamma_real,
    const T *const __restrict__ gamma_imag,
    const T *const __restrict__ delta_real,
    const T *const __restrict__ delta_imag,
    const char *const __restrict__ domainLabels,
    const T *const __restrict__ boundaryDistance,
    const int *const __restrict__ idx_in_gamma,
    const int *const __restrict__ idx_out_gamma,
    const T *const __restrict__ bnd_in_gamma_real,
    const T *const __restrict__ bnd_in_gamma_imag,
    T *const __restrict__ bnd_out_gamma_real,
    T *const __restrict__ bnd_out_gamma_imag) {
  // x-coordinate
  const int x = blockDim.x * blockIdx.x + threadIdx.x + idx_x_offset;
  // energy coordinate
  const int e_idx = blockDim.y * blockIdx.y + threadIdx.y;

  if (x < N && e_idx < Ne) {
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
      const char domainLabel = domainLabels[idx];

      if (domainLabel == DOMAIN_EXIT) {
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
  const bool isBurnIn = false;
  computeCoherenceFunctionsInternal(energy_iter, energies, deltaRot,
                                    current_angle_idx, isBurnIn);
}

template <typename T, template <typename> class POTENTIAL>
void RiccatiSolverConfined<T, POTENTIAL>::computeCoherenceFunctionsBurnIn(
    const ParameterIterator<int> &energy_iter, const GridGPU<T> *energies,
    const GridGPU<T> *deltaRot, const int current_angle_idx) {
  const bool isBurnIn = true;
  computeCoherenceFunctionsInternal(energy_iter, energies, deltaRot,
                                    current_angle_idx, isBurnIn);
}

template <typename T, template <typename> class POTENTIAL>
void RiccatiSolverConfined<T, POTENTIAL>::computeCoherenceFunctionsInternal(
    const ParameterIterator<int> &energy_iter, const GridGPU<T> *energies,
    const GridGPU<T> *deltaRot, const int current_angle_idx,
    const bool inIsBurnIn) {
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
  GridGPU<T> *inducedVectorPotential =
      C::getContext()->getInducedVectorPotential();
  computeEnergyShift<T, POTENTIAL>(this, inducedVectorPotential, eShift,
                                   current_angle_idx);

  GeometryGroup<T> *geom = R::getGeometry();
  BoundaryStorage<T> *storage = R::getBoundaryStorage();

  const auto [blocksPerGrid, threadsPerBlock] =
      R::getKernelDimensionsOptim(numEnergiesCurrentBlock, current_angle_idx);

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

  if (inIsBurnIn) {
    // Solve gamma by propagating riccati trajectories along positive y
    internal::computeCoherenceFunctionBurnIn<T, 2, 3, 1>
        <<<blocksPerGrid, threadsPerBlock, 0, s1>>>(
            // Model parameters and energy values
            stepSize, N, numEnergies_storage, energyIdxOffset_storage,
            idx_x_min, idx_x_max, idx_y_min, idx_y_max, current_angle_idx,
            energies->getDataPointer(0, Type::real) + energyIdxOffset,
            energies->getDataPointer(0, Type::imag) + energyIdxOffset,
            eShift->getDataPointer(), deltaRot->getDataPointer(0, Type::real),
            deltaRot->getDataPointer(0, Type::imag),
            // Grids for domain info/data
            geom->getDomainLabelsGrid()->getDataPointer(0, Type::real),
            geom->getBoundaryDistanceGrid()->getDataPointer(0, Type::real),
            // Data relating to sparse storage for boundary gammas
            storage->getIndexSparseIn_gamma()->getDataPointer(0, Type::real),
            storage->getIndexSparseOut_gamma()->getDataPointer(0, Type::real),
            storage->getBoundaryIn_gamma()->getDataPointer(0, Type::real),
            storage->getBoundaryIn_gamma()->getDataPointer(0, Type::imag),
            storage->getBoundaryOut_gamma()->getDataPointer(0, Type::real),
            storage->getBoundaryOut_gamma()->getDataPointer(0, Type::imag));

    // Solve gammabar by propagating riccati trajectories along negative y
    internal::computeCoherenceFunctionBurnIn<T, 3, 2, -1>
        <<<blocksPerGrid, threadsPerBlock, 0, s2>>>(
            // Model parameters and energy values
            stepSize, N, numEnergies_storage, energyIdxOffset_storage,
            idx_x_min, idx_x_max, idx_y_min, idx_y_max, current_angle_idx,
            energies->getDataPointer(0, Type::real) + energyIdxOffset,
            energies->getDataPointer(0, Type::imag) + energyIdxOffset,
            eShift->getDataPointer(), deltaRot->getDataPointer(0, Type::real),
            deltaRot->getDataPointer(0, Type::imag),
            // Grids for domain info/data
            geom->getDomainLabelsGrid()->getDataPointer(0, Type::real),
            geom->getBoundaryDistanceGrid()->getDataPointer(0, Type::real),
            // Data relating to sparse storage for boundary gammas
            storage->getIndexSparseIn_gammabar()->getDataPointer(0, Type::real),
            storage->getIndexSparseOut_gammabar()->getDataPointer(0,
                                                                  Type::real),
            storage->getBoundaryIn_gammabar()->getDataPointer(0, Type::real),
            storage->getBoundaryIn_gammabar()->getDataPointer(0, Type::imag),
            storage->getBoundaryOut_gammabar()->getDataPointer(0, Type::real),
            storage->getBoundaryOut_gammabar()->getDataPointer(0, Type::imag));
  } else {
    // Get pointers to grids
    GridGPU<T> *gamma = R::getGamma();
    GridGPU<T> *gammabar = R::getGammabar();

    // Solve gamma by propagating riccati trajectories along positive y
    internal::computeCoherenceFunction<T, 2, 3, 1>
        <<<blocksPerGrid, threadsPerBlock, 0, s1>>>(
            // Model parameters and energy values
            stepSize, N, numEnergies_storage, energyIdxOffset_storage,
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
            geom->getBoundaryDistanceGrid()->getDataPointer(0, Type::real),
            // Data relating to sparse storage for boundary gammas
            storage->getIndexSparseIn_gamma()->getDataPointer(0, Type::real),
            storage->getIndexSparseOut_gamma()->getDataPointer(0, Type::real),
            storage->getBoundaryIn_gamma()->getDataPointer(0, Type::real),
            storage->getBoundaryIn_gamma()->getDataPointer(0, Type::imag),
            storage->getBoundaryOut_gamma()->getDataPointer(0, Type::real),
            storage->getBoundaryOut_gamma()->getDataPointer(0, Type::imag));

    // Solve gammabar by propagating riccati trajectories along negative y
    internal::computeCoherenceFunction<T, 3, 2, -1>
        <<<blocksPerGrid, threadsPerBlock, 0, s2>>>(
            // Model parameters and energy values
            stepSize, N, numEnergies_storage, energyIdxOffset_storage,
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
            geom->getBoundaryDistanceGrid()->getDataPointer(0, Type::real),
            // Data relating to sparse storage for boundary gammas
            storage->getIndexSparseIn_gammabar()->getDataPointer(0, Type::real),
            storage->getIndexSparseOut_gammabar()->getDataPointer(0,
                                                                  Type::real),
            storage->getBoundaryIn_gammabar()->getDataPointer(0, Type::real),
            storage->getBoundaryIn_gammabar()->getDataPointer(0, Type::imag),
            storage->getBoundaryOut_gammabar()->getDataPointer(0, Type::real),
            storage->getBoundaryOut_gammabar()->getDataPointer(0, Type::imag));
  }

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

  const auto [blocksPerGrid, threadsPerBlock] =
      R::getKernelDimensionsOptim(numEnergiesCurrentBlock, current_angle_idx);

  // Compute energy shift from magnetic field
  GridGPU<T> *eShift = R::getEnergyShift();
  GridGPU<T> *inducedVectorPotential =
      C::getContext()->getInducedVectorPotential();
  computeEnergyShift<T, POTENTIAL>(this, inducedVectorPotential, eShift,
                                   current_angle_idx);

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
