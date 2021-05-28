//===---- ComputeFreeEnergy.h - Class for free energy compute object. -----===//
//
// Part of the SuprConGPU Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Compute object class for free energy.
///
//===----------------------------------------------------------------------===//
#ifndef COMPUTE_FREE_ENERGY_H_
#define COMPUTE_FREE_ENERGY_H_

#include "ComputeProperty.h"
#include "ComputeResidual.h"
#include "Grid.h"
#include "IntegrationIteratorOzaki.h"
#include "LocalGreens.h"
#include "OrderParameterBulk.h"
#include <cmath>
#include <cstdio>
#include <limits>
#include <thrust/complex.h>

namespace scg {
template <typename T> class ComputeFreeEnergy : public ComputeProperty<T> {
public:
  // Constructors and Destructors
  ComputeFreeEnergy();
  ~ComputeFreeEnergy();

  ComputeFreeEnergy(Context<T> *const inOutContext);

  GridGPU<T> *getResult() override { return m_freeEnergyGrid; }
  T getFreeEnergy() const { return m_freeEnergy; }
  T getFreeEnergyPerUnitArea() const { return m_freeEnergyPerUnitArea; }
  T getMagneticFreeEnergy() const { return m_magneticFreeEnergy; }
  T getMagneticFreeEnergyPerUnitArea() const {
    return m_magneticFreeEnergyPerUnitArea;
  }

  int initialize() override;
  IntegrationIterator<T> *
  createIntegrationIterator(const int inNumEnergiesPerBlock = -1) override;

  // This should be called once for every iteration, before any integration
  // takes place.
  void initializeIteration(const int inNumFields = 1) override;

  // This should be called once for every angle, before any integration takes
  // place.
  void initializeGreensAngle() override;

  // Computes greens function for current angle. This may be called multiple
  // times as we integrate over many energies. Result is in local frame and will
  // be stored in _greensCurrentAngle.
  void computeGreensAngle(ParameterIterator<int> *inParameterIterator,
                          GridGPU<T> *inGamma, GridGPU<T> *inGammabar) override;

  // Add the result from computeGreensAngle() to total. Rotates
  // _greensCurrentAngle to global frame and adds momentum dependence.
  void integrateGreensAngle(GeometryGroup<T> *inGeometryGroup,
                            const int inAngleIdx) override;

  // Compute physical property from greens function.
  void computePropertyFromGreens() override;

private:
  // Total free energy.
  T m_freeEnergy;
  T m_freeEnergyPerUnitArea;

  // Magnetic part of the free energy.
  T m_magneticFreeEnergy;
  T m_magneticFreeEnergyPerUnitArea;

  GridGPU<T> *m_freeEnergyGrid;
};

/// \brief Compute the free-energy kernel I
///
/// GPU function to compute the kernel I which enters the Eilenberger free
/// energy. This is the only object in the free energy which actually depends on
/// angle. See page 25 of P. Holmvall PhD Thesis.
///
/// \param inOrderParameter - Order parameter value
/// \param inGamma - Riccati coherence function
/// \param inGammabar - Tilde Riccati coherence function
/// \param outFreeEnergy - The free energy
template <typename T>
__device__ thrust::complex<T>
computeFreeEnergyKernelI(const thrust::complex<T> &inOrderParameter,
                         const thrust::complex<T> &inGamma,
                         const thrust::complex<T> &inGammabar) {

  return thrust::conj(inOrderParameter) * inGamma -
         inOrderParameter * inGammabar;
}

// Parallelize calculation of free-energy kernel I over spatial coordinates
template <typename T>
__global__ void computeFreeEnergy_Kernel(
    const int inNumLatticePoints, const int inNumEnergies, T *outFreeEnergy_re,
    T *outFreeEnergy_im, const T *const inGamma_re, const T *const inGamma_im,
    const T *const inGammabar_re, const T *const inGammabar_im,
    const T *const inOrderParameter_re, const T *const inOrderParameter_im,
    const T *const inOzakiResidues) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x; // x index
  const int y = blockDim.y * blockIdx.y + threadIdx.y; // y index

  // Check that thread has work to do
  if ((x < inNumLatticePoints) && (y < inNumLatticePoints)) {
    // Calculate unique index for the grain coordinate
    const int grainCoordinateIdx = y * inNumLatticePoints + x;

    // Initialize the sum (over Ozaki poles) to zero
    thrust::complex<T> freeEnergyKernelSum(static_cast<T>(0),
                                           static_cast<T>(0));

    // Order parameter at this coordinate
    const thrust::complex<T> orderParameter(
        inOrderParameter_re[grainCoordinateIdx],
        inOrderParameter_im[grainCoordinateIdx]);

    // Loop over Ozaki poles: the energy sum in the free-energy kernel I
    for (int energyIdx = 0; energyIdx < inNumEnergies; ++energyIdx) {
      // TODO(patric): This is dangerous! We should not manually calculate
      // index like this! Imagine that you want to calculate more coherence
      // functions (like gamma plus/minus instead), then you have to change
      // all these places manually?
      const int coherenceFunctionIdx =
          energyIdx * 2 * inNumLatticePoints * inNumLatticePoints +
          grainCoordinateIdx;

      // Coherence functions in the current coordinate, angle and energy
      const thrust::complex<T> gamma(inGamma_re[coherenceFunctionIdx],
                                     inGamma_im[coherenceFunctionIdx]);
      const thrust::complex<T> gammabar(inGammabar_re[coherenceFunctionIdx],
                                        inGammabar_im[coherenceFunctionIdx]);

      // Calculate I(x, y, energy, angle)
      const thrust::complex<T> freeEnergyKernelI =
          computeFreeEnergyKernelI<T>(orderParameter, gamma, gammabar);

      // Multiply by Ozaki residue and sum the contribution.
      // Note that multiplying with 'i' is done after the loop.
      freeEnergyKernelSum += inOzakiResidues[energyIdx] * freeEnergyKernelI;
    }

    // Add contribution to the free energy in this coordinate.
    // Note that a mutiplication with 'i' is done here.
    outFreeEnergy_re[grainCoordinateIdx] -= freeEnergyKernelSum.imag();
    outFreeEnergy_im[grainCoordinateIdx] += freeEnergyKernelSum.real();
  }
}

template <typename T>
ComputeFreeEnergy<T>::ComputeFreeEnergy()
    : ComputeProperty<T>(), m_freeEnergy(static_cast<T>(0)),
      m_freeEnergyPerUnitArea(static_cast<T>(0)),
      m_magneticFreeEnergy(static_cast<T>(0)),
      m_magneticFreeEnergyPerUnitArea(static_cast<T>(0)),
      m_freeEnergyGrid(nullptr) {}

template <typename T>
ComputeFreeEnergy<T>::ComputeFreeEnergy(Context<T> *const inOutContext)
    : ComputeProperty<T>(inOutContext), m_freeEnergy(static_cast<T>(0)),
      m_freeEnergyPerUnitArea(static_cast<T>(0)),
      m_magneticFreeEnergy(static_cast<T>(0)),
      m_magneticFreeEnergyPerUnitArea(static_cast<T>(0)),
      m_freeEnergyGrid(nullptr) {
  inOutContext->add(this);
}

template <typename T> ComputeFreeEnergy<T>::~ComputeFreeEnergy() {
  if (m_freeEnergyGrid) {
    delete m_freeEnergyGrid;
  }
}

template <typename T> int ComputeFreeEnergy<T>::initialize() {
#ifndef NDEBUG
  std::cout << "ComputeFreeEnergy<T>::initialize()\n";
#endif
  typedef ComputeProperty<T> C;

  // Get resolution
  const int N0 = C::getParameters()->getGridResolutionBase();
  const int N1 = C::getParameters()->getGridResolutionCoherence();

  GridGPU<T>::initializeGrid(m_freeEnergyGrid, N0, N0);

  C::initGreensAngle(N1);

  return 0;
}

template <typename T>
IntegrationIterator<T> *ComputeFreeEnergy<T>::createIntegrationIterator(
    const int inNumEnergiesPerBlock) {
  return new IntegrationIteratorOzaki<T>(ContextModule<T>::getContext(),
                                         inNumEnergiesPerBlock);
}

template <typename T>
void ComputeFreeEnergy<T>::initializeIteration(const int inNumFields) {
  // Unused.
  (void)inNumFields;

  typedef ContextModule<T> C;

  if (ComputeProperty<T>::doCompute()) {
    m_freeEnergyGrid->setZero();
  }
}

template <typename T> void ComputeFreeEnergy<T>::initializeGreensAngle() {
  if (ComputeProperty<T>::doCompute()) {
    // Set to zero.
    ComputeProperty<T>::initializeGreensAngle();
  }
}

// Computes the free-energy kernel 'I' for current angle, for every spatial grid
// point (summed over energies).
template <typename T>
void ComputeFreeEnergy<T>::computeGreensAngle(
    ParameterIterator<int> *inParameterIterator, GridGPU<T> *inGamma,
    GridGPU<T> *inGammabar) {
  typedef ContextModule<T> C;

  if (ComputeProperty<T>::doCompute()) {
    IntegrationIterator<T> *integrationIter = C::getIntegrationIterator();

    GridGPU<T> *freeEnergyKernelIntegrand =
        ComputeProperty<T>::getGreensCurrentAngle();

    GridGPU<T> *deltaRot =
        integrationIter->getOrderParameterFieldCurrentAngle();
    GridGPU<T> *ozakiResidues = integrationIter->getResidues();

    const int numEnergies = inParameterIterator->getNumInterval();
    const int energyIdxStart = inParameterIterator->getCurrentValue();

    dim3 blocksPerGrid, threadsPerBlock;
    freeEnergyKernelIntegrand->getKernelDimensions(blocksPerGrid,
                                                   threadsPerBlock);

    computeFreeEnergy_Kernel<T><<<blocksPerGrid, threadsPerBlock>>>(
        freeEnergyKernelIntegrand->dimX(), numEnergies,
        freeEnergyKernelIntegrand->getDataPointer(0, Type::real),
        freeEnergyKernelIntegrand->getDataPointer(0, Type::imag),
        inGamma->getDataPointer(0, Type::real),
        inGamma->getDataPointer(0, Type::imag),
        inGammabar->getDataPointer(0, Type::real),
        inGammabar->getDataPointer(0, Type::imag),
        deltaRot->getDataPointer(0, Type::real),
        deltaRot->getDataPointer(0, Type::imag),
        ozakiResidues->getDataPointer(0, Type::real) + energyIdxStart);
  }
}

// Sums the free-energy kernel 'I' over spatial grid points, and adds result to
// total angle integration.
template <typename T>
void ComputeFreeEnergy<T>::integrateGreensAngle(
    GeometryGroup<T> *inGeometryGroup, const int angleIdx) {
  if (ComputeProperty<T>::doCompute()) {
    GridGPU<T> *freeEnergyKernelIntegrand =
        ComputeProperty<T>::getGreensCurrentAngle();

    inGeometryGroup->extendValuesAtBoundary(freeEnergyKernelIntegrand);

    const T systemAngle =
        ComputeProperty<T>::getFermiSurface()->getSystemAngle(angleIdx);

    const T integrandFactor =
        ComputeProperty<T>::getFermiSurface()->getIntegrandFactor(angleIdx);
    *freeEnergyKernelIntegrand *= integrandFactor;

    m_freeEnergyGrid->insertRotatedGridAdditive(freeEnergyKernelIntegrand,
                                                systemAngle);
  }
}

template <typename T> void ComputeFreeEnergy<T>::computePropertyFromGreens() {
  typedef ContextModule<T> C;

  if (ComputeProperty<T>::doCompute()) {
    // This variables will contain the (unnormalized) magnetic, and
    // non-magnetic, parts of the free energy.
    T magneticSum = static_cast<T>(0);
    T nonMagneticSum = static_cast<T>(0);

    // Get properties.
    const T latticeElementSize = C::getParameters()->getGridElementSize();
    const T temperature = C::getParameters()->getTemperature();
    const int numGrainLatticeSites =
        C::getGeometry()->getNumGrainLatticeSites();

    // Compute second term of free energy not involving angle integration.
    GridCPU<T> ozakiEnergies = *(C::getIntegrationIterator()->getEnergies());
    GridCPU<T> ozakiResidues = *(C::getIntegrationIterator()->getResidues());

    const T *ozakiEnergies_im = ozakiEnergies.getDataPointer(0, Type::imag);
    const T *ozakiResidues_re = ozakiResidues.getDataPointer(0, Type::real);

    const int numEnergies = ozakiEnergies.dimX();
    T sumInverseEnergy = 0.0;
    for (int ozakiIdx = 0; ozakiIdx < numEnergies; ++ozakiIdx) {
      sumInverseEnergy +=
          ozakiResidues_re[ozakiIdx] / ozakiEnergies_im[ozakiIdx];
    }
    sumInverseEnergy *= static_cast<T>(2.0 * M_PI) * temperature;

    // Get the part of the free energy from 'I'.
    T energySummedKernelI_re = 0.0, energySummedKernelI_im = 0.0;
    C::getGeometry()->zeroValuesOutsideDomainRef(m_freeEnergyGrid);
    m_freeEnergyGrid->sumField(energySummedKernelI_re, energySummedKernelI_im);

    // Update the non-magnetic sum.
    nonMagneticSum +=
        static_cast<T>(2.0 * M_PI) * temperature * energySummedKernelI_re;

    // Add contribution from the order parameter.
    const OrderParameter<T> *const orderParameter = C::getOrderParameter();
    const int numOrderParameterComponents =
        orderParameter->getNumOrderParameterComponents();

    for (int i = 0; i < numOrderParameterComponents; ++i) {
      // Fetch a component.
      const OrderParameterComponent<T> *const component =
          orderParameter->getOrderParameterComponent(i);

      // Compute the temperature dependent factor.
      const T relativeTc = component->getRelativeTc();
      const T factor = std::log(temperature / relativeTc) + sumInverseEnergy;

      // Compute the squared absolute value.
      GridGPU<T> componentSquared =
          component->getOrderParameterField()->complexMagnitudeSq();

      // Set to zero outside of the grain.
      C::getGeometry()->zeroValuesOutsideDomainRef(&componentSquared);

      // Add contribution from this order-parameter component.
      nonMagneticSum += componentSquared.sumField() * factor;
    }

    // The Ginzburg-Landau parameter.
    const T penetrationDepth = C::getParameters()->getPenetrationDepth();

    // Add contribution from the induced magnetic field.
    if (penetrationDepth > static_cast<T>(0)) {
      // Fetch the vector potential.
      const GridGPU<T> *const inducedVectorPotential =
          C::getContext()->getInducedVectorPotential();

      // Compute the magnetic field.
      GridGPU<T> magneticField =
          inducedVectorPotential->curl(latticeElementSize);

      // Note, by setting values outside of the grain to zero, we are only
      // including the contribution of the induced magnetic field within the
      // grain to the free energy.
      // Note that for proper treatment, this might have to be done in 3D
      // (depending on assumptions/model). For a layered superconductor, taking
      // into account only 2D means the free energy per sheet/plane. Taking into
      // account 3D, on the other hand, captures e.g. a truly 2D SC or SC at a
      // surface of a 3D superconductor.
      C::getGeometry()->zeroValuesOutsideDomainRef(&magneticField);

      // Note that the prefactor depend on our choice of unit definitions, e.g.
      // (2pi) in the denominator of the coherence length. Here it is assumed
      // that it is defined without the (2pi).
      const T magneticFactor = static_cast<T>(0.5 * M_PI * M_PI) *
                               penetrationDepth * penetrationDepth;

      // Update the magnetic sum.
      magneticSum += magneticFactor * magneticField.sumSquare();
    }

    // For convenience.
    const T latticeElementArea = latticeElementSize * latticeElementSize;

    // Free energy (area integrated).
    const T magneticFreeEnergy = magneticSum * latticeElementArea;
    const T totalFreeEnergy =
        (nonMagneticSum + magneticSum) * latticeElementArea;

    // Free energy (normalized).
    const T magneticFreeEnergyPerUnitArea =
        magneticSum / static_cast<T>(numGrainLatticeSites);
    const T totalFreeEnergyPerUnitArea =
        (nonMagneticSum + magneticSum) / static_cast<T>(numGrainLatticeSites);

    // Set residual and convergence.
    const T residual =
        computeResidual(m_freeEnergyPerUnitArea, totalFreeEnergyPerUnitArea);
    const bool isConverged =
        residual < C::getParameters()->getConvergenceCriterion();
    ComputeProperty<T>::setConverged(isConverged);
    const int iterationIdx = ContextModule<T>::getContext()->getNumIterations();
    if (iterationIdx > 0) {
      ComputeProperty<T>::setResidual(residual);
    } else {
      // If this is the first iteration the residual is massive. Which to the
      // user, probably indicates that something is really wrong. So lets set
      // the residual to zero, but to non-converged.
      ComputeProperty<T>::setResidual(static_cast<T>(0));
    }

    // Update member variables.
    m_freeEnergy = totalFreeEnergy;
    m_freeEnergyPerUnitArea = totalFreeEnergyPerUnitArea;
    m_magneticFreeEnergy = magneticFreeEnergy;
    m_magneticFreeEnergyPerUnitArea = magneticFreeEnergyPerUnitArea;
  } else {
    ComputeProperty<T>::setConverged(false);
  }
}
} // namespace scg

#endif // COMPUTE_FREE_ENERGY_H_
