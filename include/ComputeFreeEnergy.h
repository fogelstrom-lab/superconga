#ifndef COMPUTE_FREE_ENERGY_H_
#define COMPUTE_FREE_ENERGY_H_

#include "ComputeProperty.h"
#include "ComputeResidual.h"
#include "IntegrationIteratorOzaki.h"
#include "LocalGreens.h"
#include "OrderParameterBulk.h"
#include "cplx.h"
#include <cmath>
#include <cstdio>
#include <limits>

namespace scg {
template <typename T> class ComputeFreeEnergy : public ComputeProperty<T> {
  typedef typename Grid<T>::GPU GridGPU;
  typedef typename Grid<T>::CPU GridCPU;

public:
  // Constructors and Destructors
  ComputeFreeEnergy();
  virtual ~ComputeFreeEnergy();

  ComputeFreeEnergy(Context<T> *const inOutContext);

  virtual GridGPU *getResult() { return m_freeEnergyGrid; }
  T getFreeEnergy() const { return m_freeEnergy; }
  T getFreeEnergyPerUnitArea() const { return m_freeEnergyPerUnitArea; }

  virtual int initialize();
  virtual IntegrationIterator<T> *
  createIntegrationIterator(const int inNumEnergiesPerBlock = -1);

  // This should be called once for every iteration, before any integration
  // takes place.
  virtual void initializeIteration(const int inNumFields = 1);

  // This should be called once for every angle, before any integration takes
  // place.
  virtual void initializeGreensAngle();

  // Computes greens function for current angle. This may be called multiple
  // times as we integrate over many energies. Result is in local frame and will
  // be stored in _greensCurrentAngle.
  virtual void computeGreensAngle(ParameterIterator<int> *inParameterIterator,
                                  GridGPU *inGamma, GridGPU *inGammabar,
                                  GeometryGroup<T> *inGeometryGroup);

  // Add the result from computeGreensAngle() to total. Rotates
  // _greensCurrentAngle to global frame and adds momentum dependence.
  virtual void integrateGreensAngle(GeometryGroup<T> *inGeometryGroup,
                                    const int inAngleIdx);

  // Compute physical property from greens function.
  virtual void computePropertyFromGreens();

private:
  // Total free energy
  T m_freeEnergy;
  T m_freeEnergyPerUnitArea;

  GridGPU *m_freeEnergyGrid;
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
__device__ void computeFreeEnergyKernelI(const cplx<T> &inOrderParameter,
                                         const cplx<T> &inGamma,
                                         const cplx<T> &inGammabar,
                                         cplx<T> &outFreeEnergy) {

  outFreeEnergy =
      (inOrderParameter.conj() * inGamma - inOrderParameter * inGammabar);
}

// Parallelize calculation of free-energy kernel I over spatial coordinates
template <typename T>
__global__ void computeFreeEnergy_Kernel(
    const int inNumLatticePoints, const int inNumEnergies, T *outFreeEnergy_re,
    T *outFreeEnergy_im, const T *const inGamma_re, const T *const inGamma_im,
    const T *const inGammabar_re, const T *const inGammabar_im,
    const char *const inDomainLabels, const T *const inOrderParameter_re,
    const T *const inOrderParameter_im, const T *const inOzakiResidues) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x; // x index
  const int y = blockDim.y * blockIdx.y + threadIdx.y; // y index

  // Check that thread has work to do
  if ((x < inNumLatticePoints) && (y < inNumLatticePoints)) {
    // Calculate unique index for the grain coordinate
    const int grainCoordinateIdx = y * inNumLatticePoints + x;

    // Initialize the sum (over Ozaki poles) to zero
    cplx<T> freeEnergyKernelSum(T(0.0), T(0.0));

    // Order parameter at this coordinate
    const cplx<T> orderParameter(inOrderParameter_re[grainCoordinateIdx],
                                 inOrderParameter_im[grainCoordinateIdx]);

    // Check that coordinate is part of the grain
    if (inDomainLabels[grainCoordinateIdx] > 0) {
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
        const cplx<T> gamma(inGamma_re[coherenceFunctionIdx],
                            inGamma_im[coherenceFunctionIdx]);
        const cplx<T> gammabar(inGammabar_re[coherenceFunctionIdx],
                               inGammabar_im[coherenceFunctionIdx]);
        cplx<T> freeEnergyKernelI;

        // Calculate I(x, y, energy, angle)
        computeFreeEnergyKernelI<T>(orderParameter, gamma, gammabar,
                                    freeEnergyKernelI);

        // Multiply by Ozaki residue and sum the contribution
        freeEnergyKernelSum +=
            inOzakiResidues[energyIdx] * cplx<T>::i() * freeEnergyKernelI;
      }
    }

    // Add contribution to the free energy in this coordinate
    outFreeEnergy_re[grainCoordinateIdx] += freeEnergyKernelSum.x;
    outFreeEnergy_im[grainCoordinateIdx] += freeEnergyKernelSum.y;
  }
}

template <typename T>
ComputeFreeEnergy<T>::ComputeFreeEnergy()
    : ComputeProperty<T>(), m_freeEnergyGrid(0) {}

template <typename T>
ComputeFreeEnergy<T>::ComputeFreeEnergy(Context<T> *const inOutContext)
    : ComputeProperty<T>(inOutContext), m_freeEnergyGrid(0) {
  inOutContext->add(this);
}

template <typename T> ComputeFreeEnergy<T>::~ComputeFreeEnergy() {
  if (m_freeEnergyGrid)
    delete m_freeEnergyGrid;
}

template <typename T> int ComputeFreeEnergy<T>::initialize() {
  std::cout << "ComputeFreeEnergy<T>::initialize()\n";

  typedef ComputeProperty<T> C;

  // Get resolution
  const int N0 = C::getParameters()->getGridResolutionBase();
  const int N1 = C::getParameters()->getGridResolutionCoherence();

  GridGPU::initializeGrid(m_freeEnergyGrid, N0, N0);

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
  if (ComputeProperty<T>::doCompute())
    ComputeProperty<T>::initializeGreensAngle(); // Sets to zero
}

// Computes the free-energy kernel 'I' for current angle, for every spatial grid
// point (summed over energies).
template <typename T>
void ComputeFreeEnergy<T>::computeGreensAngle(
    ParameterIterator<int> *inParameterIterator, GridGPU *inGamma,
    GridGPU *inGammabar, GeometryGroup<T> *inGeometryGroup) {
  typedef ContextModule<T> C;

  if (ComputeProperty<T>::doCompute()) {
    IntegrationIterator<T> *integrationIter = C::getIntegrationIterator();

    GridGPU *freeEnergyKernelIntegrand =
        ComputeProperty<T>::getGreensCurrentAngle();

    GridGPU *deltaRot = integrationIter->getOrderParameterFieldCurrentAngle();
    GridGPU *ozakiResidues = integrationIter->getResidues();

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
        inGeometryGroup->getDomainLabelsGrid()->getDataPointer(0, Type::real),
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
    GridGPU *freeEnergyKernelIntegrand =
        ComputeProperty<T>::getGreensCurrentAngle();

    inGeometryGroup->extendValuesAtBoundary(freeEnergyKernelIntegrand, 2);

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
    // Get properties
    const T latticeElementSize = C::getParameters()->getGridElementSize();
    const T temperature = C::getParameters()->getTemperature();
    const int numGrainLatticeSites =
        C::getGeometry()->getNumGrainLatticeSites();

    // Compute second term of free energy not involving angle integration
    GridCPU ozakiEnergies = *(C::getIntegrationIterator()->getEnergies());
    GridCPU ozakiResidues = *(C::getIntegrationIterator()->getResidues());

    const T *ozakiEnergies_im = ozakiEnergies.getDataPointer(0, Type::imag);
    const T *ozakiResidues_re = ozakiResidues.getDataPointer(0, Type::real);

    int numEnergies = ozakiEnergies.dimX();
    T sumInverseEnergy = 0.0;

    for (int ozakiIdx = 0; ozakiIdx < numEnergies; ++ozakiIdx) {
      sumInverseEnergy +=
          ozakiResidues_re[ozakiIdx] / ozakiEnergies_im[ozakiIdx];
    }

    // Compute free energy new way
    T energySummedKernelI_re = 0.0, energySummedKernelI_im = 0.0;
    C::getGeometry()->zeroValuesOutsideDomain(m_freeEnergyGrid);
    m_freeEnergyGrid->sumField(energySummedKernelI_re, energySummedKernelI_im);

    // Compute total free energy
    const T freeEnergyPerunitArea_old = m_freeEnergyPerUnitArea;

    OrderParameter<T> *orderParameter = C::getOrderParameter();
    GridGPU deltaField(*(orderParameter->getOrderParameterField()));
    C::getGeometry()->zeroValuesOutsideDomain(&deltaField);

    m_freeEnergyPerUnitArea =
        static_cast<T>(2.0 * M_PI) * temperature * energySummedKernelI_re;
    sumInverseEnergy *= static_cast<T>(2.0 * M_PI) * temperature;

    for (int fieldIdx = 0; fieldIdx < deltaField.numFields(); ++fieldIdx) {
      const T relativeTc =
          orderParameter->getOrderParameterComponent(fieldIdx)->getRelativeTc();
      const T integrandFactor =
          std::log(temperature / relativeTc) + sumInverseEnergy;
      m_freeEnergyPerUnitArea +=
          deltaField[fieldIdx].complexMagnitudeSq().sumField() *
          integrandFactor;
    }

    const bool hasComputedInducedVectorPotential =
        C::getRiccatiSolver()->hasComputedInducedVectorPotential();

    // Add contribution from the induced magnetic field.
    if (hasComputedInducedVectorPotential) {
      // Compute the magnetic field.
      GridGPU magneticField =
          (*(C::getContext()->getRiccatiSolver()->getInducedVectorPotential()))
              .curl(latticeElementSize);

      // Note, by setting values outside of the grain to zero, we are only
      // including the contribution of the induced magnetic field within the
      // grain to the free energy.
      // TODO(niclas): Should we include the field outside of the grain? Also,
      // should we include the field in 3D?
      C::getGeometry()->zeroValuesOutsideDomain(&magneticField);

      // The Ginzburg-Landau parameter.
      const T kappa = C::getParameters()->getPenetrationDepth();

      // Note that the prefactor depend on our choice of unit definitions, e.g.
      // (2pi) in the denominator of the coherence length. Here it is assumed
      // that it is defined without the (2pi).
      const T magneticFactor =
          static_cast<T>(0.5 * M_PI * M_PI) * kappa * kappa;

      const T magneticFreeEnergy = magneticFactor * magneticField.sumSquare();
      m_freeEnergyPerUnitArea += magneticFreeEnergy;
    }

    m_freeEnergy =
        m_freeEnergyPerUnitArea * latticeElementSize * latticeElementSize;
    m_freeEnergyPerUnitArea *=
        static_cast<T>(1.0) / static_cast<T>(numGrainLatticeSites);

    // Set residual and convergence
    const T residual =
        computeResidual(freeEnergyPerunitArea_old, m_freeEnergyPerUnitArea);
    ComputeProperty<T>::setResidual(residual);

    if (residual < C::getParameters()->getConvergenceCriterion())
      ComputeProperty<T>::setConverged();
  } else {
    m_freeEnergy = static_cast<T>(0.0);
  }
}
} // namespace scg

#endif // COMPUTE_FREE_ENERGY_H_
