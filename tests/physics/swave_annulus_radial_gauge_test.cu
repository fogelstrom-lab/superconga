#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "conga/Context.h"
#include "conga/Parameters.h"
#include "conga/Type.h"
#include "conga/accelerators/BarzilaiBorwein.h"
#include "conga/boundary/BoundaryConditionSpecular.h"
#include "conga/compute/ComputeCurrent.h"
#include "conga/compute/ComputeFreeEnergy.h"
#include "conga/compute/ComputeOrderParameter.h"
#include "conga/defines.h"
#include "conga/geometry/DiscGeometry.h"
#include "conga/geometry/GeometryGroup.h"
#include "conga/grid.h"
#include "conga/integration/IntegrationIteratorOzaki.h"
#include "conga/order_parameter/OrderParameter.h"
#include "conga/order_parameter/OrderParameterBasis.h"
#include "conga/order_parameter/OrderParameterBulk.h"
#include "conga/order_parameter/OrderParameterComponent.h"
#include "conga/order_parameter/OrderParameterInitial.h"
#include "conga/riccati/RiccatiSolverConfined.h"
#include "conga/run_iteration.h"
#include "conga/utils.h"
#include "conga/wrappers/doctest.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace helper {
/// \brief Helper function for the weights and abscissas needed for Gaussian
/// quadrature.
///
/// \param inLow The lower integration limit.
/// \param inHigh The upper integration limit.
///
/// \return The pair {weights, abscissas}.
template <typename T>
std::pair<std::vector<T>, std::vector<T>>
getWeightsAndAbscissas(const T inLow, const T inHigh) {
  // See: https://pomax.github.io/bezierinfo/legendre-gauss.html

  std::vector<T> weights = {
      static_cast<T>(0.0965400885147278), static_cast<T>(0.0965400885147278),
      static_cast<T>(0.0956387200792749), static_cast<T>(0.0956387200792749),
      static_cast<T>(0.0938443990808046), static_cast<T>(0.0938443990808046),
      static_cast<T>(0.0911738786957639), static_cast<T>(0.0911738786957639),
      static_cast<T>(0.0876520930044038), static_cast<T>(0.0876520930044038),
      static_cast<T>(0.0833119242269467), static_cast<T>(0.0833119242269467),
      static_cast<T>(0.0781938957870703), static_cast<T>(0.0781938957870703),
      static_cast<T>(0.0723457941088485), static_cast<T>(0.0723457941088485),
      static_cast<T>(0.0658222227763618), static_cast<T>(0.0658222227763618),
      static_cast<T>(0.0586840934785355), static_cast<T>(0.0586840934785355),
      static_cast<T>(0.0509980592623762), static_cast<T>(0.0509980592623762),
      static_cast<T>(0.0428358980222267), static_cast<T>(0.0428358980222267),
      static_cast<T>(0.0342738629130214), static_cast<T>(0.0342738629130214),
      static_cast<T>(0.0253920653092621), static_cast<T>(0.0253920653092621),
      static_cast<T>(0.0162743947309057), static_cast<T>(0.0162743947309057),
      static_cast<T>(0.0070186100094701), static_cast<T>(0.0070186100094701)};

  std::vector<T> abscissas = {
      static_cast<T>(-0.0483076656877383), static_cast<T>(0.0483076656877383),
      static_cast<T>(-0.1444719615827965), static_cast<T>(0.1444719615827965),
      static_cast<T>(-0.2392873622521371), static_cast<T>(0.2392873622521371),
      static_cast<T>(-0.3318686022821277), static_cast<T>(0.3318686022821277),
      static_cast<T>(-0.4213512761306353), static_cast<T>(0.4213512761306353),
      static_cast<T>(-0.5068999089322294), static_cast<T>(0.5068999089322294),
      static_cast<T>(-0.5877157572407623), static_cast<T>(0.5877157572407623),
      static_cast<T>(-0.6630442669302152), static_cast<T>(0.6630442669302152),
      static_cast<T>(-0.7321821187402897), static_cast<T>(0.7321821187402897),
      static_cast<T>(-0.7944837959679424), static_cast<T>(0.7944837959679424),
      static_cast<T>(-0.8493676137325700), static_cast<T>(0.8493676137325700),
      static_cast<T>(-0.896321155766052),  static_cast<T>(0.896321155766052),
      static_cast<T>(-0.9349060759377397), static_cast<T>(0.9349060759377397),
      static_cast<T>(-0.9647622555875064), static_cast<T>(0.9647622555875064),
      static_cast<T>(-0.9856115115452684), static_cast<T>(0.9856115115452684),
      static_cast<T>(-0.9972638618494816), static_cast<T>(0.9972638618494816)};

  // Transform the weights and abscissas to our range.
  const T scale = static_cast<T>(0.5) * (inHigh - inLow);
  const T offset = static_cast<T>(0.5) * (inHigh + inLow);

  std::transform(weights.begin(), weights.end(), weights.begin(),
                 [&](T x) -> T { return x * scale; });

  std::transform(abscissas.begin(), abscissas.end(), abscissas.begin(),
                 [&](T x) -> T { return x * scale + offset; });

  return {weights, abscissas};
}

/// \brief Get the grain and hole area of the annulus.
///
/// \param inOuterRadius The outer radius of the annulus.
/// \param inInnerRadius The inner radius of the annulus.
///
/// \return The pair {grain area, hole area}.
template <typename T>
std::pair<T, T> getAreas(const T inOuterRadius, const T inInnerRadius) {
  const T holeArea = static_cast<T>(M_PI) * inInnerRadius * inInnerRadius;
  const T grainArea =
      static_cast<T>(M_PI) * inOuterRadius * inOuterRadius - holeArea;
  return {grainArea, holeArea};
}

/// \brief Get (a,b,c) factors.
///
/// \param inOuterRadius The outer radius of the annulus.
/// \param inInnerRadius The inner radius of the annulus.
/// \param inScaledPenetrationDepth The penetration depth divided by
/// sqrt(Yoshida).
/// \param inNumFluxQuanta The number of external flux quanta.
/// \param inPhaseWinding The phase winding of the order parameter.
/// \param inChargeSign The sign of the carrier charge.
///
/// \return The tuple {a, b, c}.
template <typename T>
std::tuple<T, T, T>
getFactorsABC(const T inOuterRadius, const T inInnerRadius,
              const T inScaledPenetrationDepth, const T inNumFluxQuanta,
              const T inPhaseWinding, const int inChargeSign) {
  // The grain and hole area.
  const auto [grainArea, holeArea] = getAreas(inOuterRadius, inInnerRadius);

  // For convenience.
  const T R_k = inOuterRadius / inScaledPenetrationDepth;
  const T r_k = inInnerRadius / inScaledPenetrationDepth;

  // Relevant Bessel function.
  const T I0_r = std::cyl_bessel_i(static_cast<T>(0), r_k);
  const T I0_R = std::cyl_bessel_i(static_cast<T>(0), R_k);
  const T I2_r = std::cyl_bessel_i(static_cast<T>(2), r_k);
  const T K0_r = std::cyl_bessel_k(static_cast<T>(0), r_k);
  const T K0_R = std::cyl_bessel_k(static_cast<T>(0), R_k);
  const T K2_r = std::cyl_bessel_k(static_cast<T>(2), r_k);

  // For even more convenience.
  const T chargePhaseWinding = static_cast<T>(inChargeSign) * inPhaseWinding;

  // Make factors "a" and "b". The factor of pi in invM is due to our choice
  // of units, i.e. A_0 = \Phi_0 / (pi \xi_0).
  const T invM = static_cast<T>(M_PI) / (I0_R * K2_r - I2_r * K0_R);
  const T a = invM * (inNumFluxQuanta * K2_r / grainArea -
                      chargePhaseWinding * K0_R / holeArea);
  const T b = invM * (inNumFluxQuanta * I2_r / grainArea -
                      chargePhaseWinding * I0_R / holeArea);

  // Called beta in the paper.
  const T c =
      invM * (static_cast<T>(2 * M_PI) * inScaledPenetrationDepth *
                  inScaledPenetrationDepth * inNumFluxQuanta / grainArea +
              chargePhaseWinding * (I0_R * K0_r - I0_r * K0_R));

  return {a, b, c};
}

/// \brief Get the scaling factor for the kinetic and magnetic free energies,
/// and the magnetic moment.
///
/// \param inOuterRadius The outer radius of the annulus.
/// \param inInnerRadius The inner radius of the annulus.
/// \param inPenetrationDepth The penetration depth.
template <typename T>
T getScalingFactor(const T inOuterRadius, const T inInnerRadius,
                   const T inPenetrationDepth) {
  // The grain and hole area.
  const auto [grainArea, holeArea] = getAreas(inOuterRadius, inInnerRadius);
  const T angularFactor = static_cast<T>(2.0 * M_PI);
  return angularFactor * std::pow(inPenetrationDepth, 2) /
         (static_cast<T>(2) * grainArea);
}

/// \brief Evaluate the magnetic moment expression.
///
/// \param inOuterRadius The outer radius of the annulus [xi0].
/// \param inInnerRadius The inner radius of the annulus [xi0].
/// \param inScaledPenetrationDepth The penetration depth divided by
/// sqrt(Yoshida) [xi0].
/// \param inNumFluxQuanta The number of external flux quanta.
/// \param inPhaseWinding The phase winding of the order parameter.
/// \param inChargeSign The sign of the carrier charge.
/// \param inTemperature The temperature [Tc].
/// \param inEnergyCutoff The energy cutoff [2*pi*kB*Tc].
///
/// \return The magnetic moment [j_0*xi_0*Area].
template <typename T>
T evalMagneticMoment(const T inOuterRadius, const T inInnerRadius,
                     const T inPenetrationDepth, const T inNumFluxQuanta,
                     const T inPhaseWinding, const int inChargeSign,
                     const T inTemperature, const T inEnergyCutoff) {
  // Compute bulk order parameter.
  const T orderParameterBulk =
      conga::computeOrderParameterBulkSWave(inTemperature);

  // Yoshida function.
  const T yoshida = conga::computeYoshidaBulkSWave(
      inTemperature, orderParameterBulk, inEnergyCutoff);

  // The penetration depth in the expressions is temperature dependent.
  const T scaledPenetrationDepth = inPenetrationDepth / std::sqrt(yoshida);

  // For convenience.
  const T R_k = inOuterRadius / scaledPenetrationDepth;
  const T r_k = inInnerRadius / scaledPenetrationDepth;

  // Relevant Bessel function.
  const T I2_r = std::cyl_bessel_i(static_cast<T>(2), r_k);
  const T I2_R = std::cyl_bessel_i(static_cast<T>(2), R_k);
  const T K2_r = std::cyl_bessel_k(static_cast<T>(2), r_k);
  const T K2_R = std::cyl_bessel_k(static_cast<T>(2), R_k);

  // The (a,b,c) factors.
  const auto [a, b, c] =
      getFactorsABC(inOuterRadius, inInnerRadius, scaledPenetrationDepth,
                    inNumFluxQuanta, inPhaseWinding, inChargeSign);

  // Overall scaling factor.
  const T factor =
      getScalingFactor(inOuterRadius, inInnerRadius, inPenetrationDepth);

  // The magnetic moment.
  return -factor * (inOuterRadius * inOuterRadius * (a * I2_R - b * K2_R) -
                    inInnerRadius * inInnerRadius * (a * I2_r - b * K2_r));
}

/// \brief Evaluate the free energy expressions.
///
/// \param inOuterRadius The outer radius of the annulus [xi0].
/// \param inInnerRadius The inner radius of the annulus [xi0].
/// \param inScaledPenetrationDepth The penetration depth divided by
/// sqrt(Yoshida) [xi0].
/// \param inNumFluxQuanta The number of external flux quanta.
/// \param inPhaseWinding The phase winding of the order parameter.
/// \param inChargeSign The sign of the carrier charge.
/// \param inTemperature The temperature [Tc].
/// \param inEnergyCutoff The energy cutoff [2*pi*kB*Tc].
///
/// \return The pair {total free energy, magnetic free energy}
/// [(2*pi*kB*Tc)^2*NF*Area].
template <typename T>
std::pair<T, T> evalFreeEnergy(const T inOuterRadius, const T inInnerRadius,
                               const T inPenetrationDepth,
                               const T inNumFluxQuanta, const T inPhaseWinding,
                               const int inChargeSign, const T inTemperature,
                               const T inEnergyCutoff) {
  // Compute bulk free energy.
  const T orderParameterBulk =
      conga::computeOrderParameterBulkSWave(inTemperature);
  const T freeEnergyBulk =
      conga::computeFreeEnergyBulkSWave(inTemperature, orderParameterBulk);

  // Yoshida function.
  const T yoshida = conga::computeYoshidaBulkSWave(
      inTemperature, orderParameterBulk, inEnergyCutoff);

  // The penetration depth in the expressions is temperature dependent.
  const T scaledPenetrationDepth = inPenetrationDepth / std::sqrt(yoshida);

  // The grain and hole area.
  const auto [grainArea, holeArea] = getAreas(inOuterRadius, inInnerRadius);

  // The pi comes from our choice of units.
  const T externalFluxDensity =
      static_cast<T>(M_PI) * inNumFluxQuanta / grainArea;

  // The (a,b,c) factors.
  T a;
  T b;
  T c;
  std::tie(a, b, c) =
      getFactorsABC(inOuterRadius, inInnerRadius, scaledPenetrationDepth,
                    inNumFluxQuanta, inPhaseWinding, inChargeSign);

  // Overall scaling factor.
  const T factor =
      getScalingFactor(inOuterRadius, inInnerRadius, inPenetrationDepth);

  // Get weights and abscissas for Gaussian quadrature.
  const auto [weights, abscissas] =
      getWeightsAndAbscissas(inInnerRadius, inOuterRadius);

  // The integrand of the magnetic term in the free energy.
  auto fnMagneticIntegrand = [&](T x) {
    const T tmp =
        a * std::cyl_bessel_i(static_cast<T>(0), x / scaledPenetrationDepth) -
        b * std::cyl_bessel_k(static_cast<T>(0), x / scaledPenetrationDepth) -
        externalFluxDensity;
    return x * tmp * tmp;
  };

  // Evaluate the magnetic integrand at the abscissas.
  std::vector<T> magneticIntegrand(weights.size());
  std::transform(abscissas.begin(), abscissas.end(), magneticIntegrand.begin(),
                 fnMagneticIntegrand);

  // Multiply with the weights, sum and scale.
  const T freeEnergyMagneticGrain =
      factor * std::transform_reduce(magneticIntegrand.begin(),
                                     magneticIntegrand.end(), weights.begin(),
                                     static_cast<T>(0));

  // The free energy of the magnetic field in the hole.
  const T inducedFluxDensity = c / holeArea - externalFluxDensity;
  const T freeEnergyMagneticHole =
      factor * static_cast<T>(0.5) *
      std::pow(inducedFluxDensity * inInnerRadius, 2);

  // Total magnetic free energy.
  const T freeEnergyMagnetic = freeEnergyMagneticGrain + freeEnergyMagneticHole;

  // The integrand of the kinetic term in the free energy.
  auto fnKineticIntegrand = [&](T x) {
    const T tmp =
        a * std::cyl_bessel_i(static_cast<T>(1), x / scaledPenetrationDepth) +
        b * std::cyl_bessel_k(static_cast<T>(1), x / scaledPenetrationDepth);
    return x * tmp * tmp;
  };

  // Evaluate the kinetic integrand at the abscissas.
  std::vector<T> kineticIntegrand(weights.size());
  std::transform(abscissas.begin(), abscissas.end(), kineticIntegrand.begin(),
                 fnKineticIntegrand);

  // Multiply with the weights, sum and scale.
  const T freeEnergyKinetic =
      factor * std::transform_reduce(kineticIntegrand.begin(),
                                     kineticIntegrand.end(), weights.begin(),
                                     static_cast<T>(0));

  // Total free energy.
  const T freeEnergy = freeEnergyBulk + freeEnergyKinetic + freeEnergyMagnetic;

  return {freeEnergy, freeEnergyMagnetic};
}

/// \brief Test that the free energy, and magnetic moment, of an s-wave
/// superconductor in an annulus, with radial gauge, behaves as expected. The
/// integrands of the terms in the free energy can be computed exactly (assuming
/// constant order-parameter magnitude), but the integrals must be done
/// numerically. The magnetic moment can be computed exactly.
///
/// \return Void.
template <typename T> void testSWaveAnnulusRadial() {
  // =================== ARRANGE ==================== //
  // The accuracy of the test. The accuracy of the contribution of the magnetic
  // field in the hole is way worse than in the grain hence the large tolerance.
  // TODO(niclas): Is this due to the boundary bug with holes?
  const T tolerance = static_cast<T>(1e-3);
  const T toleranceMagnetic = static_cast<T>(0.02);
  const T scale = static_cast<T>(0);

  // The outer (R) and inner (r) radii of the annulus, and the penetration
  // depth. I.e. the relevant lengts.
  const T R = static_cast<T>(20);
  const T r = static_cast<T>(10);
  const T penetrationDepth = static_cast<T>(30);

  const int chargeSign = -1;
  const T numFluxQuanta = static_cast<T>(1);
  // Note, the charge is here just to make the test do the same thing when using
  // positive carriers.
  const T phaseWinding = static_cast<T>(2 * chargeSign);

  const T grainWidth = static_cast<T>(2) * R;
  const T pointsPerCoherenceLength = static_cast<T>(6.2);
  const int numMomenta = 32;
  const T convergenceCriterion = static_cast<T>(1e-4);
  const T temperature = static_cast<T>(0.2);
  const T energyCutoff = static_cast<T>(16);
  const T stepSize = static_cast<T>(1);
  const T vectorPotentialError = static_cast<T>(1e-3);
  const conga::ResidualNorm normType = conga::ResidualNorm::LInf;

  // Create context.
  conga::Context<T> context;

  // Set parameters.
  conga::Parameters<T> *parameters = new conga::Parameters<T>(&context);
  parameters->setGrainWidth(grainWidth);
  parameters->setPointsPerCoherenceLength(pointsPerCoherenceLength);
  parameters->setAngularResolution(numMomenta);
  parameters->setConvergenceCriterion(convergenceCriterion);
  parameters->setTemperature(temperature);
  parameters->setEnergyCutoff(energyCutoff);
  parameters->setNumFluxQuanta(numFluxQuanta);
  parameters->setPenetrationDepth(penetrationDepth);
  parameters->setChargeSign(chargeSign);
  parameters->setResidualNormType(normType);

  // Create annulus geometry.
  conga::GeometryGroup<T> *geometry = new conga::GeometryGroup<T>(&context);
  const T grainFraction = parameters->getGrainFraction();
  geometry->add(new conga::DiscGeometry<T>(grainFraction), conga::Type::add);
  const T relativeHoleRadius =
      static_cast<T>(conga::constants::MAXIMUM_COORDINATE) * r / R;
  geometry->add(new conga::DiscGeometry<T>(grainFraction, relativeHoleRadius),
                conga::Type::remove);

  // Integrator.
  new conga::IntegrationIteratorOzaki<T>(&context);

  // Create solver with radial gauge.
  conga::RiccatiSolver<T> *riccatiSolver =
      new conga::RiccatiSolverConfined<T, conga::gauge::Symmetric>(&context);
  riccatiSolver->add(new conga::BoundaryConditionSpecular<T>);

  // Create s-wave order parameter with phase winding.
  conga::OrderParameter<T> *orderParameter =
      new conga::OrderParameter<T>(&context);
  conga::OrderParameterComponent<T> *orderParameterComponent =
      new conga::OrderParameterComponent<T>(conga::BasisFunctionSymmetry::SWAVE,
                                            parameters);
  conga::Initial<T>::phaseWinding(orderParameterComponent, phaseWinding);
  orderParameter->add(orderParameterComponent);

  // Compute order parameter, current density, free energy, and vector
  // potential.
  new conga::ComputeOrderParameter<T>(&context);
  new conga::ComputeCurrent<T>(&context);
  new conga::ComputeFreeEnergy<T>(&context);
  context.add(
      std::make_unique<conga::ComputeVectorPotential<T>>(vectorPotentialError));

  // Set accelerator.
  auto accelerator =
      std::make_unique<conga::accelerators::BarzilaiBorwein<T>>();
  context.set(std::move(accelerator));

  // Run the simulation.
  context.initialize();
  const bool isConverged = conga::runCompute(0, 100, true, context);

  // The results.
  const T freeEnergy =
      context.getComputeFreeEnergy()->getFreeEnergyPerUnitArea();
  const T freeEnergyMagnetic =
      context.getComputeFreeEnergy()->getMagneticFreeEnergyPerUnitArea();
  const T latticeElementSize = context.getParameters()->getGridElementSize();
  const T magneticMoment =
      context.getComputeCurrent()->getResult()->moment(latticeElementSize) /
      context.getGeometry()->area();

  // =================== ASSERT ==================== //
  const auto [freeEnergyExpected, freeEnergyMagneticExpected] =
      evalFreeEnergy(R, r, penetrationDepth, numFluxQuanta, phaseWinding,
                     chargeSign, temperature, energyCutoff);
  const T magneticMomentExpected =
      evalMagneticMoment(R, r, penetrationDepth, numFluxQuanta, phaseWinding,
                         chargeSign, temperature, energyCutoff);

  // Compare results.
  CHECK(freeEnergy ==
        doctest::Approx(freeEnergyExpected).epsilon(tolerance).scale(scale));
  CHECK(freeEnergyMagnetic == doctest::Approx(freeEnergyMagneticExpected)
                                  .epsilon(toleranceMagnetic)
                                  .scale(scale));
  CHECK(
      magneticMoment ==
      doctest::Approx(magneticMomentExpected).epsilon(tolerance).scale(scale));

  CHECK(isConverged);
}

/// \brief Compare the expression of the magnetic moment using (a,b) factors to
/// the expression with the dependency on the external flux, and the phase
/// winding, disentangled.
///
/// \return Void.
template <typename T> void testMagneticMomentExpression() {
  // =================== ARRANGE ==================== //
  // The accuracy of the test.
  const T tolerance = static_cast<T>(100) * std::numeric_limits<T>::epsilon();
  const T scale = static_cast<T>(0);

  // The outer (R) and inner (r) radii of the annulus, and the penetration
  // depth. Do not change! The expected results where computed in Mathematica
  // using them.
  const T R = static_cast<T>(20);
  const T r = static_cast<T>(10);
  const T penetrationDepth = static_cast<T>(30);

  // General parameters.
  const int chargeSign = -1;
  const T numFluxQuanta = static_cast<T>(1);
  const T phaseWinding = static_cast<T>(2);

  // For convenience. Note that we here assume that T=0.
  const T R_k = R / penetrationDepth;
  const T r_k = r / penetrationDepth;

  // Relevant Bessel function.
  const T I0_R = std::cyl_bessel_i(static_cast<T>(0), R_k);
  const T I2_r = std::cyl_bessel_i(static_cast<T>(2), r_k);
  const T I2_R = std::cyl_bessel_i(static_cast<T>(2), R_k);
  const T K0_R = std::cyl_bessel_k(static_cast<T>(0), R_k);
  const T K2_r = std::cyl_bessel_k(static_cast<T>(2), r_k);
  const T K2_R = std::cyl_bessel_k(static_cast<T>(2), R_k);

  // The grain and hole area.
  const auto [grainArea, holeArea] = getAreas(R, r);

  // Overall scaling factor.
  const T factor = getScalingFactor(R, r, penetrationDepth);

  // M factor in the paper.
  const T M = I0_R * K2_r - I2_r * K0_R;

  // =================== ACT ==================== //
  // The magnetic moment per external flux quanta.
  const T magneticMomentFluxQuanta = factor * (holeArea + grainArea) *
                                     (I2_r * K2_R - I2_R * K2_r) /
                                     (M * grainArea);

  // The magnetic moment per phase winding.
  const T magneticMomentPhaseWinding =
      factor * (static_cast<T>(2.0 * M_PI) * penetrationDepth *
                    penetrationDepth / (M * holeArea) -
                static_cast<T>(1));

  // The total magnetic moment.
  const T magneticMoment =
      magneticMomentFluxQuanta * numFluxQuanta -
      magneticMomentPhaseWinding * phaseWinding * static_cast<T>(chargeSign);

  // =================== ASSERT ==================== //
  // Do not change! Zero temperature so that the Yoshida function is 1.
  const T temperature = static_cast<T>(0);
  const T energyCutoff = static_cast<T>(100);
  const T magneticMomentExpected =
      evalMagneticMoment(R, r, penetrationDepth, numFluxQuanta, phaseWinding,
                         chargeSign, temperature, energyCutoff);

  // Computed in Mathematica.
  const T magneticMomentFluxQuantaExpected =
      static_cast<T>(-0.1953280762339516);
  const T magneticMomentPhaseWindingExpected =
      static_cast<T>(-0.2336266542359196);

  CHECK(
      magneticMoment ==
      doctest::Approx(magneticMomentExpected).epsilon(tolerance).scale(scale));
  CHECK(magneticMomentFluxQuanta ==
        doctest::Approx(magneticMomentFluxQuantaExpected)
            .epsilon(tolerance)
            .scale(scale));
  CHECK(magneticMomentPhaseWinding ==
        doctest::Approx(magneticMomentPhaseWindingExpected)
            .epsilon(tolerance)
            .scale(scale));
}
} // namespace helper

// Test free energy and magnetic moment for an s-wave annulus.

TEST_CASE("Test s-wave annulus (radial gauge) - <float>") {
  helper::testSWaveAnnulusRadial<float>();
}

TEST_CASE("Test s-wave annulus (radial gauge) - <double>") {
  helper::testSWaveAnnulusRadial<double>();
}

// Test the magnetic moment expression.

TEST_CASE("Test magnetic moment expression - <float>") {
  helper::testMagneticMomentExpression<float>();
}

TEST_CASE("Test magnetic moment expression - <double>") {
  helper::testMagneticMomentExpression<double>();
}
