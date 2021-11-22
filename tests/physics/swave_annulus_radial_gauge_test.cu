#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "conga/Context.h"
#include "conga/Parameters.h"
#include "conga/Type.h"
#include "conga/boundary/BoundaryConditionSpecular.h"
#include "conga/compute/ComputeCurrent.h"
#include "conga/compute/ComputeFreeEnergy.h"
#include "conga/compute/ComputeOrderParameter.h"
#include "conga/defines.h"
#include "conga/geometry/DiscGeometry.h"
#include "conga/geometry/GeometryGroup.h"
#include "conga/grid.h"
#include "conga/order_parameter/OrderParameter.h"
#include "conga/order_parameter/OrderParameterBasis.h"
#include "conga/order_parameter/OrderParameterBulk.h"
#include "conga/order_parameter/OrderParameterComponent.h"
#include "conga/order_parameter/OrderParameterInitial.h"
#include "conga/riccati/RiccatiSolverConfined.h"
#include "conga/run_utils.h"
#include "conga/utils.h"
#include "conga/wrappers/doctest.h"

#include <algorithm>
#include <cmath>
#include <limits>
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

/// \brief Helper function for computing the expected free energy and magnetic
/// momentum.
///
/// \param inTemperature The temperature [Tc].
/// \param inEnergyCutoff The energy cutoff [2*pi*kB*Tc].
/// \param inPenetrationDepth The penetration depth [xi0].
/// \param inNumFluxQuanta The number of external flux quanta.
/// \param inPhaseWinding The phase winding of the order parameter.
/// \param inInnerRadius The inner radius of the annulus [xi0].
/// \param inOuterRadius The outer radius of the annulus [xi0].
///
/// \return The pair {free energy, magnetic moment}.
template <typename T>
std::pair<T, T>
computeResultsAnalytic(const T inTemperature, const T inEnergyCutoff,
                       const T inPenetrationDepth, const T inNumFluxQuanta,
                       const T inPhaseWinding, const T inInnerRadius,
                       const T inOuterRadius) {
  // Compute bulk free energy.
  const T orderParameterBulk =
      conga::computeOrderParameterBulkSWave(inTemperature);
  const T freeEnergyBulk =
      conga::computeFreeEnergyBulkSWave(inTemperature, orderParameterBulk);

  // Yoshida function.
  const T yoshida = conga::computeYoshidaBulkSWave(
      inTemperature, orderParameterBulk, inEnergyCutoff);

  // For convenience.
  const T r = inInnerRadius;
  const T R = inOuterRadius;
  const T k = inPenetrationDepth / std::sqrt(yoshida);

  // Relevant areas.
  const T grainArea = static_cast<T>(M_PI) * (R * R - r * r);
  const T holeArea = static_cast<T>(M_PI) * r * r;

  // For more convenience.
  const T R_k = R / k;
  const T r_k = r / k;

  // Relevant Bessel function.
  const T I0_R = std::cyl_bessel_i(static_cast<T>(0), R_k);
  const T I2_r = std::cyl_bessel_i(static_cast<T>(2), r_k);
  const T I2_R = std::cyl_bessel_i(static_cast<T>(2), R_k);
  const T K0_R = std::cyl_bessel_k(static_cast<T>(0), R_k);
  const T K2_r = std::cyl_bessel_k(static_cast<T>(2), r_k);
  const T K2_R = std::cyl_bessel_k(static_cast<T>(2), R_k);

  // Make factors "a" and "b". The factor of pi in invD is due to our choice of
  // units, i.e. A_0 = \Phi_0 / (pi \xi_0).
  const T invD = static_cast<T>(M_PI) / (I0_R * K2_r - I2_r * K0_R);
  const T a = invD * (inNumFluxQuanta * K2_r / grainArea -
                      inPhaseWinding * K0_R / holeArea);
  const T b = invD * (inNumFluxQuanta * I2_r / grainArea -
                      inPhaseWinding * I0_R / holeArea);

  // Common factor for the magnetic and kinetic terms of the free energy, as
  // well as the magnetic moment.
  const T angularFactor = static_cast<T>(2.0 * M_PI);
  const T factor = angularFactor * yoshida * std::pow(k, 2) /
                   (static_cast<T>(2) * grainArea);

  // The magnetic moment.
  const T magneticMoment =
      -factor * (R * R * (a * I2_R - b * K2_R) - r * r * (a * I2_r - b * K2_r));

  // Get weights and abscissas for Gaussian quadrature.
  const auto [weights, abscissas] =
      getWeightsAndAbscissas(inInnerRadius, inOuterRadius);

  // The integrand of the magnetic term in the free energy.
  auto fnMagneticIntegrand = [&](T x) {
    const T tmp = a * std::cyl_bessel_i(static_cast<T>(0), x / k) -
                  b * std::cyl_bessel_k(static_cast<T>(0), x / k) -
                  static_cast<T>(M_PI) * inNumFluxQuanta / grainArea;
    return x * tmp * tmp;
  };

  // Evaluate the magnetic integrand at the abscissas.
  std::vector<T> magneticIntegrand(weights.size());
  std::transform(abscissas.begin(), abscissas.end(), magneticIntegrand.begin(),
                 fnMagneticIntegrand);

  // Multiply with the weights, sum and scale.
  const T freeEnergyMagnetic =
      factor * std::transform_reduce(magneticIntegrand.begin(),
                                     magneticIntegrand.end(), weights.begin(),
                                     static_cast<T>(0));

  // The integrand of the kinetic term in the free energy.
  auto fnKineticIntegrand = [&](T x) {
    const T tmp = a * std::cyl_bessel_i(static_cast<T>(1), x / k) +
                  b * std::cyl_bessel_k(static_cast<T>(1), x / k);
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

  return {freeEnergy, magneticMoment};
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
  // The accuracy of the test.
  const T tolerance = static_cast<T>(1e-3);
  const T scale = static_cast<T>(0);

  // The outer (R) and inner (r) radii of the annulus.
  const T R = static_cast<T>(20);
  const T r = static_cast<T>(10);

  // General parameters.
  // TODO(niclas): For large phase windings a very high spatial resolution is
  // needed. This needs to be investigated!
  const T grainWidth = static_cast<T>(2) * R;
  const T pointsPerCoherenceLength = static_cast<T>(8);
  const int numMomenta = 32;
  const T convergenceCriterion = static_cast<T>(1e-5);
  const T temperature = static_cast<T>(0.2);
  const T energyCutoff = static_cast<T>(16);
  const T phaseWinding = static_cast<T>(2);
  const T numFluxQuanta = static_cast<T>(1);
  const T penetrationDepth = static_cast<T>(20);
  const T stepSize = static_cast<T>(0.001);
  const T vectorPotentialError = static_cast<T>(1e-3);

  // =================== ACT ==================== //
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

  // Create annulus geometry.
  conga::GeometryGroup<T> *geometry = new conga::GeometryGroup<T>(&context);
  const T grainFraction = parameters->getGrainFraction();
  geometry->add(new conga::DiscGeometry<T>(grainFraction), conga::Type::add);
  const T holeFraction = r / R;
  geometry->add(new conga::DiscGeometry<T>(grainFraction, holeFraction),
                conga::Type::remove);

  // Create solver with solenoid gauge.
  conga::RiccatiSolver<T> *riccatiSolver =
      new conga::RiccatiSolverConfined<T, conga::VectorPotential::Radial>(
          &context);
  riccatiSolver->add(new conga::BoundaryConditionSpecular<T>);

  // Create s-wave order parameter with phase winding.
  conga::OrderParameter<T> *orderParameter =
      new conga::OrderParameter<T>(&context);
  conga::OrderParameterComponent<T> *orderParameterComponent =
      new conga::OrderParameterComponent<T>(conga::BasisFunctionSymmetry::SWAVE,
                                            parameters);
  conga::Initial<T>::phaseWinding(orderParameterComponent, phaseWinding);
  orderParameter->add(orderParameterComponent);

  // Create accelerator.
  context.set(std::make_unique<conga::Accelerator<T>>(stepSize));

  // Compute order parameter, current density, free energy, and vector
  // potential.
  new conga::ComputeOrderParameter<T>(&context);
  new conga::ComputeCurrent<T>(&context);
  new conga::ComputeFreeEnergy<T>(&context);
  context.add(
      std::make_unique<conga::ComputeVectorPotential<T>>(vectorPotentialError));

  // Run the simulation.
  context.initialize();
  conga::run_utils::runSimulation(&context);

  // The results.
  const T freeEnergy =
      context.getComputeFreeEnergy()->getFreeEnergyPerUnitArea();
  const T latticeElementSize = context.getParameters()->getGridElementSize();
  const T magneticMoment =
      context.getComputeCurrent()->getResult()->moment(latticeElementSize) /
      context.getGeometry()->area();

  // =================== ASSERT ==================== //
  const auto [freeEnergyExpected, magneticMomentExpected] =
      computeResultsAnalytic(temperature, energyCutoff, penetrationDepth,
                             numFluxQuanta, phaseWinding, r, R);

  // Compare results.
  CHECK(freeEnergy ==
        doctest::Approx(freeEnergyExpected).epsilon(tolerance).scale(scale));
  CHECK(
      magneticMoment ==
      doctest::Approx(magneticMomentExpected).epsilon(tolerance).scale(scale));
}
} // namespace helper

TEST_CASE("Test s-wave annulus (radial gauge) - <float>") {
  helper::testSWaveAnnulusRadial<float>();
}

TEST_CASE("Test s-wave annulus (radial gauge) - <double>") {
  helper::testSWaveAnnulusRadial<double>();
}
