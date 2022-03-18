//===----------- main.cu - Main file for setting up simulations -----------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This is the main file for setting up simulations. Never modify this file
/// directly. Instead, to set up a simulation, specify a parameter file, and/or
/// use command-line flags. The main file parses these in turn, and sets up a
/// self-consistent simulation in a confined geometry. For more advanced
/// simulations, create a new main file.
///
//===----------------------------------------------------------------------===//

// TODO(niclas): These should not have to be included! They are needed because
// "Context" only has forward declarations.
#include "conga/compute/ComputeCurrent.h"
#include "conga/compute/ComputeFreeEnergy.h"
#include "conga/compute/ComputeOrderParameter.h"

#include "conga/Context.h"
#include "conga/Parameters.h"
#include "conga/Type.h"
#include "conga/boundary/BoundaryConditionSpecular.h"
#include "conga/compute/ComputeLDOS.h"
#include "conga/configure.h"
#include "conga/geometry/GeometryGroup.h"
#include "conga/geometry/geometry_utils.h"
#include "conga/grid.h"
#include "conga/integration/IntegrationIteratorKeldysh.h"
#include "conga/io.h"
#include "conga/order_parameter/OrderParameter.h"
#include "conga/order_parameter/OrderParameterBasis.h"
#include "conga/order_parameter/OrderParameterComponent.h"
#include "conga/order_parameter/OrderParameterInitial.h"
#include "conga/riccati/RiccatiSolverConfined.h"
#include "conga/run_simulation.h"

#include <json/json.h>

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

//*******************************************************
// PRECISION
//*******************************************************
#ifdef CONGA_SINGLE_PRECISION
typedef float T;
#else
typedef double T;
#endif

//*******************************************************
// MAIN
//*******************************************************
int main(int argc, char *argv[]) {
  //*******************************************************
  // PARSE SPECTROSCOPY CONFIG FILE
  //*******************************************************
  Json::Value postprocessConfig;
  {
    if (argc < 2) {
      std::cerr << "-- ERROR! Please provide a config file." << std::endl;
      return EXIT_FAILURE;
    }
    const std::filesystem::path postprocessConfigPath = argv[1];
    const bool success =
        conga::io::read(postprocessConfigPath, postprocessConfig);
    if (!success) {
      std::cerr << "-- ERROR! Could not find/open config file '"
                << postprocessConfigPath << "'." << std::endl;
      return EXIT_FAILURE;
    }
#ifndef NDEBUG
    std::cout << "-- Post-process configuration parameters:" << std::endl;
    std::cout << postprocessConfig << std::endl;
#endif
  }

  // Path to directory containing data to load from.
  const std::filesystem::path loadDataPath(
      postprocessConfig["misc"]["load_path"].asString());

  //*******************************************************
  // PARSE MAGNETOSTATICS CONFIG FILE
  //*******************************************************
  Json::Value simulationConfig;
  {
    // TODO(Niclas): Support old filename.

    // Path to simulation config.
    const std::filesystem::path simulationConfigPath =
        loadDataPath / "simulation_config.json";
    const bool success =
        conga::io::read(simulationConfigPath, simulationConfig);
    if (!success) {
      std::cerr << "-- ERROR! Could not find/open config file '"
                << simulationConfigPath << "'." << std::endl;
      return EXIT_FAILURE;
    }
#ifndef NDEBUG
    std::cout << "-- Simulation configuration parameters:" << std::endl;
    std::cout << simulationConfig << std::endl;
#endif
  }

  // TODO(niclas): Add sanity check of all parameters here!?

  // Fetch the verbosity so we know whether to print stuff or not.
  const bool verbose = postprocessConfig["misc"]["verbose"].asBool();

  //***************************************************
  // CREATE CONTEXT
  //***************************************************
  conga::Context<T> context;

  //***************************************************
  // APPLY PARAMETERS
  //***************************************************
  // Storage for system simulation parameters (e.g. temperature, lattice size).
  conga::Parameters<T> *parameters = new conga::Parameters<T>(&context);

  // Set temperature.
  const T temperature =
      static_cast<T>(simulationConfig["physics"]["temperature"].asDouble());
  parameters->setTemperature(temperature);

  // Set the number of lattice points per coherence length.
  const T pointsPerCoherenceLength = static_cast<T>(
      simulationConfig["numerics"]["points_per_coherence_length"].asDouble());
  parameters->setPointsPerCoherenceLength(pointsPerCoherenceLength);

  // Set parameters relevant for real-axis energies.
  const T minEnergy = static_cast<T>(
      postprocessConfig["spectroscopy"]["energy_min"].asDouble());
  const T maxEnergy = static_cast<T>(
      postprocessConfig["spectroscopy"]["energy_max"].asDouble());
  const int numEnergies =
      postprocessConfig["spectroscopy"]["num_energies"].asInt();
  const T energyBroadening = static_cast<T>(
      postprocessConfig["spectroscopy"]["energy_broadening"].asDouble());
  parameters->setEnergyRange(minEnergy, maxEnergy, numEnergies,
                             energyBroadening);

  // Number of discrete steps in momentum angle.
  const int numFermiMomenta =
      postprocessConfig["numerics"]["num_fermi_momenta"].asInt();
  parameters->setAngularResolution(numFermiMomenta);

  // Set convergence criterion.
  const T convergenceCriterion = static_cast<T>(
      postprocessConfig["numerics"]["convergence_criterion"].asDouble());
  parameters->setConvergenceCriterion(convergenceCriterion);

  // Set penetration depth.
  const T penetrationDepth = static_cast<T>(
      simulationConfig["physics"]["penetration_depth"].asDouble());
  parameters->setPenetrationDepth(penetrationDepth);

  // Set external magnetic field strength.
  const T externalFluxQuanta = static_cast<T>(
      simulationConfig["physics"]["external_flux_quanta"].asDouble());
  parameters->setNumFluxQuanta(externalFluxQuanta);

  // Set sign of charge.
  const int chargeSign = simulationConfig["physics"]["charge_sign"].asInt();
  parameters->setChargeSign(chargeSign);

  // Set which norm to use when computing residuals.
  const std::string residualNorm(
      postprocessConfig["numerics"]["norm"].asString());
  if (residualNorm == "l1") {
    parameters->setResidualNormType(conga::ResidualNorm::L1);
  } else if (residualNorm == "l2") {
    parameters->setResidualNormType(conga::ResidualNorm::L2);
  } else if (residualNorm == "linf") {
    parameters->setResidualNormType(conga::ResidualNorm::LInf);
  } else {
    std::cerr << "-- ERROR! Norm type, " << residualNorm << ", not understod"
              << std::endl;
    return EXIT_FAILURE;
  }

  //***************************************************
  // CREATE GEOMETRY
  //***************************************************
  // Create geometry group: add/remove geometry objects (e.g. discs, polygons)
  // from the geometry group.
  conga::GeometryGroup<T> *geometry = new conga::GeometryGroup<T>(&context);

  // Parse user-defined geometry. Compute factors to scale and translate
  // geometry to internal coordinate system.
  T scalingFactor;
  T xOffset;
  T yOffset;
  conga::geometry_utils::computeGeometryScaling(simulationConfig, scalingFactor,
                                                xOffset, yOffset);

  // Set grain sidelength in coherence length.
  parameters->setGrainWidth(scalingFactor);

  // Fetch relative size of geometry to available simulation space, to account
  // for padding for boundary interpolation.
  const T grainFraction = parameters->getGrainFraction();

  // Parse user-defined geometry, loop through geometry component list and
  // create compund geometry. Scale and translate to internal coordinate system.
  conga::geometry_utils::addGeometryComponents(geometry, simulationConfig,
                                               scalingFactor, xOffset, yOffset,
                                               grainFraction);

  //***************************************************
  // SET UP INTEGRATOR
  //***************************************************
  const int numEnergiesPerBlock =
      postprocessConfig["numerics"]["num_energies_per_block"].asInt();
  new conga::IntegrationIteratorKeldysh<T>(&context, numEnergiesPerBlock);

  //***************************************************
  // SET UP ORDER PARAMETER
  //***************************************************
  // Create order parameter (which might contain multiple components, i.e.
  // d+is).
  conga::OrderParameter<T> *orderParameter =
      new conga::OrderParameter<T>(&context);

  // Set rotation of crystal axes with respect to the grain geometry.
  const T crystalAxesRotation = static_cast<T>(
      2.0 * M_PI *
      simulationConfig["physics"]["crystal_axes_rotation"].asDouble());
  orderParameter->setCrystalAxesRotation(crystalAxesRotation);

  // For convenience.
  const Json::Value opParams = simulationConfig["physics"]["order_parameter"];

  // An array of all supported symmetries.
  // TODO(niclas): Move to OrderParameterBasis.h?
  const std::vector<std::string> symmetries = {"dx2-y2", "dxy", "g", "s"};

  // Loop over symmetries and add them if present.
  bool orderParameterSuccess = false;
  for (const std::string &symmetryStr : symmetries) {
    if (opParams.isMember(symmetryStr)) {
      // There is at least one component specified.
      orderParameterSuccess = true;

      // TODO(niclas): Move to OrderParameterBasis.h?
      conga::BasisFunctionSymmetry symmetry;
      if (symmetryStr == "dx2-y2") {
        symmetry = conga::BasisFunctionSymmetry::DWAVE_X2_Y2;
      } else if (symmetryStr == "dxy") {
        symmetry = conga::BasisFunctionSymmetry::DWAVE_XY;
      } else if (symmetryStr == "g") {
        symmetry = conga::BasisFunctionSymmetry::GWAVE;
      } else {
        symmetry = conga::BasisFunctionSymmetry::SWAVE;
      }

      // For convenience.
      const Json::Value opCompParams = opParams[symmetryStr];

      // Create component.
      const T criticalTemperature =
          static_cast<T>(opCompParams["critical_temperature"].asDouble());
      conga::OrderParameterComponent<T> *orderParameterComponent =
          new conga::OrderParameterComponent<T>(symmetry, parameters,
                                                criticalTemperature);

      // Add the initialized component to the total order parameter.
      orderParameter->add(orderParameterComponent);
    }
  }

  if (!orderParameterSuccess) {
    std::cerr << "-- ERROR! No order-parameter component set." << std::endl;
    return EXIT_FAILURE;
  }

  //***************************************************
  // READ ORDER PARAMETER AND VECTOR POTENTIAL FROM FILE
  //***************************************************
  {
    // Create the placeholder grid for vector potential.
    const int gridResolution = parameters->getGridResolutionBase();
    auto vectorPotential = std::make_unique<conga::GridGPU<T>>(
        gridResolution, gridResolution, 2, conga::Type::real);

    bool successOrderParameter = false;
    bool successVectorPotential = false;

#ifdef CONGA_USE_HDF5
    // Check if the simulation file exists.
    const std::filesystem::path simulationDataPath =
        loadDataPath / "simulation.h5";
    const bool simulationExists = std::filesystem::exists(simulationDataPath);
    if (simulationExists) {
      // Open file.
      const auto file = conga::io::fileReadOnly(simulationDataPath);

      // Read order parameter.
      successOrderParameter =
          conga::io::read(file, "order_parameter", *orderParameter);
      conga::io::printRead(verbose, successOrderParameter, "order parameter",
                           simulationDataPath);

      // Read vector potential.
      successVectorPotential =
          conga::io::read(file, "vector_potential", *(vectorPotential.get()));
      conga::io::printRead(verbose, successVectorPotential, "vector potential",
                           simulationDataPath);
    } else
#endif // CONGA_USE_HDF5
    {
      // Read order parameter.
      const std::filesystem::path orderParameterPath =
          loadDataPath / "order_parameter.csv";
      successOrderParameter =
          conga::io::read(orderParameterPath, *orderParameter);
      conga::io::printRead(verbose, successOrderParameter, "order parameter",
                           orderParameterPath);

      // Read vector potential.
      const std::filesystem::path vectorPotentialPath =
          loadDataPath / "vector_potential.csv";
      successVectorPotential =
          conga::io::read(vectorPotentialPath, *(vectorPotential.get()));
      conga::io::printRead(verbose, successVectorPotential, "vector potential",
                           vectorPotentialPath);
    }

    // Abort if not successful.
    if (!successOrderParameter) {
      std::cerr << "-- ERROR! Could not find order parameter in "
                << loadDataPath << std::endl;
      return EXIT_FAILURE;
    }

    // Abort if not successful.
    if (!successVectorPotential) {
      std::cerr << "-- ERROR! Could not find vector potential in "
                << loadDataPath << std::endl;
      return EXIT_FAILURE;
    }

    // Set the vector potential in context.
    if (penetrationDepth > static_cast<T>(0)) {
      // Internally the induced vector potential is scaled by the squared
      // penetration depth.
      *vectorPotential *= (penetrationDepth * penetrationDepth);
    } else {
      vectorPotential->setZero();
    }
    context.set(std::move(vectorPotential));
  }

  //***************************************************
  // CHOOSE QUANTITIES TO CALCULATE
  //***************************************************
  // Add LDOS compute object.
  new conga::ComputeLDOS<T>(&context);
  // TODO(niclas): Add spectral current?

  //***************************************************
  // CHOOSE RICCATI SOLVER AND BOUNDARY CONDITIONS
  //***************************************************
  conga::RiccatiSolver<T> *riccatiSolver;

  // Gauge choice.
  const std::string gauge(simulationConfig["physics"]["gauge"].asString());
  if (gauge == "solenoid") {
    riccatiSolver =
        new conga::RiccatiSolverConfined<T, conga::gauge::Solenoid>(&context);
  } else if (gauge == "symmetric") {
    riccatiSolver =
        new conga::RiccatiSolverConfined<T, conga::gauge::Symmetric>(&context);
  } else {
    std::cerr << "-- ERROR! Gauge, " << gauge << ", not understod" << std::endl;
    return EXIT_FAILURE;
  }

  // Add specular BC.
  riccatiSolver->add(new conga::BoundaryConditionSpecular<T>);

  //***************************************************
  // INITIALIZE SPECTROSCOPY
  //***************************************************
  context.initialize();

  //***********************************************
  // RUN SPECTROSCOPY UNTIL END CRITERIA
  //***********************************************
  // Path to directory containing data to save to.
  const std::filesystem::path saveDataPath(
      postprocessConfig["misc"]["save_path"].asString());
  const bool saveData = !saveDataPath.empty();

  // Make iteration writer.
  std::shared_ptr<conga::io::CSVIterationWriter<T>> iterationWriter;
  if (saveData) {
    const std::filesystem::path convergencePath =
        saveDataPath / "postprocess_convergence.csv";
    iterationWriter =
        std::make_shared<conga::io::CSVPostprocessIterationWriter<T>>(
            convergencePath);
  }

  // Run everything.
  const int numIterationsBurnIn =
      postprocessConfig["numerics"]["num_iterations_burnin"].asInt();
  const int numIterationsMin =
      postprocessConfig["numerics"]["num_iterations_min"].asInt();
  const int numIterationsMax =
      postprocessConfig["numerics"]["num_iterations_max"].asInt();
  conga::runPostprocess(numIterationsBurnIn, numIterationsMin, numIterationsMax,
                        verbose, iterationWriter, context);

  //***********************************************
  // SAVE DATA
  //***********************************************

  if (saveData) {
    // Create directories.
    std::filesystem::create_directories(saveDataPath);

    // Save post-process config.
    const std::filesystem::path postprocessConfigPath =
        saveDataPath / "postprocess_config.json";
    conga::io::write(postprocessConfigPath, postprocessConfig);
    conga::io::printWrite(verbose, true, "postprocess configuration",
                          postprocessConfigPath);

    // Get data format.
    const conga::io::DataFormat dataFormat = conga::io::getDataFormat(
        postprocessConfig["misc"]["data_format"].asString());

    // Save the LDOS.
    const conga::GridGPU<T> &ldos = (*(context.getComputeLDOS()->getResult()));
    switch (dataFormat) {
#ifdef CONGA_USE_HDF5
    case conga::io::DataFormat::H5: {
      const std::filesystem::path spectroscopyPath =
          saveDataPath / "postprocess.h5";
      auto file = conga::io::fileOverWrite(spectroscopyPath);
      conga::io::write(ldos, "ldos", file);
      conga::io::printWrite(verbose, true, "LDOS", spectroscopyPath);
      break;
    }
#endif // CONGA_USE_HDF5
    case conga::io::DataFormat::CSV: {
      // The default is CSV.
    }
    default: {
      const std::filesystem::path ldosPath = saveDataPath / "ldos.csv";
      const bool success = conga::io::write(ldos, {"ldos"}, ldosPath);
      conga::io::printWrite(verbose, success, "LDOS", ldosPath);
    }
    }
  }

  // Success!
  return EXIT_SUCCESS;
}
