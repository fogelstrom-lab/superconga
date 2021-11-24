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
#include "conga/geometry/DiscGeometry.h"
#include "conga/geometry/GeometryGroup.h"
#include "conga/geometry/PolygonGeometry.h"
#include "conga/grid.h"
#include "conga/integration/IntegrationIteratorKeldysh.h"
#include "conga/io/csv_writers.h"
#include "conga/io/read_utils.h"
#include "conga/order_parameter/OrderParameter.h"
#include "conga/order_parameter/OrderParameterBasis.h"
#include "conga/order_parameter/OrderParameterComponent.h"
#include "conga/order_parameter/OrderParameterInitial.h"
#include "conga/riccati/RiccatiSolverConfined.h"
#include "conga/run_utils.h"

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
  // PARSE POST-PROCESS CONFIG FILE
  //*******************************************************
  Json::Value postProcessConfig;
  {
    if (argc < 2) {
      std::cerr << "-- ERROR! Please provide a config file." << std::endl;
      return EXIT_FAILURE;
    }

    // Path to post-process config.
    const std::filesystem::path postProcessFile(argv[1]);

    // Check that provided config file actually exists and can be opened.
    std::ifstream ifile(postProcessFile);
    if (!ifile.good()) {
      std::cerr << "-- ERROR! Could not find/open config file '"
                << postProcessFile << "'." << std::endl;
      return EXIT_FAILURE;
    }

    // Parse JSON.
    Json::CharReaderBuilder builder;
    JSONCPP_STRING errs;
    if (!parseFromStream(builder, ifile, &postProcessConfig, &errs)) {
      std::cout << errs << std::endl;
      return EXIT_FAILURE;
    }
#ifndef NDEBUG
    std::cout << "-- Post-process configuration parameters:" << std::endl;
    std::cout << postProcessConfig << std::endl;
#endif
  }

  // Path to directory containing data to load from.
  const std::filesystem::path loadDataPath(
      postProcessConfig["load_path"].asString());

  //*******************************************************
  // PARSE SIMULATION CONFIG FILE
  //*******************************************************
  Json::Value simParams;
  {
    // Path to simulation config.
    const std::filesystem::path simulationFile = loadDataPath / "config.json";

    // Check that provided config file actually exists and can be opened.
    std::ifstream ifile(simulationFile);
    if (!ifile.good()) {
      std::cerr << "-- ERROR! Could not find/open config file '"
                << simulationFile << "'." << std::endl;
      return EXIT_FAILURE;
    }

    // Parse JSON.
    Json::CharReaderBuilder builder;
    JSONCPP_STRING errs;
    if (!parseFromStream(builder, ifile, &simParams, &errs)) {
      std::cout << errs << std::endl;
      return EXIT_FAILURE;
    }
#ifndef NDEBUG
    std::cout << "-- Simulation configuration parameters:" << std::endl;
    std::cout << simParams << std::endl;
#endif
  }

  // TODO(niclas): Add sanity check of all parameters here!?

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
      static_cast<T>(simParams["physics"]["temperature"].asDouble());
  parameters->setTemperature(temperature);

  // Set the number of lattice points per coherence length.
  const T pointsPerCoherenceLength = static_cast<T>(
      simParams["accuracy"]["points_per_coherence_length"].asDouble());
  parameters->setPointsPerCoherenceLength(pointsPerCoherenceLength);

  // Set parameters relevant for real-axis energies.
  const T minEnergy =
      static_cast<T>(postProcessConfig["energy_min"].asDouble());
  const T maxEnergy =
      static_cast<T>(postProcessConfig["energy_max"].asDouble());
  const int numEnergies = postProcessConfig["num_energies"].asInt();
  const T energySmearing =
      static_cast<T>(postProcessConfig["energy_smearing"].asDouble());
  parameters->setEnergyRange(minEnergy, maxEnergy, numEnergies, energySmearing);

  // Number of discrete steps in momentum angle.
  const int numFermiMomenta = postProcessConfig["num_fermi_momenta"].asInt();
  parameters->setAngularResolution(numFermiMomenta);

  // Set convergence criterion.
  const T convergenceCriterion =
      static_cast<T>(postProcessConfig["convergence_criterion"].asDouble());
  parameters->setConvergenceCriterion(convergenceCriterion);

  // Set penetration depth.
  const T penetrationDepth =
      static_cast<T>(simParams["physics"]["penetration_depth"].asDouble());
  parameters->setPenetrationDepth(penetrationDepth);

  // Set external magnetic field strength.
  const T externalFluxQuanta =
      static_cast<T>(simParams["physics"]["external_flux_quanta"].asDouble());
  parameters->setNumFluxQuanta(externalFluxQuanta);

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
  conga::run_utils::computeGeometryScaling(simParams, scalingFactor, xOffset,
                                           yOffset);

  // Set grain sidelength in coherence length.
  parameters->setGrainWidth(scalingFactor);

  // Fetch relative size of geometry to available simulation space, to account
  // for padding for boundary interpolation.
  const T grainFraction = parameters->getGrainFraction();

  // Parse user-defined geometry, loop through geometry component list and
  // create compund geometry. Scale and translate to internal coordinate system.
  conga::run_utils::addGeometryComponents(geometry, simParams, scalingFactor,
                                          xOffset, yOffset, grainFraction);

  //***************************************************
  // SET UP INTEGRATOR
  //***************************************************
  const int numEnergiesPerBlock =
      postProcessConfig["num_energies_per_block"].asInt();
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
      2.0 * M_PI * simParams["physics"]["crystal_axes_rotation"].asDouble());
  orderParameter->setCrystalAxesRotation(crystalAxesRotation);

  // For convenience.
  const Json::Value opParams = simParams["physics"]["order_parameter"];

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

  // Read the order parameter from file.
  {
    const std::string orderParameterFile = loadDataPath / "order_parameter.csv";
    // Read the values from file.
    const bool success =
        conga::io::readOrderParameter(orderParameterFile, *orderParameter);

    // Abort if not successful.
    if (!success) {
      return EXIT_FAILURE;
    }
  }

  //***************************************************
  // SET INDUCED VECTOR-POTENTIAL
  //***************************************************
  // Read the vector potential from file.
  const std::string vectorPotentialFile = loadDataPath / "vector_potential.csv";
  {
    // Create the placeholder grid.
    const int gridResolution = parameters->getGridResolutionBase();
    auto vectorPotential = std::make_unique<conga::GridGPU<T>>(
        gridResolution, gridResolution, 2, conga::Type::real);

    // Read the values from file.
    const bool success = conga::io::readVectorPotential(
        vectorPotentialFile, *(vectorPotential.get()));

    // Abort if not successful.
    if (!success) {
      return EXIT_FAILURE;
    }

    // Set the vector potential in context.
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
  // RiccatiSolverConfined: provides analytic solutions to the Riccati equations
  // for a piecewise constant order parameter.
  conga::RiccatiSolver<T> *riccatiSolver;

  // Gauge choice: VectorPotential::Solenoid calculates the vector potential
  // with a solenoid gauge.
  const bool solenoidGuage = simParams["gauge_solenoid"].asBool();
  if (solenoidGuage) {
    riccatiSolver =
        new conga::RiccatiSolverConfined<T, conga::VectorPotential::Solenoid>(
            &context);
  } else {
    riccatiSolver =
        new conga::RiccatiSolverConfined<T, conga::VectorPotential::Radial>(
            &context);
  }

  // Add specular BC.
  riccatiSolver->add(new conga::BoundaryConditionSpecular<T>);

  //***************************************************
  // INITIALIZE SIMULATION
  //***************************************************
  context.initialize();

  //***********************************************
  // PROCESSING: RUN SIMULATION UNTIL END CRITERIA
  //***********************************************
  conga::run_utils::runSimulation(
      &context, postProcessConfig["verbose"].asBool(),
      postProcessConfig["num_iterations_min"].asInt(),
      postProcessConfig["num_iterations_max"].asInt());

  //***********************************************
  // SAVE DATA
  //***********************************************
  // TODO(niclas): Perhaps move this to io/write_utils.h, e.g.
  // saveDensityOfStates? And do the same thing for saving other grids in
  // run_utils.h.

  // Path to directory containing data to save to.
  const std::filesystem::path saveDataPath(
      postProcessConfig["save_path"].asString());

  if (!saveDataPath.empty()) {
    // Create directories.
    std::filesystem::create_directories(saveDataPath);

    // Save post-process config.
    const std::filesystem::path configFileName =
        saveDataPath / "postprocess.json";
    conga::io::writeJSON(postProcessConfig, configFileName,
                         postProcessConfig["verbose"].asBool());

    // Get file extension.
    const conga::Type::fileFormat dataFormat = conga::Type::csv;
    const std::string extension = conga::io::getFileExtension(dataFormat);

    // Save the DOS.
    {
      // Create CSV writer.
      const bool writeIndices = false;
      const conga::io::CSVGridDataSaveStrategy<T> writer(writeIndices);

      // Create filename.
      const std::filesystem::path dosFileName =
          saveDataPath / ("dos" + extension);

      // Get the DOS.
      const conga::GridCPU<T> &dos = *(context.getComputeLDOS()->getDOS());

      // Write data.
      const bool success = writer.save(dos, {"energy", "dos"}, dosFileName);

      if (!success) {
        std::cout << "-- ERROR! Could not write: " << dosFileName << std::endl;
      } else {
        std::cout << "-- Wrote: " << dosFileName << std::endl;
      }
    }

    // Save the LDOS.
    {
      // Create CSV writer.
      const conga::io::CSVGridDataSaveStrategy<T> writer;

      // Create filename.
      const std::filesystem::path ldosFileName =
          saveDataPath / ("ldos" + extension);

      // Get the LDOS.
      const conga::GridGPU<T> &ldos =
          (*(context.getComputeLDOS()->getResult()));

      // Write data.
      const bool success = writer.save(ldos, {"ldos"}, ldosFileName);

      if (!success) {
        std::cout << "-- ERROR! Could not write: " << ldosFileName << std::endl;
      } else {
        std::cout << "-- Wrote: " << ldosFileName << std::endl;
      }
    }
  }

  // Success!
  return EXIT_SUCCESS;
}
