//===----------- main.cu - Main file for setting up simulations -----------===//
//
// Part of the SuprConGPU Project, under the GNU LGPL v3 license or higher.
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
#include "ComputeCurrent.h"
#include "ComputeFreeEnergy.h"
#include "ComputeOrderParameter.h"

#include "BoundaryConditionSpecular.h"
#include "ComputeLDOS.h"
#include "Context.h"
#include "DiscGeometry.h"
#include "GeometryGroup.h"
#include "Grid.h"
#include "Optimizer.h"
#include "OrderParameter.h"
#include "OrderParameterBasis.h"
#include "OrderParameterComponent.h"
#include "OrderParameterInitial.h"
#include "Parameters.h"
#include "PolygonGeometry.h"
#include "RiccatiSolverConfined.h"
#include "Type.h"
#include "configure.h"
#include "io/csv_writers.h"
#include "io/read_utils.h"
#include "run_utils.h"
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <json/json.h>
#include <limits>
#include <memory>
#include <string>

//*******************************************************
// PRECISION
//*******************************************************
#ifdef SCG_SINGLE_PRECISION
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
  scg::Context<T> context;

  //***************************************************
  // APPLY PARAMETERS
  //***************************************************
  // Storage for system simulation parameters (e.g. temperature, lattice size).
  scg::Parameters<T> *parameters = new scg::Parameters<T>(&context);

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
  const int numEnergiesPerBlock =
      postProcessConfig["num_energies_per_block"].asInt();
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
  scg::GeometryGroup<T> *geometry = new scg::GeometryGroup<T>(&context);

  // Parse user-defined geometry. Compute factors to scale and translate
  // geometry to internal coordinate system.
  T scalingFactor;
  T xOffset;
  T yOffset;
  scg::run_utils::computeGeometryScaling(simParams, scalingFactor, xOffset,
                                         yOffset);

  // Set grain sidelength in coherence length.
  parameters->setGrainWidth(scalingFactor);

  // Fetch relative size of geometry to available simulation space, to account
  // for padding for boundary interpolation.
  const T grainFraction = parameters->getGrainFraction();

  // Parse user-defined geometry, loop through geometry component list and
  // create compund geometry. Scale and translate to internal coordinate system.
  scg::run_utils::addGeometryComponents(geometry, simParams, scalingFactor,
                                        xOffset, yOffset, grainFraction);

  //***************************************************
  // SET UP ORDER PARAMETER
  //***************************************************
  // Create order parameter (which might contain multiple components, i.e.
  // d+is).
  scg::OrderParameter<T> *orderParameter = new scg::OrderParameter<T>(&context);

  // Get the global crystal-axes rotation.
  const T crystalAxesRotation = static_cast<T>(
      2.0 * M_PI * simParams["physics"]["crystal_axes_rotation"].asDouble());

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
      scg::BasisFunctionSymmetry symmetry;
      if (symmetryStr == "dx2-y2") {
        symmetry = scg::BasisFunctionSymmetry::DWAVE_X2_Y2;
      } else if (symmetryStr == "dxy") {
        symmetry = scg::BasisFunctionSymmetry::DWAVE_XY;
      } else if (symmetryStr == "g") {
        symmetry = scg::BasisFunctionSymmetry::GWAVE;
      } else {
        symmetry = scg::BasisFunctionSymmetry::SWAVE;
      }

      // For convenience.
      const Json::Value opCompParams = opParams[symmetryStr];

      // Create component.
      const T criticalTemperature =
          static_cast<T>(opCompParams["critical_temperature"].asDouble());
      scg::OrderParameterComponent<T> *orderParameterComponent =
          new scg::OrderParameterComponent<T>(symmetry, parameters,
                                              criticalTemperature);

      // Set rotation of crystal axes with respect to the grain geometry.
      // TODO(niclas): Move to Parameters.h?
      orderParameterComponent->setCrystalAxesRotation(crystalAxesRotation);

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
        scg::io::readOrderParameter(orderParameterFile, *orderParameter);

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
    auto vectorPotential = std::make_unique<scg::GridGPU<T>>(
        gridResolution, gridResolution, 2, scg::Type::real);

    // Read the values from file.
    const bool success = scg::io::readVectorPotential(vectorPotentialFile,
                                                      *(vectorPotential.get()));

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
  new scg::ComputeLDOS<T>(&context);
  // TODO(niclas): Add spectral current?

  //***************************************************
  // CHOOSE RICCATI SOLVER AND BOUNDARY CONDITIONS
  //***************************************************
  // RiccatiSolverConfined: provides analytic solutions to the Riccati equations
  // for a piecewise constant order parameter.
  scg::RiccatiSolver<T> *riccatiSolver;

  // Gauge choice: VectorPotential::Solenoid calculates the vector potential
  // with a solenoid gauge.
  const bool solenoidGuage = simParams["gauge_solenoid"].asBool();
  if (solenoidGuage) {
    riccatiSolver =
        new scg::RiccatiSolverConfined<T, scg::VectorPotential::Solenoid>(
            &context);
  } else {
    riccatiSolver =
        new scg::RiccatiSolverConfined<T, scg::VectorPotential::Radial>(
            &context);
  }

  // Add specular BC.
  riccatiSolver->add(new scg::BoundaryConditionSpecular<T>);

  //***************************************************
  // INITIALIZE SIMULATION
  //***************************************************
  context.initialize(numEnergiesPerBlock);

  //***********************************************
  // PROCESSING: RUN SIMULATION UNTIL END CRITERIA
  //***********************************************
  scg::run_utils::runSimulation(
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
    scg::io::writeJSON(postProcessConfig, configFileName,
                       postProcessConfig["verbose"].asBool());

    // Get file extension.
    const scg::Type::fileFormat dataFormat = scg::Type::csv;
    const std::string extension = scg::io::getFileExtension(dataFormat);

    // Save the DOS.
    {
      // Create CSV writer.
      const bool writeIndices = false;
      const scg::io::CSVGridDataSaveStrategy<T> writer(writeIndices);

      // Create filename.
      const std::filesystem::path dosFileName =
          saveDataPath / ("dos" + extension);

      // Get the DOS.
      const scg::GridCPU<T> &dos = *(context.getComputeLDOS()->getDOS());

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
      const scg::io::CSVGridDataSaveStrategy<T> writer;

      // Create filename.
      const std::filesystem::path ldosFileName =
          saveDataPath / ("ldos" + extension);

      // Get the LDOS.
      const scg::GridGPU<T> &ldos = (*(context.getComputeLDOS()->getResult()));

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
