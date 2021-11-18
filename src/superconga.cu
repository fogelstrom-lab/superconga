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

#include "conga/Accelerator.h"
#include "conga/Context.h"
#include "conga/Parameters.h"
#include "conga/Type.h"
#include "conga/boundary/BoundaryConditionSpecular.h"
#include "conga/compute/ComputeCurrent.h"
#include "conga/compute/ComputeFreeEnergy.h"
#include "conga/compute/ComputeOrderParameter.h"
#include "conga/compute/ComputeVectorPotential.h"
#include "conga/configure.h"
#include "conga/geometry/DiscGeometry.h"
#include "conga/geometry/GeometryGroup.h"
#include "conga/geometry/PolygonGeometry.h"
#include "conga/io/read_utils.h"
#include "conga/order_parameter/OrderParameter.h"
#include "conga/order_parameter/OrderParameterBasis.h"
#include "conga/order_parameter/OrderParameterComponent.h"
#include "conga/order_parameter/OrderParameterInitial.h"
#include "conga/riccati/RiccatiSolverConfined.h"
#include "conga/run_utils.h"
#include "conga/visualization/Visualizer.h"

#include <json/json.h>

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

//*******************************************************
// FORWARD DECLARATIONS
//*******************************************************
void display();

//*******************************************************
// PRECISION
//*******************************************************
#ifdef CONGA_SINGLE_PRECISION
typedef float T;
#else
typedef double T;
#endif

//*******************************************************
// GLOBAL OBJECTS: glut does not accept arguments and relies on global/static
// variables.
//*******************************************************
namespace {
// Create context object.
conga::Context<T> *context = new conga::Context<T>();
// Create visualization object.
conga::Visualizer<T> *visualizer;
} // namespace

//*******************************************************
// MAIN
//*******************************************************
int main(int argc, char *argv[]) {
  //*******************************************************
  // PARSE SIMULATION CONFIG FILE
  //*******************************************************
  Json::Value simParams;
  {
    if (argc < 2) {
      std::cerr << "-- ERROR! Please provide a config file." << std::endl;
      conga::run_utils::cleanup(context, visualizer);
      return EXIT_FAILURE;
    }
    const std::string filename = argv[1];

    // Check that provided config file actually exists and can be opened.
    std::ifstream ifile(filename);
    if (!ifile.good()) {
      std::cerr << "-- ERROR! Could not find/open config file '" << filename
                << "'." << std::endl;
      conga::run_utils::cleanup(context, visualizer);
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
    std::cout << "-- Configuration parameters:" << std::endl;
    std::cout << simParams << std::endl;
#endif
  }

  // TODO(niclas): Add sanity check of all parameters here!?

  //***************************************************
  // APPLY PARAMETERS
  //***************************************************
  // Storage for system simulation parameters (e.g. temperature, lattice size).
  conga::Parameters<T> *parameters = new conga::Parameters<T>(context);

  // Set temperature.
  const T temperature =
      static_cast<T>(simParams["physics"]["temperature"].asDouble());
  parameters->setTemperature(temperature);

  // Set the number of lattice points per coherence length.
  const T pointsPerCoherenceLength = static_cast<T>(
      simParams["accuracy"]["points_per_coherence_length"].asDouble());
  parameters->setPointsPerCoherenceLength(pointsPerCoherenceLength);

  // Set energy cutoff.
  const T energyCutoff =
      static_cast<T>(simParams["accuracy"]["energy_cutoff"].asDouble());
  parameters->setEnergyCutoff(energyCutoff);

  // Number of discrete steps in momentum angle.
  const int numFermiMomenta =
      simParams["accuracy"]["num_fermi_momenta"].asInt();
  parameters->setAngularResolution(numFermiMomenta);

  // Set convergence criterion.
  const T convergenceCriterion =
      static_cast<T>(simParams["accuracy"]["convergence_criterion"].asDouble());
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
  conga::GeometryGroup<T> *geometry = new conga::GeometryGroup<T>(context);

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
  // SET UP ORDER PARAMETER
  //***************************************************
  // Create order parameter (which might contain multiple components, i.e.
  // d+is).
  conga::OrderParameter<T> *orderParameter =
      new conga::OrderParameter<T>(context);

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

      // Get initial phase parameters.
      const T phaseWinding =
          static_cast<T>(opCompParams["initial_phase_winding"].asDouble());
      const T noiseStdDev =
          static_cast<T>(opCompParams["initial_noise_stddev"].asDouble());
      const T phaseShift = static_cast<T>(
          2.0 * M_PI * opCompParams["initial_phase_shift"].asDouble());

      // Check if we should initialize to bulk.
      const T eps = std::numeric_limits<T>::epsilon();
      const bool useBulk = (std::abs(phaseWinding) < eps) &&
                           (std::abs(noiseStdDev) < eps) &&
                           (std::abs(phaseShift) < eps);

      // Initialize order parameter with a guess.
      if (useBulk) {
        conga::Initial<T>::bulk(orderParameterComponent);
      } else {
        conga::Initial<T>::perturbedPhaseWinding(
            orderParameterComponent, phaseWinding, noiseStdDev, phaseShift);
      }

      // Add the initialized component to the total order parameter.
      orderParameter->add(orderParameterComponent);
    }
  }

  if (!orderParameterSuccess) {
    std::cerr << "-- ERROR! No order-parameter component set." << std::endl;
    conga::run_utils::cleanup(context, visualizer);
    return EXIT_FAILURE;
  }

  // Path to directory containing data to load from.
  const std::filesystem::path loadDataPath(
      simParams["misc"]["load_path"].asString());

  // Read the order parameter from file, if the path is not empty.
  // TODO(niclas): This is a bit ugly perhaps. If the file contains data for a
  // component, it is overwritten. If it does not, it is left untouched.
  if (!loadDataPath.empty()) {
    const std::string orderParameterFile = loadDataPath / "order_parameter.csv";

    // Read the values from file.
    const bool success =
        conga::io::readOrderParameter(orderParameterFile, *orderParameter);

    // Abort if not successful.
    if (!success) {
      conga::run_utils::cleanup(context, visualizer);
      return EXIT_FAILURE;
    }
  }

  //***************************************************
  // SET INDUCED VECTOR-POTENTIAL
  //***************************************************
  // Read the vector potential from file, if the path is not empty.
  if (!loadDataPath.empty()) {
    const std::string vectorPotentialFile =
        loadDataPath / "vector_potential.csv";

    // Create the placeholder grid.
    const int gridResolution = parameters->getGridResolutionBase();
    auto vectorPotential = std::make_unique<conga::GridGPU<T>>(
        gridResolution, gridResolution, 2, conga::Type::real);

    // Read the values from file.
    const bool success = conga::io::readVectorPotential(
        vectorPotentialFile, *(vectorPotential.get()));

    // Abort if not successful.
    if (!success) {
      conga::run_utils::cleanup(context, visualizer);
      return 1;
    }

    // Set the vector potential in context.
    context->set(std::move(vectorPotential));
  }

  //***************************************************
  // ACCELERATOR
  //***************************************************
  const T stepSize =
      static_cast<T>(simParams["accelerator"]["step_size"].asDouble());
  const T inertia =
      static_cast<T>(simParams["accelerator"]["inertia"].asDouble());
  const T adaptive = simParams["accelerator"]["adaptive"].asBool();
  const std::size_t rank =
      static_cast<std::size_t>(simParams["accelerator"]["rank"].asInt());
  const bool replaceOldest =
      simParams["accelerator"]["replace_oldest"].asBool();
  const bool andersonAlways =
      simParams["accelerator"]["anderson_always"].asBool();
  const T minMixing =
      static_cast<T>(simParams["accelerator"]["min_mixing"].asDouble());
  const T maxMixing =
      static_cast<T>(simParams["accelerator"]["max_mixing"].asDouble());
  auto accelerator = std::make_unique<conga::Accelerator<T>>(
      stepSize, inertia, adaptive, rank, replaceOldest, andersonAlways,
      minMixing, maxMixing);
  context->set(std::move(accelerator));

  //***************************************************
  // CHOOSE QUANTITIES TO CALCULATE
  //***************************************************
  // Order parameter compute object.
  new conga::ComputeOrderParameter<T>(context);
  // Add current compute object.
  new conga::ComputeCurrent<T>(context);
  // Add free energy compute object.
  new conga::ComputeFreeEnergy<T>(context);

  // Add vector-potential compute object.
  if (penetrationDepth > static_cast<T>(0)) {
    const T maxError = static_cast<T>(
        simParams["accuracy"]["vector_potential_error"].asDouble());
    auto computeVectorPotential =
        std::make_unique<conga::ComputeVectorPotential<T>>(maxError);
    context->add(std::move(computeVectorPotential));
  }

  //***************************************************
  // CHOOSE RICCATI SOLVER AND BOUNDARY CONDITIONS
  //***************************************************
  // RiccatiSolverConfined: provides analytic solutions to the Riccati equations
  // for a piecewise constant order parameter.
  conga::RiccatiSolver<T> *riccatiSolver;

  // Gauge choice: VectorPotential::Solenoid calculates the vector potential
  // with a solenoid gauge.
  const bool solenoidGuage = simParams["physics"]["gauge_solenoid"].asBool();
  if (solenoidGuage) {
    riccatiSolver =
        new conga::RiccatiSolverConfined<T, conga::VectorPotential::Solenoid>(
            context);
  } else {
    riccatiSolver =
        new conga::RiccatiSolverConfined<T, conga::VectorPotential::Radial>(
            context);
  }

  // Add specular BC.
  riccatiSolver->add(new conga::BoundaryConditionSpecular<T>);

  //***************************************************
  // INITIALIZE SIMULATION
  //***************************************************
  context->initialize();

  //***********************************************
  // SET UP OPENGL VISUALIZATION
  //***********************************************
  const bool visualize = simParams["misc"]["visualize"].asBool();
  if (visualize) {
    visualizer = conga::run_utils::createVisualization(context);
    conga::initGL(argc, argv, visualizer->getNumPixelsWindowX(),
                  visualizer->getNumPixelsWindowY());
    glutKeyboardFunc(conga::run_utils::myKeyboardFunc);
    glutDisplayFunc(display);
  }

  //***********************************************
  // CREATE DIRECTORIES AND SAVE PARAMETERS
  //***********************************************
  const std::filesystem::path saveDataPath(
      simParams["misc"]["save_path"].asString());

  // Set the save frequency. If the argument is negative we only save the grid
  // data at the very end.
  const int saveFrequencyArg = simParams["misc"]["save_frequency"].asInt();
  const int saveFrequency = (saveFrequencyArg < 0)
                                ? std::numeric_limits<int>::max()
                                : saveFrequencyArg;

  const bool saveData = !saveDataPath.empty() && (saveFrequency > 0);
  const bool verbose = simParams["misc"]["verbose"].asBool();
  if (saveData) {
    // Create directories.
    std::filesystem::create_directories(saveDataPath);

    // Save config parameters to JSON file.
    const std::filesystem::path configFileName = saveDataPath / "config.json";
    conga::io::writeJSON(simParams, configFileName, verbose);

    // TODO(niclas): Write internal parameters here as well? They do not change
    // during the simulation.

    // Save domain.
    const std::filesystem::path domainFileName = saveDataPath / "domain.csv";
    const conga::GridGPU<char> &domain = *(geometry->getDomainGrid());
    const conga::io::CSVGridDataSaveStrategy<char> writer;
    const bool success = writer.save(domain, {"domain"}, domainFileName);
    if (verbose) {
      if (!success) {
        std::cout << "-- ERROR! Could not write: " << domainFileName
                  << std::endl;
      } else {
        std::cout << "-- Wrote: " << domainFileName << std::endl;
      }
    }
  }

  //***********************************************
  // PROCESSING: RUN SIMULATION UNTIL END CRITERIA
  //***********************************************
  const conga::Type::fileFormat dataFormat = conga::Type::csv;
  conga::run_utils::runSimulation(
      context, verbose, simParams["accuracy"]["num_iterations_min"].asInt(),
      simParams["accuracy"]["num_iterations_max"].asInt(), visualize,
      saveDataPath, saveData, saveFrequency, dataFormat);

  //***********************************************
  // SAVE RESULTS
  //***********************************************
  // Save at the very end.
  if (saveData) {
    conga::run_utils::saveDataWrapper(context, saveDataPath, verbose,
                                      dataFormat);
  }

  // Cleanup dynamically allocated memory.
  conga::run_utils::cleanup(context, visualizer);
  return EXIT_SUCCESS;
}

// Plot data in OpenGL GUI.
void display() {
  conga::run_utils::display(context, visualizer);

  // Tell Glut to update window.
  glutPostRedisplay();
}
