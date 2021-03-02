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

#include "BoundaryConditionSpecular.h" // Boundary condition for perfect specular reflection
#include "ComputeCurrent.h"        // Compute the total charge current density
#include "ComputeFreeEnergy.h"     // Compute the (Eilenberger) free energy
#include "ComputeOrderParameter.h" // Compute order parameter
#include "Context.h" // The context that connects all classes (helpful since enacpsulation is used)
#include "DiscGeometry.h" // Used to constructed simulated grain from discs
#include "FermiSurfaceCircle.h" // Standard circle Fermi surface
#include "GeometryGroup.h" // The geometry group stores a list of polygons/discs/... that together define the sample geometry (container class for GeometryComponent)
#include "Optimizer.h"
#include "OrderParameter.h" // Container class for order parameter components
#include "OrderParameterBasis.h"
#include "OrderParameterComponent.h"
#include "OrderParameterInitial.h" // Used for generating initial guess for order parameter
#include "Parameters.h" // Parameters specifying which the superconducting system to simulate
#include "PolygonGeometry.h" // Used to constructed simulated grain from polygons
#include "RiccatiSolverConfined.h" // Riccati solver for confined geometries
#include "Type.h"                  // Custom types (real, complex, ...)
#include "Visualization.h" // Live visualization of desired observables through OpenGL
#include "configure.h" // Variables and defines exported by cmake, e.g. version number and project name
#include "io/read_utils.h"
#include "run_utils.h" // Commonly used functions in run files

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <json/json.h>
#include <limits>
#include <memory>
#include <string>

//*******************************************************
// FORWARD DECLARATIONS
//*******************************************************
void display();

//*******************************************************
// TYPEDEFS
//*******************************************************
// Run with double or single floating point precision, double is highly
// recommended.
typedef double T;

//*******************************************************
// GLOBAL OBJECTS: glut does not accept arguments and relies on global/static
// variables.
//*******************************************************
namespace {
// Create context object.
scg::Context<T> *context = new scg::Context<T>();
// Create visualization object.
scg::Visualization<T> *gui;
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
      scg::run_utils::cleanup(context, gui);
      return 1;
    }
    const std::string filename = argv[1];

    // Check that provided config file actually exists and can be opened.
    std::ifstream ifile(filename);
    if (!ifile.good()) {
      std::cerr << "-- ERROR! Could not find/open config file '" << filename
                << "'." << std::endl;
      scg::run_utils::cleanup(context, gui);
      return 1;
    }

    // Parse JSON.
    Json::CharReaderBuilder builder;
    JSONCPP_STRING errs;
    if (!parseFromStream(builder, ifile, &simParams, &errs)) {
      std::cout << errs << std::endl;
      return 1;
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
  scg::Parameters<T> *parameters = new scg::Parameters<T>(context);

  // Set temperature.
  const T temperature = static_cast<T>(simParams["temperature"].asDouble());
  parameters->setTemperature(temperature);

  // Set grain sidelength in coherence length.
  // Note, the internal coherence length is defined as xi_0 = hbar*<vF>/(kB*Tc).
  // But externally we use xi_0 = hbar*<vF>/(2*pi*kB*Tc), so we must scale it
  // appropriately.
  const T grainSideLength =
      static_cast<T>(simParams["grain_side_length"].asDouble() / (2.0 * M_PI));
  parameters->setGrainWidth(grainSideLength);

  // Set the number of lattice points per coherence length.
  // Note, the internal coherence length is defined as xi_0 = hbar*<vF>/(kB*Tc).
  // But externally we use xi_0 = hbar*<vF>/(2*pi*kB*Tc), so we must scale it
  // appropriately.
  const T pointsPerCoherenceLength = static_cast<T>(
      simParams["points_per_coherence_length"].asDouble() * 2.0 * M_PI);
  parameters->setPointsPerCoherenceLength(pointsPerCoherenceLength);

  // Set number of Ozaki energies.
  const int numOzakiEnergies = simParams["num_ozaki_energies"].asInt();
  parameters->setNumMatsubaraEnergies(numOzakiEnergies);

  // Number of discrete steps in momentum angle.
  const int numFermiMomenta = simParams["num_fermi_momenta"].asInt();
  parameters->setAngularResolution(numFermiMomenta);

  // Set convergence criterion.
  const T convergenceCriterion =
      static_cast<T>(simParams["convergence_criterion"].asDouble());
  parameters->setConvergenceCriterion(convergenceCriterion);

  // Set penetration depth.
  parameters->setPenetrationDepth(
      static_cast<T>(simParams["penetration_depth"].asDouble()));

  //***************************************************
  // CREATE GEOMETRY
  //***************************************************
  // Create geometry group: add/remove geometry objects (e.g. discs, polygons)
  // from the geometry group.
  scg::GeometryGroup<T> *geometry = new scg::GeometryGroup<T>(context);

  // Check if to set up a disc or polygon.
  const int numVertices = simParams["geometry"]["num_outer_edges"].asInt();
  const bool isDisc = numVertices < 0;

  // Get the grain fraction, which is needed for scaling the geometry
  // components.
  const T grainFracton = parameters->getGrainFraction();

  // Make the grain a disc or a square.
  if (isDisc) {
    geometry->add(new scg::DiscGeometry<T>(grainFracton), scg::Type::add);
  } else {
    geometry->add(new scg::PolygonGeometry<T>(grainFracton, numVertices),
                  scg::Type::add);
  }

  // Remove a portion to create a solenoid.
  const T solenoidScale =
      static_cast<T>(simParams["geometry"]["hole_scale"].asDouble());
  const bool isSolenoid = solenoidScale > static_cast<T>(0);
  if (isSolenoid) {
    const int numSolenoidVertices =
        simParams["geometry"]["num_inner_edges"].asInt();
    const bool isDisc = numSolenoidVertices < 0;

    // Circular or polygon hole.
    if (isDisc) {
      geometry->add(new scg::DiscGeometry<T>(grainFracton, solenoidScale),
                    scg::Type::remove);
    } else {
      geometry->add(new scg::PolygonGeometry<T>(
                        grainFracton, numSolenoidVertices, solenoidScale),
                    scg::Type::remove);
    }
  }

  //***************************************************
  // SET UP ORDER PARAMETER
  //***************************************************
  // Create order parameter (which might contain multiple components, i.e.
  // d+is).
  scg::OrderParameter<T> *orderParameter = new scg::OrderParameter<T>(context);

  const T crystalAxesRotation = static_cast<T>(
      2.0 * M_PI * simParams["crystal_axes_rotation"].asDouble());

  const Json::Value opCompParamsVec = simParams["order_parameter"];

  for (int i = 0; i < opCompParamsVec.size(); ++i) {
    const Json::Value opCompParams = opCompParamsVec[i];

    // Add a d-wave or an s-wave order-parameter component.
    const std::string symmetryStr = opCompParams["symmetry"].asString();

    // TODO(niclas): Move this to OrderParameterBasis.h?
    scg::BasisFunctionSymmetry symmetry;
    if (symmetryStr == "s") {
      symmetry = scg::BasisFunctionSymmetry::SWAVE;
    } else if (symmetryStr == "dx2-y2") {
      symmetry = scg::BasisFunctionSymmetry::DWAVE_X2_Y2;
    } else if (symmetryStr == "dxy") {
      symmetry = scg::BasisFunctionSymmetry::DWAVE_XY;
    } else if (symmetryStr == "g") {
      symmetry = scg::BasisFunctionSymmetry::GWAVE;
    } else {
      std::cerr << "-- Order parameter symmetry '" << symmetryStr
                << "' not understood." << std::endl;
      scg::run_utils::cleanup(context, gui);
      return 1;
    }

    const T criticalTemperature =
        static_cast<T>(opCompParams["critical_temperature"].asDouble());
    scg::OrderParameterComponent<T> *orderParameterComponent =
        new scg::OrderParameterComponent<T>(symmetry, parameters,
                                            criticalTemperature);

    // Set rotation of crystal axes with respect to the grain geometry.
    orderParameterComponent->setCrystalAxesRotation(crystalAxesRotation);

    // Get initial phase parameters.
    const T phaseWinding =
        static_cast<T>(opCompParams["initial_phase_winding"].asDouble());
    const T randomPhaseWeight =
        static_cast<T>(opCompParams["initial_random_phase_weight"].asDouble());
    const T phaseShift = static_cast<T>(
        2.0 * M_PI * opCompParams["initial_phase_shift"].asDouble());

    // Check if we should initialize to bulk.
    const T eps = std::numeric_limits<T>::epsilon();
    const bool useBulk = (std::abs(phaseWinding) < eps) &&
                         (std::abs(randomPhaseWeight) < eps) &&
                         (std::abs(phaseShift) < eps);

    // Initialize order parameter with a guess.
    if (useBulk) {
      scg::Initial<T>::bulk(orderParameterComponent);
    } else {
      scg::Initial<T>::perturbedPhaseWinding(
          orderParameterComponent, phaseWinding, randomPhaseWeight, phaseShift);
    }

    // Add the initialized component to the total order parameter.
    orderParameter->add(orderParameterComponent);
  }

  // Read the order parameter from file, if the path is not empty.
  // TODO(niclas): This is a bit ugly perhaps. If the file contains data for a
  // component, it is overwritten. If it does not, it is left untouched.
  const std::string orderParameterFile =
      std::filesystem::path(simParams["order_parameter_path"].asString());
  if (!orderParameterFile.empty()) {
    // Read the values from file.
    const bool success =
        scg::io::readOrderParameter(orderParameterFile, *orderParameter);

    // Abort if not successful.
    if (!success) {
      scg::run_utils::cleanup(context, gui);
      return 1;
    }
  }

  //***************************************************
  // SET INDUCED VECTOR-POTENTIAL
  //***************************************************
  // Read the vector potential from file, if the path is not empty.
  const std::string vectorPotentialFile =
      std::filesystem::path(simParams["vector_potential_path"].asString());
  if (!vectorPotentialFile.empty()) {
    // Create the placeholder grid.
    const int gridResolution = parameters->getGridResolutionBase();
    auto vectorPotential = std::make_unique<scg::GridGPU<T>>(
        gridResolution, gridResolution, 2, scg::Type::real);

    // Read the values from file.
    const bool success = scg::io::readVectorPotential(vectorPotentialFile,
                                                      *(vectorPotential.get()));

    // Abort if not successful.
    if (!success) {
      scg::run_utils::cleanup(context, gui);
      return 1;
    }
  }

  //***************************************************
  // CHOOSE FERMI SURFACE
  //***************************************************
  scg::FermiSurfaceCircle<T> *fermiSurface = new scg::FermiSurfaceCircle<T>();
  context->set(fermiSurface);

  //***************************************************
  // OPTIMIZER
  //***************************************************
  const T stepSize =
      static_cast<T>(simParams["optimizer"]["step_size"].asDouble());
  const T inertia =
      static_cast<T>(simParams["optimizer"]["inertia"].asDouble());
  const std::size_t rank =
      static_cast<std::size_t>(simParams["optimizer"]["rank"].asInt());
  const bool replaceOldest = simParams["optimizer"]["replace_oldest"].asBool();
  const bool andersonAlways =
      simParams["optimizer"]["anderson_always"].asBool();
  const T minMixing =
      static_cast<T>(simParams["optimizer"]["min_mixing"].asDouble());
  const T maxMixing =
      static_cast<T>(simParams["optimizer"]["max_mixing"].asDouble());
  auto optimizer = std::make_unique<scg::Optimizer<T>>(
      stepSize, inertia, rank, replaceOldest, andersonAlways, minMixing,
      maxMixing);
  context->set(std::move(optimizer));

  //***************************************************
  // CHOOSE QUANTITIES TO CALCULATE
  //***************************************************
  // Order parameter compute object.
  new scg::ComputeOrderParameter<T>(context);
  // Add current compute object.
  new scg::ComputeCurrent<T>(context);
  // Add free energy compute object.
  new scg::ComputeFreeEnergy<T>(context);

  //***************************************************
  // CHOOSE RICCATI SOLVER AND BOUNDARY CONDITIONS
  //***************************************************
  // RiccatiSolverConfined: provides analytic solutions to the Riccati equations
  // for a piecewise constant order parameter.
  scg::RiccatiSolver<T> *riccatiSolver;

  // Gauge choice: VectorPotential::Solenoid calculates the vector potential
  // with a solenoid gague.
  if (isSolenoid) {
    riccatiSolver =
        new scg::RiccatiSolverConfined<T, scg::Gamma::General,
                                       scg::VectorPotential::Solenoid>(context);
  } else {
    riccatiSolver =
        new scg::RiccatiSolverConfined<T, scg::Gamma::General,
                                       scg::VectorPotential::Radial>(context);
  }

  // Add specular BC.
  riccatiSolver->add(new scg::BoundaryConditionSpecular<T>);

  //***************************************************
  // INITIALIZE SIMULATION
  //***************************************************

  // Set external magnetic field strength.
  const T externalFluxQuanta =
      static_cast<T>(simParams["external_flux_quanta"].asDouble());
  if (isSolenoid) {
    // External field in flux quanta.
    parameters->setMagneticFieldStrength(externalFluxQuanta);
  } else {
    // Todo(patric): The field is set after the first initialization since we
    // cannot get geometry->area() without having initialized the geometry. Find
    // a way around this if possible.
    context->initialize();

    // External field in flux quanta per unit area.
    const T externalMagneticFluxDensity = externalFluxQuanta / geometry->area();

    parameters->setMagneticFieldStrength(externalMagneticFluxDensity);
  }

  // Initialize.
  context->initialize();

  //***********************************************
  // SET UP OPENGL VISUALIZATION
  //***********************************************
  const bool visualize = simParams["visualize"].asBool();
  if (visualize) {
    gui = scg::run_utils::createVisualization(context);
    scg::initGL(argc, argv, gui->getNumPixelsWindowX(),
                gui->getNumPixelsWindowY());
    glutKeyboardFunc(scg::run_utils::myKeyboardFunc);
    glutDisplayFunc(display);
  }

  //***********************************************
  // CREATE DIRECTORIES AND SAVE PARAMETERS
  //***********************************************
  // Construct full directory path.
  const std::filesystem::path dataBasePath(simParams["data_path"].asString());
  const std::filesystem::path dataPath =
      dataBasePath /
      ("T" + std::to_string(simParams["temperature"].asDouble()));

  const bool saveData = simParams["save_data"].asBool();
  const bool verbose = simParams["verbose"].asBool();
  if (saveData) {
    // Create directories.
    std::filesystem::create_directories(dataPath);

    // Save config parameters to JSON file.
    const std::filesystem::path configFileName = dataPath / "config.json";
    scg::io::writeJSON(simParams, configFileName, verbose);

    // TODO(niclas): Write internal parameters here as well? They do not change
    // during the simulation.

    // Save domain.
    const std::filesystem::path domainFileName = dataPath / "domain.csv";
    const scg::GridGPU<char> &domain = *(geometry->getDomainGrid());
    const scg::io::CSVGridDataSaveStrategy<char> writer;
    const bool success = writer.save(domain, {"domain"}, domainFileName);
    if (!success) {
      std::cout << "-- ERROR! Could not save: " << domainFileName << std::endl;
    } else {
      std::cout << "-- Wrote: " << domainFileName << std::endl;
    }
  }

  //***********************************************
  // PROCESSING: RUN SIMULATION UNTIL END CRITERIA
  //***********************************************
  const scg::Type::fileFormat dataFormat = scg::Type::csv;
  scg::run_utils::runSimulation(
      context, verbose, simParams["num_iterations_min"].asInt(),
      simParams["num_iterations_max"].asInt(), visualize, dataPath, saveData,
      simParams["save_data_interval"].asInt(), dataFormat);

  //***********************************************
  // SAVE RESULTS
  //***********************************************
  // Save at the very end.
  if (saveData) {
    scg::run_utils::saveDataWrapper(context, dataPath, verbose, dataFormat);
  }

  // Cleanup dynamically allocated memory.
  scg::run_utils::cleanup(context, gui);
  return 0;
}

// Plot data in OpenGL GUI.
void display() {
  scg::run_utils::display(context, gui);

  // Tell Glut to update window.
  glutPostRedisplay();
}
