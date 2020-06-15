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
#include "OptimizerRollingBall.h" // Convergence accelerator for self-consistency solver
#include "OrderParameter.h" // Container class for order parameter components
#include "OrderParameterBasis.h"
#include "OrderParameterComponent.h"
#include "OrderParameterInitial.h" // Used for generating initial guess for order parameter
#include "Parameters.h" // Parameters specifying which the superconducting system to simulate
#include "PolygonGeometry.h" // Used to constructed simulated grain from polygons
#include "RiccatiSolverConfined.h" // Riccati solver for confined geometries
#include "Visualization.h" // Live visualization of desired observables through OpenGL
#include "cli_utils.h" // Helper functions to parse command-line, and json-files.
#include "configure.h" // Variables and defines exported by cmake, e.g. version number and project name
#include "run_utils.h" // Commonly used functions in run files

#include "nlohmann_json_wrapper.h"
#include <CLI/CLI.hpp>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

//*******************************************************
// FORWARD DECLARATIONS
//*******************************************************
void display();
void saveDataWrapper(nlohmann::json &inOutJSON);

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
Context<T> *context = new Context<T>();
// Create visualization object.
Visualization<T> *gui;
} // namespace

//*******************************************************
// MAIN
//*******************************************************
int main(int argc, char *argv[]) {
  // CLI11 application that parses all arguments.
  CLI::App app;

  // JSON parameters which will contain the parsed arguments.
  nlohmann::json simParams;

  // Only applicable from the command line. They will be parsed by the CLI11
  // app.
  bool printVersion = false;
  bool dumpConfig = false;

  // Setup the application and default parameter values.
  cli_utils::makeDefaultApp(app, simParams, printVersion, dumpConfig);

  // Parse the arguments.
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  }

  if (printVersion) {
    // Print version number and exit.
    std::cout << configure::projectName << " " << configure::version
              << std::endl;
    std::cout << configure::licenseStatement << std::endl;
    return 0;
  } else if (dumpConfig) {
    // Dump config and exit.
    const int indentation = 4;
    std::cout << simParams.dump(indentation) << std::endl;
    return 0;
  }

  //***************************************************
  // APPLY PARAMETERS
  //***************************************************
  // Storage for system simulation parameters (e.g. temperature, lattice size).
  Parameters<T> *parameters = new Parameters<T>(context);
  // Set temperature.
  parameters->setTemperature(simParams["temperature"].get<T>());
  // Set grain sidelength in coherence lengths: hbar*vF/(2*pi*kB*Tc).
  parameters->setGrainWidthCoherenceLengths(
      simParams["grain_side_length"].get<T>());
  // Set grain resolution in number of lattice sites.
  parameters->setGridResolution(simParams["lattice_side_length"].get<int>());
  // Set number of Ozaki energies.
  parameters->setNumMatsubaraEnergies(
      simParams["num_ozaki_energies"].get<int>());
  // Number of discrete steps in momentum angle.
  parameters->setAngularResolution(simParams["num_fermi_momenta"].get<int>());
  // Set convergence criterion.
  parameters->setConvergenceCriterion(
      simParams["convergence_criterion"].get<T>());

  //***************************************************
  // CREATE GEOMETRY
  //***************************************************
  // Create geometry group: add/remove geometry objects (e.g. discs, polygons)
  // from the geometry group.
  GeometryGroup<T> *geometry = new GeometryGroup<T>(context);

  // Add polygon to simulation space, scale to fit in simulation domain
  // (grainScale 1.0 is 100% of simulation space).
  const T grainScale = static_cast<T>(1.0);

  // Number of vertices in a square.
  const int numVerticesSquare = 4;

  // Make the grain a disc or a square.
  if (simParams["disc_geometry"].get<bool>()) {
    geometry->add(new DiscGeometry<T>(grainScale), Type::add);
  } else {
    geometry->add(new PolygonGeometry<T>(numVerticesSquare, grainScale),
                  Type::add);
  }

  // Remove a portion to create a solenoid.
  if (simParams["solenoid"].get<bool>()) {
    // Relative size of grain hole for solenoid.
    const T holeScale = static_cast<T>(0.2);

    // Circular or square hole.
    if (simParams["disc_geometry"].get<bool>()) {
      geometry->add(new DiscGeometry<T>(holeScale), Type::remove);
    } else {
      geometry->add(new PolygonGeometry<T>(numVerticesSquare, holeScale),
                    Type::remove);
    }
  }

  //***************************************************
  // SET UP ORDER PARAMETER
  //***************************************************
  // Create order parameter (which might contain multiple components, i.e.
  // d+is).
  OrderParameter<T> *orderParameter = new OrderParameter<T>(context);

  // Add a d-wave or an s-wave order-parameter component.
  const BasisFunctionSymmetry symmetry = simParams["dwave_symmetry"].get<bool>()
                                             ? BasisFunctionSymmetry::DWAVE
                                             : BasisFunctionSymmetry::SWAVE;
  OrderParameterComponent<T> *orderParameterComponent =
      new OrderParameterComponent<T>(symmetry, parameters);

  // Set rotation of crystal axes with respect to the grain geometry.
  const T crystalAxesRotation =
      static_cast<T>(M_PI) * simParams["crystal_axes_rotation"].get<T>();
  orderParameterComponent->setCrystalAxesRotation(crystalAxesRotation);

  // Initialize order parameter with a guess.
  if (simParams["solenoid"].get<bool>()) {
    // Initialize order-parameter component with a phase winding corresponding
    // to the number of external flux quanta. If there is a mismatch, the system
    // might converge to a meta-stable state.
    Initial<T>::phaseWinding(
        orderParameterComponent,
        static_cast<int>(simParams["external_flux_quanta"].get<T>()));
  } else {
    // Initialize order-parameter component to the (real) bulk value.
    Initial<T>::bulk(orderParameterComponent);
  }

  // Add the initialized component to the total order parameter. This
  orderParameter->add(orderParameterComponent);

  //***************************************************
  // CHOOSE FERMI SURFACE
  //***************************************************
  FermiSurfaceCircle<T> *fermiSurface = new FermiSurfaceCircle<T>();
  context->set(fermiSurface);

  //***************************************************
  // CHOOSE QUANTITIES TO CALCULATE
  //***************************************************
  // Order parameter compute object.
  ComputeOrderParameter<T> *computeOrderParameter =
      new ComputeOrderParameter<T>(context);
  // Add current compute object.
  new ComputeCurrent<T>(context);
  // Add free energy compute object.
  new ComputeFreeEnergy<T>(context);

  //***************************************************
  // CHOOSE RICCATI SOLVER AND BOUNDARY CONDITIONS
  //***************************************************
  // RiccatiSolverConfined: provides analytic solutions to the Riccati equations
  // for a piecewise constant order parameter.
  RiccatiSolver<T> *riccatiSolver;

  // Gauge choice: VectorPotential::Solenoid calculates the vector potential
  // with a solenoid gague.
  if (simParams["solenoid"].get<bool>()) {
    riccatiSolver =
        new RiccatiSolverConfined<T, Gamma::General, VectorPotential::Solenoid>(
            context);
  } else {
    riccatiSolver =
        new RiccatiSolverConfined<T, Gamma::General, VectorPotential::Radial>(
            context);
  }

  // Add specular BC.
  riccatiSolver->add(new BoundaryConditionSpecular<T>);

  //***************************************************
  // CHOOSE CONVERGENCE ACCELERATOR
  //***************************************************
  OptimizerRollingBall<T> *optimizer = new OptimizerRollingBall<T>();
  computeOrderParameter->setOptimizer(optimizer);

  //***************************************************
  // INITIALIZE SIMULATION
  //***************************************************
  context->initialize();

  // Set external magnetic field strength.
  if (simParams["solenoid"].get<bool>()) {
    // External field in flux quanta.
    parameters->setMagneticFieldStrength(
        simParams["external_flux_quanta"].get<T>());
  } else {
    // External field in fux quanta per unit area.
    const T externalMagneticFluxDensity =
        simParams["external_flux_quanta"].get<T>() / geometry->area();

    parameters->setMagneticFieldStrength(externalMagneticFluxDensity);
  }

  // Re-initialize
  // Todo(patric): the field is set after the first initialization since we
  // cannot get geometry->area() without having initialized the geometry. Find a
  // way around this if possible.
  context->initialize();

  //***********************************************
  // SET UP OPENGL VISUALIZATION
  //***********************************************
  if (simParams["visualize"].get<bool>()) {
    gui = run_utils::createVisualization(context);
    initGL(argc, argv, gui->getNumPixelsWindowX(), gui->getNumPixelsWindowY());
    glutKeyboardFunc(run_utils::myKeyboardFunc);
    glutDisplayFunc(display);
  }

  //***********************************************
  // PROCESSING: RUN SIMULATION UNTIL END CRITERIA
  //***********************************************
  run_utils::runSimulation(context, simParams["visualize"].get<bool>(),
                           simParams["verbose"].get<bool>(),
                           simParams["num_iterations_min"].get<int>(),
                           simParams["num_iterations_max"].get<int>(),
                           simParams["save_data"].get<bool>(),
                           simParams["save_data_interval"].get<int>(),
                           simParams);

  //***********************************************
  // POST-PROCESSING
  //***********************************************
  // Save at the very end.
  if (simParams["save_data"].get<bool>()) {
    run_utils::saveDataWrapper(context, simParams,
                               simParams["verbose"].get<bool>());
  }

  // Cleanup dynamically allocated memory.
  run_utils::cleanup(context, gui);
  return 0;
}

// Plot data in OpenGL GUI.
void display() {
  run_utils::display(context, gui);

  // Tell Glut to update window.
  glutPostRedisplay();
}
