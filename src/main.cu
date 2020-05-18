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
#include "OrderParameterComponent_D.h" // d-wave
#include "OrderParameterComponent_S.h" // s-wave
#include "OrderParameterInitial.h" // Used for generating initial guess for order parameter
#include "Parameters.h" // Parameters specifying which the superconducting system to simulate
#include "PolygonGeometry.h" // Used to constructed simulated grain from polygons
#include "RiccatiSolverConfined.h" // Riccati solver for confined geometries
#include "Visualization.h" // Live visualization of desired observables through OpenGL
#include "run_utils.h" // Commonly used functions in run files

#include <CLI/CLI.hpp>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

//*******************************************************
// FORWARD DECLARATIONS
//*******************************************************
void display();
void saveDataWrapper();

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
// JSON parameters.
nlohmann::json jParameters;
} // namespace

//*******************************************************
// MAIN
//*******************************************************
int main(int argc, char *argv[]) {
  //***************************************************
  // READ PARAMETERS
  //***************************************************
  CLI::App app{
      "SuprConGPU main file for setting up self-consistent simulations in a "
      "finite geometry (a mesoscopic grain). Different superconducting "
      "order-paramers and geometries can be specified, as well as physical "
      "parameters such as temperature and external fields. To set these, do "
      "NOT modify the source file (i.e. the .cu file) directly. Instead, to "
      "set up simulation properties and parameters, use the command-line "
      "options specified below, or set up a .json parameter-file as in "
      "examples/ and  give the path to this file with the -f option (see "
      "below). Command-line options have highest priority."};

  std::string filename;
  std::string dataBasePath;
  T temperature;
  T externalMagneticFluxQuanta;
  T grainSideLength;
  T crystalAxesRotation;
  T convergenceCriterion;
  int gridResolution;
  int numNumOzakiEnergies;
  int numAngles;
  int maxIterations;
  int minIterations;
  bool useSolenoid;
  bool visualize;
  bool verbose;
  bool printVersion;

  app.add_option(
      "-f,--file", filename,
      "path to JSON file with parameters specifying the simulation settings.");
  app.add_option(
      "-o,--out", dataBasePath,
      "base folder for output (data and parameter files), default: ...");
  app.add_option("-T,--temperature", temperature,
                 "temperature in units of T_c. Valid range: (0,1].");
  app.add_option(
      "-B,--mfield", externalMagneticFluxQuanta,
      "external magnetic field, perpendicular to sample, in units of "
      "flux quanta. For non-solenoid field, it is distributed uniformly over "
      "the sample. Default: 0.");
  app.add_option("-s,--size", grainSideLength,
                 "side length of grain in units of coherence lengths "
                 "(hbar*v_F)/(2*pi*k_B*T_c).");
  app.add_option(
      "-N,--resolution", gridResolution,
      "number of discrete lattice points in the grain. Set to odd value "
      "for best convergence. Recommended to be at least 10 times the grain "
      "size.");
  app.add_option("-e,--energies", numNumOzakiEnergies,
                 "number of Ozaki energies (similar to Matsubara energies, see "
                 "the documentation).");
  app.add_option("-a,--angles", numAngles,
                 "number of discrete angles in Fermi-surface average. Set to "
                 "odd value for best convergence. Recommended minumum: 101.");
  app.add_option("-r,--rotation", crystalAxesRotation,
                 "rotation between grain and crystal ab-axes. Default: 0.");
  app.add_option("-c,--convergence", convergenceCriterion,
                 "convergence criteron for self-consistency. Default: ...");
  app.add_option(
      "-M,--max-iter", maxIterations,
      "maximum number of self-consistent iterations to run. Default: ...");
  app.add_option(
      "-m,--min-iter", minIterations,
      "minimum number of self-consistent iterations to run. Default: ...");
  app.add_flag("-S,--solenoid", useSolenoid,
               "set up a solenoid, and use solenoid gauge for external field.");
  app.add_flag("-V,--visualize", visualize, "run with real-time plotting.");
  app.add_flag("-v,--verbose", verbose,
               "print iteration status and other output to terminal.");
  app.add_flag("--version", printVersion, "print version number.");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  }

  // Read the parameter file.
  std::ifstream file(filename);
  file >> jParameters;

  // Update the jParameters with command-line values

  //***************************************************
  // APPLY PARAMETERS
  //***************************************************
  // Storage for system simulation parameters (e.g. temperature, lattice size).
  Parameters<T> *parameters = new Parameters<T>(context);
  // Set temperature.
  parameters->setTemperature(jParameters["temperature"].get<T>());
  // Set grain sidelength in coherence lengths: hbar*vF/(2*pi*kB*Tc).
  parameters->setGrainWidthCoherenceLengths(
      jParameters["grain_side_length"].get<T>());
  // Set grain resolution in number of lattice sites.
  parameters->setGridResolution(jParameters["grid_resolution"].get<int>());
  // Set number of Ozaki energies.
  parameters->setNumMatsubaraEnergies(jParameters["num_energies"].get<int>());
  // Number of discrete steps in momentum angle.
  parameters->setAngularResolution(jParameters["num_angles"].get<int>());
  // Set convergence criterion.
  parameters->setConvergenceCriterion(
      jParameters["convergence_criterion"].get<T>());

  //***************************************************
  // CREATE GEOMETRY
  //***************************************************
  // Create geometry group: add/remove geometry objects (e.g. discs, polygons)
  // from the geometry group.
  GeometryGroup<T> *geometry = new GeometryGroup<T>(context);

  // Add an square to simulation space, scale to fit in simulation domain, then
  // remove a square-shaped hole
  // TODO: (Patric) Different polygon objects need different scaling in order to
  // not fall outside simulation domain. This should however be done
  // automatically (if possible) and hidden from the user.

  // Relative size of grain (1.0 is 100% of simulation space).
  const T grainScale = 1.0;

  // Number of vertices in a square.
  const int numVerticesSquare = 4;

  // Make the grain a disc or a square.
  if (jParameters["disc"].get<bool>()) {
    geometry->add(new DiscGeometry<T>(grainScale), Type::add);
  } else {
    geometry->add(new PolygonGeometry<T>(numVerticesSquare, grainScale),
                  Type::add);
  }

  // Remove a portion to create a solenoid.
  if (jParameters["solenoid"].get<bool>()) {
    // Relative size of grain hole for solenoid.
    const T holeScale = 0.2;

    // Circular or square hole.
    if (jParameters["disc"].get<bool>()) {
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
  OrderParameterComponent<T> *orderParameterComponent;
  if (jParameters["d_wave"].get<bool>()) {
    orderParameterComponent = new OrderParameterComponent_D<T>(parameters);
  } else {
    orderParameterComponent = new OrderParameterComponent_S<T>(parameters);
  }

  // Set rotation of crystal axes with respect to the grain geometry.
  crystalAxesRotation =
      static_cast<T>(M_PI) * jParameters["crystal_axes_rotation"].get<T>();
  orderParameterComponent->setCrystalAxesRotation(crystalAxesRotation);

  // Initialize order parameter with a guess.
  if (jParameters["solenoid"].get<bool>()) {
    // Initialize order-parameter component with a phase winding corresponding
    // to the number of external flux quanta. If there is a mismatch, the system
    // might converge to a meta-stable state.
    Initial<T>::phaseWinding(
        orderParameterComponent,
        static_cast<int>(
            jParameters["external_magnetic_flux_quanta"].get<T>()));
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
  if (jParameters["solenoid"].get<bool>()) {
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
  if (jParameters["solenoid"].get<bool>()) {
    // External field in flux quanta.
    parameters->setMagneticFieldStrength(
        jParameters["external_magnetic_flux_quanta"].get<T>());
  } else {
    // External field in fux quanta per unit area.
    const T externalMagneticFluxDensity =
        jParameters["external_magnetic_flux_quanta"].get<T>() /
        geometry->area();

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
  visualize = jParameters["visualize"].get<bool>();
  if (visualize) {
    gui = run_utils::createVisualization(context);
    initGL(argc, argv, gui->getNumPixelsWindowX(), gui->getNumPixelsWindowY());
    glutKeyboardFunc(run_utils::myKeyboardFunc);
    glutDisplayFunc(display);
  }

  //***********************************************
  // PROCESSING: RUN SIMULATION UNTIL END CRITERIA
  //***********************************************
  run_utils::runSimulation(context, visualize,
                           jParameters["print_iteration_status"].get<bool>(),
                           jParameters["num_iterations_min"].get<int>(),
                           jParameters["num_iterations_max"].get<int>());

  //***********************************************
  // POST-PROCESSING
  //***********************************************
  // Save at the very end.
  if (jParameters["save_data"].get<bool>()) {
    saveDataWrapper();
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

void saveDataWrapper() {

  char dataPath[512];
  const std::string dataBasePath =
      jParameters["data_base_path"].get<std::string>();
  snprintf(dataPath, sizeof(dataPath), "%sT%2.4f/", dataBasePath.c_str(),
           jParameters["temperature"].get<T>());

  // Todo(patric): Implement helper functions to save data.
}
