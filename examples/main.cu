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

#include <iostream>
#include <stdlib.h>
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
} // namespace

//*******************************************************
// SIMULATION PARAMETERS
//*******************************************************
namespace simParams {
//******************************
// Physics
//******************************
// Temperature in units of T_c.
const T TEMPERATURE = 0.1;
// Run with d-wave or s-wave.
const bool D_WAVE = true;
// Rotation between crystal axes and grain geometry.
const T CRYSTAL_AXES_ROTATION = 0.25 * M_PI;
// External magnetic flux in units of flux quanta.
const T EXTERNAL_MAGNETIC_FLUX_QUANTA = 6.0;
// Size of superconducting grain in coherence lengths hbar*vF/(2*pi*kBTc).
const T GRAIN_SIDE_LENGTH = 30.0;

//******************************
// Geometry
//******************************
// Run with circular or square grain.
const bool DISC = true;
// Run with a solenoid or not.
const bool SOLENOID = true;

//******************************
// Numeric details
//******************************
// Number of lattice points per grain side length.
const int GRID_RESOLUTION = 201;
// Number of discrete trajectory angles (on the Fermi-surface).
const int NUM_ANGLES = 64;
// Number of discrete energies (poles).
const int NUM_ENERGIES = 20;

//******************************
// Self-consistency and convergence
//******************************
// Convergence criterion for order-parameter residual.
const T CONVERGENCE_CRITERION = 1E-6;
// Maximum number of iterations.
const int NUM_ITERATIONS_MAX = 1000;
// Minimum number of iterations.
const int NUM_ITERATIONS_MIN = 20;

//******************************
// Output and visualization
//******************************
// Print iteration status to terminal.
const bool PRINT_ITERATION_STATUS = true;
// Run with OpenGL visualization or in terminal.
const bool VISUALIZE = true;
// Plot the order parameter
const bool PLOT_ORDER_PARAMETER = true;
// Plot current density (x,y)-components and magnitude.
const bool PLOT_CURRENT_DENSITY = true;
// Save data to file upon completion.
const bool SAVE_DATA = true;
// Path to data, relative to build directory.
const std::string DATA_BASE_PATH = "../examples/data/";
} // namespace simParams

//*******************************************************
// MAIN
//*******************************************************
int main(int argc, char *argv[]) {
  //***************************************************
  // APPLY PARAMETERS
  //***************************************************
  // Storage for system simulation parameters (e.g. temperature, lattice size).
  Parameters<T> *parameters = new Parameters<T>(context);
  // Set temperature.
  parameters->setTemperature(simParams::TEMPERATURE);
  // Set grain sidelength in coherence lengths: hbar*vF/(2*pi*kB*Tc).
  parameters->setGrainWidthCoherenceLengths(simParams::GRAIN_SIDE_LENGTH);
  // Set grain resolution in number of lattice sites.
  parameters->setGridResolution(simParams::GRID_RESOLUTION);
  // Set number of Matsubara energies (Ozaki poles).
  parameters->setNumMatsubaraEnergies(simParams::NUM_ENERGIES);
  // Number of discrete steps in momentum angle.
  parameters->setAngularResolution(simParams::NUM_ANGLES);
  // Set convergence criterion.
  parameters->setConvergenceCriterion(simParams::CONVERGENCE_CRITERION);

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
  if (simParams::DISC) {
    geometry->add(new DiscGeometry<T>(grainScale), Type::add);
  } else {
    geometry->add(new PolygonGeometry<T>(numVerticesSquare, grainScale),
                  Type::add);
  }

  // Remove a portion to create a solenoid.
  if (simParams::SOLENOID) {
    // Relative size of grain hole for solenoid.
    const T holeScale = 0.2;

    // Circular or square hole.
    if (simParams::DISC) {
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
  if (simParams::D_WAVE) {
    orderParameterComponent = new OrderParameterComponent_D<T>(parameters);

    // Set rotation of crystal axes with respect to the grain geometry.
    orderParameterComponent->setCrystalAxesRotation(
        simParams::CRYSTAL_AXES_ROTATION);
  } else {
    orderParameterComponent = new OrderParameterComponent_S<T>(parameters);
  }

  // Initialize order parameter with a guess.
  if (simParams::SOLENOID) {
    // Initialize order-parameter component with a phase winding corresponding
    // to the number of external flux quanta. If there is a mismatch, the system
    // might converge to a meta-stable state.
    Initial<T>::phaseWinding(orderParameterComponent,
                             simParams::EXTERNAL_MAGNETIC_FLUX_QUANTA);
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
  if (simParams::SOLENOID) {
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
  if (simParams::SOLENOID) {
    // External field in flux quanta.
    parameters->setMagneticFieldStrength(
        simParams::EXTERNAL_MAGNETIC_FLUX_QUANTA);
  } else {
    // External field in fux quanta per unit area.
    const T EXTERNAL_MAGNETIC_FLUX_DENSITY =
        simParams::EXTERNAL_MAGNETIC_FLUX_QUANTA / geometry->area();

    parameters->setMagneticFieldStrength(EXTERNAL_MAGNETIC_FLUX_DENSITY);
  }

  // Re-initialize
  // Todo(patric): the field is set after the first initialization since we
  // cannot get geometry->area() without having initialized the geometry. Find a
  // way around this if possible.
  context->initialize();

  //***********************************************
  // SET UP OPENGL VISUALIZATION
  //***********************************************
  if (simParams::VISUALIZE) {
    gui =
        run_utils::createVisualization(context, simParams::PLOT_ORDER_PARAMETER,
                                       simParams::PLOT_CURRENT_DENSITY);
    initGL(argc, argv, gui->getNumPixelsWindowX(), gui->getNumPixelsWindowY());
    glutKeyboardFunc(run_utils::myKeyboardFunc);
    glutDisplayFunc(display);
  }

  //***********************************************
  // PROCESSING: RUN SIMULATION UNTIL END CRITERIA
  //***********************************************
  run_utils::runSimulation(
      context, simParams::VISUALIZE, simParams::PRINT_ITERATION_STATUS,
      simParams::NUM_ITERATIONS_MIN, simParams::NUM_ITERATIONS_MAX);

  //***********************************************
  // POST-PROCESSING
  //***********************************************
  // Save at the very end.
  if (simParams::SAVE_DATA) {
    saveDataWrapper();
  }

  // Cleanup dynamically allocated memory.
  run_utils::cleanup(context, gui);
  return 0;
}

// Plot data in OpenGL GUI.
void display() {
  run_utils::display(context, gui, simParams::PLOT_ORDER_PARAMETER,
                     simParams::PLOT_CURRENT_DENSITY);

  // Tell Glut to update window.
  glutPostRedisplay();
}

void saveDataWrapper() {

  char dataPath[512];
  snprintf(dataPath, sizeof(dataPath), "%sT%2.4f/",
           simParams::DATA_BASE_PATH.c_str(), simParams::TEMPERATURE);

  // Todo(patric): Implement helper functions to save data.
}
