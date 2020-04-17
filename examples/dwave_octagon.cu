//===-- dwave_octagon.cu - Example run file of a d-wave octagon.
//----------------------------------===//
//
// Part of the SuprConGPU Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This is a simple example that simulates a d-wave octagon at a finite
/// temperature and displays the value of the order parameter magnitude. The
/// expected result is to see a bulk order parameter everywhere (with
/// magnitude 1.51 at low temperatures), except at pairbreaking edges defined by
/// the crystal ab-axes rotation.
///
//===----------------------------------------------------------------------===//

#include "BoundaryConditionSpecular.h" // Boundary condition for perfect specular reflection
#include "ComputeOrderParameter.h" // Compute order parameter
#include "Context.h" // The context that connects all classes (helpful since enacpsulation is used)
#include "FermiSurfaceCircle.h" // Standard circle Fermi surface
#include "GeometryGroup.h" // The geometry group stores a list of polygons/discs/... that together define the sample geometry (container class for GeometryComponent)
#include "OptimizerRollingBall.h" // Convergence accelerator for self-consistency solver
#include "OrderParameter.h" // Container class for order parameter components
#include "OrderParameterComponent_D.h" // d-wave
#include "OrderParameterInitial.h" // Used for generating initial guess for order parameter
#include "Parameters.h" // Parameters specifying which the superconducting system to simulate
#include "PolygonGeometry.h" // Used to constructed simulated grain from polygons
#include "RiccatiSolverConfined.h" // Riccati solver for confined geometries
#include "Visualization.h" // Live visualization of desired observables through OpenGL
#include "run_utils.h" // Commonly used functions in run files

#include <iostream>
#include <stdlib.h>

//*******************************************************
// FORWARD DECLARATIONS
//*******************************************************
void display();

//*******************************************************
// TYPEDEFS
//*******************************************************
// Run with double or single floating point precision, double is highly
// recommended
typedef double T;

//*******************************************************
// GLOBAL OBJECTS: glut does not accept arguments and relies on global/static
// variables.
//*******************************************************
namespace {
Context<T> *context = new Context<T>();
Visualization<T> *gui;
} // namespace

//*******************************************************
// SIMULATION PARAMETERS
//*******************************************************
namespace simParams {
// Temperature in units of T_c
const T TEMPERATURE = 0.5;
// Size of superconducting grain in coherence lengths hbar*v_F/(2*pi*k_BT_c)
const T GRAIN_SIDE_LENGTH = 30.0;
// Crystal ab-axis rotation relative to simulation grain. Zero means that the
// ab-axes are aligned with the edges of the screen.
const T CRYSTAL_AXES_ROTATION = 0.25 * M_PI;
// Number of lattice points per grain side length
const int GRID_RESOLUTION = 201;
// Number of discrete trajectory angles (on the Fermi-surface)
const int NUM_ANGLES = 64;
// Number of discrete energies (poles)
const int NUM_ENERGIES = 20;
// Convergence criterion for order-parameter residual
const T CONVERGENCE_CRITERION = 1E-6;
// Maximum number of iterations
const int NUM_ITERATIONS_MAX = 1000;
// Minimum number of iterations
const int NUM_ITERATIONS_MIN = 20;
// Print iteration status to terminal
const bool PRINT_ITERATION_STATUS = true;
// Plot the order parameter magnitude (and phase if also plotting the current
// density).
const bool PLOT_ORDER_PARAMETER = true;
// Plot current density (x,y)-components and magnitude
const bool PLOT_CURRENT_DENSITY = false;
} // namespace simParams

//*******************************************************
// MAIN
//*******************************************************
int main(int argc, char *argv[]) {
  //***************************************************
  // APPLY PARAMETERS
  //***************************************************
  // Storage for system simulation parameters (e.g. temperature, lattice size)
  Parameters<T> *parameters = new Parameters<T>(context);
  // Set grain resolution in number of lattice sites
  parameters->setGridResolution(simParams::GRID_RESOLUTION);
  // Set grain sidelength in coherence lengths: hbar*vF/(2*pi*kB*Tc)
  parameters->setGrainWidthCoherenceLengths(simParams::GRAIN_SIDE_LENGTH);
  // Number of discrete steps in momentum angle
  parameters->setAngularResolution(simParams::NUM_ANGLES);
  // Set convergence criterion
  parameters->setConvergenceCriterion(simParams::CONVERGENCE_CRITERION);
  // Set number of Ozaki poles
  parameters->setNumMatsubaraEnergies(simParams::NUM_ENERGIES);
  // Set temperature
  parameters->setTemperature(simParams::TEMPERATURE);

  //***************************************************
  // CREATE GEOMETRY
  //***************************************************
  // Create geometry group: add/remove geometry objects (e.g. discs, polygons)
  // from the geometry group
  GeometryGroup<T> *geometry = new GeometryGroup<T>(context);

  // Add an octagon to simulation space, scale to fit in simulation domain
  // TODO: (Patric) Different polygon objects need different scaling in order to
  // not fall outside simulation domain. This should however be done
  // automatically (if possible) and hidden from the user.
  const int NUM_VERTICES_OCTAGON = 8;
  const T OCTAGON_GRAIN_SCALE = 0.75;
  geometry->add(
      new PolygonGeometry<T>(NUM_VERTICES_OCTAGON, OCTAGON_GRAIN_SCALE),
      Type::add);

  //***************************************************
  // CHOOSE RICCATI SOLVER AND BOUNDARY CONDITIONS
  //***************************************************
  // RiccatiSolverConfined: provides analytic solutions to the Riccati equations
  // for a piecewise constant order parameter Gauge choice:
  // VectorPotential::None for never calculating the vector potential (no
  // external or induced fields)
  RiccatiSolver<T> *riccatiSolver =
      new RiccatiSolverConfined<T, Gamma::General, VectorPotential::None>(
          context);
  // Add specular BC
  riccatiSolver->add(new BoundaryConditionSpecular<T>);

  //***************************************************
  // SET UP ORDER PARAMETER
  //***************************************************
  // Create order parameter (which might contain multiple components, i.e. d+is)
  OrderParameter<T> *orderParameter = new OrderParameter<T>(context);
  // Add a d-wave component to the total order-parameter
  OrderParameterComponent_D<T> *orderParameterComponent =
      new OrderParameterComponent_D<T>(parameters);
  // Rotate d-wave lobes (ab-axes) relative to grain
  orderParameterComponent->setLatticeOrientation(
      simParams::CRYSTAL_AXES_ROTATION);
  Initial<T>::bulk(orderParameterComponent);
  // Add the initialized d-wave component to the total order parameter
  orderParameter->add(orderParameterComponent);

  //***************************************************
  // CHOOSE QUANTITIES TO CALCULATE
  //***************************************************
  // Create compute object(s) for observable(s)
  ComputeOrderParameter<T> *computeOrderParameter =
      new ComputeOrderParameter<T>(context);

  //***************************************************
  // CHOOSE CONVERGENCE ACCELERATOR
  //***************************************************
  OptimizerRollingBall<T> *optimizer = new OptimizerRollingBall<T>();
  computeOrderParameter->setOptimizer(optimizer);

  //***************************************************
  // CHOOSE FERMI SURFACE
  //***************************************************
  FermiSurfaceCircle<T> *fermiSurface = new FermiSurfaceCircle<T>();
  context->set(fermiSurface);

  //***************************************************
  // INITIALIZE SIMULATION
  //***************************************************
  context->initialize();

  //***********************************************
  // SET UP OPENGL VISUALIZATION
  //***********************************************
  const bool visualize =
      simParams::PLOT_ORDER_PARAMETER || simParams::PLOT_CURRENT_DENSITY;
  if (visualize) {
    gui =
        run_utils::createVisualization(context, simParams::PLOT_ORDER_PARAMETER,
                                       simParams::PLOT_CURRENT_DENSITY);
    initGL(argc, argv, gui->getNumPixelsWindowX(), gui->getNumPixelsWindowY());
    glutKeyboardFunc(run_utils::myKeyboardFunc);
    glutDisplayFunc(display);
  }

  //***********************************************
  // RUN SIMULATION UNTIL END CRITERIA
  //***********************************************
  run_utils::runSimulation(
      context, visualize, simParams::PRINT_ITERATION_STATUS,
      simParams::NUM_ITERATIONS_MIN, simParams::NUM_ITERATIONS_MAX);
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
