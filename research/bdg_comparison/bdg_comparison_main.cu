//===---- circular_FS.cu - Stabilize phase crystal with a circular FS. ----===//
//
// Part of the SuprConGPU Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file stabilizes the phase crystal in a triangular grain with a single
/// pairbreaking edge, with a circular Fermi surface.
///
//===----------------------------------------------------------------------===//
#include "BoundaryConditionSpecular.h" // Boundary condition for perfect specular reflection
#include "ComputeCurrent.h"        // Compute the total charge current density
#include "ComputeFreeEnergy.h"     // Compute the (Eilenberger) free energy
#include "ComputeOrderParameter.h" // Compute order parameter
#include "Context.h" // The context that connects all classes (helpful since enacpsulation is used)
#include "FermiSurfaceCircle.h"       // Standard circular Fermi surface
#include "FermiSurfaceTightBinding.h" // Tight binding Fermi surface
#include "GeometryGroup.h" // The geometry group stores a list of polygons/discs/... that together define the sample geometry (container class for GeometryComponent)
#include "OptimizerRollingBall.h" // Convergence accelerator for self-consistency solver
#include "OrderParameter.h" // Container class for order parameter components
#include "OrderParameterComponent_D.h" // d-wave
#include "OrderParameterInitial.h" // Used for generating initial guess for order parameter
#include "Parameters.h" // Parameters specifying which the superconducting system to simulate
#include "PolygonGeometry.h" // Used to constructed simulated grain from polygons
#include "RiccatiSolverConfined.h" // Riccati solver for confined geometries
#include "Visualization.h" // Live visualization of desired observables through OpenGL

#include "bdg_comparison_utils.h" // Common utils for this project

#include <cuda.h>
#include <iostream>

//*******************************************************
// FORWARD DECLARATIONS
//*******************************************************
void runInTerminal();
void myKeyboardFunc(const unsigned char key, const int x, const int y);

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
// Context object
Context<T> *context = new Context<T>();
// Visualization object
Visualization<T> *gui;
// Free energy object
ComputeFreeEnergy<T> *computeFreeEnergy;
// Keep track of user key press in OpenGL mode
bool gUserQuit = false;
bool gUserSaveData = false;
// Used to construct data file path
char dataPath[512];
} // namespace

//*******************************************************
// SIMULATION PARAMETERS
//*******************************************************
// TODO(patric): Make into a struct
namespace simParams {
// Run in OpenGL mode or terminal mode
const bool RUN_VISUAL = true;
// Temperature in units of T_c
const T TEMPERATURE = static_cast<T>(0.1);
// Size of pairbreaking edge in coherence lengths hbar*v_F/(2*pi*k_BT_c)
T GRAIN_DIAGONAL = static_cast<T>(40.0);
// Size (in coherence lengths) that defines the grain geometry: 0 gives a
// triangle, >0 gives an irregular pentagon
T GRAIN_CORNER_SIZE = static_cast<T>(4.0);
const T GRAIN_SIDE_LENGTH =
    GRAIN_CORNER_SIZE + GRAIN_DIAGONAL / static_cast<T>(std::sqrt(2.0));
// Number of lattice points per grain side length
const int GRID_RESOLUTION = 401;
// Number of discrete trajectory angles (on the Fermi-surface)
const int NUM_ANGLES = 128;
// Number of discrete energies (poles)
const int NUM_ENERGIES = 20;
// Convergence criterion for order-parameter residual
const T CONVERGENCE_CRITERION = static_cast<T>(1E-6);
// Maximum number of iterations
const int NUM_ITERATIONS_MAX = 5000;
// Minimum number of iterations
const int NUM_ITERATIONS_MIN = 300;
// Path to data, relative to build directory!
const char *DATA_BASE_PATH = (char *)"../research/bdg_comparison/data/qc/";
// Number of iterations between regular saving of data
const int SAVE_DATA_INTERVAL = 200;
// Label which fermi surface it is:
// 1 - Circular
// 2 - Tight Binding: NNN= -0.25, NNNN=0, mu=0
// 3 - Tight Binding: NNN= -0.477, NNNN=0.156, mu=-1.267
int fermiSurfaceLabel = 1;
} // namespace simParams

//*******************************************************
// MAIN
//*******************************************************
int main(int argc, char *argv[]) {
  //***************************************************
  // APPLY PARAMETERS
  //***************************************************
  // Update parameters from command line
  // TODO(patric): replace with proper argument parser and put in utils_io.h
  bdg_comp_utils::readCommandLineArguments<T>(
      argc, argv, simParams::fermiSurfaceLabel, simParams::GRAIN_DIAGONAL,
      simParams::GRAIN_CORNER_SIZE);

  // Check that fermiSurfaceLabel has a proper value
  if (simParams::fermiSurfaceLabel < 1 || simParams::fermiSurfaceLabel > 3) {
    std::cout << "ERROR! Illegal value for simParams::fermiSurfaceLabel: "
              << simParams::fermiSurfaceLabel << std::endl;
    return 0;
  }

  // Check that grain sizes are valid
  if (simParams::GRAIN_CORNER_SIZE >
      (simParams::GRAIN_DIAGONAL * static_cast<T>(std::sqrt(2.0)))) {
    std::cout << "ERROR! GRAIN_CORNER_SIZE too large:"
              << simParams::GRAIN_CORNER_SIZE << std::endl;
    return 0;
  }
  if (simParams::GRAIN_CORNER_SIZE < static_cast<T>(0.0)) {
    std::cout << "ERROR! GRAIN_CORNER_SIZE negative: "
              << simParams::GRAIN_CORNER_SIZE << std::endl;
    return 0;
  }

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
  // Set filename
  sprintf(dataPath, "%sfs%d_R%2.3f_r%2.3f_T%2.4f/", simParams::DATA_BASE_PATH,
          simParams::fermiSurfaceLabel, simParams::GRAIN_DIAGONAL,
          simParams::GRAIN_CORNER_SIZE, simParams::TEMPERATURE);
  // Create data path if it does not exist
  bdg_comp_utils::createProjectDirectory(dataPath);

  //***************************************************
  // CREATE GEOMETRY
  //***************************************************
  // Create geometry group, add polygon and remove corners
  GeometryGroup<T> *geometry = new GeometryGroup<T>(context);
  const int numVertices = 5;
  const T outerEdgeCoordinate = static_cast<T>(1.0);
  const T diagonalCornerCoordinate =
      static_cast<T>(1.0) - static_cast<T>(2.0) * simParams::GRAIN_CORNER_SIZE /
                                simParams::GRAIN_SIDE_LENGTH;

  // DIAGONAL DOES NOT GO THROUGH THE CENTER
  // Vertex coords: start w. top-right corner (counter-clockwise order)
  const T xVertices[numVertices] = {
      outerEdgeCoordinate, -outerEdgeCoordinate, -outerEdgeCoordinate,
      -diagonalCornerCoordinate, outerEdgeCoordinate};
  const T yVertices[numVertices] = {outerEdgeCoordinate, outerEdgeCoordinate,
                                    -outerEdgeCoordinate, -outerEdgeCoordinate,
                                    diagonalCornerCoordinate};

  geometry->add(new PolygonGeometry<T>(numVertices, xVertices, yVertices),
                Type::add);

  // Horizontal distance to pairbreaking edge from the middle of the grain
  const T pbEdgeOffset =
      simParams::GRAIN_CORNER_SIZE / static_cast<T>(std::sqrt(2.0));
  // Convert distance to coordinates
  const T pbEdgeCoordinate =
      pbEdgeOffset / (simParams::GRAIN_SIDE_LENGTH * static_cast<T>(0.5));
  // Coordinates to the midpoint of the pairbreaking edge
  const T pbMidPointX = pbEdgeCoordinate;
  const T pbMidPointY = -pbEdgeCoordinate;
  // Rotation of pairbreaking edge with respect to the screen
  const T pbEdgeRotation = -static_cast<T>(0.25 * M_PI);

  //***************************************************
  // CHOOSE RICCATI SOLVER AND BOUNDARY CONDITIONS
  //***************************************************
  // RiccatiSolverConfined: provides analytic solutions to the Riccati equations
  // for a piecewise constant order parameter Gauge choice:
  // VectorPotential::Radial used to have a weak external field for seeding the
  // phase
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
  // Period of phase crystal, converted to coordinate units. In coherence
  // lengths, it is typically 2*pi coherence lengths in QC simulations (see the
  // Nat. Comm. and PRR papers).
  const T startPhasePeriod =
      static_cast<T>(4.0 * M_PI) / simParams::GRAIN_SIDE_LENGTH;
  // Decay length of phase crystal, converted to coordinate units.
  const T startPhaseDecay = static_cast<T>(1.0) / simParams::GRAIN_SIDE_LENGTH;

  Initial<T>::edgePhaseCrystal(orderParameterComponent, startPhasePeriod,
                               startPhaseDecay, pbMidPointX, pbMidPointY,
                               pbEdgeRotation);
  // Add the initialized d-wave component to the total order parameter
  orderParameter->add(orderParameterComponent);

  //***************************************************
  // CHOOSE QUANTITIES TO CALCULATE
  //***************************************************
  // Create compute object(s) for observable(s)
  ComputeOrderParameter<T> *computeOrderParameter =
      new ComputeOrderParameter<T>(context);
  // Current compute object
  new ComputeCurrent<T>(context);
  // Free energy compute object
  computeFreeEnergy = new ComputeFreeEnergy<T>(context);

  //***************************************************
  // CHOOSE CONVERGENCE ACCELERATOR
  //***************************************************
  OptimizerRollingBall<T> *optimizer = new OptimizerRollingBall<T>();
  computeOrderParameter->setOptimizer(optimizer);

  //***************************************************
  // CHOOSE FERMI SURFACE
  //***************************************************

  switch (simParams::fermiSurfaceLabel) {
  case 1: {
    new FermiSurfaceCircle<T>(context);
    break;
  }
  case 2: {
    const T hoppingNNN = static_cast<T>(-0.25);
    const T hoppingNNNN = static_cast<T>(0.0);
    const T chemicalPotential = static_cast<T>(0.0);
    new FermiSurfaceTightBinding<T>(context, hoppingNNN, hoppingNNNN,
                                    chemicalPotential);
    break;
  }
  case 3: {
    const T hoppingNNN = static_cast<T>(-0.477);
    const T hoppingNNNN = static_cast<T>(0.156);
    const T chemicalPotential = static_cast<T>(-1.267);
    new FermiSurfaceTightBinding<T>(context, hoppingNNN, hoppingNNNN,
                                    chemicalPotential);
    break;
  }
  default:
    std::cout << "ERROR! Illegal value for simParams::fermiSurfaceLabel:"
              << simParams::fermiSurfaceLabel << std::endl;
    cleanup();
    return 0;
  }

  //***************************************************
  // INITIALIZE SIMULATION
  //***************************************************
  context->initialize();

  //***************************************************
  // SAVE PARAMETERS TO FILE
  //***************************************************
  bdg_comp_utils::saveParamsMetaData<T>(
      context, dataPath, simParams::GRAIN_SIDE_LENGTH,
      simParams::GRAIN_CORNER_SIZE, simParams::GRAIN_DIAGONAL);

  //***************************************************
  // RUN SIMULATION UNTIL END CRITERIA
  //***************************************************
  if (simParams::RUN_VISUAL) {
    //***********************************************
    // SET UP OPENGL VISUALIZATION
    //***********************************************
    // Number of plots in visualization = NUM_PLOTS_Y * NUM_PLOTS_X
    const int NUM_PLOTS_Y = 1;
    const int NUM_PLOTS_X = 3;

    // Create visualization object
    gui =
        new Visualization<T>(orderParameterComponent->getOrderParameterField(),
                             NUM_PLOTS_X, NUM_PLOTS_Y);
    gui->setGeometry(geometry);
    gui->setPrintParameters(parameters);

    initGL(argc, argv, gui->getNumPixelsWindowX(), gui->getNumPixelsWindowY());
    glutKeyboardFunc(myKeyboardFunc);
    glutMainLoop();
  } else {
    // Run in terminal mode
    runInTerminal();
  }

  std::cout << "Successful termination!\n";
  return 0;
}

// Run a single kernel iteration.
void runIteration() {
  // Run kernel, solving equations for all compute objects
  context->compute();

  // Output iteration status to terminal
  bdg_comp_utils::printIterationStatus<T>(context);

  // Plot data in OpenGL
  if (simParams::RUN_VISUAL) {
    bdg_comp_utils::plotData<T>(context, gui);
  }
}

// Cleanup of context and visualization objects.
// Automatically called by OpenGL upon its completion.
void cleanup() {
  std::cout << "Cleaning up...\n";

  delete gui;
  std::cout << "Cleaned visualization object.\n";
  delete context;
  std::cout << "Cleaned context object.\n";

  std::cout << "Done cleaning up...\n";
}

void saveDataWrapper() {
  // Save order parameter and current
  bdg_comp_utils::saveGridData<T>(context, dataPath);
  // Update meta data file
  bdg_comp_utils::saveParamsMetaData<T>(
      context, dataPath, simParams::GRAIN_SIDE_LENGTH,
      simParams::GRAIN_CORNER_SIZE, simParams::GRAIN_DIAGONAL);
  // Save induced magnetic field
  bdg_comp_utils::saveInducedFluxDensity<T>(context, dataPath);
  // Save free energy and iteration status
  bdg_comp_utils::saveFreeEnergy<T>(
      context, dataPath, computeFreeEnergy, simParams::GRAIN_SIDE_LENGTH,
      simParams::GRAIN_CORNER_SIZE, simParams::GRAIN_DIAGONAL);
}

// Wrapper for running program.
// Return true if program is done (converged or reach maximum number of
// iterations).
bool runWrapper() {
  // Run single kernel iteration
  runIteration();

  // Check for convergence or maximum number of iterations (can add user
  // termination through OpenGL hook - check myKeyboardFunc)
  if (((context->isConverged() ||
        context->getNumIterations() > simParams::NUM_ITERATIONS_MAX) &&
       (context->getNumIterations() > simParams::NUM_ITERATIONS_MIN))) {
    std::cout << "Simulation finished." << std::endl;

    // Save data to file
    saveDataWrapper();

    return true;
  } else if (gUserQuit) {
    return true;
  } else {
    // Check if user has queued save data, or it is time for regular data save
    if (gUserSaveData ||
        ((!(context->getNumIterations() % simParams::SAVE_DATA_INTERVAL)) &&
         (context->getNumIterations() > 0))) {

      // Save data to file
      saveDataWrapper();

      // Reset save data key stroke
      gUserSaveData = false;
    }
  }

  return false;
}

// OpenGL display loop
void display() {
  // Run one iteration
  bool allConverged = runWrapper();

  if (allConverged) {
    exit(0);
  }
}

// Terminal mode wrapper
void runInTerminal() {
  bool allConverged = false;

  while (!allConverged) {
    // Run one iteration
    allConverged = runWrapper();
  }

  // Clean up context and visualization
  cleanup();
}

// Detect keypresses in OpenGL instance - useful for changing parameters live,
// saving data or quitting the program
void myKeyboardFunc(const unsigned char inKey, const int inX, const int inY) {
  std::cout << "key pressed: " << inKey << ", at (" << inX << ", " << inY
            << ")\n";
  if (inKey == 'q' || inKey == 'Q') {
    gUserQuit = true;
  } else if (inKey == 's' || inKey == 'S') {
    gUserSaveData = true;
  }
}
