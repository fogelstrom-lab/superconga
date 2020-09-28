//=== bdg_comparison_main.cu - Stabilize phase crystal with a circular FS. ===//
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
#include "ComputeCurrent.h"    // Compute the total charge current density
#include "ComputeFreeEnergy.h" // Compute the (Eilenberger) free energy
#include "ComputeLDOS.h"
#include "ComputeOrderParameter.h" // Compute order parameter
#include "Context.h" // The context that connects all classes (helpful since enacpsulation is used)
#include "FermiSurface.h"
#include "FermiSurfaceCircle.h"       // Standard circular Fermi surface
#include "FermiSurfaceTightBinding.h" // Tight binding Fermi surface
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
#include "run_utils.h" // Commonly used functions in run files

#include "bdg_comparison_utils.h" // Common utils for this project

#include <cstdlib>
#include <cuda.h>
#include <iostream>

//*******************************************************
// FORWARD DECLARATIONS
//*******************************************************
void display();
void saveDataWrapper();
void postProcessing();

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
scg::Context<T> *context = new scg::Context<T>();
// Visualization object
scg::Visualization<T> *gui;
// Free energy object
scg::ComputeFreeEnergy<T> *computeFreeEnergy;
// Used to construct data file path
char dataPath[512];
} // namespace

//*******************************************************
// SIMULATION PARAMETERS
//*******************************************************
// TODO(patric): Make into a struct
namespace simParams {
// Run in OpenGL mode or terminal mode.
const bool RUN_VISUAL = false;
// Print iteration status to terminal
const bool PRINT_ITERATION_STATUS = true;
// Run metastable state without current-loops (real order parameter).
bool RUN_METASTABLE = false;
// Load the order parameter from file.
const bool LOAD_ORDER_PARAMETER = false;
// Whether to compute the order parameter or just go to post-processing.
const bool COMPUTE_ORDER_PARAMETER = true;
// Whether to compute the LDOS as post-processing.
const bool COMPUTE_LDOS = true;

// Temperature in units of T_c
T TEMPERATURE = static_cast<T>(0.1);
// Size of pairbreaking edge in coherence lengths hbar*v_F/(2*pi*k_BT_c)
T GRAIN_DIAGONAL = static_cast<T>(60.0);
// Size (in coherence lengths) that defines the grain geometry: 0 gives a
// triangle, >0 gives an irregular pentagon
T GRAIN_CORNER_SIZE = static_cast<T>(4.0);
const T GRAIN_SIDE_LENGTH =
    GRAIN_CORNER_SIZE + GRAIN_DIAGONAL / static_cast<T>(std::sqrt(2.0));
// Number of lattice points per grain side length
const int GRID_RESOLUTION = 801;
// Number of discrete trajectory angles (on the Fermi-surface)
const int NUM_ANGLES = 128;
// Number of discrete energies (poles)
const int NUM_ENERGIES = 20;
// Convergence criterion for order-parameter residual
const T CONVERGENCE_CRITERION = static_cast<T>(1E-6);
// Minimum number of iterations
const int NUM_ITERATIONS_MIN = 300;
// Maximum number of iterations
const int NUM_ITERATIONS_MAX = 5000;
// Path to data, relative to build directory!
const char *DATA_BASE_PATH = (char *)"data/qc/";
// Whether or not to save data.
const bool SAVE_DATA = true;
// Number of iterations between regular saving of data
const int SAVE_DATA_INTERVAL = 200;
// Label which fermi surface it is:
// 1 - Circular
// 2 - Tight Binding: NNN= -0.25, NNNN=0, mu=0
// 3 - Tight Binding: NNN= -0.495, NNNN=0.156, mu=-1.267
int fermiSurfaceLabel = 1;

// ================= LDOS ================= //
const int NUM_LDOS_ANGLES = 5 * 360;
const int NUM_ENERGIES_PER_BLOCK = 8;
const T ENERGY_MIN = static_cast<T>(0.0);
const T ENERGY_MAX = static_cast<T>(3.0);
const T ENERGY_SMEARING = static_cast<T>(0.05);
const int NUM_LDOS_ENERGIES = 512;
const int NUM_LDOS_ITERATIONS_MAX = 1000;
const T LDOS_CONVERGENCE_CRITERION = static_cast<T>(1E-3);
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
      simParams::GRAIN_CORNER_SIZE, simParams::TEMPERATURE,
      simParams::RUN_METASTABLE);

  // Check that fermiSurfaceLabel has a proper value
  if (simParams::fermiSurfaceLabel < 1 || simParams::fermiSurfaceLabel > 3) {
    std::cerr << "ERROR! Illegal value for simParams::fermiSurfaceLabel: "
              << simParams::fermiSurfaceLabel << std::endl;
    scg::run_utils::cleanup(context, gui);
    return 1;
  }

  // Check that grain sizes are valid
  if (simParams::GRAIN_CORNER_SIZE >
      (simParams::GRAIN_DIAGONAL * static_cast<T>(std::sqrt(2.0)))) {
    std::cerr << "ERROR! GRAIN_CORNER_SIZE too large:"
              << simParams::GRAIN_CORNER_SIZE << std::endl;
    scg::run_utils::cleanup(context, gui);
    return 1;
  }
  if (simParams::GRAIN_CORNER_SIZE < static_cast<T>(0.0)) {
    std::cerr << "ERROR! GRAIN_CORNER_SIZE negative: "
              << simParams::GRAIN_CORNER_SIZE << std::endl;
    scg::run_utils::cleanup(context, gui);
    return 1;
  }

  // Storage for system simulation parameters (e.g. temperature, lattice size)
  scg::Parameters<T> *parameters = new scg::Parameters<T>(context);
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
  if (simParams::RUN_METASTABLE) {
    sprintf(dataPath, "%sfs%d_R%2.3f_r%2.3f_T%2.4f_ms/",
            simParams::DATA_BASE_PATH, simParams::fermiSurfaceLabel,
            simParams::GRAIN_DIAGONAL, simParams::GRAIN_CORNER_SIZE,
            simParams::TEMPERATURE);
  } else {
    sprintf(dataPath, "%sfs%d_R%2.3f_r%2.3f_T%2.4f/", simParams::DATA_BASE_PATH,
            simParams::fermiSurfaceLabel, simParams::GRAIN_DIAGONAL,
            simParams::GRAIN_CORNER_SIZE, simParams::TEMPERATURE);
  }
  // Create data path if it does not exist
  scg::run_utils::createDirectory(dataPath);

  //***************************************************
  // CREATE GEOMETRY
  //***************************************************
  // Create geometry group, add polygon and remove corners
  scg::GeometryGroup<T> *geometry = new scg::GeometryGroup<T>(context);
  const int numVertices = 5;
  // The coordinate system is in the range:
  // [-scg::constants::MAXIMUM_COORDINATE,scg::constants::MAXIMUM_COORDINATE].
  const T outerEdgeCoordinate =
      static_cast<T>(scg::constants::MAXIMUM_COORDINATE);
  const T diagonalCornerCoordinate =
      outerEdgeCoordinate - static_cast<T>(2.0) * outerEdgeCoordinate *
                                simParams::GRAIN_CORNER_SIZE /
                                simParams::GRAIN_SIDE_LENGTH;

  // DIAGONAL DOES NOT GO THROUGH THE CENTER
  // Vertex coords: start w. top-right corner (counter-clockwise order)
  const T xVertices[numVertices] = {
      outerEdgeCoordinate, -outerEdgeCoordinate, -outerEdgeCoordinate,
      -diagonalCornerCoordinate, outerEdgeCoordinate};
  const T yVertices[numVertices] = {outerEdgeCoordinate, outerEdgeCoordinate,
                                    -outerEdgeCoordinate, -outerEdgeCoordinate,
                                    diagonalCornerCoordinate};

  geometry->add(new scg::PolygonGeometry<T>(numVertices, xVertices, yVertices),
                scg::Type::add);

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
  scg::RiccatiSolver<T> *riccatiSolver =
      new scg::RiccatiSolverConfined<T, scg::Gamma::General,
                                     scg::VectorPotential::None>(context);
  // Add specular BC
  riccatiSolver->add(new scg::BoundaryConditionSpecular<T>);

  //***************************************************
  // SET UP ORDER PARAMETER
  //***************************************************
  // Create order parameter (which might contain multiple components, i.e. d+is)
  scg::OrderParameter<T> *orderParameter = new scg::OrderParameter<T>(context);
  // Add a d-wave component to the total order-parameter
  scg::OrderParameterComponent<T> *orderParameterComponent =
      new scg::OrderParameterComponent<T>(
          scg::BasisFunctionSymmetry::DWAVE_X2_Y2, parameters);
  // Period of phase crystal, converted to coordinate units. In coherence
  // lengths, it is typically 2*pi coherence lengths in QC simulations (see the
  // Nat. Comm. and PRR papers).
  const T startPhasePeriod =
      static_cast<T>(4.0 * M_PI) / simParams::GRAIN_SIDE_LENGTH;
  // Decay length of phase crystal, converted to coordinate units.
  const T startPhaseDecay = static_cast<T>(1.0) / simParams::GRAIN_SIDE_LENGTH;

  // Run with or without current loops:
  // Useful when comparing free energy to find T^*.
  if (simParams::LOAD_ORDER_PARAMETER) {
    scg::Grid<T>::GPU orderParameterLoaded;
    // Read orderparameter from file.
    char orderParameterFile[512];
    sprintf(orderParameterFile, "%sorder_parameter.grd", dataPath);
    if (orderParameterLoaded.read(orderParameterFile, parameters)) {
      std::cerr << "Error reading file: " << orderParameterFile << std::endl;
      scg::run_utils::cleanup(context, gui);
      return 1;
    }
    std::cout << "Initialize the order parameter from file." << std::endl;
    scg::Initial<T>::grid(orderParameterComponent, orderParameterLoaded);
  } else if (simParams::RUN_METASTABLE) {
    // This will set the imagniary part to zero, and it will not change during
    // the simulation.
    std::cout << "Initialize the order parameter with constant bulk value."
              << std::endl;
    scg::Initial<T>::bulk(orderParameterComponent);
  } else {
    scg::Initial<T>::edgePhaseCrystal(orderParameterComponent, startPhasePeriod,
                                      startPhaseDecay, pbMidPointX, pbMidPointY,
                                      pbEdgeRotation);
  }

  // Add the initialized d-wave component to the total order parameter
  orderParameter->add(orderParameterComponent);

  //***************************************************
  // CHOOSE QUANTITIES TO CALCULATE
  //***************************************************
  // Create compute object(s) for observable(s)
  scg::ComputeOrderParameter<T> *computeOrderParameter =
      new scg::ComputeOrderParameter<T>(context);
  // Current compute object
  new scg::ComputeCurrent<T>(context);
  // Free energy compute object
  computeFreeEnergy = new scg::ComputeFreeEnergy<T>(context);

  //***************************************************
  // CHOOSE CONVERGENCE ACCELERATOR
  //***************************************************
  scg::OptimizerRollingBall<T> *optimizer = new scg::OptimizerRollingBall<T>();
  computeOrderParameter->setOptimizer(optimizer);

  //***************************************************
  // CHOOSE FERMI SURFACE
  //***************************************************

  scg::FermiSurface<T> *fermiSurface;
  switch (simParams::fermiSurfaceLabel) {
  case 1: {
    fermiSurface = new scg::FermiSurfaceCircle<T>();
    break;
  }
  case 2: {
    const T hoppingNNN = static_cast<T>(-0.25);
    const T hoppingNNNN = static_cast<T>(0.0);
    const T chemicalPotential = static_cast<T>(0.0);
    fermiSurface = new scg::FermiSurfaceTightBinding<T>(hoppingNNN, hoppingNNNN,
                                                        chemicalPotential);
    break;
  }
  case 3: {
    const T hoppingNNN = static_cast<T>(-0.495);
    const T hoppingNNNN = static_cast<T>(0.156);
    const T chemicalPotential = static_cast<T>(-1.267);
    fermiSurface = new scg::FermiSurfaceTightBinding<T>(hoppingNNN, hoppingNNNN,
                                                        chemicalPotential);
    break;
  }
  default:
    std::cerr << "ERROR! Illegal value for simParams::fermiSurfaceLabel:"
              << simParams::fermiSurfaceLabel << std::endl;
    scg::run_utils::cleanup(context, gui);
    return 1;
  }
  context->set(fermiSurface);

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

  //***********************************************
  // SET UP OPENGL VISUALIZATION
  //***********************************************
  if (simParams::RUN_VISUAL) {
    const bool plotOrderParameter = true;
    const bool plotCurentDensity = true;
    gui = scg::run_utils::createVisualization(context, plotOrderParameter,
                                              plotCurentDensity);
    scg::initGL(argc, argv, gui->getNumPixelsWindowX(),
                gui->getNumPixelsWindowY());
    glutKeyboardFunc(scg::run_utils::myKeyboardFunc);
    glutDisplayFunc(display);
  }

  //***************************************************
  // RUN SIMULATION UNTIL END CRITERIA
  //***************************************************
  if (simParams::COMPUTE_ORDER_PARAMETER) {
    scg::run_utils::Status status = scg::run_utils::Status::kContinue;
    while (status != scg::run_utils::Status::kQuit) {
      // Run one iteration,
      status = scg::run_utils::runIteration(
          context, simParams::PRINT_ITERATION_STATUS,
          simParams::NUM_ITERATIONS_MIN, simParams::NUM_ITERATIONS_MAX,
          simParams::SAVE_DATA, simParams::SAVE_DATA_INTERVAL);

      if (status == scg::run_utils::Status::kSave) {
        saveDataWrapper();
      }

      if (simParams::RUN_VISUAL) {
        // Check if the Glut window is open.
        const int windowID = glutGetWindow();
        const bool openWindow = windowID > 0;

        // Plot data in OpenGL.
        if (openWindow) {
          glutMainLoopEvent();
        } else {
          // The user closed the window so lets abort.
          break;
        }
      }
    }

    // Save at the very end.
    saveDataWrapper();
  }

  postProcessing();

  scg::run_utils::cleanup(context, gui);
  return 0;
}

// Plot data in OpenGL GUI.
void display() {
  const bool plotOrderParameter = true;
  const bool plotCurentDensity = true;
  scg::run_utils::display(context, gui, plotOrderParameter, plotCurentDensity);

  // Tell Glut to update window.
  glutPostRedisplay();
}

void postProcessing() {
  if (simParams::COMPUTE_LDOS) {
    // Delete all compute objects, e.g. current, free energy.
    context->deleteComputeObjects();

    // Set new parameters relevant for a Keldysh integration iterator.
    context->getParameters()->setEnergyRange(
        simParams::ENERGY_MIN, simParams::ENERGY_MAX,
        simParams::NUM_LDOS_ENERGIES, simParams::ENERGY_SMEARING);

    // The LDOS needs more angles compared to computing the order parameter.
    context->getParameters()->setAngularResolution(simParams::NUM_LDOS_ANGLES);

    context->getParameters()->setConvergenceCriterion(
        simParams::LDOS_CONVERGENCE_CRITERION);

    // Add LDOS compute object.
    new scg::ComputeLDOS<T>(context);

    // Re-initialize everything.
    context->initialize(simParams::NUM_ENERGIES_PER_BLOCK);

    // Compute the LDOS.
    for (int i = 0; i < simParams::NUM_LDOS_ITERATIONS_MAX; ++i) {
      context->compute();

      // Output iteration status to terminal.
      scg::run_utils::printIterationStatus(context);

      const bool isConverged = context->isConverged();
      if (isConverged) {
        break;
      }
    }

    // Save the LDOS.
    bdg_comp_utils::saveParametersLDOS(context, dataPath);
    bdg_comp_utils::saveDataLDOS(context, dataPath);
  }
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
