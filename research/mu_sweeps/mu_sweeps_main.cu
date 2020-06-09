//=== mu_sweeps_main.cu - Phase diagram with subdominant order. ===//
//
// Part of the SuprConGPU Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Run with both d-wave and extended s-wave order parameter. The relative T_c
/// is extracted from tight-binding calculations. Sweep mu and establish the
/// phase diagram for phase crystal vs regular superconductivity vs
/// translational invariant superflow due to the extended s-wave.
///
//===----------------------------------------------------------------------===//
#include "BoundaryConditionSpecular.h" // Boundary condition for perfect specular reflection
#include "ComputeCurrent.h"        // Compute the total charge current density
#include "ComputeFreeEnergy.h"     // Compute the (Eilenberger) free energy
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
#include "bdg_utils.h"
#include "dispersion.h"
#include "run_utils.h" // Commonly used functions in run files

#include "mu_sweeps_utils.h" // Common utils for this project

#include <cstdlib>
#include <cuda.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <utility>

//*******************************************************
// TYPEDEFS
//*******************************************************
// Run with double or single floating point precision, double is highly
// recommended
typedef double T;

//*******************************************************
// FORWARD DECLARATIONS
//*******************************************************
void display();
void saveDataWrapper(const T inGrainSideLength, const T inGrainCornerLength,
                     const T ingrainDiagonalLength);

//*******************************************************
// GLOBAL OBJECTS: glut does not accept arguments and relies on global/static
// variables.
//*******************************************************
namespace {
// Context object
Context<T> *context = new Context<T>();
// Visualization object
Visualization<T> *gui;
// Used to construct data file path
char dataPath[512];

// Fermi surface parameters.
const std::map<std::string, T> FS1 = {{"hoppingNNN", static_cast<T>(-0.25)},
                                      {"hoppingNNNN", static_cast<T>(0.0)},
                                      {"potential", static_cast<T>(1.4)}};
const std::map<std::string, T> FS2 = {{"hoppingNNN", static_cast<T>(-0.495)},
                                      {"hoppingNNNN", static_cast<T>(0.156)},
                                      {"potential", static_cast<T>(1.279)}};
} // namespace

//*******************************************************
// MAIN
//*******************************************************
int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Please provide config file.\n";
    exit(1);
  }
  const std::string filename = argv[1];

  // Parse JSON parameter file.
  nlohmann::json simParams;
  std::ifstream ifile(filename);
  ifile >> simParams;

  //***************************************************
  // APPLY PARAMETERS
  //***************************************************
  // Define some nice consts: don't want to write simParams getters everywhere
  const T grainDiagonalLength = simParams["grain_diagonal_length"].get<T>();
  const T grainCornerLength = simParams["grain_corner_length"].get<T>();
  // Calculate sidelength
  const T grainSideLength =
      grainCornerLength + grainDiagonalLength / static_cast<T>(std::sqrt(2.0));

  // Check that grain sizes are valid
  if (grainCornerLength >
      (grainCornerLength * static_cast<T>(std::sqrt(2.0)))) {
    std::cout << "ERROR! GRAIN_CORNER_SIZE too large:" << grainCornerLength
              << std::endl;
    return 1;
  }
  if (grainCornerLength < static_cast<T>(0.0)) {
    std::cout << "ERROR! GRAIN_CORNER_SIZE negative: " << grainCornerLength
              << std::endl;
    return 1;
  }

  Parameters<T> *parameters = new Parameters<T>(context);
  // Set temperature.
  parameters->setTemperature(simParams["temperature"].get<T>());
  // Set grain sidelength in coherence lengths: hbar*vF/(2*pi*kB*Tc).
  parameters->setGrainWidthCoherenceLengths(grainSideLength);
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

  // Set filename
  if (simParams["meta_stable"].get<bool>()) {
    snprintf(dataPath, sizeof(dataPath),
             "%s/fs%d_mu%2.4f/R%2.3f_r%2.3f_T%2.4f_ms",
             simParams["data_path"].get<std::string>().c_str(),
             simParams["fermi_surface_label"].get<int>(),
             simParams["chemical_potential"].get<T>(), grainCornerLength,
             grainCornerLength, simParams["temperature"].get<T>());
  } else {
    snprintf(dataPath, sizeof(dataPath),
             "%s/fs%d_mu%2.4f/_R%2.3f_r%2.3f_T%2.4f",
             simParams["data_path"].get<std::string>().c_str(),
             simParams["fermi_surface_label"].get<int>(),
             simParams["chemical_potential"].get<T>(), grainCornerLength,
             grainCornerLength, simParams["temperature"].get<T>());
  }

  // Create data path if it does not exist
  run_utils::createDirectory(dataPath);

  //***************************************************
  // CREATE GEOMETRY
  //***************************************************
  // Create geometry group, add polygon and remove corners
  GeometryGroup<T> *geometry = new GeometryGroup<T>(context);
  const int numVertices = 5;
  // The coordinate system is in the range:
  // [-qcs::MAXIMUM_COORDINATE,qcs::MAXIMUM_COORDINATE].
  const T outerEdgeCoordinate = static_cast<T>(qcs::MAXIMUM_COORDINATE);
  const T diagonalCornerCoordinate =
      outerEdgeCoordinate - static_cast<T>(2.0) * outerEdgeCoordinate *
                                grainCornerLength / grainSideLength;

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
  const T pbEdgeOffset = grainCornerLength / static_cast<T>(std::sqrt(2.0));
  // Convert distance to coordinates
  const T pbEdgeCoordinate =
      pbEdgeOffset / (grainSideLength * static_cast<T>(0.5));
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
  // SET DISPERSION
  //***************************************************
  const bool isFS1 = simParams["fermi_surface_label"].get<int>() == 1;
  const auto fermiSurfaceParams = isFS1 ? FS1 : FS2;
  const DispersionTightBinding<T> dispersion = {
      fermiSurfaceParams.at("hoppingNNN"), fermiSurfaceParams.at("hoppingNNNN"),
      simParams["chemical_potential"].get<T>()};

  //***************************************************
  // SET FERMI SURFACE
  //***************************************************
  FermiSurface<T> *fermiSurface = new FermiSurfaceTightBinding<T>(dispersion);
  context->set(fermiSurface);

  //***************************************************
  // SET UP ORDER PARAMETER
  //***************************************************
  // Create order parameter (which might contain multiple components, i.e. d+is)
  OrderParameter<T> *orderParameter = new OrderParameter<T>(context);

  // Compute the critical temperatures.
  // D-wave.
  const std::pair<bool, T> dWaveTmp = bdg::computeCriticalTemperature(
      dispersion, fermiSurfaceParams.at("potential"),
      BasisFunctionSymmetry::DWAVE_X2_Y2);
  const bool dWaveSuccess = std::get<0>(dWaveTmp);
  const T dWaveTcRaw = std::get<1>(dWaveTmp);
  std::cout << "Tc: d = " << dWaveTcRaw << std::endl;

  // S-wave.
  const std::pair<bool, T> sWaveTmp = bdg::computeCriticalTemperature(
      dispersion, fermiSurfaceParams.at("potential"),
      BasisFunctionSymmetry::SWAVE_EXT);
  const bool sWaveSuccess = std::get<0>(sWaveTmp);
  const T sWaveTcRaw = std::get<1>(sWaveTmp);
  std::cout << "Tc: s = " << sWaveTcRaw << std::endl;

  const T dWaveTc = dWaveTcRaw / std::max(dWaveTcRaw, sWaveTcRaw);
  const T sWaveTc = sWaveTcRaw / std::max(dWaveTcRaw, sWaveTcRaw);
  std::cout << "Tc: d = " << dWaveTc << " : s = " << sWaveTc << std::endl;

  if (!(dWaveSuccess && sWaveSuccess)) {
    run_utils::cleanup(context, gui);
    return 1;
  }

  // Add the initialized components to the total order parameter
  const T TcLimit = static_cast<T>(1e-3);

  if (dWaveTc > TcLimit) {
    std::cout << "Adding d-wave component:" << std::endl;
    std::cout << "Tc: raw = " << dWaveTcRaw << " : scaled = " << dWaveTc
              << std::endl;
    // Add a d-wave component to the total order-parameter
    OrderParameterComponent<T> *orderParameterComponentD =
        new OrderParameterComponent<T>(BasisFunctionSymmetry::DWAVE_X2_Y2,
                                       parameters, dWaveTc);

    // Period of phase crystal, converted to coordinate units. In coherence
    // lengths, it is typically 2*pi coherence lengths in QC simulations (see
    // the Nat. Comm. and PRR papers).
    const T startPhasePeriod = static_cast<T>(4.0 * M_PI) / grainSideLength;
    // Decay length of phase crystal, converted to coordinate units.
    const T startPhaseDecay = static_cast<T>(1.0) / grainSideLength;

    // Run with or without vortices: Useful when comparing free energy to find
    // T^*
    if (simParams["meta_stable"].get<bool>()) {
      Initial<T>::bulk(orderParameterComponentD);
    } else {
      Initial<T>::edgePhaseCrystal(orderParameterComponentD, startPhasePeriod,
                                   startPhaseDecay, pbMidPointX, pbMidPointY,
                                   pbEdgeRotation);
    }

    orderParameter->add(orderParameterComponentD);
  }

  if (sWaveTc > TcLimit) {
    std::cout << "Adding s-wave component:" << std::endl;
    std::cout << "Tc: raw = " << sWaveTc << " : scaled = " << sWaveTc
              << std::endl;
    OrderParameterComponent<T> *orderParameterComponentS =
        new OrderParameterComponent<T>(BasisFunctionSymmetry::SWAVE_EXT,
                                       parameters, sWaveTc);
    Initial<T>::bulk(orderParameterComponentS);
    orderParameter->add(orderParameterComponentS);
  }

  //***************************************************
  // CHOOSE QUANTITIES TO CALCULATE
  //***************************************************
  // Create compute object(s) for observable(s)
  ComputeOrderParameter<T> *computeOrderParameter =
      new ComputeOrderParameter<T>(context);
  // Current compute object
  new ComputeCurrent<T>(context);
  // Free energy compute object
  new ComputeFreeEnergy<T>(context);

  //***************************************************
  // CHOOSE CONVERGENCE ACCELERATOR
  //***************************************************
  OptimizerRollingBall<T> *optimizer = new OptimizerRollingBall<T>();
  computeOrderParameter->setOptimizer(optimizer);

  //***************************************************
  // INITIALIZE SIMULATION
  //***************************************************
  context->initialize();

  //***************************************************
  // SAVE PARAMETERS TO FILE
  //***************************************************
  mu_sweeps_utils::saveParamsMetaData(context, dataPath, grainSideLength,
                                      grainCornerLength, grainCornerLength);

  //***********************************************
  // SET UP OPENGL VISUALIZATION
  //***********************************************
  if (simParams["visualize"].get<bool>()) {
    const bool plotOrderParameter = true;
    const bool plotCurentDensity = true;
    gui = run_utils::createVisualization(context, plotOrderParameter,
                                         plotCurentDensity);
    initGL(argc, argv, gui->getNumPixelsWindowX(), gui->getNumPixelsWindowY());
    glutKeyboardFunc(run_utils::myKeyboardFunc);
    glutDisplayFunc(display);
  }

  //***************************************************
  // RUN SIMULATION UNTIL END CRITERIA
  //***************************************************
  run_utils::Status status = run_utils::Status::kContinue;
  while (status != run_utils::Status::kQuit) {
    // Run one iteration,
    status =
        run_utils::runIteration(context, simParams["verbose"].get<bool>(),
                                simParams["num_iterations_min"].get<int>(),
                                simParams["num_iterations_max"].get<int>(),
                                simParams["save_data"].get<bool>(),
                                simParams["save_data_interval"].get<int>());

    if (status == run_utils::Status::kSave) {
      saveDataWrapper(grainSideLength, grainCornerLength, grainDiagonalLength);
    }

    if (simParams["visualize"].get<bool>()) {
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
  saveDataWrapper(grainSideLength, grainCornerLength, grainDiagonalLength);

  run_utils::cleanup(context, gui);
  return 0;
}

// Plot data in OpenGL GUI.
void display() {
  const bool plotOrderParameter = true;
  const bool plotCurentDensity = true;
  run_utils::display(context, gui, plotOrderParameter, plotCurentDensity);

  // Tell Glut to update window.
  glutPostRedisplay();
}

void saveDataWrapper(const T inGrainSideLength, const T inGrainCornerLength,
                     const T ingrainDiagonalLength) {
  // Save order parameter and current
  mu_sweeps_utils::saveGridData(context, dataPath);
  // Update meta data file
  mu_sweeps_utils::saveParamsMetaData(context, dataPath, inGrainSideLength,
                                      inGrainCornerLength,
                                      ingrainDiagonalLength);
  // Save induced magnetic field
  mu_sweeps_utils::saveInducedFluxDensity(context, dataPath);
  // Save free energy and iteration status
  mu_sweeps_utils::saveFreeEnergy(context, dataPath, inGrainSideLength,
                                  inGrainCornerLength, ingrainDiagonalLength);
}
