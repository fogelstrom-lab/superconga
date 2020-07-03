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
                     const T inGrainDiagonalLength, const T inDWaveTc,
                     const T inSWaveTc, const bool inTcSucces);

//*******************************************************
// GLOBAL OBJECTS: glut does not accept arguments and relies on global/static
// variables.
//*******************************************************
namespace {
// Context object
scg::Context<T> *context = new scg::Context<T>();
// Visualization object
scg::Visualization<T> *gui;
// Used to construct data file path
char DATA_PATH[512];

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
    scg::run_utils::cleanup(context, gui);
    return 1;
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
    scg::run_utils::cleanup(context, gui);
    return 1;
  }
  if (grainCornerLength < static_cast<T>(0.0)) {
    std::cout << "ERROR! GRAIN_CORNER_SIZE negative: " << grainCornerLength
              << std::endl;
    scg::run_utils::cleanup(context, gui);
    return 1;
  }

  scg::Parameters<T> *parameters = new scg::Parameters<T>(context);
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
  snprintf(DATA_PATH, sizeof(DATA_PATH),
           "%s/fs%d_mu%2.4f_sl%d/R%2.3f_r%2.3f_T%2.4f",
           simParams["data_path"].get<std::string>().c_str(),
           simParams["fermi_surface_label"].get<int>(),
           simParams["chemical_potential"].get<T>(),
           simParams["start_guess_label"].get<int>(), grainCornerLength,
           grainCornerLength, simParams["temperature"].get<T>());

  // Create data path if it does not exist
  scg::run_utils::createDirectory(DATA_PATH);

  //***************************************************
  // CREATE GEOMETRY
  //***************************************************
  // Create geometry group, add polygon and remove corners
  scg::GeometryGroup<T> *geometry = new scg::GeometryGroup<T>(context);
  const int numVertices = 5;
  // The coordinate system is in the range:
  // [-MAXIMUM_COORDINATE, MAXIMUM_COORDINATE].
  const T outerEdgeCoordinate =
      static_cast<T>(scg::constants::MAXIMUM_COORDINATE);
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

  geometry->add(new scg::PolygonGeometry<T>(numVertices, xVertices, yVertices),
                scg::Type::add);

  // Horizontal distance to pairbreaking edge from the middle of the grain
  const T pbEdgeOffset = grainCornerLength / static_cast<T>(std::sqrt(2.0));
  // Convert distance to coordinates
  const T pbEdgeCoordinate =
      pbEdgeOffset / (grainSideLength * static_cast<T>(0.5));

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
  // SET DISPERSION
  //***************************************************
  const bool isFS1 = simParams["fermi_surface_label"].get<int>() == 1;
  const auto fermiSurfaceParams = isFS1 ? FS1 : FS2;
  const scg::DispersionTightBinding<T> dispersion = {
      fermiSurfaceParams.at("hoppingNNN"), fermiSurfaceParams.at("hoppingNNNN"),
      simParams["chemical_potential"].get<T>()};

  //***************************************************
  // SET FERMI SURFACE
  //***************************************************
  scg::FermiSurface<T> *fermiSurface =
      new scg::FermiSurfaceTightBinding<T>(dispersion);
  context->set(fermiSurface);

  //***************************************************
  // SET UP ORDER PARAMETER
  //***************************************************
  // Create order parameter (which might contain multiple components, i.e. d+is)
  scg::OrderParameter<T> *orderParameter = new scg::OrderParameter<T>(context);

  // Compute the critical temperatures.
  // D-wave.
  const std::pair<bool, T> dWaveTmp = scg::bdg::computeCriticalTemperature(
      dispersion, fermiSurfaceParams.at("potential"),
      scg::BasisFunctionSymmetry::DWAVE_X2_Y2);
  const bool dWaveSuccess = std::get<0>(dWaveTmp);
  const T dWaveTcRaw = std::get<1>(dWaveTmp);
  std::cout << "Tc: d = " << dWaveTcRaw << std::endl;

  // S-wave.
  const std::pair<bool, T> sWaveTmp = scg::bdg::computeCriticalTemperature(
      dispersion, fermiSurfaceParams.at("potential"),
      scg::BasisFunctionSymmetry::SWAVE_EXT);
  const bool sWaveSuccess = std::get<0>(sWaveTmp);
  const T sWaveTcRaw = std::get<1>(sWaveTmp);
  std::cout << "Tc: s = " << sWaveTcRaw << std::endl;

  const T dWaveTc = dWaveTcRaw / std::max(dWaveTcRaw, sWaveTcRaw);
  const T sWaveTc = sWaveTcRaw / std::max(dWaveTcRaw, sWaveTcRaw);
  std::cout << "Tc: d = " << dWaveTc << " : s = " << sWaveTc << std::endl;

  const bool TcSuccess = dWaveSuccess && sWaveSuccess;

  // Add the initialized components to the total order parameter
  const T TcLimit = static_cast<T>(1e-3);

  // Parameters for random initial guess
  const T startRandomPhaseMix = static_cast<T>(0.02);
  const T startRandomPhaseShift = static_cast<T>(0.25 * M_PI);

  if (dWaveTc > TcLimit) {
    std::cout << "Adding d-wave component:" << std::endl;
    std::cout << "Tc: raw = " << dWaveTcRaw << " : scaled = " << dWaveTc
              << std::endl;
    // Add a d-wave component to the total order-parameter
    scg::OrderParameterComponent<T> *orderParameterComponentD =
        new scg::OrderParameterComponent<T>(
            scg::BasisFunctionSymmetry::DWAVE_X2_Y2, parameters, dWaveTc);

    // Different start guesses: useful when comparing free energy to find T^*.
    // start guess label:
    // 1 - both real,
    // 2 - both random phase,
    // 3 - s-wave real, d-wave phase crystal (from converged file)
    // 4 - s-wave real, d-wave phase crystal guess
    switch (simParams["start_guess_label"].get<int>()) {
    case 1: {
      scg::Initial<T>::bulk(orderParameterComponentD);
      break;
    }
    case 2: {
      scg::Initial<T>::random(orderParameterComponentD, startRandomPhaseMix,
                              startRandomPhaseShift);
      break;
    }
    case 3: {
      // Read orderparameter from file.
      char orderParameterStartFile[512];
      snprintf(orderParameterStartFile, sizeof(orderParameterStartFile),
               "../../research/mu_sweeps/data/op_phase_crystal_fs%d_t0-1.grd",
               simParams["fermi_surface_label"].get<int>());
      scg::Initial<T>::read(orderParameterComponentD, orderParameterStartFile);
      break;
    }
    case 4: {
      // Period of phase crystal, converted to coordinate units. In coherence
      // lengths, it is typically 2*pi coherence lengths in QC simulations (see
      // the Nat. Comm. and PRR papers).
      const T startPhasePeriod = static_cast<T>(4.0 * M_PI) / grainSideLength;
      // Decay length of phase crystal, converted to coordinate units.
      const T startPhaseDecay = static_cast<T>(1.0) / grainSideLength;
      // Coordinates to the midpoint of the pairbreaking edge
      const T pbMidPointX = pbEdgeCoordinate;
      const T pbMidPointY = -pbEdgeCoordinate;
      // Rotation of pairbreaking edge with respect to the screen
      const T pbEdgeRotation = -static_cast<T>(0.25 * M_PI);
      scg::Initial<T>::edgePhaseCrystal(
          orderParameterComponentD, startPhasePeriod, startPhaseDecay,
          pbMidPointX, pbMidPointY, pbEdgeRotation);
      break;
    }
    default:
      std::cout << "Error! Start guess '"
                << simParams["start_guess_label"].get<int>()
                << "' not implemented." << std::endl;
      scg::run_utils::cleanup(context, gui);
      return 1;
    }

    orderParameter->add(orderParameterComponentD);
  }

  if (sWaveTc > TcLimit) {
    std::cout << "Adding s-wave component:" << std::endl;
    std::cout << "Tc: raw = " << sWaveTcRaw << " : scaled = " << sWaveTc
              << std::endl;
    scg::OrderParameterComponent<T> *orderParameterComponentS =
        new scg::OrderParameterComponent<T>(
            scg::BasisFunctionSymmetry::SWAVE_EXT, parameters, sWaveTc);

    switch (simParams["start_guess_label"].get<int>()) {
    case 1: {
      scg::Initial<T>::bulk(orderParameterComponentS);
      break;
    }
    case 2: {
      scg::Initial<T>::random(orderParameterComponentS, startRandomPhaseMix,
                              startRandomPhaseShift);
      break;
    }
    case 3: {
      scg::Initial<T>::bulk(orderParameterComponentS);
      break;
    }
    case 4: {
      scg::Initial<T>::bulk(orderParameterComponentS);
      break;
    }
    default:
      std::cout << "Error! Start guess '"
                << simParams["start_guess_label"].get<int>()
                << "' not implemented." << std::endl;
      scg::run_utils::cleanup(context, gui);
      return 1;
    }

    orderParameter->add(orderParameterComponentS);
  }

  //***************************************************
  // CHOOSE QUANTITIES TO CALCULATE
  //***************************************************
  // Create compute object(s) for observable(s)
  scg::ComputeOrderParameter<T> *computeOrderParameter =
      new scg::ComputeOrderParameter<T>(context);
  // Current compute object
  new scg::ComputeCurrent<T>(context);
  // Free energy compute object
  new scg::ComputeFreeEnergy<T>(context);

  //***************************************************
  // CHOOSE CONVERGENCE ACCELERATOR
  //***************************************************
  scg::OptimizerRollingBall<T> *optimizer = new scg::OptimizerRollingBall<T>();
  computeOrderParameter->setOptimizer(optimizer);

  //***************************************************
  // INITIALIZE SIMULATION
  //***************************************************
  context->initialize();

  //***************************************************
  // SAVE PARAMETERS TO FILE
  //***************************************************
  mu_sweeps_utils::saveParamsMetaData(context, DATA_PATH, grainSideLength,
                                      grainCornerLength, grainDiagonalLength,
                                      dWaveTc, sWaveTc, TcSuccess);
  if (!TcSuccess) {
    scg::run_utils::cleanup(context, gui);
    return 1;
  }

  //***********************************************
  // SET UP OPENGL VISUALIZATION
  //***********************************************
  if (simParams["visualize"].get<bool>()) {
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
  scg::run_utils::Status status = scg::run_utils::Status::kContinue;
  while (status != scg::run_utils::Status::kQuit) {
    // Run one iteration,
    status = scg::run_utils::runIteration(
        context, simParams["verbose"].get<bool>(),
        simParams["num_iterations_min"].get<int>(),
        simParams["num_iterations_max"].get<int>(),
        simParams["save_data"].get<bool>(),
        simParams["save_data_interval"].get<int>());

    if (status == scg::run_utils::Status::kSave) {
      saveDataWrapper(grainSideLength, grainCornerLength, grainDiagonalLength,
                      dWaveTc, sWaveTc, TcSuccess);
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
  saveDataWrapper(grainSideLength, grainCornerLength, grainDiagonalLength,
                  dWaveTc, sWaveTc, TcSuccess);

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

void saveDataWrapper(const T inGrainSideLength, const T inGrainCornerLength,
                     const T inGrainDiagonalLength, const T inDWaveTc,
                     const T inSWaveTc, const bool inTcSuccess) {
  // Save order parameter and current
  mu_sweeps_utils::saveGridData(context, DATA_PATH);
  // Update meta data file
  mu_sweeps_utils::saveParamsMetaData(
      context, DATA_PATH, inGrainSideLength, inGrainCornerLength,
      inGrainDiagonalLength, inDWaveTc, inSWaveTc, inTcSuccess);
  // Save induced magnetic field
  mu_sweeps_utils::saveInducedFluxDensity(context, DATA_PATH);
  // Save free energy and iteration status
  mu_sweeps_utils::saveFreeEnergy(context, DATA_PATH, inGrainSideLength,
                                  inGrainCornerLength, inGrainDiagonalLength);
}
