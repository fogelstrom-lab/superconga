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
#include "configure.h" // Variables and defines exported by cmake, e.g. version number and project name
#include "run_utils.h" // Commonly used functions in run files

#include "nlohmann_json_wrapper.h"
#include <cstdlib>
#include <iomanip>
#include <iostream>
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

struct OrderParameterJSON {
  std::string symmetry;
  T criticalTemperature;
  T initialPhaseShift;
  T initialPhaseWinding;
  T initialRandomPhaseWeight;
};

void from_json(const nlohmann::json &j, OrderParameterJSON &op) {
  j.at("symmetry").get_to(op.symmetry);
  j.at("critical_temperature").get_to(op.criticalTemperature);
  j.at("initial_phase_shift").get_to(op.initialPhaseShift);
  j.at("initial_phase_winding").get_to(op.initialPhaseWinding);
  j.at("initial_random_phase_weight").get_to(op.initialRandomPhaseWeight);
}
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

  // Parse JSON.
  nlohmann::json simParams;
  std::ifstream ifile(filename);

  // Check that provided config file actually exists and can be opened.
  if (!ifile.good()) {
    std::cerr << "Could not find/open parameter file: " << filename << "\n";
    scg::run_utils::cleanup(context, gui);
    return 1;
  }

  ifile >> simParams;

  // TODO(niclas): Add sanity check of all parameters here!?

  //***************************************************
  // APPLY PARAMETERS
  //***************************************************
  // Storage for system simulation parameters (e.g. temperature, lattice size).
  scg::Parameters<T> *parameters = new scg::Parameters<T>(context);
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
  scg::GeometryGroup<T> *geometry = new scg::GeometryGroup<T>(context);

  // Add polygon to simulation space, scale to fit in simulation domain
  // (grainScale 1.0 is 100% of simulation space).
  const T grainScale = static_cast<T>(1.0);

  // Check if to set up a disc or polygon.
  const int numVertices = simParams["num_polygon_edges"].get<int>();
  const bool isDisc = numVertices < 0;

  // Make the grain a disc or a square.
  if (isDisc) {
    geometry->add(new scg::DiscGeometry<T>(grainScale), scg::Type::add);
  } else {
    geometry->add(new scg::PolygonGeometry<T>(numVertices, grainScale),
                  scg::Type::add);
  }

  // Remove a portion to create a solenoid.
  if (simParams["solenoid"].get<bool>()) {
    // Relative size of grain hole for solenoid.
    const T holeScale = static_cast<T>(0.2);

    // Circular or square hole.
    if (isDisc) {
      geometry->add(new scg::DiscGeometry<T>(holeScale), scg::Type::remove);
    } else {
      geometry->add(new scg::PolygonGeometry<T>(numVertices, holeScale),
                    scg::Type::remove);
    }
  }

  //***************************************************
  // SET UP ORDER PARAMETER
  //***************************************************
  // Create order parameter (which might contain multiple components, i.e.
  // d+is).
  scg::OrderParameter<T> *orderParameter = new scg::OrderParameter<T>(context);

  const T crystalAxesRotation =
      static_cast<T>(2.0 * M_PI) * simParams["crystal_axes_rotation"].get<T>();

  const auto opCompParamsVec =
      simParams["order_parameter"].get<std::vector<::OrderParameterJSON>>();

  for (const auto &opCompParams : opCompParamsVec) {
    // Add a d-wave or an s-wave order-parameter component.
    const std::string symmetryStr = opCompParams.symmetry;

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
      std::cout << "Order parameter symmetry '" << symmetryStr
                << "' not understood." << std::endl;
      scg::run_utils::cleanup(context, gui);
      return 1;
    }

    scg::OrderParameterComponent<T> *orderParameterComponent =
        new scg::OrderParameterComponent<T>(symmetry, parameters,
                                            opCompParams.criticalTemperature);

    // Set rotation of crystal axes with respect to the grain geometry.
    orderParameterComponent->setCrystalAxesRotation(crystalAxesRotation);

    const T phaseShift =
        static_cast<T>(2.0 * M_PI) * opCompParams.initialPhaseShift;

    // Initialize order parameter with a guess.
    scg::Initial<T>::perturbedPhaseWinding(
        orderParameterComponent, opCompParams.initialPhaseWinding,
        opCompParams.initialRandomPhaseWeight, phaseShift);

    // Add the initialized component to the total order parameter.
    orderParameter->add(orderParameterComponent);
  }

  //***************************************************
  // CHOOSE FERMI SURFACE
  //***************************************************
  scg::FermiSurfaceCircle<T> *fermiSurface = new scg::FermiSurfaceCircle<T>();
  context->set(fermiSurface);

  //***************************************************
  // CHOOSE QUANTITIES TO CALCULATE
  //***************************************************
  // Order parameter compute object.
  scg::ComputeOrderParameter<T> *computeOrderParameter =
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
  if (simParams["solenoid"].get<bool>()) {
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
  // CHOOSE CONVERGENCE ACCELERATOR
  //***************************************************
  scg::OptimizerRollingBall<T> *optimizer = new scg::OptimizerRollingBall<T>();
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
    gui = scg::run_utils::createVisualization(context);
    scg::initGL(argc, argv, gui->getNumPixelsWindowX(),
                gui->getNumPixelsWindowY());
    glutKeyboardFunc(scg::run_utils::myKeyboardFunc);
    glutDisplayFunc(display);
  }

  //***********************************************
  // PROCESSING: RUN SIMULATION UNTIL END CRITERIA
  //***********************************************
  scg::run_utils::runSimulation(context, simParams["visualize"].get<bool>(),
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
    scg::run_utils::saveDataWrapper(context, simParams,
                                    simParams["verbose"].get<bool>());
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
