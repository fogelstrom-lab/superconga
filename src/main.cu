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
#include "Visualization.h" // Live visualization of desired observables through OpenGL
#include "configure.h" // Variables and defines exported by cmake, e.g. version number and project name
#include "run_utils.h" // Commonly used functions in run files

#include <cstdlib>
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
  if (argc < 2) {
    std::cerr << "Please provide config file.\n";
    scg::run_utils::cleanup(context, gui);
    return 1;
  }
  const std::string filename = argv[1];

  // Check that provided config file actually exists and can be opened.
  std::ifstream ifile(filename);
  if (!ifile.good()) {
    std::cerr << "Could not find/open parameter file: " << filename << "\n";
    scg::run_utils::cleanup(context, gui);
    return 1;
  }

  // Parse JSON.
  Json::Value simParams;
  Json::CharReaderBuilder builder;
  JSONCPP_STRING errs;
  if (!parseFromStream(builder, ifile, &simParams, &errs)) {
    std::cout << errs << std::endl;
    return 1;
  }
  std::cout << simParams << std::endl;

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
  const int numVertices = simParams["num_polygon_edges"].asInt();
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
  if (simParams["solenoid"].asBool()) {
    // Relative size of grain hole for solenoid.
    const T holeScale = static_cast<T>(0.2);

    // Circular or square hole.
    if (isDisc) {
      geometry->add(new scg::DiscGeometry<T>(grainFracton, holeScale),
                    scg::Type::remove);
    } else {
      geometry->add(
          new scg::PolygonGeometry<T>(grainFracton, numVertices, holeScale),
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
      std::cout << "Order parameter symmetry '" << symmetryStr
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
  auto optimizer = std::make_unique<scg::Optimizer<T>>(stepSize, inertia, rank);
  context->set(std::move(optimizer));

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
  const bool solenoid = simParams["solenoid"].asBool();
  if (solenoid) {
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
  context->initialize();

  // Set external magnetic field strength.
  const T externalFluxQuanta =
      static_cast<T>(simParams["external_flux_quanta"].asDouble());
  if (solenoid) {
    // External field in flux quanta.
    parameters->setMagneticFieldStrength(externalFluxQuanta);
  } else {
    // External field in fux quanta per unit area.
    const T externalMagneticFluxDensity = externalFluxQuanta / geometry->area();

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
  if (simParams["visualize"].asBool()) {
    gui = scg::run_utils::createVisualization(context);
    scg::initGL(argc, argv, gui->getNumPixelsWindowX(),
                gui->getNumPixelsWindowY());
    glutKeyboardFunc(scg::run_utils::myKeyboardFunc);
    glutDisplayFunc(display);
  }

  //***********************************************
  // PROCESSING: RUN SIMULATION UNTIL END CRITERIA
  //***********************************************
  scg::run_utils::runSimulation(
      context, simParams["visualize"].asBool(), simParams["verbose"].asBool(),
      simParams["num_iterations_min"].asInt(),
      simParams["num_iterations_max"].asInt(), simParams["save_data"].asBool(),
      simParams["save_data_interval"].asInt(), simParams);

  //***********************************************
  // POST-PROCESSING
  //***********************************************
  // Save at the very end.
  if (simParams["save_data"].asBool()) {
    scg::run_utils::saveDataWrapper(context, simParams,
                                    simParams["verbose"].asBool());
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
