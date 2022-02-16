//===----------- main.cu - Main file for setting up simulations -----------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
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

#include "conga/Context.h"
#include "conga/Parameters.h"
#include "conga/Type.h"
#include "conga/accelerators.h"
#include "conga/boundary/BoundaryConditionSpecular.h"
#include "conga/compute/ComputeCurrent.h"
#include "conga/compute/ComputeFreeEnergy.h"
#include "conga/compute/ComputeOrderParameter.h"
#include "conga/compute/ComputeVectorPotential.h"
#include "conga/configure.h"
#include "conga/geometry/GeometryGroup.h"
#include "conga/geometry/geometry_utils.h"
#include "conga/integration/IntegrationIteratorOzaki.h"
#include "conga/io.h"
#include "conga/order_parameter/OrderParameter.h"
#include "conga/order_parameter/OrderParameterBasis.h"
#include "conga/order_parameter/OrderParameterComponent.h"
#include "conga/order_parameter/OrderParameterInitial.h"
#include "conga/riccati/RiccatiSolverConfined.h"
#include "conga/run_simulation.h"

#ifdef CONGA_VISUALIZE
#include "conga/visualization/Visualizer.h"
#endif // CONGA_VISUALIZE

#include <json/json.h>

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

//*******************************************************
// PRECISION
//*******************************************************
#ifdef CONGA_SINGLE_PRECISION
typedef float T;
#else
typedef double T;
#endif

//*******************************************************
// MAIN
//*******************************************************
int main(int argc, char *argv[]) {
  //*******************************************************
  // PARSE SIMULATION CONFIG FILE
  //*******************************************************
  Json::Value simParams;
  {
    if (argc < 2) {
      std::cerr << "-- ERROR! Please provide a config file." << std::endl;
      return EXIT_FAILURE;
    }
    const std::filesystem::path configPath = argv[1];
    const bool success = conga::io::read(configPath, simParams);
    if (!success) {
      std::cerr << "-- ERROR! Could not find/open config file '" << configPath
                << "'." << std::endl;
      return EXIT_FAILURE;
    }
#ifndef NDEBUG
    std::cout << "-- Configuration parameters:" << std::endl;
    std::cout << simParams << std::endl;
#endif
  }

  // TODO(niclas): Add sanity check of all parameters here!?

  // Fetch the verbosity so we know whether to print stuff or not.
  const bool verbose = simParams["misc"]["verbose"].asBool();

  //***************************************************
  // CREATE CONTEXT
  //***************************************************
  conga::Context<T> context;

  //***************************************************
  // APPLY PARAMETERS
  //***************************************************
  // Storage for system simulation parameters (e.g. temperature, lattice size).
  conga::Parameters<T> *parameters = new conga::Parameters<T>(&context);

  // Set temperature.
  const T temperature =
      static_cast<T>(simParams["physics"]["temperature"].asDouble());
  parameters->setTemperature(temperature);

  // Set the number of lattice points per coherence length.
  const T pointsPerCoherenceLength = static_cast<T>(
      simParams["accuracy"]["points_per_coherence_length"].asDouble());
  parameters->setPointsPerCoherenceLength(pointsPerCoherenceLength);

  // Set energy cutoff.
  const T energyCutoff =
      static_cast<T>(simParams["accuracy"]["energy_cutoff"].asDouble());
  parameters->setEnergyCutoff(energyCutoff);

  // Number of discrete steps in momentum angle.
  const int numFermiMomenta =
      simParams["accuracy"]["num_fermi_momenta"].asInt();
  parameters->setAngularResolution(numFermiMomenta);

  // Set convergence criterion.
  const T convergenceCriterion =
      static_cast<T>(simParams["accuracy"]["convergence_criterion"].asDouble());
  parameters->setConvergenceCriterion(convergenceCriterion);

  // Set penetration depth.
  const T penetrationDepth =
      static_cast<T>(simParams["physics"]["penetration_depth"].asDouble());
  parameters->setPenetrationDepth(penetrationDepth);

  // Set external magnetic field strength.
  const T externalFluxQuanta =
      static_cast<T>(simParams["physics"]["external_flux_quanta"].asDouble());
  parameters->setNumFluxQuanta(externalFluxQuanta);

  // Set sign of charge.
  const int chargeSign = simParams["physics"]["charge_sign"].asInt();
  parameters->setChargeSign(chargeSign);

  // Set which norm to use when computing residuals.
  const std::string residualNorm(simParams["accuracy"]["norm"].asString());
  if (residualNorm == "L1") {
    parameters->setResidualNormType(conga::ResidualNorm::L1);
  } else if (residualNorm == "L2") {
    parameters->setResidualNormType(conga::ResidualNorm::L2);
  } else if (residualNorm == "LInf") {
    parameters->setResidualNormType(conga::ResidualNorm::LInf);
  } else {
    std::cerr << "-- ERROR! Norm type, " << residualNorm << ", not understod"
              << std::endl;
    return EXIT_FAILURE;
  }

  //***************************************************
  // CREATE GEOMETRY
  //***************************************************
  // Create geometry group: add/remove geometry objects (e.g. discs, polygons)
  // from the geometry group.
  conga::GeometryGroup<T> *geometry = new conga::GeometryGroup<T>(&context);

  // Parse user-defined geometry. Compute factors to scale and translate
  // geometry to internal coordinate system.
  T scalingFactor;
  T xOffset;
  T yOffset;
  conga::geometry_utils::computeGeometryScaling(simParams, scalingFactor,
                                                xOffset, yOffset);

  // Set grain sidelength in coherence length.
  parameters->setGrainWidth(scalingFactor);

  // Fetch relative size of geometry to available simulation space, to account
  // for padding for boundary interpolation.
  const T grainFraction = parameters->getGrainFraction();

  // Parse user-defined geometry, loop through geometry component list and
  // create compund geometry. Scale and translate to internal coordinate system.
  conga::geometry_utils::addGeometryComponents(
      geometry, simParams, scalingFactor, xOffset, yOffset, grainFraction);

  //***************************************************
  // SET UP INTEGRATOR
  //***************************************************
  // A negative number means doing all of them at once.
  const int numEnergiesPerBlock = -1;
  const int numBurnInIterations =
      simParams["accuracy"]["num_iterations_burnin"].asInt();
  new conga::IntegrationIteratorOzaki<T>(&context, numEnergiesPerBlock,
                                         numBurnInIterations);

  //***************************************************
  // SET UP ORDER PARAMETER
  //***************************************************
  // Create order parameter (which might contain multiple components, i.e.
  // d+is).
  conga::OrderParameter<T> *orderParameter =
      new conga::OrderParameter<T>(&context);

  // Set rotation of crystal axes with respect to the grain geometry.
  const T crystalAxesRotation = static_cast<T>(
      2.0 * M_PI * simParams["physics"]["crystal_axes_rotation"].asDouble());
  orderParameter->setCrystalAxesRotation(crystalAxesRotation);

  // For convenience.
  const Json::Value opParams = simParams["physics"]["order_parameter"];

  // An array of all supported symmetries.
  // TODO(niclas): Move to OrderParameterBasis.h?
  const std::vector<std::string> symmetries = {"dx2-y2", "dxy", "g", "s"};

  // Loop over symmetries and add them if present.
  bool orderParameterSuccess = false;
  for (const std::string &symmetryStr : symmetries) {
    if (opParams.isMember(symmetryStr)) {
      // There is at least one component specified.
      orderParameterSuccess = true;

      // TODO(niclas): Move to OrderParameterBasis.h?
      conga::BasisFunctionSymmetry symmetry;
      if (symmetryStr == "dx2-y2") {
        symmetry = conga::BasisFunctionSymmetry::DWAVE_X2_Y2;
      } else if (symmetryStr == "dxy") {
        symmetry = conga::BasisFunctionSymmetry::DWAVE_XY;
      } else if (symmetryStr == "g") {
        symmetry = conga::BasisFunctionSymmetry::GWAVE;
      } else {
        symmetry = conga::BasisFunctionSymmetry::SWAVE;
      }

      // For convenience.
      const Json::Value opCompParams = opParams[symmetryStr];

      // Create component.
      const T criticalTemperature =
          static_cast<T>(opCompParams["critical_temperature"].asDouble());
      conga::OrderParameterComponent<T> *orderParameterComponent =
          new conga::OrderParameterComponent<T>(symmetry, parameters,
                                                criticalTemperature);

      // Get initial phase parameters.
      const T phaseWinding =
          static_cast<T>(opCompParams["initial_phase_winding"].asDouble());
      const T noiseStdDev =
          static_cast<T>(opCompParams["initial_noise_stddev"].asDouble());
      const T phaseShift = static_cast<T>(
          2.0 * M_PI * opCompParams["initial_phase_shift"].asDouble());

      // Check if we should initialize to bulk.
      const T eps = std::numeric_limits<T>::epsilon();
      const bool useBulk = (std::abs(phaseWinding) < eps) &&
                           (std::abs(noiseStdDev) < eps) &&
                           (std::abs(phaseShift) < eps) &&
                           (!simParams["physics"].isMember("vortex_lattice"));

      // Initialize order parameter with a guess.
      if (useBulk) {
        conga::Initial<T>::bulk(orderParameterComponent);
      } else {
        // Check if vortex-lattice start guess was supplied.
        if (simParams["physics"].isMember("vortex_lattice")) {

          // Get coordinates and  winding-numbers for vortex lattice start
          // guess.
          const Json::Value jsonVorticesCentersX =
              simParams["physics"]["vortex_lattice"]["vortices_centers_x"];
          const Json::Value jsonVorticesCentersY =
              simParams["physics"]["vortex_lattice"]["vortices_centers_y"];
          const Json::Value jsonVorticesWindingNumbers =
              simParams["physics"]["vortex_lattice"]
                       ["vortices_winding_numbers"];

          std::vector<T> vorticesCentersX;
          std::vector<T> vorticesCentersY;
          std::vector<T> vorticesWindingNumbers;

          // Scale to internal coordinates and add to vectors.
          for (int vortexIdx = 0; vortexIdx < jsonVorticesCentersX.size();
               ++vortexIdx) {
            // TODO(patric): fix bug when both coordinates are offset (but
            // unequally). The bug seems to lie in the code that centers the
            // coordinate system(?).
            const T xTranslatedScaled =
                (static_cast<T>(jsonVorticesCentersX[vortexIdx].asDouble()) -
                 xOffset) /
                (static_cast<T>(0.5) * scalingFactor);
            const T yTranslatedScaled =
                (static_cast<T>(jsonVorticesCentersY[vortexIdx].asDouble()) -
                 yOffset) /
                (static_cast<T>(0.5) * scalingFactor);

            vorticesCentersX.push_back(xTranslatedScaled);
            vorticesCentersY.push_back(yTranslatedScaled);
            vorticesWindingNumbers.push_back(static_cast<T>(
                jsonVorticesWindingNumbers[vortexIdx].asDouble()));
          }

          // Overwrite phase-winding from order-parameter components.
          conga::Initial<T>::perturbedPhaseWindings(
              orderParameterComponent, vorticesWindingNumbers, vorticesCentersX,
              vorticesCentersY, noiseStdDev, phaseShift);
        } else {
          // No vortex-lattice start guess, use order-parameter component value.
          conga::Initial<T>::perturbedPhaseWinding(
              orderParameterComponent, phaseWinding, noiseStdDev, phaseShift);
        }
      }

      // Add the initialized component to the total order parameter.
      orderParameter->add(orderParameterComponent);
    }
  }

  if (!orderParameterSuccess) {
    std::cerr << "-- ERROR! No order-parameter component set." << std::endl;
    return EXIT_FAILURE;
  }

  //***************************************************
  // READ ORDER PARAMETER AND VECTOR POTENTIAL FROM FILE
  //***************************************************
  // Path to directory containing data to load from.
  const std::filesystem::path loadDataPath(
      simParams["misc"]["load_path"].asString());

  if (!loadDataPath.empty()) {
    // Create the placeholder grid for vector potential.
    const int gridResolution = parameters->getGridResolutionBase();
    auto vectorPotential = std::make_unique<conga::GridGPU<T>>(
        gridResolution, gridResolution, 2, conga::Type::real);

    bool successOrderParameter = false;
    bool successVectorPotential = false;

#ifdef CONGA_USE_HDF5
    // Check if the simulation file exists.
    const std::filesystem::path simulationPath = loadDataPath / "simulation.h5";
    const bool simulationExists = std::filesystem::exists(simulationPath);
    if (simulationExists) {
      // Open file.
      const auto file = conga::io::fileReadOnly(simulationPath);

      // Read order parameter.
      successOrderParameter =
          conga::io::read(file, "order_parameter", *orderParameter);
      conga::io::printRead(verbose, successOrderParameter, "order parameter",
                           simulationPath);

      // Read vector potential.
      successVectorPotential =
          conga::io::read(file, "vector_potential", *(vectorPotential.get()));
      conga::io::printRead(verbose, successVectorPotential, "vector potential",
                           simulationPath);

    } else
#endif // CONGA_USE_HDF5
    {
      // Read order parameter.
      const std::filesystem::path orderParameterPath =
          loadDataPath / "order_parameter.csv";
      successOrderParameter =
          conga::io::read(orderParameterPath, *orderParameter);
      conga::io::printRead(verbose, successOrderParameter, "order parameter",
                           orderParameterPath);

      // Read vector potential.
      const std::filesystem::path vectorPotentialPath =
          loadDataPath / "vector_potential.csv";
      successVectorPotential =
          conga::io::read(vectorPotentialPath, *(vectorPotential.get()));
      conga::io::printRead(verbose, successVectorPotential, "vector potential",
                           vectorPotentialPath);
    }

    // Abort if not successful.
    if (!successOrderParameter) {
      std::cerr << "-- ERROR! Could not find order parameter in "
                << loadDataPath << std::endl;
      return EXIT_FAILURE;
    }

    // Abort if not successful.
    if (!successVectorPotential) {
      std::cerr << "-- ERROR! Could not find vector potential in "
                << loadDataPath << std::endl;
      return EXIT_FAILURE;
    }

    // Set the vector potential in context.
    if (penetrationDepth > static_cast<T>(0)) {
      // Internally the induced vector potential is scaled by the squared
      // penetration depth.
      *vectorPotential *= (penetrationDepth * penetrationDepth);
    } else {
      vectorPotential->setZero();
    }
    context.set(std::move(vectorPotential));
  }

  //***************************************************
  // ACCELERATOR
  //***************************************************
  // Get the name.
  const std::string acceleratorName(
      simParams["accelerator"]["name"].asString());

  // Make accelerator.
  std::unique_ptr<conga::accelerators::Accelerator<T>> accelerator;
  if (acceleratorName == "Adaptive-Ball" || acceleratorName == "AB") {
    // Create Adaptive-Ball accelerator.
    accelerator = std::make_unique<conga::accelerators::AdaptiveBall<T>>(
        simParams["accelerator"]);
  } else if (acceleratorName == "Anderson") {
    // Create Anderson accelerator.
    accelerator = std::make_unique<conga::accelerators::Anderson<T>>(
        simParams["accelerator"]);
  } else if (acceleratorName == "Barzilai-Borwein" || acceleratorName == "BB") {
    // Create Barzilai-Borwein accelerator.
    accelerator = std::make_unique<conga::accelerators::BarzilaiBorwein<T>>(
        simParams["accelerator"]);
  } else if (acceleratorName == "Picard") {
    // Create Picard accelerator.
    accelerator = std::make_unique<conga::accelerators::Picard<T>>(
        simParams["accelerator"]);
  } else if (acceleratorName == "Polyak") {
    // Create Polyak accelerator.
    accelerator = std::make_unique<conga::accelerators::Polyak<T>>(
        simParams["accelerator"]);
  } else {
    std::cerr << "-- ERROR! Accelerator name, " << acceleratorName
              << ", not understod" << std::endl;
    return EXIT_FAILURE;
  }

  // Move to context.
  context.set(std::move(accelerator));

  //***************************************************
  // CHOOSE QUANTITIES TO CALCULATE
  //***************************************************
  // Order parameter compute object.
  new conga::ComputeOrderParameter<T>(&context);
  // Add current compute object.
  new conga::ComputeCurrent<T>(&context);
  // Add free energy compute object.
  new conga::ComputeFreeEnergy<T>(&context);

  // Add vector-potential compute object.
  if (penetrationDepth > static_cast<T>(0)) {
    const T maxError = static_cast<T>(
        simParams["accuracy"]["vector_potential_error"].asDouble());
    auto computeVectorPotential =
        std::make_unique<conga::ComputeVectorPotential<T>>(maxError);
    context.add(std::move(computeVectorPotential));
  }

  //***************************************************
  // CHOOSE RICCATI SOLVER AND BOUNDARY CONDITIONS
  //***************************************************
  // RiccatiSolverConfined: provides analytic solutions to the Riccati equations
  // for a piecewise constant order parameter.
  conga::RiccatiSolver<T> *riccatiSolver;

  // Gauge choice.
  const std::string gauge(simParams["physics"]["gauge"].asString());
  if (gauge == "radial") {
    riccatiSolver =
        new conga::RiccatiSolverConfined<T, conga::VectorPotential::Radial>(
            &context);
  } else if (gauge == "solenoid") {
    riccatiSolver =
        new conga::RiccatiSolverConfined<T, conga::VectorPotential::Solenoid>(
            &context);
  } else {
    std::cerr << "-- ERROR! Gauge, " << gauge << ", not understod" << std::endl;
    return EXIT_FAILURE;
  }

  // Add specular BC.
  riccatiSolver->add(new conga::BoundaryConditionSpecular<T>);

  //***************************************************
  // INITIALIZE SIMULATION
  //***************************************************
  context.initialize();

  //***********************************************
  // CREATE DIRECTORIES AND SAVE PARAMETERS
  //***********************************************
  const std::filesystem::path saveDataPath(
      simParams["misc"]["save_path"].asString());

  // Set the save frequency. If the argument is negative we only save the grid
  // data at the very end.
  const int saveFrequencyArg = simParams["misc"]["save_frequency"].asInt();
  const int saveFrequency = (saveFrequencyArg < 0)
                                ? std::numeric_limits<int>::max()
                                : saveFrequencyArg;

  const bool saveData = !saveDataPath.empty() && (saveFrequency > 0);
  if (saveData) {
    // Create directories.
    std::filesystem::create_directories(saveDataPath);

    // Save config parameters to JSON file.
    const std::filesystem::path configPath = saveDataPath / "config.json";
    conga::io::write(configPath, simParams);
    conga::io::printWrite(verbose, true, "configuration", configPath);

    // TODO(niclas): Write internal parameters here as well? They do not change
    // during the simulation.
  }

  //***********************************************
  // PROCESSING: RUN SIMULATION UNTIL END CRITERIA
  //***********************************************
  const bool visualize = simParams["misc"]["visualize"].asBool();
  const int numIterationsMin =
      simParams["accuracy"]["num_iterations_min"].asInt();
  const int numIterationsMax =
      simParams["accuracy"]["num_iterations_max"].asInt();
  const conga::io::DataFormat dataFormat =
      conga::io::getDataFormat(simParams["misc"]["data_format"].asString());
#ifdef CONGA_VISUALIZE
  if (visualize) {
    // Create visualizer.
    const int numOrderParameterComponents =
        context.getOrderParameter()->getNumComponents();
    const conga::GridGPU<char> &domain =
        *(context.getGeometry()->getDomainGrid());
    // Magnitude and xy-components.
    const int rows = 3;

    // Extra columns: current density, vector potential, and flux density.
    const int extraCols = (penetrationDepth > static_cast<T>(0)) ? 3 : 1;
    const int cols = numOrderParameterComponents + extraCols;
    conga::Visualizer<T> visualizer("SuperConga", cols, rows, domain);

    conga::runSimulation(&context, visualizer, verbose, numIterationsMin,
                         numIterationsMax, saveDataPath, saveFrequency,
                         dataFormat);
  } else {
    conga::runSimulation(&context, verbose, numIterationsMin, numIterationsMax,
                         saveDataPath, saveFrequency, dataFormat);
  }
#else
  conga::runSimulation(&context, verbose, numIterationsMin, numIterationsMax,
                       saveDataPath, saveFrequency, dataFormat);
#endif // CONGA_VISUALIZE

  //***********************************************
  // SAVE RESULTS
  //***********************************************
  // Save at the very end.
  if (saveData) {
    conga::saveDataWrapper(&context, saveDataPath, dataFormat, verbose);
  }

  // Yay!
  return EXIT_SUCCESS;
}
