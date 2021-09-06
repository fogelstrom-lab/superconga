//===----- run_utils.h - Useful functions for run file examples --===//
//
// Part of the SuprConGPU Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains useful functions used in example run files.
///
//===----------------------------------------------------------------------===//
#ifndef RUN_UTILS_H_
#define RUN_UTILS_H_

#include "Context.h"
#include "DiscGeometry.h"
#include "GeometryGroup.h"
#include "Grid.h"
#include "Parameters.h"
#include "PolygonGeometry.h"
#include "Type.h"
#include "Visualization.h"
#include "io/csv_writers.h"
#include "io/write_utils.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <json/json.h>
#include <limits>
#include <memory>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <vector>

#ifdef __linux__
#include <GL/freeglut.h>
#else
#include <GLUT/glut.h>
#endif

namespace scg {
namespace run_utils {
namespace internal {
//===----------------------------- Variables -----------------------------===//
// Constant used to check if user presses 'q' or 's' in the GUI to quit or save.
bool guiUserQuit = false;
bool guiUserSaveData = false;
} // namespace internal

//===----------------------------- Functions -----------------------------===//

/// \brief Calculate scaling and offset for user-provided geometry.
///
/// Parse geometry dictionary for a list of geometry components, provided e.g.
/// via simulation parameters from user. Calculate the bounding box
/// of the compound geometry, and the offset to the geometry center point. The
/// bounding box is used to calculate the scaling factor to internal
/// coordinates. Return scaling and offsets.
///
/// \param inSimulationParameters User-defined simulation parameters.
/// \param outScalingFactor Scaling factor for geometry to internal coordinates.
/// \param outOffsetX Distance to geometry center point along x.
/// \param outOffsetY Distance to geometry center point along y.
///
/// \return void.
template <typename T>
void computeGeometryScaling(const Json::Value &inSimulationParameters,
                            T &outScalingFactor, T &outOffsetX, T &outOffsetY) {
  // Access geometry dictionary in simulation paramters.
  const Json::Value geometryComponentVector =
      inSimulationParameters["geometry"];

  // Find out bounding box for compound geometry.
  T xMax = std::numeric_limits<T>::min();
  T xMin = std::numeric_limits<T>::max();
  T yMax = std::numeric_limits<T>::min();
  T yMin = std::numeric_limits<T>::max();
  for (int i = 0; i < geometryComponentVector.size(); ++i) {
    // Fetch geometry component type (e.g. disc).
    const std::vector<std::string> geometryComponentNames =
        geometryComponentVector[i].getMemberNames();
    const std::string geometryComponentName = geometryComponentNames[0];

    // Only consider components that are added (not removed).
    if (!geometryComponentVector[i][geometryComponentName]["add"].asBool()) {
      continue;
    }

    // Update maxima and minima of geometry extent.
    if (geometryComponentName == "disc") {
      //***************************************************
      // DISC: GET EXTREMUM COORDINATES
      //***************************************************
      const T radius = static_cast<T>(
          0.5 * geometryComponentVector[i][geometryComponentName]["diameter"]
                    .asDouble());
      const T centerX = static_cast<T>(
          geometryComponentVector[i][geometryComponentName]["center_x"]
              .asDouble());
      const T centerY = static_cast<T>(
          geometryComponentVector[i][geometryComponentName]["center_y"]
              .asDouble());

      xMax = std::max(xMax, centerX + radius);
      xMin = std::min(xMin, centerX - radius);
      yMax = std::max(yMax, centerY + radius);
      yMin = std::min(yMin, centerY - radius);
    } else if (geometryComponentName == "polygon") {
      //***************************************************
      // POLYGON: GET EXTREMUM COORDINATES
      //***************************************************
      // Loop over vertex coordinates.
      const Json::Value xVerticesDict =
          inSimulationParameters["geometry"][i][geometryComponentName]
                                ["vertices_x"];
      const Json::Value yVerticesDict =
          inSimulationParameters["geometry"][i][geometryComponentName]
                                ["vertices_y"];

      // Check that the length is the same for the x and y vertices.
      if (xVerticesDict.size() != yVerticesDict.size()) {
        // Unknown geometry component. Print error and exit.
        std::cerr << "-- ERROR! Different number of x- and y-components given "
                     "for polygon vertices: "
                  << xVerticesDict.size() << " != " << yVerticesDict.size()
                  << "." << std::endl;
        std::exit(EXIT_FAILURE);
      }

      // Note, having n as std::size_t causes a bug in JsonCpp...
      for (int n = 0; n < xVerticesDict.size(); ++n) {
        const T x = static_cast<T>(xVerticesDict[n].asDouble());
        const T y = static_cast<T>(yVerticesDict[n].asDouble());

        xMax = std::max(xMax, x);
        xMin = std::min(xMin, x);
        yMax = std::max(yMax, y);
        yMin = std::min(yMin, y);
      }
    } else if (geometryComponentName == "regular_polygon") {
      //***************************************************
      // REGULAR POLYGON: GET EXTREMUM COORDINATES
      //***************************************************
      const T sideLength = static_cast<T>(
          geometryComponentVector[i][geometryComponentName]["side_length"]
              .asDouble());
      const T centerX = static_cast<T>(
          geometryComponentVector[i][geometryComponentName]["center_x"]
              .asDouble());
      const T centerY = static_cast<T>(
          geometryComponentVector[i][geometryComponentName]["center_y"]
              .asDouble());
      const T rotation = static_cast<T>(
          2.0 * M_PI *
          geometryComponentVector[i][geometryComponentName]["rotation"]
              .asDouble());
      const T numEdges =
          geometryComponentVector[i][geometryComponentName]["num_edges"]
              .asInt();

      const T angleStep = static_cast<T>(2.0 * M_PI) / static_cast<T>(numEdges);
      const T radius =
          sideLength / std::hypot(std::cos(angleStep) - static_cast<T>(1),
                                  std::sin(angleStep));
      const T angleOffset = static_cast<T>(0.5) * angleStep + rotation;

      for (int n = 0; n < numEdges; ++n) {
        const T angle = angleOffset + angleStep * static_cast<T>(n);
        const T x = centerX + radius * std::cos(angle);
        const T y = centerY + radius * std::sin(angle);

        xMax = std::max(xMax, x);
        xMin = std::min(xMin, x);
        yMax = std::max(yMax, y);
        yMin = std::min(yMin, y);
      }
    } else {
      // Unknown geometry component. Print error and exit.
      std::cout << "-- ERROR! Unknown geometry type: '" << geometryComponentName
                << "'." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // Calculate extent for scaling to internal coordinate system.
  const T xExtent = xMax - xMin;
  const T yExtent = yMax - yMin;
  // Calculate offset for translating to internal coordinate system.
  outOffsetX = xMin + static_cast<T>(0.5) * xExtent;
  outOffsetY = yMin + static_cast<T>(0.5) * yExtent;

  // Calculate scaling factor for scaling to internal coordinate system.
  // Consequently, this is the side length of the compound geometry - this is
  // returned so that it can be used for the Parameter class instance.
  outScalingFactor = std::max(xExtent, yExtent);
}

/// \brief Create geometry from list of geometry components.
///
/// Parse geometry dictionary for a list of geometry components, provided e.g.
/// via simulation parameters from user. Start by calculating the bounding box
/// of the compound geometry. Translate and scale the compound geometry to fit
/// in internal geometry coordinate system. Loop through list and add/remove
/// geometry objects as specified.
///
/// \param outGeometry Geometry container: instance of GeometryGroup class.
/// \param inSimulationParameters User-defined simulation parameters.
/// \param inScalingFactor Factor to scale geometry to internal coordinates.
/// \param inOffsetX Distance along x to translate geometry to origin.
/// \param inOffsetY Distance along y to translate geometry to origin.
/// \param inGrainFraction Simulation space size, minus margin.
///
/// \return void.
template <typename T>
void addGeometryComponents(GeometryGroup<T> *outGeometry,
                           const Json::Value &inSimulationParameters,
                           const T inScalingFactor, const T inOffsetX,
                           const T inOffsetY, const T inGrainFraction) {
  // Access geometry dictionary in simulation paramters.
  const Json::Value geometryComponentVector =
      inSimulationParameters["geometry"];

  // Loop over list of geometry components.
  for (int i = 0; i < geometryComponentVector.size(); ++i) {
    // Fetch geometry component type (e.g. disc).
    const std::vector<std::string> geometryComponentNames =
        geometryComponentVector[i].getMemberNames();
    const std::string geometryComponentName = geometryComponentNames[0];

    // Check whether component is to be added or removed.
    const scg::Type::type addOrRemove =
        (geometryComponentVector[i][geometryComponentName]["add"].asBool())
            ? scg::Type::add
            : scg::Type::remove;

    // Add/remove the appropriate geometry component.
    if (geometryComponentName == "disc") {
      //***************************************************
      // DISC: ADD/REMOVE
      //***************************************************

      // Translate and scale center coordinates.
      const T xCenterTranslatedScaled =
          (static_cast<T>(
               geometryComponentVector[i]["disc"]["center_x"].asDouble()) -
           inOffsetX) /
          inScalingFactor;
      const T yCenterTranslatedScaled =
          (static_cast<T>(
               geometryComponentVector[i]["disc"]["center_y"].asDouble()) -
           inOffsetY) /
          inScalingFactor;
      // Disc diameter in internal coordinate system.
      const T diameter =
          static_cast<T>(
              geometryComponentVector[i]["disc"]["diameter"].asDouble()) /
          inScalingFactor;

      // Add/remove disc.
      outGeometry->add(new scg::DiscGeometry<T>(inGrainFraction, diameter,
                                                xCenterTranslatedScaled,
                                                yCenterTranslatedScaled),
                       addOrRemove);
    } else if (geometryComponentVector[i].isMember("polygon")) {
      //***************************************************
      // POLYGON: ADD/REMOVE
      //***************************************************
      // Check whether component is to be added or removed.

      // Fetch vertices.
      const int numVertices =
          int(geometryComponentVector[i]["polygon"]["vertices_x"].size());

      T xVertices[numVertices];
      T yVertices[numVertices];

      for (int vertex_idx = 0; vertex_idx < numVertices; ++vertex_idx) {
        // Translate and scale polygon vertex coordinates.
        xVertices[vertex_idx] =
            (static_cast<T>(
                 geometryComponentVector[i]["polygon"]["vertices_x"][vertex_idx]
                     .asDouble()) -
             inOffsetX) /
            inScalingFactor;
        yVertices[vertex_idx] =
            (static_cast<T>(
                 geometryComponentVector[i]["polygon"]["vertices_y"][vertex_idx]
                     .asDouble()) -
             inOffsetY) /
            inScalingFactor;
      }

      // Add/remove polygon.
      outGeometry->add(new scg::PolygonGeometry<T>(inGrainFraction, numVertices,
                                                   xVertices, yVertices),
                       addOrRemove);
    } else if (geometryComponentVector[i].isMember("regular_polygon")) {
      //***************************************************
      // REGULAR POLYGON: ADD/REMOVE
      //***************************************************
      // Check whether component is to be added or removed.

      // Translate and scale center coordinates.
      const T xCenterTranslatedScaled =
          (static_cast<T>(
               geometryComponentVector[i]["regular_polygon"]["center_x"]
                   .asDouble()) -
           inOffsetX) /
          inScalingFactor;
      const T yCenterTranslatedScaled =
          (static_cast<T>(
               geometryComponentVector[i]["regular_polygon"]["center_y"]
                   .asDouble()) -
           inOffsetY) /
          inScalingFactor;
      // Side length of the regular polygon, in internal coordinate system.
      const T sideLength =
          static_cast<T>(
              geometryComponentVector[i]["regular_polygon"]["side_length"]
                  .asDouble()) /
          inScalingFactor;

      const int numEdges =
          geometryComponentVector[i]["regular_polygon"]["num_edges"].asInt();

      const T rotation = static_cast<T>(
          2.0 * M_PI *
          geometryComponentVector[i]["regular_polygon"]["rotation"].asDouble());

      // Add/remove regular polygon.
      outGeometry->add(new scg::PolygonGeometry<T>(
                           inGrainFraction, numEdges, sideLength, rotation,
                           xCenterTranslatedScaled, yCenterTranslatedScaled),
                       addOrRemove);
    } else {
      // Unknown geometry component. Print error and exit.
      const std::vector<std::string> geometryComponentNames =
          geometryComponentVector[i].getMemberNames();
      std::cout << "-- ERROR! Unknown geometry type: '"
                << geometryComponentNames[i][0] << "'." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
}

/// \brief Detect OpenGL key presses
///
/// Detect keypresses in OpenGL instance - useful e.g. for changing
/// parameters live, saving data, quitting the program
///
/// \param inKey - Pressed key.
/// \param inCoordinateX - The GUI x-coordinate of the mouse at the key
/// press. \param inCoordinateY - The GUI y-coordinate of the mouse at the
/// key press.
///
/// \return - void
inline void myKeyboardFunc(const unsigned char inKey, const int inX,
                           const int inY) {
#ifndef NDEBUG
  std::cout << "-- Key pressed: " << inKey << ", at (" << inX << ", " << inY
            << ")\n";
#endif
  if (inKey == 'q' || inKey == 'Q') {
    internal::guiUserQuit = true;
  } else if (inKey == 's' || inKey == 'S') {
    internal::guiUserSaveData = true;
  }
}

/// \brief Save config params, internal params, and results to JSON files.
///
/// This function saves two json files: one for internal parameters, and one
/// for scalar results. Motivation, there are some internal parameters that
/// are not possible to infer without running the program (the exact number of
/// lattice points inside the grain), these are very useful to have access to
/// when post-processing (e.g. for doing area normalization). Finally, there
/// are some scalar results (e.g. area-averaged quantities) that are more
/// appropriate and convenient to have in json rather than csv.
///
/// \param inContext - Context object for simulation.
/// \param inDataPath - Directory where to put JSON files.
///
/// \return - void
template <typename T>
void saveParametersWrapper(const Context<T> *const inContext,
                           const std::filesystem::path &inDataPath,
                           const bool inVerbose = true) {
  //***************************************************
  // CONFIG PARAMETERS
  //***************************************************
  // Construct full filepath to JSON-file.
  const std::filesystem::path fullPathInternalParamsJSON =
      inDataPath / "internal_parameters.json";
  const std::filesystem::path fullPathResultsJSON = inDataPath / "results.json";

  // Number of lattice sites in grain.
  const int numGrainLatticeSites =
      inContext->getGeometry()->getNumGrainLatticeSites();

  // Json for various results.
  Json::Value resultsJSON;

  // Free energy per unit area.
  resultsJSON["free_energy_per_unit_area"] =
      inContext->getComputeFreeEnergy()->getFreeEnergyPerUnitArea();
  resultsJSON["magnetic_free_energy_per_unit_area"] =
      inContext->getComputeFreeEnergy()->getMagneticFreeEnergyPerUnitArea();

  // Average order-parameter magnitude.
  // TODO(niclas): Is this correct for multi-component OP?
  resultsJSON["order_parameter_average_magnitude"] =
      inContext->getComputeOrderParameter()->getResult()->sumMagnitude() /
      static_cast<T>(numGrainLatticeSites);

  // Average current-density magnitude.
  resultsJSON["current_density_average_magnitude"] =
      inContext->getComputeCurrent()->getResult()->sumMagnitude() /
      static_cast<T>(numGrainLatticeSites);

  // Get the vector potential. It is used both for the average vector potential,
  // and average flux density.
  GridGPU<T> vectorPotential = *(inContext->getInducedVectorPotential());

  // Average flux density.
  // TODO(niclas): Should this be computed outside of the grain?
  const T latticeElementSize = inContext->getParameters()->getGridElementSize();
  GridGPU<T> fluxDensity = vectorPotential.curl(latticeElementSize);
  inContext->getGeometry()->zeroValuesOutsideDomainRef(&fluxDensity);
  resultsJSON["flux_density_average"] =
      fluxDensity.sumField() / static_cast<T>(numGrainLatticeSites);

  // Average vector-potential magnitude.
  // TODO(niclas): Should this be computed outside of the grain?
  inContext->getGeometry()->zeroValuesOutsideDomainRef(&vectorPotential);
  resultsJSON["vector_potential_average_magnitude"] =
      vectorPotential.sumMagnitude() / static_cast<T>(numGrainLatticeSites);

  // Save various results to JSON file.
  io::writeJSON(resultsJSON, fullPathResultsJSON, inVerbose);

  // Internal parameters useful for post-processing.
  Json::Value internalParamJSON;
  internalParamJSON["area_compute"] = inContext->getGeometry()->area();
  internalParamJSON["internal_lattice_points"] = numGrainLatticeSites;
  internalParamJSON["gauge"] =
      inContext->getRiccatiSolver()->getVectorPotentialType();
  internalParamJSON["grain_margin"] = constants::GRAIN_SIZE_MARGIN;
  const int numLatticeSitesPerSide =
      inContext->getParameters()->getGridResolutionBase();
  internalParamJSON["num_lattice_points"] =
      numLatticeSitesPerSide * numLatticeSitesPerSide;

  // Save internal parameters to JSON file.
  io::writeJSON(internalParamJSON, fullPathInternalParamsJSON, inVerbose);
}

/// \brief Save convergence status (residuals vs iteration) to file.
///
/// \param inContext - Context object for simulation.
/// \param inDataPath - Directory where to put file.
/// \param inVerbose - Print success to terminal.
/// \param inIterationDataSaveFormat - Enum specifying which format to save
/// the per-iteration data in (see Type::fileFormat).
///
/// \return - void
template <typename T>
void saveConvergenceStatus(
    const Context<T> *const inContext, const std::filesystem::path &inDataPath,
    const bool inVerbose = true,
    const Type::fileFormat inIterationDataSaveFormat = Type::csv) {
  const std::filesystem::path convergenceFile =
      inDataPath /
      ("convergence" + io::getFileExtension(inIterationDataSaveFormat));

  std::unique_ptr<io::IIterationDataSaveStrategy<T>> iterationDataSaver =
      io::getIterationDataSaver<T>(inIterationDataSaveFormat);

  const bool success = iterationDataSaver &&
                       iterationDataSaver->save(inContext, convergenceFile);
  if (!success) {
    std::cout << "-- ERROR! Failed to write: " << convergenceFile << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

/// \brief Save the order parameter, the current density, the induced vector
/// potential, and the induced magnetic-flux density to file.
///
/// \param inContext - Context object for simulation.
/// \param inDataPath - JSON object with parameter values.
/// \param inVerbose - Print success to terminal.
/// \param inGridDataSaveFormat - Enum specifying which format to save the
/// GridData in (see Type::fileFormat).
///
/// \return - void
template <typename T>
void saveGridData(const Context<T> *const inContext,
                  const std::filesystem::path &inDataPath,
                  const bool inVerbose = true,
                  const Type::fileFormat inGridDataSaveFormat = Type::csv) {
  // Fetch run parameters from context.
  Parameters<T> *parameters = inContext->getParameters();

  // Set file extension.
  const std::string extension = io::getFileExtension(inGridDataSaveFormat);

  // Fetch order parameter.
  GridGPU<T> &orderParameterSave =
      *(inContext->getComputeOrderParameter()->getResult());

  // Set order parameter to zero outside simulation domain (non-zero values
  // are used for interpolation).
  inContext->getGeometry()->zeroValuesOutsideDomainRef(&orderParameterSave);

  // Construct full file name for order-parameter file.
  const std::filesystem::path orderParameterFile =
      inDataPath / ("order_parameter" + extension);

  // Try to save to file. Exit program if fail.
  bool success = false;
  if (inGridDataSaveFormat == Type::binary) {
    success = orderParameterSave.write(orderParameterFile, parameters) == 0;
  } else {
    std::unique_ptr<io::ITextGridDataSaveStrategy<T>> textDataSaver =
        io::getTextGridDataSaver<T>(inGridDataSaveFormat);

    success = bool(textDataSaver) &&
              textDataSaver->save(orderParameterSave,
                                  inContext->getOrderParameter()->getTypes(),
                                  orderParameterFile);
  }
  if (!success) {
    std::cout << "-- ERROR! Could not write: " << orderParameterFile
              << std::endl;
    std::exit(EXIT_FAILURE);
  } else {
    if (inVerbose) {
      std::cout << "-- Wrote: " << orderParameterFile << std::endl;
    }
  }

  // Fetch current density.
  GridGPU<T> &currentSave = *(inContext->getComputeCurrent()->getResult());

  // Set current density to zero outside simulation domain (non-zero values
  // are used for interpolation).
  inContext->getGeometry()->zeroValuesOutsideDomainRef(&currentSave);

  // Construct full file name for current density.
  const std::filesystem::path currentDensityFile =
      inDataPath / ("current_density" + extension);

  // Try to save to file. Exit program if fail.
  success = false;
  if (inGridDataSaveFormat == Type::binary) {
    success = currentSave.write(currentDensityFile, parameters) == 0;
  } else {
    std::unique_ptr<io::ITextGridDataSaveStrategy<T>> textDataSaver =
        io::getTextGridDataSaver<T>(inGridDataSaveFormat);

    success =
        bool(textDataSaver) &&
        textDataSaver->save(currentSave, {"j_x", "j_y"}, currentDensityFile);
  }
  if (!success) {
    std::cout << "-- ERROR! Could not write: " << currentDensityFile
              << std::endl;
    std::exit(EXIT_FAILURE);
  } else {
    if (inVerbose) {
      std::cout << "-- Wrote: " << currentDensityFile << std::endl;
    }
  }

  // Fetch the vector potential.
  const GridGPU<T> &vectorPotentialSave =
      *(inContext->getInducedVectorPotential());

  // Construct full file name for vector potential
  const std::filesystem::path vectorPotentialFile =
      inDataPath / ("vector_potential" + extension);

  // Try to save to file. Exit program if fail.
  success = false;
  if (inGridDataSaveFormat == Type::binary) {
    success = vectorPotentialSave.write(vectorPotentialFile, parameters) == 0;
  } else {
    std::unique_ptr<io::ITextGridDataSaveStrategy<T>> textDataSaver =
        io::getTextGridDataSaver<T>(inGridDataSaveFormat);

    success = bool(textDataSaver) &&
              textDataSaver->save(vectorPotentialSave, {"a_x", "a_y"},
                                  vectorPotentialFile);
  }
  if (!success) {
    std::cout << "-- ERROR! Could not write: " << vectorPotentialFile
              << std::endl;
    std::exit(EXIT_FAILURE);
  } else {
    if (inVerbose) {
      std::cout << "-- Wrote: " << vectorPotentialFile << std::endl;
    }
  }

  // TODO(niclas): Convert to (reduced) coherence length? I.e. multiply by
  // 2pi.
  const T latticeSpacing = parameters->getGridElementSize();

  // The magnetic flux-density is simply the curl of the vector potential.
  const GridGPU<T> magneticFluxDensitySave =
      vectorPotentialSave.curl(latticeSpacing);

  // Construct full file name.
  const std::filesystem::path fluxDensityFile =
      inDataPath / ("magnetic_flux_density" + extension);

  // Try to save to file. Exit program if fail.
  success = false;
  if (inGridDataSaveFormat == Type::binary) {
    success = magneticFluxDensitySave.write(fluxDensityFile, parameters) == 0;
  } else {
    std::unique_ptr<io::ITextGridDataSaveStrategy<T>> textDataSaver =
        io::getTextGridDataSaver<T>(inGridDataSaveFormat);

    success =
        bool(textDataSaver) &&
        textDataSaver->save(magneticFluxDensitySave, {"b_z"}, fluxDensityFile);
  }
  if (!success) {
    std::cout << "-- ERROR! Could not write: " << fluxDensityFile << std::endl;
    std::exit(EXIT_FAILURE);
  } else {
    if (inVerbose) {
      std::cout << "-- Wrote: " << fluxDensityFile << std::endl;
    }
  }
}

/// \brief Wrapper to save data to file.
///
/// Wrapper that calls functions to save metadata parameters, free energy, and
/// observables to file.
///
/// \param inContext - Context object for simulation.
/// \param inDataPath - The path to data directory.
/// \param inGridDataSaveFormat - Enum specifying which format to save the
/// GridData in (see Type::fileFormat).
/// \param inIterationDataSaveFormat - Enum specifying which format to save
/// the per-iteration data in (see Type::fileFormat).
///
/// \return - void
template <typename T>
void saveDataWrapper(
    const Context<T> *const inContext, const std::filesystem::path inDataPath,
    const bool inVerbose,
    const Type::fileFormat inGridDataSaveFormat = Type::csv,
    const Type::fileFormat inIterationDataSaveFormat = Type::csv) {

  // Save parameters to JSON file.
  saveParametersWrapper(inContext, inDataPath, inVerbose);

  // Save order parameter, current density, induced vector-potential, and
  // induced magnetic flux-density, to file.
  saveGridData(inContext, inDataPath, inVerbose, inGridDataSaveFormat);
}

/// \brief Print iteration status.
///
/// Output status to terminal in each iteration.
///
/// \param inContext - Context object to delete.
///
/// \return - void
template <typename T> void printIterationStatus(Context<T> *inContext) {
  // Get the number of iterations so far.
  const int numIterations = inContext->getNumIterations();

  // Get the max residual.
  const T maxResidual = inContext->getResidual();

  // Get the computation time for this iteration.
  const T computeTime = inContext->getComputeTime();
  const T iterationsPerSecond = static_cast<T>(1) / computeTime;

  // Get the free-energy computation object.
  const ComputeFreeEnergy<T> *const computeFreeEnergy =
      inContext->getComputeFreeEnergy();

  if (computeFreeEnergy) {
    // Get the free energy.
    const T freeEnergy = computeFreeEnergy->getFreeEnergyPerUnitArea();

    // Print values.
    std::printf("Iteration = %d | Max Residual = %e | Free Energy = %f | "
                "Iterations / Second = %f\n",
                numIterations, maxResidual, freeEnergy, iterationsPerSecond);
  } else {
    // Print values.
    std::printf(
        "Iteration = %d | Max Residual = %e | Iterations / Second = %f\n",
        numIterations, maxResidual, iterationsPerSecond);
  }
}

/// \brief Create visualization object
///
/// \param inContext - Context object.
///
/// \return - Visualization object pointer.
// TODO(patric): Move this functionality to visualization class.
template <typename T>
Visualization<T> *createVisualization(const Context<T> *const inContext) {
  // Compute the number of columns and rows. The number of plots in
  // visualization = numRows * numCols.
  int numCols = 0;
  int numRows = 0;

  // Add order parameter.
  if (inContext->getComputeOrderParameter()) {
    // One column for each order-parameter component.
    const int numOrderParameterComponents =
        inContext->getOrderParameter()->getNumOrderParameterComponents();
    numCols += numOrderParameterComponents;

    // Two rows: magnitude and phase.
    numRows = 2;
  }

  // Add current density.
  if (inContext->getComputeCurrent()) {
    // A single column for the current density.
    numCols++;

    // Three rows: magnitude and x/y-components.
    numRows = 3;
  }

  // Add vector potential (and magnetic field).
  const T penetrationDepth = inContext->getParameters()->getPenetrationDepth();
  if (penetrationDepth > static_cast<T>(0)) {
    // One column for the induced vector potential, and one for the induced
    // magnetic field.
    numCols += 2;

    // Three rows: magnitude and x/y-components.
    numRows = 3;
  }

  // Create visualization object.
  const int gridResolutionBase =
      inContext->getParameters()->getGridResolutionBase();
  Visualization<T> *gui = new Visualization<T>(
      gridResolutionBase, gridResolutionBase, numCols, numRows);

  gui->setGeometry(inContext->getGeometry());
  gui->setPrintParameters(inContext->getParameters());
  return gui;
}

/// \brief Plot data in OpenGL GUI.
///
/// \param inContext - Context object.
/// \param inGUI - Visualization object.
///
/// \return - void
template <typename T>
void display(Context<T> *inContext, Visualization<T> *inGUI) {
  // Clear the plotting region.
  inGUI->clearImage();

  // Column index of index.
  int colIdx = 0;

  // Get compute order-parameter object.
  const ComputeOrderParameter<T> *const computeOrderParameter =
      inContext->getComputeOrderParameter();

  if (computeOrderParameter) {
    // Plot the order parameter.
    const int numOrderParameterComponents =
        inContext->getOrderParameter()->getNumOrderParameterComponents();
    for (; colIdx < numOrderParameterComponents; ++colIdx) {
      // Get the order parameter component.
      const GridGPU<T> component =
          *(inContext->getOrderParameter()->getOrderParameterComponentField(
              colIdx));

      // Plot the order parameter.
      inGUI->label("|Delta_" + std::to_string(colIdx + 1) + "|/k_BT_c", colIdx,
                   inGUI->getNumImagesY() - 1);
      inGUI->label("Chi_" + std::to_string(colIdx + 1), colIdx,
                   inGUI->getNumImagesY() - 2);
      inGUI->plotImage(component, colIdx);
    }
  }

  // Get compute current object.
  const ComputeCurrent<T> *const computeCurrent =
      inContext->getComputeCurrent();

  if (computeCurrent) {
    // Get the current.
    const GridGPU<T> J = *(inContext->getComputeCurrent()->getResult());

    // Plot the current.
    inGUI->label("|J|/j_d", colIdx, inGUI->getNumImagesY() - 1);
    inGUI->label("J_x/j_d", colIdx, inGUI->getNumImagesY() - 2);
    inGUI->label("J_y/j_d", colIdx, inGUI->getNumImagesY() - 3);
    inGUI->plotImage(J, colIdx);
    colIdx++;

    // The London penetration depth.
    const T penetrationDepth =
        inContext->getParameters()->getPenetrationDepth();

    if (penetrationDepth > static_cast<T>(0)) {
      // Just to set "erase = false", i.e. plot everywhere, not just in the
      // grain.
      const bool erase = false;

      // Get the vector potential.
      const GridGPU<T> A = *(inContext->getInducedVectorPotential());

      // Plot the induced vector potential.
      inGUI->label("|A|/A_0", colIdx, inGUI->getNumImagesY() - 1);
      inGUI->label("A_x/A_0", colIdx, inGUI->getNumImagesY() - 2);
      inGUI->label("A_y/A_0", colIdx, inGUI->getNumImagesY() - 3);
      inGUI->plotImage(A, colIdx, erase);
      colIdx++;

      // Plot the induced magnetic field.
      const T latticeSpacing = inContext->getParameters()->getGridElementSize();
      const GridGPU<T> divA = A.div(latticeSpacing);
      const GridGPU<T> magneticFluxDensity = A.curl(latticeSpacing);
      inGUI->label("B/B_0", colIdx, inGUI->getNumImagesY() - 1);
      inGUI->plotImage(magneticFluxDensity, colIdx, erase);
    }
  }

  // Render the plots.
  inGUI->draw();
}

/// \brief Iteration status enum class
///
/// kContinue - Everything is fine, carry on.
/// kQuit - We have either converged or lost patience.
/// kSave - Time to save the results so far.
enum class Status { kContinue, kQuit, kSave };

/// \brief Single iteration simulation wrapper
///
/// Wrapper for running one iteration without plotting results.
///
/// \param inOutContext - Context object for simulation.
/// \param inVerbose - Print iteration status to terminal or not.
/// \param inNumIterationsMin - The minimum number of iterations.
/// \param inNumIterationsMax - The maximum number of iterations.
/// \param inSaveDataInterval - How frequently to save data.
///
/// \return - Simulation status.
template <typename T>
Status runIteration(Context<T> *inOutContext, const bool inVerbose = true,
                    const int inNumIterationsMin = 0,
                    const int inNumIterationsMax = 10000,
                    const int inSaveDataInterval = 0) {
  // Solve equations for all compute objects.
  inOutContext->compute();

  // Output iteration status to the terminal.
  if (inVerbose) {
    printIterationStatus<T>(inOutContext);
  }

  const int numIterations = inOutContext->getNumIterations();

  // Check for convergence.
  const bool isConverged =
      (inOutContext->isConverged()) && (numIterations >= inNumIterationsMin);

  // Check for maximum number of iterations.
  const bool patienceReached = (numIterations >= inNumIterationsMax);

  // Print status if simulation finished.
  if (inVerbose) {
    if (isConverged) {
      std::cout << "-- SUCCESS! Simulation converged." << std::endl;
    } else if (patienceReached) {
      std::cout << "-- FAILURE! Simulation did not converge." << std::endl;
    }
  }

  // Determine what to do.
  const bool quit = isConverged || patienceReached || internal::guiUserQuit;
  const bool saveInterval = inSaveDataInterval > 0
                                ? (numIterations % inSaveDataInterval == 0)
                                : false;
  const bool saveData = internal::guiUserSaveData || saveInterval;

  // Set status.
  Status status;
  if (quit) {
    status = Status::kQuit;
  } else if (saveData) {
    status = Status::kSave;
    // Reset save data key stroke.
    internal::guiUserSaveData = false;
  } else {
    status = Status::kContinue;
  }

  return status;
}

/// \brief Simulation wrapper
///
/// Wrapper for running the simulation until convergence.
///
/// \param inOutContext Context object for simulation.
/// \param inVerbose Verbosity.
/// \param inNumIterationsMin The minimum number of iterations.
/// \param inNumIterationsMax The maximum number of iterations.
/// \param inVisualize Run with graphics or in terminal mode.
/// \param inDataPath Path to data directory.
/// \param inSaveData Whether to save data or not.
/// \param inSaveDataInterval How frequently to save data.
/// \param inGridDataSaveFormat Enum specifying which format to save the
/// GridData in (see Type::fileFormat).
///
/// \return - void.
template <typename T>
void runSimulation(Context<T> *inOutContext, const bool inVerbose = true,
                   const int inNumIterationsMin = 0,
                   const int inNumIterationsMax = 10000,
                   const bool inVisualize = false,
                   const std::filesystem::path &inDataPath = {},
                   const bool inSaveData = false,
                   const int inSaveDataInterval = 0,
                   const Type::fileFormat inGridDataSaveFormat = Type::csv) {
  Status status = Status::kContinue;
  bool quit = false;

  while (!quit) {
    if (status != Status::kQuit) {
      // Run one iteration.
      status = runIteration(inOutContext, inVerbose, inNumIterationsMin,
                            inNumIterationsMax, inSaveDataInterval);

      if (inSaveData) {
        // Append convergence status to file.
        saveConvergenceStatus(inOutContext, inDataPath, inVerbose,
                              inGridDataSaveFormat);

        // Save grid data and results.
        if (status == Status::kSave) {
          saveDataWrapper(inOutContext, inDataPath, inVerbose,
                          inGridDataSaveFormat);
        }
      }
    } else {
      // When no computation is done it is unnecessary to update the window as
      // fast as possible.
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (inVisualize) {
      // Check if the Glut window is open.
      const int windowID = glutGetWindow();
      const bool noWindow = windowID == 0;

      // Set quit flag.
      quit = noWindow || internal::guiUserQuit;

      // Plot data in OpenGL.
      if (!quit) {
        glutMainLoopEvent();
      }
    } else {
      quit = (status == Status::kQuit);
    }
  }
}

/// \brief Cleanup simulation
///
/// Just deleting the context and visualization objects.
///
/// \param inContext - Context object for simulation.
/// \param inGUI - Visualization GUI object.
///
/// \return - void.
template <typename T>
void cleanup(Context<T> *inContext, Visualization<T> *inGUI = nullptr) {
#ifndef NDEBUG
  std::cout << "-- Cleaning up..." << std::endl;
#endif

  delete inGUI;
  delete inContext;

#ifndef NDEBUG
  std::cout << "-- Done!" << std::endl;
#endif
}
} // namespace run_utils
} // namespace scg

#endif // RUN_UTILS_H_
