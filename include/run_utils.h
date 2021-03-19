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
#include "Grid.h"
#include "Parameters.h" // Parameters specifying which the superconducting system to simulate
#include "Type.h" // Custom types (real, complex, ...)
#include "Visualization.h"
#include "io/csv_writers.h"
#include "io/write_utils.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <errno.h> // errno, ENOENT, EEXIST
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <json/json.h>
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
  std::cout << "key pressed: " << inKey << ", at (" << inX << ", " << inY
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
/// This function saves two json files: one for internal parameters, and one for
/// scalar results. Motivation, there are some internal parameters that are not
/// possible to infer without running the program (the exact number of lattice
/// points inside the grain), these are very useful to have access to when
/// post-processing (e.g. for doing area normalization). Finally, there are some
/// scalar results (e.g. area-averaged quantities) that are more appropriate and
/// convenient to have in json rather than csv.
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

  // Average vector-potential magnitude.
  // TODO(niclas): Should this be computed outside of the grain?
  GridGPU<T> vectorPotential = *(inContext->getInducedVectorPotential());
  inContext->getGeometry()->zeroValuesOutsideDomainRef(&vectorPotential);
  resultsJSON["vector_potential_average_magnitude"] =
      vectorPotential.sumMagnitude() / static_cast<T>(numGrainLatticeSites);

  // Average flux density.
  // TODO(niclas): Should this be computed outside of the grain?
  const T latticeElementSize = inContext->getParameters()->getGridElementSize();
  resultsJSON["flux_density_average"] =
      vectorPotential.curl(latticeElementSize).sumField() /
      static_cast<T>(numGrainLatticeSites);

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
/// \param inIterationDataSaveFormat - Enum specifying which format to save the
/// per-iteration data in (see Type::fileFormat).
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

  if (iterationDataSaver &&
      iterationDataSaver->save(inContext, convergenceFile)) {
    if (inVerbose) {
      std::cout << "Wrote convergence status to file: " << convergenceFile
                << std::endl;
    }
  } else {
    std::cout << "ERROR! Failed to save " << convergenceFile << "file."
              << std::endl;
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

  // Set order parameter to zero outside simulation domain (non-zero values are
  // used for interpolation).
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
    std::cout << "ERROR! Could not save Order Parameter to file: "
              << orderParameterFile << std::endl;
    std::exit(EXIT_FAILURE);
  } else {
    if (inVerbose) {
      std::cout << "Saved order parameter to file: " << orderParameterFile
                << std::endl;
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
    std::cout << "Could not save current density to file: "
              << currentDensityFile << std::endl;
    std::exit(EXIT_FAILURE);
  } else {
    if (inVerbose) {
      std::cout << "Saved current density to file: " << currentDensityFile
                << std::endl;
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
    std::cout << "Could not save induced vector-potential to file: "
              << vectorPotentialFile << std::endl;
    std::exit(EXIT_FAILURE);
  } else {
    if (inVerbose) {
      std::cout << "Saved induced vector-potential  to file: "
                << vectorPotentialFile << std::endl;
    }
  }

  // TODO(niclas): Convert to (reduced) coherence length? I.e. multiply by 2pi.
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
    std::cout << "Could not save induced flux-density to file: "
              << fluxDensityFile << std::endl;
    std::exit(EXIT_FAILURE);
  } else {
    if (inVerbose) {
      std::cout << "Saved induced flux-density to file: " << fluxDensityFile
                << std::endl;
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
/// \param inIterationDataSaveFormat - Enum specifying which format to save the
/// per-iteration data in (see Type::fileFormat).
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

  // Append convergence status to file.
  saveConvergenceStatus(inContext, inDataPath, inVerbose,
                        inIterationDataSaveFormat);

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
      inGUI->plotImage(A, colIdx, erase);
      colIdx++;

      // Plot the induced magnetic field.
      const T latticeSpacing = inContext->getParameters()->getGridElementSize();
      const GridGPU<T> divA = A.div(latticeSpacing);
      const GridGPU<T> magneticFluxDensity = A.curl(latticeSpacing);
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
/// \param inSaveData - Whether to save data or not.
/// \param inSaveDataInterval - How frequently to save data.
///
/// \return - Simulation status.
template <typename T>
Status runIteration(Context<T> *inOutContext, const bool inVerbose = true,
                    const int inNumIterationsMin = 0,
                    const int inNumIterationsMax = 10000,
                    const bool inSaveData = false,
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
  const bool patienceReached = (numIterations > inNumIterationsMax);

  // Print status if simulation finished.
  if (isConverged) {
    std::cout << "SUCCESS! Simulation converged." << std::endl;
  } else if (patienceReached) {
    std::cout << "FAILURE! Simulation did not converge." << std::endl;
  }

  // Determine what to do.
  const bool quit = isConverged || patienceReached || internal::guiUserQuit;
  const bool saveInterval = inSaveDataInterval > 0
                                ? (numIterations % inSaveDataInterval == 0)
                                : false;
  const bool saveData =
      (internal::guiUserSaveData || saveInterval) && inSaveData;

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
                            inNumIterationsMax, inSaveData, inSaveDataInterval);

      if (status == Status::kSave) {
        saveDataWrapper(inOutContext, inDataPath, inVerbose,
                        inGridDataSaveFormat);
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
  std::cout << "Cleaning up..." << std::endl;
#endif

  delete inGUI;
  delete inContext;

#ifndef NDEBUG
  std::cout << "Done!" << std::endl;
#endif
}
} // namespace run_utils
} // namespace scg

#endif // RUN_UTILS_H_
