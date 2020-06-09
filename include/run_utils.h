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

#include "ComputeVectorPotential.h" // Compute the vector potential induced by currents
#include "Context.h"
#include "Grid.h"     // Grid and GridData
#include "GridData.h" // Grid and GridData
#include "Parameters.h" // Parameters specifying which the superconducting system to simulate
#include "Type.h" // Custom types (real, complex, ...)
#include "Visualization.h"

#include "nlohmann_json_wrapper.h"
#include <boost/filesystem.hpp>
#include <cmath>
#include <iomanip>

#ifdef __linux__
#include <GL/freeglut.h>
#else
#include <GLUT/glut.h>
#endif

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
void myKeyboardFunc(const unsigned char inKey, const int inX, const int inY) {
  std::cout << "key pressed: " << inKey << ", at (" << inX << ", " << inY
            << ")\n";
  if (inKey == 'q' || inKey == 'Q') {
    internal::guiUserQuit = true;
  } else if (inKey == 's' || inKey == 'S') {
    internal::guiUserSaveData = true;
  }
}

/// \brief Create directory if not existing.
///
/// \param inDirectory - Path to directory.
///
/// \return - void
void createDirectory(const char *const inDirectory) {
  // TODO(patric): Replace boost with standard functions.
  const boost::filesystem::path rootPath(inDirectory);
  boost::filesystem::create_directories(rootPath);
}

/// \brief Create directory if not existing.
///
/// \param inDirectory - Path to directory.
///
/// \return - void
void createDirectory(const std::string inDirectory) {
  createDirectory(inDirectory.c_str());
}

/// \brief Write line to file
///
/// \param inFile - File to write to.
/// \param inOStream - ostream object containing the line to write.
/// \param inFileMode - Writing mode (e.g. append, truncate).
///
/// \return - void
void writeLineToFile(
    const char *const inFile, const std::ostream &inOStream,
    const std::ios_base::openmode inFileMode = std::ios_base::app) {
  std::ofstream outfile;

  std::stringstream oss;
  oss << inOStream.rdbuf();
  outfile.open(inFile, inFileMode);
  outfile << oss.str() << std::endl;
  outfile.close();
}

/// \brief Save simulation parameters to JSON file.
///
/// \param inJSON - JSON object with parameter values.
/// \param inFile - Full path and filename of file to save.
/// \param inVerbose - Whether or not to print when file is written.
///
/// \return - void
void saveParameterFileJSON(const nlohmann::json &inJSON,
                           const char *const inFile,
                           const bool inVerbose = false) {

  std::ofstream ofstreamJSON(inFile);
  const int indentation = 4;
  ofstreamJSON << std::setw(indentation) << inJSON << std::endl;

  if (inVerbose) {
    std::cout << "Wrote parameters to JSON file: " << inFile << std::endl;
  }
}

/// \brief Save simulation parameters to JSON file.
///
/// \param inContext - Context object for simulation.
/// \param inJSON - JSON object with parameter values.
/// \param inDataPath - Directory where to put JSON file.
/// \param inDataFileName - Name of JSON file to write.
///
/// \return - void
template <typename T>
void saveParametersWrapper(
    const Context<T> *const inContext, const nlohmann::json &inJSON,
    const char *const inDataPath, const bool inVerbose = true,
    const char *const inDataFileName = (char *)"config") {
  //***************************************************
  // CONFIG PARAMETERS
  //***************************************************
  // Construct full filepath to JSON-file.
  char fullPathJSON[512];
  snprintf(fullPathJSON, sizeof(fullPathJSON), "%s/%s.json", inDataPath,
           inDataFileName);

  // Save JSON parameters (if not already saved or updated)
  saveParameterFileJSON(inJSON, fullPathJSON, inVerbose);

  //***************************************************
  // ADDITIONAL USEFUL INTERNAL PARAMETERS
  //***************************************************
  // Construct full filepath for extra internal parameters.
  char fullPathInternalParameters[512];
  snprintf(fullPathInternalParameters, sizeof(fullPathInternalParameters),
           "%s/%s_internal_parameters.json", inDataPath, inDataFileName);

  // Add other useful parameters for post-processing.
  nlohmann::json internalParameters;
  internalParameters["area_compute"] = inContext->getGeometry()->area();
  internalParameters["internal_lattice_points"] =
      inContext->getGeometry()->getNumGrainLatticeSites();
  internalParameters["gauge"] =
      inContext->getRiccatiSolver()->getVectorPotentialType();
  internalParameters["external_flux_density"] =
      inContext->getParameters()->getMagneticFieldStrength();
  internalParameters["grain_margin"] = qcs::GRAIN_SIZE_MARGIN;

  // Save additional JSON parameters (if not already saved or updated)
  saveParameterFileJSON(internalParameters, fullPathInternalParameters,
                        inVerbose);
}

/// \brief Save free energy to text file.
///
/// \param inContext - Context object for simulation.
/// \param inDataPath - Directory where to put free-energy file.
/// \param inVerbose - Print success to terminal.
/// \param inDataFileName - Name of free-energy file to write.
/// \param dataFloatDecimals - Number of floating point digits to write in file.
///
/// \return - void
template <typename T>
void saveFreeEnergy(const Context<T> *const inContext,
                    const char *const inDataPath, const bool inVerbose = true,
                    const char *const inDataFileName = (char *)"free_energy",
                    const int dataFloatDecimals = 10) {
  // Construct full file name for free energy
  char freeEnergyFile[512];
  snprintf(freeEnergyFile, sizeof(freeEnergyFile), "%s/%s.dat", inDataPath,
           inDataFileName);

  // Add header if file does not exist
  if (!boost::filesystem::exists(freeEnergyFile)) {
    std::stringstream ss_header;
    ss_header << "# iterationNumber residual freeEnergyPerUnitArea";
    writeLineToFile(freeEnergyFile, ss_header);
  }

  // Append line with energy and other information
  std::stringstream ss;
  ss << inContext->getNumIterations() << " ";
  ss << inContext->getResidual(0) << " ";
  ss.precision(dataFloatDecimals);
  ss << inContext->getComputeFreeEnergy()->getFreeEnergyPerUnitArea() << " ";
  writeLineToFile(freeEnergyFile, ss);

  if (inVerbose) {
    std::cout << "Wrote free energy to file: " << freeEnergyFile << std::endl;
  }
}

/// \brief Save induced magnetic flux density to file
///
/// \param inContext - Context object
/// \param inDataPath - Path where to save save grid data file.
/// \param inVerbose - Print success to terminal.
/// \param inDataFileName - File name of induced flux density file.
// Todo(patric): Create utils_io.h and move there.
template <typename T>
void saveInducedFluxDensity(
    const Context<T> *const inContext, const char *const inDataPath,
    const bool inVerbose = true,
    const char *const inDataFileName = (char *)"magnetic_flux_density") {

  // Construct full file name.
  char gridDataFile[512];
  snprintf(gridDataFile, sizeof(gridDataFile), "%s/%s.grd", inDataPath,
           inDataFileName);

  // Fetch parameters from context.
  Parameters<T> *parameters = inContext->getParameters();

  // Fetch current density and set to zero outside grain.
  typename Grid<T>::GPU *current = inContext->getComputeCurrent()->getResult();
  inContext->getGeometry()->zeroValuesOutsideDomain(current);
  const int numLatticePoints = current->dimX();

  // Domain structure for vector potential calculation.
  Grid<char>::GPU *domain =
      inContext->getGeometry()->createDomain(numLatticePoints);

  // Create GPU storage for vector potential.
  typename Grid<T>::GPU *vectorPotential = new
      typename Grid<T>::GPU(numLatticePoints, numLatticePoints, 2, Type::real);

  // Convert lattice spacing back to proper coherence lengths:
  // hbar*vF/(2*Pi*kB*Tc).
  const T latticeSpacing =
      parameters->getGridElementSize() * static_cast<T>(2.0 * M_PI);

  // Calculate coefficients for Poissons equation
  // TODO(patric): this should be done inside computeVectorPotential!
  const T penetrationDepth = parameters->getPenetrationDepth();
  const T minusInvKappaSquared =
      -static_cast<T>(1.0) / (penetrationDepth * penetrationDepth);

  // Solve the vector potential from the current.
  computeVectorPotential(vectorPotential, current, domain, latticeSpacing,
                         minusInvKappaSquared);

  // Solve the magnetic flux density from the vector potential through Maxwell.
  typename Grid<T>::GPU magneticFluxDensity =
      vectorPotential->curl(latticeSpacing);

  delete vectorPotential;

  // Try to save induced flux density to file. Exit program if fail.
  if (magneticFluxDensity.write(gridDataFile, parameters)) {
    std::cout << "ERROR! Could not save induced flux density to file: "
              << gridDataFile << std::endl;
    exit(1);
  } else {
    if (inVerbose) {
      std::cout << "Saved induced flux density to file: " << gridDataFile
                << std::endl;
    }
  }
}

/// \brief Save order parameter and current to file.
///
/// \param inContext - Context object for simulation.
/// \param inDataPath - JSON object with parameter values.
/// \param inVerbose - Print success to terminal.
/// \param inOrderParameterFileName - File name of order-parameter file.
/// \param inCurrentDensityFileName - File name of current-density file.
///
/// \return - void
template <typename T>
void saveGridData(
    const Context<T> *const inContext, const char *const inDataPath,
    const bool inVerbose = true,
    const char *const inOrderParameterFileName = (char *)"order_parameter",
    const char *const inCurrentDensityFileName = (char *)"current_density") {
  // Fetch run parameters from context.
  Parameters<T> *parameters = inContext->getParameters();

  // Fetch order parameter.
  typename Grid<T>::GPU *const orderParameterSave =
      inContext->getComputeOrderParameter()->getResult();

  // Set order parameter to zero outside simulation domain (non-zero values are
  // used for interpolation).
  inContext->getGeometry()->zeroValuesOutsideDomain(orderParameterSave);

  // Construct full file name for order-parameter file.
  char orderParameterFile[512];
  snprintf(orderParameterFile, sizeof(orderParameterFile), "%s/%s.grd",
           inDataPath, inOrderParameterFileName);

  // Try to save order parameter to file. Exit program if fail.
  if (orderParameterSave->write(orderParameterFile, parameters)) {
    std::cout << "ERROR! Could not save Order Parameter to file: "
              << orderParameterFile << std::endl;
    exit(1);
  } else {
    if (inVerbose) {
      std::cout << "Saved order parameter to file: " << orderParameterFile
                << std::endl;
    }
  }

  // Fetch current density.
  typename Grid<T>::GPU *const currentSave =
      inContext->getComputeCurrent()->getResult();

  // Set current density to zero outside simulation domain (non-zero values are
  // used for interpolation).
  inContext->getGeometry()->zeroValuesOutsideDomain(currentSave);

  // Construct full file name for current density.
  char currentDensityFile[512];
  snprintf(currentDensityFile, sizeof(currentDensityFile), "%s/%s.grd",
           inDataPath, inCurrentDensityFileName);

  // Try to save order parameter to file. Exit program if fail.
  if (currentSave->write(currentDensityFile, parameters)) {
    std::cout << "Could not save current density to file: "
              << currentDensityFile << std::endl;
    exit(1);
  } else {
    if (inVerbose) {
      std::cout << "Saved current density to file: " << currentDensityFile
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
/// \param inJSON - JSON object with parameter values.
///
/// \return - void
template <typename T>
void saveDataWrapper(const Context<T> *const inContext,
                     const nlohmann::json &inJSON, const bool inVerbose) {
  // Construct full directory path.
  char dataPath[512];
  const std::string dataBasePath = inJSON["data_path"].get<std::string>();
  snprintf(dataPath, sizeof(dataPath), "%s/T%2.4f", dataBasePath.c_str(),
           inJSON["temperature"].get<double>());

  // Create directory if not existing.
  createDirectory(dataPath);

  // Save parameters to JSON file.
  saveParametersWrapper(inContext, inJSON, dataPath, inVerbose);

  // Append free energy and iteration status to file.
  saveFreeEnergy(inContext, dataPath, inVerbose);

  // Save order parameter and current to file.
  saveGridData(inContext, dataPath, inVerbose);

  // Save induced magnetic field to file.
  saveInducedFluxDensity(inContext, dataPath, inVerbose);
}

/// \brief Print iteration status.
///
/// Output status to terminal in each iteration.
///
/// \param inContext - Context object to delete.
///
/// \return - void
template <typename T> void printIterationStatus(Context<T> *inContext) {
  std::cout << "Iteration = " << inContext->getNumIterations() << ", ";
  std::cout << "Residual = " << inContext->getResidual(0) << ", ";
  std::cout << "Free Energy = "
            << inContext->getComputeFreeEnergy()->getFreeEnergyPerUnitArea()
            << ", ";
  std::cout << "Iteration time = " << inContext->getComputeTime() << std::endl;
}

/// \brief Create visualization object
///
/// \param inContext - Context object.
/// \param inPlotOrderParameter - Whether to plot the order parameter or
/// not.
/// \param inPlotCurrentDensity - Whether to plot the current density
///
/// \return - Visualization object pointer.
// TODO(patric): Move this functionality to visualization class.
template <typename T>
Visualization<T> *createVisualization(const Context<T> *const inContext,
                                      const bool inPlotOrderParameter = true,
                                      const bool inPlotCurrentDensity = true) {

  const int numOrderParameterComponents =
      inContext->getOrderParameter()->getNumOrderParameterComponents();

  // Number of plots in visualization = numPlotsY * numPlotsX.
  const int numPlotsX = (inPlotCurrentDensity ? 3 : 2);
  const int numPlotsY =
      static_cast<int>(inPlotOrderParameter) * numOrderParameterComponents +
      static_cast<int>(inPlotCurrentDensity);

  // Create visualization object.
  const int gridResolutionBase =
      inContext->getParameters()->getGridResolutionBase();
  Visualization<T> *gui = new Visualization<T>(
      gridResolutionBase, gridResolutionBase, numPlotsX, numPlotsY);

  gui->setGeometry(inContext->getGeometry());
  gui->setPrintParameters(inContext->getParameters());
  return gui;
}

/// \brief Plot data in OpenGL GUI.
///
/// \param inContext - Context object.
/// \param inGUI - Visualization object.
/// \param inPlotOrderParameter - Whether to plot the order parameter or
/// not.
/// \param inPlotCurrentDensity - Whether to plot the current density
/// or not.
///
/// \return - void
template <typename T>
void display(Context<T> *inContext, Visualization<T> *inGUI,
             const bool inPlotOrderParameter = true,
             const bool inPlotCurrentDensity = true) {
  // Clear the plotting region
  inGUI->clearImage();

  // Plot the current density.
  if (inPlotCurrentDensity) {
    // Plot the x-component of the current density.
    inGUI->plotImage(*(inContext->getComputeCurrent()->getResult()), 0,
                     Type::real, 0, 0);

    // Plot the y-component of the current density.
    inGUI->plotImage(*(inContext->getComputeCurrent()->getResult()), 1,
                     Type::real, 1, 0);

    // Plot the magnitude of the current density.
    inGUI->plotImage(
        inContext->getComputeCurrent()->getResult()->fieldMagnitude(), 0,
        Type::real, 2, 0);
  }

  // Plot the order parameter.
  if (inPlotOrderParameter) {
    const int numOrderParameterComponents =
        inContext->getOrderParameter()->getNumOrderParameterComponents();
    for (int i = 0; i < numOrderParameterComponents; ++i) {
      // Plot the magnitude of the order parameter.
      inGUI->plotImage(
          inContext->getComputeOrderParameter()->getResult()->abs(), i,
          Type::real, 0, i + 1);

      // Plot the phase of the order parameter.
      inGUI->plotImage(
          inContext->getComputeOrderParameter()->getResult()->complexPhase(), i,
          Type::real, 1, i + 1);
    }
  }

  // Render the plots
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
/// \param inPrintStatus - Print iteration status to terminal or not.
/// \param inNumIterationsMin - The minimum number of iterations.
/// \param inNumIterationsMax - The maximum number of iterations.
/// \param inSaveData - Whether to save data or not.
/// \param inSaveDataInterval - How frequently to save data.
///
/// \return - Simulation status.
template <typename T>
Status runIteration(Context<T> *inOutContext, const bool inPrintStatus = true,
                    const int inNumIterationsMin = 100,
                    const int inNumIterationsMax = 1000,
                    const bool inSaveData = false,
                    const int inSaveDataInterval = 200) {
  // Solve equations for all compute objects.
  inOutContext->compute();

  // Output iteration status to the terminal.
  if (inPrintStatus) {
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
  const bool saveData = (internal::guiUserSaveData ||
                         (numIterations % inSaveDataInterval == 0)) &&
                        inSaveData;

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
/// \param inOutContext - Context object for simulation.
/// \param inVisualize - Run with graphics or in terminal mode.
/// \param inPrintStatus - Print iteration status to terminal or not.
/// \param inNumIterationsMin - The minimum number of iterations.
/// \param inNumIterationsMax - The maximum number of iterations.
/// \param inSaveData - Whether to save data or not.
/// \param inSaveDataInterval - How frequently to save data.
/// \param inJSON - JSON object with parameter values.
///
/// \return - void.
template <typename T>
void runSimulation(Context<T> *inOutContext, const bool inVisualize,
                   const bool inPrintStatus = true,
                   const int inNumIterationsMin = 100,
                   const int inNumIterationsMax = 1000,
                   const bool inSaveData = false,
                   const int inSaveDataInterval = 200,
                   const nlohmann::json &inJSON = {}) {
  run_utils::Status status = run_utils::Status::kContinue;
  while (status != Status::kQuit) {
    // Run one iteration,
    status = runIteration(inOutContext, inPrintStatus, inNumIterationsMin,
                          inNumIterationsMax, inSaveData, inSaveDataInterval);

    if (status == Status::kSave) {
      if (inJSON.empty()) {
        std::cout
            << "ERROR! Attempting to save data, but JSON parameter "
               "object was empty! (call to runSimulation() in run_utils.h)"
            << std::endl;
      } else {
        saveDataWrapper(inOutContext, inJSON, inPrintStatus);
      }
    }

    if (inVisualize) {
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
void cleanup(Context<T> *inContext, Visualization<T> *inGUI = NULL) {
  std::cout << "Cleaning up..." << std::endl;
  delete inGUI;
  delete inContext;
  std::cout << "Done!" << std::endl;
}
} // namespace run_utils

#endif // RUN_UTILS_H_
