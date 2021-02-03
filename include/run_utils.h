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

inline std::string getFileExtension(Type::fileFormat format) {
  switch (format) {
  case Type::binary:
    return ".grd";
  case Type::csv:
    return ".csv";
  default:
    return "";
  }
}
} // namespace internal

//===------------------------------ Classes ------------------------------===//
/// \brief Interface for saving per-iteration data
///
/// Interface for a class which encapsulates data saving on a per-iteration
/// basis. Should be implemented for each save strategy (e.g. new save format,
/// or different type of info to be stored).
template <typename T> class IIterationDataSaveStrategy {
public:
  virtual bool save(const Context<T> *const inContext,
                    const std::filesystem::path &inFilePath) = 0;

  virtual ~IIterationDataSaveStrategy() = default;
};

template <typename T>
class CSVIterationDataSaveStrategy : public IIterationDataSaveStrategy<T> {
public:
  CSVIterationDataSaveStrategy() = default;

  CSVIterationDataSaveStrategy(const std::string &delimiter,
                               const int floatPrecision = 10)
      : floatPrecision(floatPrecision), delimiter(delimiter) {}

  bool save(const Context<T> *const inContext,
            const std::filesystem::path &inFilePath) override {

    if (!std::filesystem::exists(inFilePath)) {
      if (!appendHeader(inFilePath)) {
        return false;
      };
    }
    return appendRow(inContext, inFilePath);
  }

private:
  const int floatPrecision = 10;
  const std::string delimiter = ",";

  bool appendHeader(const std::filesystem::path &inFilePath) const {
    std::ostringstream oss;
    oss << "iteration" << delimiter << "residual" << delimiter
        << "free_energy_per_unit_area";
    return appendToFile(oss.str(), inFilePath);
  };

  bool appendRow(const Context<T> *const inContext,
                 const std::filesystem::path &inFilePath) const {

    std::ostringstream oss;
    oss << inContext->getNumIterations() << delimiter
        << inContext->getResidual() << delimiter
        << std::setprecision(floatPrecision)
        << inContext->getComputeFreeEnergy()->getFreeEnergyPerUnitArea();
    return appendToFile(oss.str(), inFilePath);
  };

  bool appendToFile(const std::string &inLine,
                    std::filesystem::path inFilePath) const {
    std::ofstream outFile(inFilePath, std::ios_base::app);
    if (!outFile.good()) {
      return false;
    }
    outFile << inLine << std::endl;
    return true;
  }
};

/// \brief Interface for saving GridGPU data type in textual format
///
/// Interface for a class which encapsulates saving GridGPU instances.
/// Should be implemented for each save strategy (e.g. new save format,
/// or different type of info to be stored).
template <typename T> class ITextGridDataSaveStrategy {
public:
  virtual bool save(const GridGPU<T> *const inGridGPU,
                    const std::vector<std::string> &inFieldTitles,
                    const std::filesystem::path &inFilePath) = 0;

  virtual ~ITextGridDataSaveStrategy() = default;
};

template <typename T>
class CSVGridDataSaveStrategy : public ITextGridDataSaveStrategy<T> {
public:
  CSVGridDataSaveStrategy() = default;

  CSVGridDataSaveStrategy(const std::string &delimiter,
                          const int floatPrecision = 10)
      : floatPrecision(floatPrecision), delimiter(delimiter) {}

  bool save(const GridGPU<T> *const inGridGPU,
            const std::vector<std::string> &inFieldTitles,
            const std::filesystem::path &inFilePath) override {

    if (inFieldTitles.size() != inGridGPU->numFields()) {
      return false;
    }

    if (!std::filesystem::exists(inFilePath)) {
      if (!appendHeader(inFieldTitles, inGridGPU->isComplex(), inFilePath)) {
        return false;
      };
    }
    return appendData(inGridGPU, inFilePath);
  }

private:
  const int floatPrecision = 10;
  const std::string delimiter = ",";

  bool appendHeader(const std::vector<std::string> &inColumnTitles,
                    const bool isComplex,
                    const std::filesystem::path &inFilePath) const {
    std::ostringstream oss;
    oss << "x_idx" << delimiter << "y_idx" << delimiter;
    for (const auto &columnTitle : inColumnTitles) {
      oss << columnTitle;

      if (isComplex) {
        oss << "_re" << delimiter << columnTitle << "_im";
      }

      if (columnTitle != inColumnTitles.back()) {
        oss << delimiter;
      }
    }
    return appendToFile(oss.str(), inFilePath);
  };

  bool appendData(const GridGPU<T> *const inGridGPU,
                  const std::filesystem::path &inFilePath) const {
    const thrust::host_vector<T> gridData = *(inGridGPU->getVector());
    const T *data = thrust::raw_pointer_cast(gridData.data());

    const bool isComplex = inGridGPU->isComplex();
    const int numFields = inGridGPU->numFields();
    std::vector<const T *> fieldPtrs;
    fieldPtrs.reserve(numFields * (1 + int(isComplex)));
    for (int f = 0; f < numFields; ++f) {
      fieldPtrs.emplace_back(data +
                             inGridGPU->getFieldComponentOffset(f, Type::real));
      if (isComplex) {
        fieldPtrs.emplace_back(
            data + inGridGPU->getFieldComponentOffset(f, Type::imag));
      }
    }

    const int dimX = inGridGPU->dimX();
    const int dimY = inGridGPU->dimY();
    for (int j = 0; j < dimY; ++j) {
      for (int i = 0; i < dimX; ++i) {
        const int idx = j * dimX + i;
        std::ostringstream oss;
        oss << i << delimiter << j << delimiter;
        for (auto &&fieldPtr : fieldPtrs) {
          oss << fieldPtr[idx];
          if (fieldPtr != fieldPtrs.back()) {
            oss << delimiter;
          }
        }
        if (!appendToFile(oss.str(), inFilePath)) {
          return false;
        }
      }
    }
    return true;
  }

  bool appendToFile(const std::string &inLine,
                    std::filesystem::path inFilePath) const {
    std::ofstream outFile(inFilePath, std::ios_base::app);
    if (!outFile.good()) {
      return false;
    }
    outFile << inLine << std::endl;
    return true;
  }
};

//===----------------------------- Functions -----------------------------===//
/// \brief Get IIterationDataSaveStrategy<T>* for concrete implementation
///
/// Return a pointer to the singleton instace of the interface implementation.
/// The implementation should be stateless in order to be used safely this way.
///
/// \param format - save format enum (see Type::fileFormat).
///
/// \return - IIterationDataSaveStrategy<T>*
template <typename T>
std::unique_ptr<IIterationDataSaveStrategy<T>>
getIterationDataSaver(const Type::fileFormat format) {
  if (format == Type::csv) {
    return std::make_unique<CSVIterationDataSaveStrategy<T>>();
  }
  return nullptr;
}

/// \brief Get ITextGridDataSaveStrategy<T>* for concrete implementation
///
/// Return a pointer to the singleton instace of the interface implementation.
/// The implementation should be stateless in order to be used safely this way.
///
/// \param format - save format enum (see Type::fileFormat).
///
/// \return - ITextGridDataSaveStrategy<T>*
template <typename T>
std::unique_ptr<ITextGridDataSaveStrategy<T>>
getTextGridDataSaver(const Type::fileFormat format) {
  if (format == Type::csv) {
    return std::make_unique<CSVGridDataSaveStrategy<T>>();
  }
  return nullptr;
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
  std::cout << "key pressed: " << inKey << ", at (" << inX << ", " << inY
            << ")\n";
  if (inKey == 'q' || inKey == 'Q') {
    internal::guiUserQuit = true;
  } else if (inKey == 's' || inKey == 'S') {
    internal::guiUserSaveData = true;
  }
}

/// \brief Save simulation parameters to JSON file.
///
/// \param inJSON - JSON object with parameter values.
/// \param inFile - Full path and filename of file to save.
/// \param inVerbose - Whether or not to print when file is written.
///
/// \return - void
inline void saveParameterFileJSON(const Json::Value &inJSON,
                                  const std::filesystem::path &inFile,
                                  const bool inVerbose = false) {

  std::ofstream ofstreamJSON(inFile);
  const int indentation = 4;
  ofstreamJSON << std::setw(indentation) << inJSON << std::endl;

  if (inVerbose) {
    std::cout << "Wrote parameters to JSON file: " << inFile << std::endl;
  }
}

/// \brief Save config params, internal params, and results to JSON files.
///
/// This function saves three json files: one for config parameters, one for
/// internal parameters, and one for scalar results.
/// Motivation: It is useful to have the exact config parameters saved, so that
/// a run can be resumed/continued/post-processed. These config parameters are
/// saved to a JSON file. Furthermore, there are some internal parameters that
/// are not possible to infer without running the program (the exact number of
/// lattice points inside the grain), these are very useful to have access to
/// when post-processing (e.g. for doing area normalization). Finally, there are
/// some scalar results (e.g. area-averaged quantities) that are more
/// appropriate and convenient to have in json rather than csv.
///
/// \param inContext - Context object for simulation.
/// \param inJSON - JSON object with parameter values.
/// \param inDataPath - Directory where to put JSON files.
/// \param inConfigFileName - Filename for config parameters.
/// \param inInternalParametersFileName - Filename for internal parameters.
/// \param inResultsFileName - Filename for scalar results.
///
/// \return - void
template <typename T>
void saveParametersWrapper(
    const Context<T> *const inContext, const Json::Value &inJSON,
    const std::filesystem::path &inDataPath, const bool inVerbose = true,
    const std::string &inConfigFileName = "config",
    const std::string &inInternalParametersFileName = "internal_parameters",
    const std::string &inResultsFileName = "results") {
  //***************************************************
  // CONFIG PARAMETERS
  //***************************************************
  // Construct full filepath to JSON-file.
  const std::filesystem::path fullPathConfigJSON =
      inDataPath / (inConfigFileName + ".json");
  const std::filesystem::path fullPathInternalParamsJSON =
      inDataPath / (inInternalParametersFileName + ".json");
  const std::filesystem::path fullPathResultsJSON =
      inDataPath / (inResultsFileName + ".json");

  // Save config parameters to JSON file.
  saveParameterFileJSON(inJSON, fullPathConfigJSON, inVerbose);

  // Number of lattice sites in grain.
  const int numGrainLatticeSites =
      inContext->getGeometry()->getNumGrainLatticeSites();

  // Various results.
  Json::Value resultsJSON;
  resultsJSON["free_energy_per_unit_area"] =
      inContext->getComputeFreeEnergy()->getFreeEnergyPerUnitArea();
  resultsJSON["order_parameter_average_magnitude"] =
      inContext->getComputeOrderParameter()->getResult()->sumMagnitude() /
      static_cast<T>(numGrainLatticeSites);
  resultsJSON["current_density_average_magnitude"] =
      inContext->getComputeCurrent()->getResult()->sumMagnitude() /
      static_cast<T>(numGrainLatticeSites);

  // Save various results to JSON file.
  saveParameterFileJSON(resultsJSON, fullPathResultsJSON, inVerbose);

  // Internal parameters useful for post-processing.
  Json::Value internalParamJSON;
  internalParamJSON["area_compute"] = inContext->getGeometry()->area();
  internalParamJSON["internal_lattice_points"] = numGrainLatticeSites;
  internalParamJSON["gauge"] =
      inContext->getRiccatiSolver()->getVectorPotentialType();
  internalParamJSON["external_flux_density"] =
      inContext->getParameters()->getMagneticFieldStrength();
  internalParamJSON["grain_margin"] = constants::GRAIN_SIZE_MARGIN;
  const int numLatticeSitesPerSide =
      inContext->getParameters()->getGridResolutionBase();
  internalParamJSON["num_lattice_points"] =
      numLatticeSitesPerSide * numLatticeSitesPerSide;

  // Save internal parameters to JSON file.
  saveParameterFileJSON(internalParamJSON, fullPathInternalParamsJSON,
                        inVerbose);
}

/// \brief Save free energy to text file.
///
/// \param inContext - Context object for simulation.
/// \param inDataPath - Directory where to put free-energy file.
/// \param inVerbose - Print success to terminal.
/// \param inIterationDataSaveFormat - Enum specifying which format to save the
/// per-iteration data in (see Type::fileFormat).
/// \param inDataFileName - Name of free-energy file to write.
///
/// \return - void
template <typename T>
void saveFreeEnergy(
    const Context<T> *const inContext, const std::filesystem::path &inDataPath,
    const bool inVerbose = true,
    const Type::fileFormat inIterationDataSaveFormat = Type::csv,
    const std::string &inDataFileName = "free_energy") {
  const std::filesystem::path freeEnergyFile =
      inDataPath /
      (inDataFileName + internal::getFileExtension(inIterationDataSaveFormat));

  std::unique_ptr<IIterationDataSaveStrategy<T>> iterationDataSaver =
      getIterationDataSaver<T>(inIterationDataSaveFormat);

  if (iterationDataSaver &&
      iterationDataSaver->save(inContext, freeEnergyFile)) {
    if (inVerbose) {
      std::cout << "Wrote free energy to file: " << freeEnergyFile << std::endl;
    }
  } else {
    std::cout << "ERROR! Failed to save " << freeEnergyFile << "file."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

/// \brief Save induced magnetic flux density to file
///
/// \param inContext - Context object
/// \param inDataPath - Path where to save save grid data file.
/// \param inVerbose - Print success to terminal.
/// \param inGridDataSaveFormat - Enum specifying which format to save the
/// GridData in (see Type::fileFormat).
/// \param inDataFileName - File name of induced flux density file.
// Todo(patric): Create utils_io.h and move there.
template <typename T>
void saveInducedFluxDensity(
    const Context<T> *const inContext, const std::filesystem::path &inDataPath,
    const bool inVerbose = true,
    const Type::fileFormat inGridDataSaveFormat = Type::binary,
    const std::string &inDataFileName = "magnetic_flux_density") {

  // Fetch the vector potential.
  const GridGPU<T> *const inducedVectorPotential =
      inContext->getInducedVectorPotential();

  // Fetch parameters from context.
  Parameters<T> *parameters = inContext->getParameters();

  // TODO(niclas): Convert to (reduced) coherence length? I.e. divide by 2pi.
  const T latticeSpacing = parameters->getGridElementSize();

  // The magnetic flux-density is simply the curl of the vector potential.
  const GridGPU<T> magneticFluxDensity =
      inducedVectorPotential->curl(latticeSpacing);

  // Try to save induced flux density to file. Exit program if fail.
  const std::filesystem::path gridDataFile =
      inDataPath /
      (inDataFileName + internal::getFileExtension(inGridDataSaveFormat));

  bool success = false;
  if (inGridDataSaveFormat == Type::binary) {
    success = magneticFluxDensity.write(gridDataFile, parameters) == 0;
  } else {
    std::unique_ptr<ITextGridDataSaveStrategy<T>> textDataSaver =
        getTextGridDataSaver<T>(inGridDataSaveFormat);

    success = bool(textDataSaver) &&
              textDataSaver->save(&magneticFluxDensity,
                                  {"induced_flux_density"}, gridDataFile);
  }
  if (!success) {
    std::cout << "ERROR! Could not save induced flux density to file: "
              << gridDataFile << std::endl;
    std::exit(EXIT_FAILURE);
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
/// \param inGridDataSaveFormat - Enum specifying which format to save the
/// GridData in (see Type::fileFormat).
/// \param inOrderParameterFileName - File name of order-parameter file.
/// \param inCurrentDensityFileName - File name of current-density file.
///
/// \return - void
template <typename T>
void saveGridData(
    const Context<T> *const inContext, const std::filesystem::path &inDataPath,
    const bool inVerbose = true,
    const Type::fileFormat inGridDataSaveFormat = Type::binary,
    const std::string &inOrderParameterFileName = "order_parameter",
    const std::string &inCurrentDensityFileName = "current_density") {
  // Fetch run parameters from context.
  Parameters<T> *parameters = inContext->getParameters();

  // Fetch order parameter.
  GridGPU<T> *const orderParameterSave =
      inContext->getComputeOrderParameter()->getResult();

  // Set order parameter to zero outside simulation domain (non-zero values are
  // used for interpolation).
  inContext->getGeometry()->zeroValuesOutsideDomainRef(orderParameterSave);

  // Construct full file name for order-parameter file.
  const std::string extension =
      internal::getFileExtension(inGridDataSaveFormat);
  const std::filesystem::path orderParameterFile =
      inDataPath / (inOrderParameterFileName + extension);

  // Try to save order parameter to file. Exit program if fail.
  bool success = false;
  if (inGridDataSaveFormat == Type::binary) {
    success = orderParameterSave->write(orderParameterFile, parameters) == 0;
  } else {
    std::unique_ptr<ITextGridDataSaveStrategy<T>> textDataSaver =
        getTextGridDataSaver<T>(inGridDataSaveFormat);

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
  GridGPU<T> *const currentSave = inContext->getComputeCurrent()->getResult();

  // Set current density to zero outside simulation domain (non-zero values
  // are used for interpolation).
  inContext->getGeometry()->zeroValuesOutsideDomainRef(currentSave);

  // Construct full file name for current density.
  const std::filesystem::path currentDensityFile =
      inDataPath / (inCurrentDensityFileName + extension);

  // Try to save order parameter to file. Exit program if fail.
  success = false;
  if (inGridDataSaveFormat == Type::binary) {
    success = currentSave->write(currentDensityFile, parameters) == 0;
  } else {
    std::unique_ptr<ITextGridDataSaveStrategy<T>> textDataSaver =
        getTextGridDataSaver<T>(inGridDataSaveFormat);

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
}

/// \brief Wrapper to save data to file.
///
/// Wrapper that calls functions to save metadata parameters, free energy, and
/// observables to file.
///
/// \param inContext - Context object for simulation.
/// \param inJSON - JSON object with parameter values.
/// \param inGridDataSaveFormat - Enum specifying which format to save the
/// GridData in (see Type::fileFormat).
/// \param inIterationDataSaveFormat - Enum specifying which format to save the
/// per-iteration data in (see Type::fileFormat).
///
/// \return - void
template <typename T>
void saveDataWrapper(
    const Context<T> *const inContext, const Json::Value &inJSON,
    const bool inVerbose,
    const Type::fileFormat inGridDataSaveFormat = Type::binary,
    const Type::fileFormat inIterationDataSaveFormat = Type::csv) {
  // Construct full directory path.
  const std::filesystem::path dataBasePath(inJSON["data_path"].asString());
  const std::filesystem::path dataPath =
      dataBasePath / ("T" + std::to_string(inJSON["temperature"].asDouble()));

  // Create directories.
  std::filesystem::create_directories(dataPath);

  // Save parameters to JSON file.
  saveParametersWrapper(inContext, inJSON, dataPath, inVerbose);

  // Append free energy and iteration status to file.
  saveFreeEnergy(inContext, dataPath, inVerbose, inIterationDataSaveFormat);

  // Save order parameter and current to file.
  saveGridData(inContext, dataPath, inVerbose, inGridDataSaveFormat);

  // Save induced magnetic field to file.
  saveInducedFluxDensity(inContext, dataPath, inVerbose, inGridDataSaveFormat);
}

/// \brief Print iteration status.
///
/// Output status to terminal in each iteration.
///
/// \param inContext - Context object to delete.
///
/// \return - void
template <typename T> void printIterationStatus(Context<T> *inContext) {
  const int numIterations = inContext->getNumIterations();
  const T residual = inContext->getResidual();
  const T freeEnergy =
      inContext->getComputeFreeEnergy()->getFreeEnergyPerUnitArea();
  const T computeTime = inContext->getComputeTime();

  std::cout << "Iteration = " << numIterations << ", Residual = " << residual
            << ", Free Energy = " << freeEnergy
            << ", Iteration time = " << computeTime << std::endl;
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
/// \param inGridDataSaveFormat - Enum specifying which format to save the
/// GridData in (see Type::fileFormat).
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
                   const Json::Value &inJSON = {},
                   const Type::fileFormat inGridDataSaveFormat = Type::binary) {
  Status status = Status::kContinue;
  bool quit = false;

  while (!quit) {
    if (status != Status::kQuit) {
      // Run one iteration.
      status = runIteration(inOutContext, inPrintStatus, inNumIterationsMin,
                            inNumIterationsMax, inSaveData, inSaveDataInterval);

      if (status == Status::kSave) {
        if (inJSON.empty()) {
          std::cout
              << "ERROR! Attempting to save data, but JSON parameter "
                 "object was empty! (call to runSimulation() in run_utils.h)"
              << std::endl;
        } else {
          saveDataWrapper(inOutContext, inJSON, inPrintStatus,
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
void cleanup(Context<T> *inContext, Visualization<T> *inGUI = NULL) {
  std::cout << "Cleaning up..." << std::endl;
  delete inGUI;
  delete inContext;
  std::cout << "Done!" << std::endl;
}
} // namespace run_utils
} // namespace scg

#endif // RUN_UTILS_H_
