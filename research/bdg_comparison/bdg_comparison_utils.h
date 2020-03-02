//===------ bdg_comparison_utils.h - Collection of useful functions -------===//
//
// Part of the SuprConGPU Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains utilities that are common for the research project "BdG
/// Comparison".
///
//===----------------------------------------------------------------------===//
#ifndef BDG_COMPARISON_UTILS_H_
#define BDG_COMPARISON_UTILS_H_

#include "ComputeVectorPotential.h" // Compute the vector potential induced by currents
#include "Context.h" // The context that connects all classes (helpful since enacpsulation is used)
#include "Grid.h"     // Grid and GridData
#include "GridData.h" // Grid and GridData
#include "Parameters.h" // Parameters specifying which the superconducting system to simulate
#include "Type.h" // Custom types (real, complex, ...)
#include "Visualization.h"

#include <boost/filesystem.hpp>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <sstream>

namespace bdg_comp_utils {
/// \brief Save induced magnetic flux density to file
///
/// \param inContext - Context object
/// \param inDataPath - Path where to save save grid data file.
// Todo(patric): Create utils_io.h and move there.
template <typename T>
void saveInducedFluxDensity(
    const Context<T> *const inContext, const char *const inDataPath,
    const char *const inDataFileName = (char *)"magnetic_flux_density") {

  // Construct full file name
  char gridDataFile[512];
  sprintf(gridDataFile, "%s%s.grd", inDataPath, inDataFileName);

  // Fetch parameters
  Parameters<T> *parameters = inContext->getParameters();

  // Fetch current and set to zero outside grain
  typename Grid<T>::GPU *current = inContext->getComputeCurrent()->getResult();
  inContext->getGeometry()->zeroValuesOutsideDomain(current);
  const int numLatticePoints = current->dimX();

  // Domain structure for vector potential calculation
  Grid<char>::GPU *domain =
      inContext->getGeometry()->createDomain(numLatticePoints);

  // Create GPU storage for vector potential
  typename Grid<T>::GPU *vectorPotential = new
      typename Grid<T>::GPU(numLatticePoints, numLatticePoints, 2, Type::real);

  // Convert lattice spacing back to proper coherence lengths:
  // hbar*vF/(2*Pi*kB*Tc)
  const T latticeSpacing =
      parameters->getGridElementSize() * static_cast<T>(2.0 * M_PI);

  // Calculate coefficients for Poissons equation
  // TODO(patric): this should be done inside computeVectorPotential!
  const T penetrationDepth = parameters->getPenetrationDepth();
  const T minusInvKappaSquared =
      -static_cast<T>(1.0) / (penetrationDepth * penetrationDepth);

  computeVectorPotential(vectorPotential, current, domain, latticeSpacing,
                         minusInvKappaSquared);

  typename Grid<T>::GPU magneticFluxDensity =
      vectorPotential->curl(latticeSpacing);

  delete vectorPotential;

  magneticFluxDensity.write(gridDataFile, parameters);
}

// Use boost to create directory if not existing
// TODO(patric): Replace boost with standard functions. Create utils_io.h in lib
// and move there
void createProjectDirectory(const char *const inDirectory) {
  boost::filesystem::path rootPath(inDirectory);
  boost::filesystem::create_directories(rootPath);
}

// Output status to terminal
// TODO(patric): Create utils_io.h and move there. Set flags what to print.
template <typename T>
void printIterationStatus(const Context<T> *const inContext) {
  std::cout << "Iteration = " << inContext->getNumIterations() << ", ";
  std::cout << "Residual = " << inContext->getResidual(0) << ", ";
  std::cout << "Iteration time = " << inContext->getComputeTime() << std::endl;
}

// Plot data in OpenGL GUI
// TODO(patric): Merge into Visualization class, with flags for what to plot.
template <typename T>
void plotData(const Context<T> *const inContext,
              Visualization<T> *const inGUI) {
  // Extract computed data
  const typename Grid<T>::CPU orderParameter =
      *(inContext->getComputeOrderParameter()->getResult());
  // Extract computed current
  const typename Grid<T>::CPU current =
      *(inContext->getComputeCurrent()->getResult());

  // Clear the plotting region
  inGUI->clearImage();

  // Plot the magnitude of the order parameter
  inGUI->plotImage(orderParameter.abs(), 0, Type::real, 0, 0);
  // Plot the phase of the order parameter
  inGUI->plotImage(orderParameter.complexPhase(), 0, Type::real, 1, 0);
  // Plot the magnitude of the current density
  inGUI->plotImage(current.fieldMagnitude(), 0, Type::real, 2, 0);

  // Render the plots
  inGUI->draw();
}

// Append data to text file
// TODO(patric): Create utils_io.h and move there.
void appendLineToTextFile(
    const char *const inFile, const std::ostream &inOStream,
    const std::ios_base::openmode inFileMode = std::ios_base::app) {
  std::ofstream outfile;

  std::stringstream oss;
  oss << inOStream.rdbuf();
  outfile.open(inFile, inFileMode);
  outfile << oss.str() << std::endl;
  outfile.close();
}

// Save params to text file as METADATA
template <typename T>
void saveParamsMetaData(
    const Context<T> *const inContext, const char *const inDataPath,
    const T inGrainSideLength, const T inGrainCornerSize,
    const T inGrainDiagonal,
    const char *const inDataFileName = (char *)"parameters") {
  // Fetch simulation parameters
  Parameters<T> *parameters = inContext->getParameters();

  // Construct full file name for meta data file
  char metaDataFile[512];
  sprintf(metaDataFile, "%s%s.dat", inDataPath, inDataFileName);

  // Add header
  std::stringstream ss_header;
  ss_header
      << "# sideLength cornerSize pairBreakingEdge temperature pairingSymmetry "
         "areaCompute internalLatticePoints latticeRotation gauge "
         "externalFluxQuanta externalFluxDensity numLatticePoints numAngles "
         "numOzakiPoles energyCutoff";
  appendLineToTextFile(metaDataFile, ss_header, std::ios_base::trunc);

  // Append line with energy and other information
  std::stringstream ss;
  ss << inGrainSideLength << " ";
  ss << inGrainCornerSize << " ";
  ss << inGrainDiagonal << " ";
  ss << parameters->getTemperature() << " ";
  // TODO(patric): Fetch OP symmetry from a getter (i.e. in OrderParameter)
  ss << "d ";
  ss << inContext->getGeometry()->area() << " ";
  ss << inContext->getGeometry()->area(1) << " ";
  ss << "0 ";
  // TODO(patric): Fetch Gauge from getter (i.e. in RiccatiSolver)
  ss << "none ";
  ss << parameters->getMagneticFieldStrength() *
            inContext->getGeometry()->area();
  ss << parameters->getMagneticFieldStrength();
  ss << parameters->getGridResolutionBase() << " ";
  ss << parameters->getAngularResolution() << " ";
  ss << parameters->getNumMatsubaraEnergies() << " ";
  ss << parameters->getCutOff();
  appendLineToTextFile(metaDataFile, ss);
}

// Save free energy to text file
// TODO(patric): create methods in context for add() and get ComputeFreeEnergy
template <typename T>
void saveFreeEnergy(const Context<T> *const inContext,
                    const char *const inDataPath,
                    const ComputeFreeEnergy<T> *const inComputeFreeEnergy,
                    const T inGrainSideLength, const T inGrainCornerSize,
                    const T inGrainDiagonal,
                    const char *const inDataFileName = (char *)"free_energy",
                    const int dataFloatDecimals = 10) {
  // Fetch simulation parameters
  Parameters<T> *parameters = inContext->getParameters();

  // Construct full file name for free energy
  char freeEnergyFile[512];
  sprintf(freeEnergyFile, "%s%s.dat", inDataPath, inDataFileName);

  // Add header if file does not exist
  if (!boost::filesystem::exists(freeEnergyFile)) {
    std::stringstream ss_header;
    ss_header << "# iterationNumber residual freeEnergyPerUnitArea "
                 "externalMagneticFluxDensity grainSideLength cornerLength "
                 "pairBreakingLength areaInternalUnits internalLatticePoints";
    appendLineToTextFile(freeEnergyFile, ss_header);
  }

  // Append line with energy and other information
  std::stringstream ss;
  const std::streamsize defaultPrecision = ss.precision();

  ss << inContext->getNumIterations() << " ";
  ss << inContext->getResidual(0) << " ";
  ss.precision(dataFloatDecimals);
  ss << inComputeFreeEnergy->getFreeEnergyPerUnitArea() << " ";
  ss.precision(defaultPrecision);
  ss << parameters->getMagneticFieldStrength();
  ss << inGrainSideLength << " ";
  ss << inGrainCornerSize << " ";
  ss << inGrainDiagonal << " ";
  ss << inContext->getGeometry()->area() << " ";
  ss << inContext->getGeometry()->area(1);
  appendLineToTextFile(freeEnergyFile, ss);
}

// Save data to file
template <typename T>
void saveGridData(
    const Context<T> *const inContext, const char *const inDataPath,
    const char *const inOrderParameterFileName = (char *)"order_parameter",
    const char *const inCurrentDensityFileName = (char *)"current_density") {
  // Fetch run parameters from context
  Parameters<T> *parameters = inContext->getParameters();

  // Save Order Parameter
  typename Grid<T>::GPU *const orderParameterSave =
      inContext->getComputeOrderParameter()->getResult();
  inContext->getGeometry()->zeroValuesOutsideDomain(orderParameterSave);

  // Construct full file name for order parameter
  char orderParameterFile[512];
  sprintf(orderParameterFile, "%s%s.grd", inDataPath, inOrderParameterFileName);

  if (orderParameterSave->write(orderParameterFile, parameters)) {
    std::cout << "Could not save Order Parameter to file.\n";
    exit(0);
  }

  // Save Current
  typename Grid<T>::GPU *const currentSave =
      inContext->getComputeCurrent()->getResult();
  inContext->getGeometry()->zeroValuesOutsideDomain(currentSave);

  // Construct full file name for current density
  char currentDensityFile[512];
  sprintf(currentDensityFile, "%s%s.grd", inDataPath, inCurrentDensityFileName);

  if (currentSave->write(currentDensityFile, parameters)) {
    std::cout << "Could not save Current to file.\n";
    exit(0);
  }
}

// TODO(patric): replace with proper argument parser and put in utils_io.h
template <typename T>
void readCommandLineArguments(const int inNumArguments,
                              char *const *const inArguments,
                              int &outFermiSurfaceLabel,
                              T &outGrainDiagonalLength,
                              T &outGrainCornerLength) {
  // Check that there are arguments to parse (need at least a flag and a value)
  if (inNumArguments > 1) {
    // Loop over command line arguments
    for (int argumentIdx = 0; argumentIdx < inNumArguments; argumentIdx++) {
      // Check that there is one more argument following, since there needs to
      // be both a flag and a value
      if (argumentIdx + 1 != inNumArguments) {
        std::string thisArg = inArguments[argumentIdx];
        std::string nextArg = inArguments[argumentIdx + 1];

        if (thisArg == "--fs") {
          outFermiSurfaceLabel = std::stoi(nextArg);
        } else if (thisArg == "--R") {
          outGrainDiagonalLength = static_cast<T>(std::stod(nextArg));
        } else if (thisArg == "--r") {
          outGrainCornerLength = static_cast<T>(std::stod(nextArg));
        }
      }
    }
  }
}

} // namespace bdg_comp_utils
#endif // BDG_COMPARISON_UTILS_H_
