//===------ mu_sweeps_utils_utils.h - Collection of useful functions
//-------===//
//
// Part of the SuprConGPU Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains utilities that are common for the research project "Mu
/// sweeps".
///
//===----------------------------------------------------------------------===//
#ifndef MU_SWEEPS_UTILS_H_
#define MU_SWEEPS_UTILS_H_

#include "ComputeVectorPotential.h" // Compute the vector potential induced by currents
#include "Context.h" // The context that connects all classes (helpful since enacpsulation is used)
#include "Grid.h"     // Grid and GridData
#include "GridData.h" // Grid and GridData
#include "Parameters.h" // Parameters specifying which the superconducting system to simulate
#include "Type.h" // Custom types (real, complex, ...)
#include "Visualization.h"
#include "defines.h"

#include <cstdlib>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <sstream>

namespace mu_sweeps_utils {
/// \brief Save induced magnetic flux density to file
///
/// \param inContext - Context object
/// \param inDataPath - Path where to save save grid data file.
// Todo(patric): Create utils_io.h and move there.
template <typename T>
void saveInducedFluxDensity(
    const scg::Context<T> *const inContext, const char *const inDataPath,
    const char *const inDataFileName = (char *)"magnetic_flux_density") {

  // Construct full file name
  char gridDataFile[512];
  sprintf(gridDataFile, "%s/%s.grd", inDataPath, inDataFileName);

  // Fetch parameters
  scg::Parameters<T> *parameters = inContext->getParameters();

  // Fetch current and set to zero outside grain
  typename scg::Grid<T>::GPU *current =
      inContext->getComputeCurrent()->getResult();
  inContext->getGeometry()->zeroValuesOutsideDomain(current);
  const int numLatticePoints = current->dimX();

  // Domain structure for vector potential calculation
  scg::Grid<char>::GPU *domain =
      inContext->getGeometry()->createDomain(numLatticePoints);

  // Create GPU storage for vector potential
  typename scg::Grid<T>::GPU *vectorPotential = new typename scg::Grid<T>::GPU(
      numLatticePoints, numLatticePoints, 2, scg::Type::real);

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

  typename scg::Grid<T>::GPU magneticFluxDensity =
      vectorPotential->curl(latticeSpacing);

  delete vectorPotential;

  magneticFluxDensity.write(gridDataFile, parameters);
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
    const scg::Context<T> *const inContext, const char *const inDataPath,
    const T inGrainSideLength, const T inGrainCornerSize,
    const T inGrainDiagonal, const T inDWaveTc, const T inSWaveTc,
    const bool inTcSuccess,
    const char *const inDataFileName = (char *)"parameters") {
  // Fetch simulation parameters
  scg::Parameters<T> *parameters = inContext->getParameters();

  // Construct full file name for meta data file
  char metaDataFile[512];
  sprintf(metaDataFile, "%s/%s.dat", inDataPath, inDataFileName);

  // Add header
  std::stringstream ss_header;
  ss_header
      << "# sideLength cornerSize pairBreakingEdge temperature pairingSymmetry "
         "areaCompute internalLatticePoints latticeRotation gauge "
         "externalFluxQuanta externalFluxDensity numLatticePoints numAngles "
         "numOzakiPoles energyCutoff grainMargin dWaveTc sWaveTc TcSuccess";
  appendLineToTextFile(metaDataFile, ss_header, std::ios_base::trunc);

  // Append line with energy and other information
  std::stringstream ss;
  ss << inGrainSideLength << " ";
  ss << inGrainCornerSize << " ";
  ss << inGrainDiagonal << " ";
  ss << parameters->getTemperature() << " ";
  ss << inContext->getOrderParameter()->getTypes() << " ";
  ss << inContext->getGeometry()->area() << " ";
  ss << inContext->getGeometry()->getNumGrainLatticeSites() << " ";
  ss << "0 ";
  ss << inContext->getRiccatiSolver()->getVectorPotentialType() << " ";
  ss << parameters->getMagneticFieldStrength() *
            inContext->getGeometry()->area()
     << " ";
  ss << parameters->getMagneticFieldStrength() << " ";
  ss << parameters->getGridResolutionBase() << " ";
  ss << parameters->getAngularResolution() << " ";
  ss << parameters->getNumMatsubaraEnergies() << " ";
  ss << parameters->getCutOff() << " ";
  ss << scg::constants::GRAIN_SIZE_MARGIN << " ";
  ss << inDWaveTc << " ";
  ss << inSWaveTc << " ";
  ss << inTcSuccess;
  appendLineToTextFile(metaDataFile, ss);
}

// Save free energy to text file
template <typename T>
void saveFreeEnergy(const scg::Context<T> *const inContext,
                    const char *const inDataPath, const T inGrainSideLength,
                    const T inGrainCornerSize, const T inGrainDiagonal,
                    const char *const inDataFileName = (char *)"free_energy",
                    const int dataFloatDecimals = 10) {
  // Fetch simulation parameters
  scg::Parameters<T> *parameters = inContext->getParameters();

  // Construct full file name for free energy
  char freeEnergyFile[512];
  sprintf(freeEnergyFile, "%s/%s.dat", inDataPath, inDataFileName);

  // Add header if file does not exist.
  std::ifstream freeEnergyFileStream(freeEnergyFile);
  if (!freeEnergyFileStream.good()) {
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
  ss << inContext->getComputeFreeEnergy()->getFreeEnergyPerUnitArea() << " ";
  ss.precision(defaultPrecision);
  ss << parameters->getMagneticFieldStrength() << " ";
  ss << inGrainSideLength << " ";
  ss << inGrainCornerSize << " ";
  ss << inGrainDiagonal << " ";
  ss << inContext->getGeometry()->area() << " ";
  ss << inContext->getGeometry()->getNumGrainLatticeSites();
  appendLineToTextFile(freeEnergyFile, ss);
}

// Save data to file
template <typename T>
void saveGridData(
    const scg::Context<T> *const inContext, const char *const inDataPath,
    const char *const inOrderParameterFileName = (char *)"order_parameter",
    const char *const inCurrentDensityFileName = (char *)"current_density") {
  // Fetch run parameters from context
  scg::Parameters<T> *parameters = inContext->getParameters();

  // Save Order Parameter
  typename scg::Grid<T>::GPU *const orderParameterSave =
      inContext->getComputeOrderParameter()->getResult();
  inContext->getGeometry()->zeroValuesOutsideDomain(orderParameterSave);

  // Construct full file name for order parameter
  char orderParameterFile[512];
  sprintf(orderParameterFile, "%s/%s.grd", inDataPath,
          inOrderParameterFileName);

  if (orderParameterSave->write(orderParameterFile, parameters)) {
    std::cerr << "ERROR! Could not save Order Parameter to file.\n";
    std::exit(EXIT_FAILURE);
  }

  // Save Current
  typename scg::Grid<T>::GPU *const currentSave =
      inContext->getComputeCurrent()->getResult();
  inContext->getGeometry()->zeroValuesOutsideDomain(currentSave);

  // Construct full file name for current density
  char currentDensityFile[512];
  sprintf(currentDensityFile, "%s/%s.grd", inDataPath,
          inCurrentDensityFileName);

  if (currentSave->write(currentDensityFile, parameters)) {
    std::cerr << "ERROR! Could not save Current to file.\n";
    std::exit(EXIT_FAILURE);
  }
}

} // namespace mu_sweeps_utils
#endif // MU_SWEEPS_UTILS_H_
