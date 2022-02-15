//===----- run_simulation.h - Useful functions for run file examples --===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains useful functions used in example run files.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_RUN_SIMULATION_H_
#define CONGA_RUN_SIMULATION_H_

#include "Context.h"
#include "Parameters.h"
#include "grid.h"
#include "io.h"
#include "run_iteration.h"

#ifdef CONGA_VISUALIZE
#include "visualization/Visualizer.h"
#endif // CONGA_VISUALIZE

#include <json/json.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <vector>

namespace conga {
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
/// \param inContext Context object for simulation.
/// \param inDataPath Directory where to put JSON files.
/// \param inVerbose Print to terminal.
///
/// \return Void
template <typename T>
void saveParametersWrapper(const Context<T> *const inContext,
                           const std::filesystem::path &inDataPath,
                           const bool inVerbose) {
  //***************************************************
  // CONFIG PARAMETERS
  //***************************************************
  // Number of lattice sites in grain.
  const int numGrainLatticeSites =
      inContext->getGeometry()->getNumGrainLatticeSites();

  // Side length of lattice site.
  const T latticeElementSize = inContext->getParameters()->getGridElementSize();

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

  // Magnetic moment per unit area.
  resultsJSON["magnetic_moment_per_unit_area"] =
      inContext->getComputeCurrent()->getResult()->moment(latticeElementSize) /
      inContext->getGeometry()->area();

  // Get the vector potential. It is used both for the average vector potential,
  // and average flux density.
  GridGPU<T> vectorPotential = *(inContext->getInducedVectorPotential());

  // Average flux density.
  // TODO(niclas): Should this be computed outside of the grain?
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
  const std::filesystem::path resultsPath = inDataPath / "results.json";
  io::write(resultsPath, resultsJSON);
  conga::io::printWrite(inVerbose, true, "scalar results", resultsPath);

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
  const std::filesystem::path internalParametersPath =
      inDataPath / "internal_parameters.json";
  io::write(internalParametersPath, internalParamJSON);
  conga::io::printWrite(inVerbose, true, "internal parameters",
                        internalParametersPath);
}

/// \brief Save convergence status (residuals vs iteration) to file.
///
/// \param inContext Context object for simulation.
/// \param inDataPath Directory where to put file.
/// \param inVerbose Print to terminal.
///
/// \return Void
template <typename T>
void saveConvergenceStatus(const Context<T> *const inContext,
                           const std::filesystem::path &inDataPath,
                           const bool inVerbose) {
  const std::filesystem::path convergenceFile = inDataPath / "convergence.csv";

  io::CSVIterationWriter<T> iterationDataSaver;

  const bool success = iterationDataSaver.write(inContext, convergenceFile);
  if (!success) {
    std::cout << "-- ERROR! Failed to write: " << convergenceFile << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

/// \brief Save the order parameter, the current density, the induced vector
/// potential, and the induced magnetic-flux density to file.
///
/// \param inContext Context object for simulation.
/// \param inDataPath Path of data directory.
/// \param inDataFormat The format to use when writing data.
/// \param inVerbose Print to terminal.
///
/// \return Void.
template <typename T>
void saveGridData(const Context<T> *const inContext,
                  const std::filesystem::path &inDataPath,
                  const io::DataFormat inDataFormat, const bool inVerbose) {
  // All objects to save.
  const GridGPU<char> &domain = *(inContext->getGeometry()->getDomainGrid());
  const OrderParameter<T> &orderParameter = *(inContext->getOrderParameter());
  const GridGPU<T> &currentDensity =
      *(inContext->getComputeCurrent()->getResult());

  // Scale vector potential to our external units.
  const T penetrationDepth = inContext->getParameters()->getPenetrationDepth();
  const T vectorPotentialScale =
      penetrationDepth > static_cast<T>(0)
          ? static_cast<T>(1) / (penetrationDepth * penetrationDepth)
          : static_cast<T>(0);
  const GridGPU<T> vectorPotential =
      *(inContext->getInducedVectorPotential()) * vectorPotentialScale;

  // Compute flux density.
  const T latticeSpacing = inContext->getParameters()->getGridElementSize();
  const GridGPU<T> fluxDensity = vectorPotential.curl(latticeSpacing);

  switch (inDataFormat) {
#ifdef CONGA_USE_HDF5
  case io::DataFormat::H5: {
    const std::filesystem::path fileName =
        inDataPath / std::string("simulation.h5");
    auto file = io::fileOverWrite(fileName);

    // Write.
    io::write(domain, "domain", file);
    io::write(orderParameter, "order_parameter", file);
    io::write(currentDensity, "current_density", file);
    io::write(vectorPotential, "vector_potential", file);
    io::write(fluxDensity, "magnetic_flux_density", file);

    // Can it never fail?
    const bool success = true;
    io::printWrite(inVerbose, success, "all simulation data", fileName);
    break;
  }
#endif // CONGA_USE_HDF5
  case io::DataFormat::CSV: {
    // CSV is the default.
  }
  default: {
    // Write domain.
    {
      const std::filesystem::path fileName = inDataPath / "domain.csv";
      const bool success = io::write(domain, {"domain"}, fileName);
      io::printWrite(inVerbose, success, "domain", fileName);
    }

    // Write order parameter.
    {
      // This is an ugly hack due to the CSVWriter not accepting an order
      // parameter as input.
      const GridGPU<T> &orderParameterGrid =
          *(inContext->getComputeOrderParameter()->getResult());
      const std::filesystem::path fileName = inDataPath / "order_parameter.csv";
      const bool success =
          io::write(orderParameterGrid, orderParameter.getTypes(), fileName);
      io::printWrite(inVerbose, success, "order parameter", fileName);
    }

    // Write curent density.
    {
      const std::filesystem::path fileName = inDataPath / "current_density.csv";
      const bool success = io::write(currentDensity, {"j_x", "j_y"}, fileName);
      io::printWrite(inVerbose, success, "current density", fileName);
    }

    // Write vector potential.
    {
      const std::filesystem::path fileName =
          inDataPath / "vector_potential.csv";
      const bool success = io::write(vectorPotential, {"a_x", "a_y"}, fileName);
      io::printWrite(inVerbose, success, "vector potential", fileName);
    }

    // Write flux density.
    {
      const std::filesystem::path fileName =
          inDataPath / "magnetic_flux_density.csv";
      const bool success = io::write(fluxDensity, {"b_z"}, fileName);
      io::printWrite(inVerbose, success, "magnetic flux density", fileName);
    }
    break;
  }
  }
}

/// \brief Wrapper to save data to file.
///
/// Wrapper that calls functions to save metadata parameters, free energy, and
/// observables to file.
///
/// \param inContext Context object for simulation.
/// \param inDataPath The path to data directory.
/// \param inDataFormat The format to use when writing data.
/// \param inVerbose Print to terminal.
///
/// \return - void
template <typename T>
void saveDataWrapper(const Context<T> *const inContext,
                     const std::filesystem::path inDataPath,
                     const io::DataFormat inDataFormat, const bool inVerbose) {

  // Save parameters to JSON file.
  saveParametersWrapper(inContext, inDataPath, inVerbose);

  // Save order parameter, current density, induced vector-potential, and
  // induced magnetic flux-density, to file.
  saveGridData(inContext, inDataPath, inDataFormat, inVerbose);
}

/// \brief Simulation wrapper
///
/// Wrapper for running the simulation until convergence.
///
/// \param inOutContext Context object for simulation.
/// \param inVerbose Verbosity.
/// \param inNumIterationsMin The minimum number of iterations.
/// \param inNumIterationsMax The maximum number of iterations.
/// \param inDataPath Path to data directory.
/// \param inSaveDataInterval How frequently to save data.
///
/// \return Void.
template <typename T>
void runSimulation(Context<T> *inOutContext, const bool inVerbose = true,
                   const int inNumIterationsMin = 0,
                   const int inNumIterationsMax = 10000,
                   const std::filesystem::path &inDataPath = {},
                   const int inSaveDataInterval = 0,
                   const io::DataFormat inDataFormat = io::DataFormat::H5) {
  Status status = Status::kContinue;
  while (status != Status::kQuit) {
    // Run one iteration.
    status = runIteration(inOutContext, inVerbose, inNumIterationsMin,
                          inNumIterationsMax, inSaveDataInterval);

    if (inSaveDataInterval > 0) {
      // Append convergence status to file.
      saveConvergenceStatus(inOutContext, inDataPath, inVerbose);

      // Save grid data and results.
      if (status == Status::kSave) {
        saveDataWrapper(inOutContext, inDataPath, inDataFormat, inVerbose);
      }
    }
  }
}

#ifdef CONGA_VISUALIZE
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
/// \param inSaveDataInterval How frequently to save data.
///
/// \return - void.
template <typename T>
void runSimulation(Context<T> *inOutContext, Visualizer<T> &inOutVisualizer,
                   const bool inVerbose = true,
                   const int inNumIterationsMin = 0,
                   const int inNumIterationsMax = 10000,
                   const std::filesystem::path &inDataPath = {},
                   const int inSaveDataInterval = 0,
                   const io::DataFormat inDataFormat = io::DataFormat::H5) {
  Status status = Status::kContinue;
  bool quit = false;

  while (!quit) {
    if (status != Status::kQuit) {
      // Run one iteration.
      status = runIteration(inOutContext, inVerbose, inNumIterationsMin,
                            inNumIterationsMax, inSaveDataInterval);

      if (inSaveDataInterval > 0) {
        // Append convergence status to file.
        saveConvergenceStatus(inOutContext, inDataPath, inVerbose);

        // Save grid data and results.
        if (status == Status::kSave) {
          saveDataWrapper(inOutContext, inDataPath, inDataFormat, inVerbose);
        }
      }
    } else {
      // When no computation is done it is unnecessary to update the window as
      // fast as possible.
      std::this_thread::sleep_for(std::chrono::milliseconds(42));
    }

    // Check if the window is closed.
    quit = inOutVisualizer.close();

    if (!quit) {
      int col = 0;
      const int numOrderParameterComponents =
          inOutContext->getOrderParameter()->getNumComponents();
      for (; col < numOrderParameterComponents; ++col) {
        // Get order parameter.
        const GridGPU<T> &orderParameter =
            *(inOutContext->getOrderParameter()->getComponentGrid(col));

        // Component type.
        const std::string type =
            inOutContext->getOrderParameter()->getComponent(col)->getType();

        // Draw.
        inOutVisualizer.draw(type, col, orderParameter, true);
      }

      // Draw the current density.
      const GridGPU<T> &current =
          *(inOutContext->getComputeCurrent()->getResult());
      inOutVisualizer.draw(std::string("J"), col, current, true);
      col++;

      const T penetrationDepth =
          inOutContext->getParameters()->getPenetrationDepth();
      if (penetrationDepth > static_cast<T>(0)) {
        // Draw the induced vector-potential.
        const GridGPU<T> &vectorPotential =
            *(inOutContext->getInducedVectorPotential());
        inOutVisualizer.draw(std::string("A"), col, vectorPotential, false);
        col++;

        // Draw the induced flux-density.
        const T latticeElementSize =
            inOutContext->getParameters()->getGridElementSize();
        const GridGPU<T> fluxDensity = vectorPotential.curl(latticeElementSize);
        inOutVisualizer.draw(std::string("B_z"), col, fluxDensity, false);
      }

      const bool running = status != Status::kQuit;
      inOutVisualizer.setInfo(running);
      inOutVisualizer.swapBuffers();
    }
  }
}
#endif // CONGA_VISUALIZE
} // namespace conga

#endif // CONGA_RUN_SIMULATION_H_
