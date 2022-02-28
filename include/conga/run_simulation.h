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
#include "io/CSVIterationWriter.h"
#include "io/SimulationWriter.h"
#include "run_iteration.h"

#ifdef CONGA_VISUALIZE
#include "visualization/Visualizer.h"
#endif // CONGA_VISUALIZE

#include <chrono>
#include <filesystem>
#include <memory>
#include <thread>

namespace conga {
/// \brief Postprocess wrapper.
///
/// \param inNumBurnInIterations How many iterations to run just converging the
/// boundary.
/// \param inNumIterationsMin The minimum number of iterations.
/// \param inNumIterationsMax The maximum number of iterations.
/// \param inVerbose Verbosity.
/// \param inIterationWriter Iteration writer for saving iteration data to file.
/// \param inOutContext Context object for postprocessing.
///
/// \return Void.
template <typename T>
void inline runPostprocess(
    const int inNumBurnInIterations, const int inNumIterationsMin,
    const int inNumIterationsMax, const bool inVerbose,
    const std::shared_ptr<io::CSVIterationWriter<T>> &inIterationWriter,
    Context<T> &inOutContext) {
  bool isConverged = false;
  while (!isConverged) {
    if (inVerbose) {
      // Get info about the energy block.
      const ParameterIterator<int> *const energyIterator =
          inOutContext.getIntegrationIterator()->getEnergyIterator();
      const int currentEnergyIndex = energyIterator->getCurrentValue();
      const int numEnergiesCurrentBlock = energyIterator->getNumInterval();
      const int numEnergies = energyIterator->getRange();

      // Print info about the energy block.
      const T startFraction =
          static_cast<T>(currentEnergyIndex) / static_cast<T>(numEnergies);
      const T stopFraction =
          static_cast<T>(currentEnergyIndex + numEnergiesCurrentBlock) /
          static_cast<T>(numEnergies);
      const T toPercent = static_cast<T>(100);
      std::printf("-- Energy block %.2f%% --> %.2f%%.\n",
                  startFraction * toPercent, stopFraction * toPercent);
    }

    // Run burn-in iterations.
    // How the number of burn-in iterations is exposed it is both the mininum
    // and maximum number of iterations.
    const bool burnInConverged =
        runBurnIn(inNumBurnInIterations, inNumBurnInIterations, inVerbose,
                  inIterationWriter, inOutContext);

    // Run until partial convergence.
    const bool energyBlockConverged =
        runCompute(inNumIterationsMin, inNumIterationsMax, inVerbose,
                   inIterationWriter, inOutContext);

    // Check total convergence.
    isConverged = inOutContext.isConverged();

    // Print info.
    if (inVerbose) {
      if (energyBlockConverged) {
        std::cout << "-- Energy block converged." << std::endl;
      } else {
        std::cout << "-- Energy block did not converge." << std::endl;
      }
    }

    // Abort if the block did not converge.
    if (!energyBlockConverged) {
      break;
    }
  }

  // Print end status.
  if (inVerbose) {
    if (isConverged) {
      std::cout << "-- SUCCESS! Postprocess converged." << std::endl;
    } else {
      std::cout << "-- FAILURE! Postprocess did not converge." << std::endl;
    }
  }
}

/// \brief Simulation wrapper
///
/// \param inNumBurnInIterations How many iterations to run just converging the
/// boundary.
/// \param inNumIterationsMin The minimum number of iterations.
/// \param inNumIterationsMax The maximum number of iterations.
/// \param inVerbose Verbosity.
/// \param inIterationWriter Iteration writer for saving iteration data to file.
/// \param inDataPath Path to data directory.
/// \param inSaveFrequency At which frequency to save data.
/// \param inDataFormat Which data format to use.
/// \param inOutContext Context object for postprocessing.
///
/// \return Void.
template <typename T>
void inline runSimulation(
    const int inNumBurnInIterations, const int inNumIterationsMin,
    const int inNumIterationsMax, const bool inVerbose,
    const std::shared_ptr<io::CSVIterationWriter<T>> &inIterationWriter,
    const std::filesystem::path &inDataPath, const int inSaveFrequency,
    const io::DataFormat inDataFormat, Context<T> &inOutContext) {
  // Run burn-in iterations.
  // How the number of burn-in iterations is exposed it is both the mininum and
  // maximum number of iterations.
  const bool burnInConverged =
      runBurnIn(inNumBurnInIterations, inNumBurnInIterations, inVerbose,
                inIterationWriter, inOutContext);

  // Create simulation writer.
  io::SimulationWriter<T> simulationWriter(inDataPath, inDataFormat, inVerbose);

  bool isConverged = false;
  if (inSaveFrequency > 0) {
    bool quit = false;
    while (!quit) {
      // Run a single iteration.
      isConverged =
          runCompute(0, 1, inVerbose, inIterationWriter, inOutContext);

      // Check iteration number.
      const int numIterations = inOutContext.getNumIterations();

      // Determine if to quit.
      const bool minReached = numIterations >= inNumIterationsMin;
      const bool maxReached = (inNumIterationsMax < 0)
                                  ? false
                                  : (numIterations >= inNumIterationsMax);
      const bool quit = (isConverged && minReached) || maxReached;
      if (quit) {
        break;
      }

      // Do not save any data until the minimum number of iterations is reached.
      const bool write = numIterations % inSaveFrequency == 0;
      if (write) {
        simulationWriter.write(inOutContext);
      }
    }
  } else {
    // Run until convergence.
    isConverged = runCompute(inNumIterationsMin, inNumIterationsMax, inVerbose,
                             inIterationWriter, inOutContext);
  }

  // Print end status.
  if (inVerbose) {
    if (isConverged) {
      std::cout << "-- SUCCESS! Simulation converged." << std::endl;
    } else {
      std::cout << "-- FAILURE! Simulation did not converge." << std::endl;
    }
  }

  // Save when done.
  if (inSaveFrequency != 0) {
    simulationWriter.write(inOutContext);
  }
}

#ifdef CONGA_VISUALIZE
/// \brief Simulation wrapper with live visualization.
///
/// \param inNumBurnInIterations How many iterations to run just converging the
/// boundary.
/// \param inNumIterationsMin The minimum number of iterations.
/// \param inNumIterationsMax The maximum number of iterations.
/// \param inVerbose Verbosity.
/// \param inIterationWriter Iteration writer for saving iteration data to file.
/// \param inDataPath Path to data directory.
/// \param inSaveFrequency At which frequency to save data.
/// \param inDataFormat Which data format to use.
/// \param inOutVisualizer Visualizer object for real-time plotting.
/// \param inOutContext Context object.
///
/// \return Void.
template <typename T>
void inline runSimulation(
    const int inNumBurnInIterations, const int inNumIterationsMin,
    const int inNumIterationsMax, const bool inVerbose,
    const std::shared_ptr<io::CSVIterationWriter<T>> &inIterationWriter,
    const std::filesystem::path &inDataPath, const int inSaveFrequency,
    const io::DataFormat inDataFormat, Visualizer<T> &inOutVisualizer,
    Context<T> &inOutContext) {
  // Run burn-in iterations.
  // How the number of burn-in iterations is exposed it is both the mininum and
  // maximum number of iterations.
  const bool burnInConverged =
      runBurnIn(inNumBurnInIterations, inNumBurnInIterations, inVerbose,
                inIterationWriter, inOutContext);

  // Create simulation writer.
  io::SimulationWriter<T> simulationWriter(inDataPath, inDataFormat, inVerbose);

  // Show the window after burn-in.
  inOutVisualizer.show();

  bool isConverged = false;
  bool quit = false;
  bool windowClosed = false;
  while (!windowClosed) {
    // Check if the window is closed.
    windowClosed = inOutVisualizer.close();

    if (!windowClosed) {
      int col = 0;
      const int numOrderParameterComponents =
          inOutContext.getOrderParameter()->getNumComponents();
      for (; col < numOrderParameterComponents; ++col) {
        // Get order parameter.
        const GridGPU<T> &orderParameter =
            *(inOutContext.getOrderParameter()->getComponentGrid(col));

        // Component type.
        const std::string type =
            inOutContext.getOrderParameter()->getComponent(col)->getType();

        // Draw.
        inOutVisualizer.draw(type, col, orderParameter, true);
      }

      // Draw the current density.
      const GridGPU<T> &current =
          *(inOutContext.getComputeCurrent()->getResult());
      inOutVisualizer.draw(std::string("J"), col, current, true);
      col++;

      const T penetrationDepth =
          inOutContext.getParameters()->getPenetrationDepth();
      if (penetrationDepth > static_cast<T>(0)) {
        // Draw the induced vector-potential.
        const GridGPU<T> &vectorPotential =
            *(inOutContext.getInducedVectorPotential());
        inOutVisualizer.draw(std::string("A"), col, vectorPotential, false);
        col++;

        // Draw the induced flux-density.
        const T latticeElementSize =
            inOutContext.getParameters()->getGridElementSize();
        const GridGPU<T> fluxDensity = vectorPotential.curl(latticeElementSize);
        inOutVisualizer.draw(std::string("B_z"), col, fluxDensity, false);
      }

      const bool running = !isConverged;
      inOutVisualizer.setInfo(running);
      inOutVisualizer.swapBuffers();
    }

    if (quit) {
      // When no computation is done it is unnecessary to update the window as
      // fast as possible.
      std::this_thread::sleep_for(std::chrono::milliseconds(42));
    } else {
      // Run a single iteration.
      isConverged =
          runCompute(0, 1, inVerbose, inIterationWriter, inOutContext);

      // Check iteration number.
      const int numIterations = inOutContext.getNumIterations();

      // Save data..
      const bool write =
          inSaveFrequency > 0 ? numIterations % inSaveFrequency == 0 : false;
      if (write) {
        simulationWriter.write(inOutContext);
      }

      // Determine if to quit.
      const bool minReached = numIterations >= inNumIterationsMin;
      const bool maxReached = (inNumIterationsMax < 0)
                                  ? false
                                  : (numIterations >= inNumIterationsMax);
      quit = (isConverged && minReached) || maxReached;

      // Print end status.
      if (quit && inVerbose) {
        if (isConverged) {
          std::cout << "-- SUCCESS! Simulation converged." << std::endl;
        } else {
          std::cout << "-- FAILURE! Simulation did not converge." << std::endl;
        }
      }
    }
  }

  // Save when done.
  if (inSaveFrequency != 0) {
    simulationWriter.write(inOutContext);
  }
}
#endif // CONGA_VISUALIZE
} // namespace conga

#endif // CONGA_RUN_SIMULATION_H_
