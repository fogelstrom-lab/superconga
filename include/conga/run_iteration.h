//===----- run_iteration.h - Helper functions for a single iteration. --===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Helper functions for a single iteration.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_RUN_ITERATION_H_
#define CONGA_RUN_ITERATION_H_

#include "Context.h"
#include "compute/ComputeFreeEnergy.h"

#include <iostream>

namespace conga {
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
  const T speed = static_cast<T>(numIterations) / computeTime;

  // Convert to hours or minutes if more appropriate.
  const T secondsPerHour = static_cast<T>(3600);
  const T secondsPerMinute = static_cast<T>(60);
  const int hours = static_cast<int>(computeTime / static_cast<T>(3600));
  const int minutes = static_cast<int>(computeTime / secondsPerMinute) -
                      hours * static_cast<T>(60);
  const int seconds =
      static_cast<int>(computeTime) - hours * 3600 - minutes * 60;

  // Get the free-energy computation object.
  const ComputeFreeEnergy<T> *const computeFreeEnergy =
      inContext->getComputeFreeEnergy();

  if (computeFreeEnergy) {
    // Get the free energy.
    const T freeEnergy = computeFreeEnergy->getFreeEnergyPerUnitArea();

    // Print values.
    std::printf(
        "Iteration: %d | Time: %d:%d:%d [h:m:s] | Speed: %0.2f [Hz] | Max "
        "Residual: %e | Free Energy: %e\n",
        numIterations, hours, minutes, seconds, speed, maxResidual, freeEnergy);
  } else {
    // Print values.
    std::printf(
        "Iteration: %d | Time: %d:%d:%d [h:m:s] | Speed: %0.2f [Hz] | Max "
        "Residual: %e\n",
        numIterations, hours, minutes, seconds, speed, maxResidual);
  }
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
/// \param inOutContext Context object for simulation.
/// \param inVerbose Print iteration status to terminal or not.
/// \param inNumIterationsMin The minimum number of iterations.
/// \param inNumIterationsMax The maximum number of iterations.
/// \param inSaveDataInterval How frequently to save data.
///
/// \return Simulation status.
template <typename T>
Status runIteration(Context<T> *inOutContext, const bool inVerbose = true,
                    const int inNumIterationsMin = 0,
                    const int inNumIterationsMax = 10000,
                    const int inSaveDataInterval = 0) {
  // Solve equations for all compute objects.
  inOutContext->compute();

  // Output iteration status to the terminal.
  if (inVerbose) {
    printIterationStatus(inOutContext);
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
  const bool quit = isConverged || patienceReached;
  const bool saveData = inSaveDataInterval > 0
                            ? (numIterations % inSaveDataInterval == 0)
                            : false;

  // Set status.
  Status status;
  if (quit) {
    status = Status::kQuit;
  } else if (saveData) {
    status = Status::kSave;
  } else {
    status = Status::kContinue;
  }

  return status;
}
} // namespace conga

#endif // CONGA_RUN_ITERATION_H_
