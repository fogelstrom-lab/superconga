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
#include "io/CSVIterationWriter.h"

#include <iostream>

namespace conga {
namespace iteration_internal {
/// \brief Print iteration status.
///
/// Output status to terminal in each iteration.
///
/// \param inContext Context object.
///
/// \return Void
template <typename T>
void inline printIterationStatus(const Context<T> &inContext) {
  // Get the number of iterations so far.
  const int numIterations = inContext.getNumIterations();

  // Get the max residual.
  const T maxResidual = inContext.getResidual();

  // Get the computation time for this iteration.
  const T computeTime = inContext.getComputeTime();
  const T speed = static_cast<T>(numIterations) / computeTime;

  // Convert to hours or minutes if more appropriate.
  const int secondsPerMinute = 60;
  const int minutesPerHour = 60;
  const int secondsPerHour = secondsPerMinute * minutesPerHour;
  const int hours = static_cast<int>(computeTime) / secondsPerHour;
  const int minutes =
      static_cast<int>(computeTime) / secondsPerMinute - hours * minutesPerHour;
  const int seconds = static_cast<int>(computeTime) - hours * secondsPerHour -
                      minutes * secondsPerMinute;

  // Get the free-energy computation object.
  const ComputeFreeEnergy<T> *const computeFreeEnergy =
      inContext.getComputeFreeEnergy();

  if (computeFreeEnergy) {
    // Get the free energy.
    const T freeEnergy = computeFreeEnergy->getFreeEnergyPerUnitArea();

    // Print values.
    std::printf(
        "Iteration: %d | Time: %d:%d:%d [h:m:s] | Avg Speed: %0.2f [Hz] | "
        "Residual: %e | Free Energy: %e\n",
        numIterations, hours, minutes, seconds, speed, maxResidual, freeEnergy);
  } else {
    // Print values.
    std::printf(
        "Iteration: %d | Time: %d:%d:%d [h:m:s] | Avg Speed: %0.2f [Hz] | "
        "Residual: %e\n",
        numIterations, hours, minutes, seconds, speed, maxResidual);
  }
}

/// \brief Wrapper for running "burn-in" or "compute" a number of iterations.
///
/// \param inIsCompute Whether to do compute or burn-in iterations.
/// \param inNumIterationsMin The minimum number of iterations.
/// \param inNumIterationsMax The maximum number of iterations.
/// \param inVerbose Whether to print to the terminal or not.
/// \param inIterationWriter Iteration writer for saving iteration data to file.
/// \param inOutContext The context object.
///
/// \return Success.
template <typename T>
bool inline runIterations(
    const bool inIsCompute, const int inNumIterationsMin,
    const int inNumIterationsMax, const bool inVerbose,
    const std::shared_ptr<io::CSVIterationWriter<T>> &inIterationWriter,
    Context<T> &inOutContext) {
  if ((inNumIterationsMin == 0) && (inNumIterationsMax == 0)) {
    // Nothing to do.
    return false;
  }

  bool isConverged = false;

  // Start counter at one because we count the number of done iterations from
  // one.
  for (int i = 1; i < std::numeric_limits<int>::max(); ++i) {
    if (inIsCompute) {
      // Solve equations for all compute objects.
      inOutContext.compute();

      // Check convergence. Partial convergence for e.g. the LDoS means that one
      // energy block out of many has converged, but for e.g. the order
      // parameter it means that everything is converged.
      isConverged = inOutContext.isPartiallyConverged();
    } else {
      // Update boundary coherence functions.
      inOutContext.burnIn();

      // Check convergence.
      const T convergenceCriterion =
          inOutContext.getParameters()->getConvergenceCriterion();
      const T residual = inOutContext.getRiccatiSolver()->getResidual();
      isConverged = residual < convergenceCriterion;
    }

    // Output iteration status to the terminal.
    if (inVerbose) {
      printIterationStatus(inOutContext);
    }

    // Save iteration data to file.
    if (inIterationWriter.get()) {
      inIterationWriter->write(inOutContext);
    }

    // Determine if to quit.
    const bool minReached = i >= inNumIterationsMin;
    const bool maxReached =
        (inNumIterationsMax < 0) ? false : (i >= inNumIterationsMax);
    const bool quit = (isConverged && minReached) || maxReached;

    if (quit) {
      break;
    }
  }

  return isConverged;
}
} // namespace iteration_internal

/// \brief Wrapper for running "burn-in" a number of iterations.
///
/// \param inNumIterationsMin The minimum number of iterations.
/// \param inNumIterationsMax The maximum number of iterations.
/// \param inVerbose Whether to print to the terminal or not.
/// \param inIterationWriter Iteration writer for saving iteration data to file.
/// \param inOutContext The context object.
///
/// \return Success.
template <typename T>
bool inline runBurnIn(
    const int inNumIterationsMin, const int inNumIterationsMax,
    const bool inVerbose,
    const std::shared_ptr<io::CSVIterationWriter<T>> &inIterationWriter,
    Context<T> &inOutContext) {
  if (inVerbose) {
    std::cout << "-- Starting burn-in (converging boundary)." << std::endl;
  }

  const bool isCompute = false;
  const bool isConverged = iteration_internal::runIterations(
      isCompute, inNumIterationsMin, inNumIterationsMax, inVerbose,
      inIterationWriter, inOutContext);

  if (inVerbose) {
    if (isConverged) {
      std::cout << "-- Stopping burn-in (converged)." << std::endl;
    } else {
      std::cout << "-- Stopping burn-in (unconverged)." << std::endl;
    }
  }

  return isConverged;
}

/// \brief Wrapper for running "burn-in" a number of iterations.
///
/// \param inNumIterationsMin The minimum number of iterations.
/// \param inNumIterationsMax The maximum number of iterations.
/// \param inVerbose Whether to print to the terminal or not.
/// \param inOutContext The context object.
///
/// \return Success.
template <typename T>
bool inline runBurnIn(const int inNumIterationsMin,
                      const int inNumIterationsMax, const bool inVerbose,
                      Context<T> &inOutContext) {
  // Nullptr.
  const std::shared_ptr<io::CSVIterationWriter<T>> iterationWriter;
  const bool isConverged = runBurnIn(inNumIterationsMin, inNumIterationsMax,
                                     inVerbose, iterationWriter, inOutContext);
  return isConverged;
}

/// \brief Wrapper for running "compute" a number of iterations.
///
/// \param inNumIterationsMin The minimum number of iterations.
/// \param inNumIterationsMax The maximum number of iterations.
/// \param inVerbose Whether to print to the terminal or not.
/// \param inIterationWriter Iteration writer for saving iteration data to file.
/// \param inOutContext The context object.
///
/// \return Success.
template <typename T>
bool inline runCompute(
    const int inNumIterationsMin, const int inNumIterationsMax,
    const bool inVerbose,
    const std::shared_ptr<io::CSVIterationWriter<T>> &inIterationWriter,
    Context<T> &inOutContext) {
  const bool isCompute = true;
  const bool isConverged = iteration_internal::runIterations(
      isCompute, inNumIterationsMin, inNumIterationsMax, inVerbose,
      inIterationWriter, inOutContext);
  return isConverged;
}

/// \brief Wrapper for running "compute" a number of iterations.
///
/// \param inNumIterationsMin The minimum number of iterations.
/// \param inNumIterationsMax The maximum number of iterations.
/// \param inVerbose Whether to print to the terminal or not.
/// \param inOutContext The context object.
///
/// \return Success.
template <typename T>
bool inline runCompute(const int inNumIterationsMin,
                       const int inNumIterationsMax, const bool inVerbose,
                       Context<T> &inOutContext) {
  // Nullptr.
  const std::shared_ptr<io::CSVIterationWriter<T>> iterationWriter;
  const bool isConverged = runCompute(inNumIterationsMin, inNumIterationsMax,
                                      inVerbose, iterationWriter, inOutContext);
  return isConverged;
}
} // namespace conga

#endif // CONGA_RUN_ITERATION_H_
