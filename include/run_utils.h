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
#include "Visualization.h"

#ifdef __linux__
#include <GL/freeglut.h>
#else
#include <GLUT/glut.h>
#endif
#include <cmath>

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
  // Number of plots in visualization = numPlotsY * numPlotsX.
  const int numPlotsX = (inPlotCurrentDensity ? 3 : 1);
  const int numPlotsY = static_cast<int>(inPlotOrderParameter) +
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

  // Plot the order parameter.
  if (inPlotOrderParameter) {
    // Plot the magnitude of the order parameter.
    inGUI->plotImage(inContext->getComputeOrderParameter()->getResult()->abs(),
                     0, Type::real, 0, 0);

    if (inPlotCurrentDensity) {
      // Plot the phase of the order parameter.
      inGUI->plotImage(
          inContext->getComputeOrderParameter()->getResult()->complexPhase(), 0,
          Type::real, 1, 0);
    }
  }

  // Plot the current density.
  if (inPlotCurrentDensity) {
    // Plot the x-component of the current density.
    inGUI->plotImage(*(inContext->getComputeCurrent()->getResult()), 0,
                     Type::real, 0, 1);

    // Plot the y-component of the current density.
    inGUI->plotImage(*(inContext->getComputeCurrent()->getResult()), 1,
                     Type::real, 1, 1);

    // Plot the magnitude of the current density.
    inGUI->plotImage(
        inContext->getComputeCurrent()->getResult()->fieldMagnitude(), 0,
        Type::real, 2, 1);
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
/// \param inContext - Context object for simulation.
/// \param inPrintStatus - Print iteration status to terminal or not.
/// \param inNumIterationsMin - The minimum number of iterations.
/// \param inNumIterationsMax - The maximum number of iterations.
///
/// \return - Simulation status.
template <typename T>
Status runIteration(Context<T> *inContext, const bool inPrintStatus = true,
                    const int inNumIterationsMin = 100,
                    const int inNumIterationsMax = 1000,
                    const int inSaveDataInterval = 200) {
  // Solve equations for all compute objects.
  inContext->compute();

  // Output iteration status to the terminal.
  if (inPrintStatus) {
    printIterationStatus<T>(inContext);
  }

  const int numIterations = inContext->getNumIterations();

  // Check for convergence.
  const bool isConverged =
      (inContext->isConverged()) && (numIterations >= inNumIterationsMin);

  // Check for maximum number of iterations.
  const bool patienceReached = (numIterations > inNumIterationsMax);

  // Print status if simulation finished.
  if (isConverged) {
    std::cout << "SUCCESS! Simulation did converge." << std::endl;
  } else if (patienceReached) {
    std::cout << "FAILURE! Simulation did not converge." << std::endl;
  }

  // Determine what to do.
  const bool quit = isConverged || patienceReached || internal::guiUserQuit;
  const bool saveData =
      internal::guiUserSaveData || (numIterations % inSaveDataInterval == 0);

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
/// \param inContext - Context object for simulation.
/// \param inVisualize - Visualize results in OpenGL or not.
/// \param inPrintStatus - Print iteration status to terminal or not.
/// \param inNumIterationsMin - The minimum number of iterations.
/// \param inNumIterationsMax - The maximum number of iterations.
///
/// \return - void.
template <typename T>
void runSimulation(Context<T> *inContext, const bool inVisualize,
                   const bool inPrintStatus = true,
                   const int inNumIterationsMin = 100,
                   const int inNumIterationsMax = 1000) {
  run_utils::Status status = run_utils::Status::kContinue;
  while (status != Status::kQuit) {
    // Run one iteration,
    status = runIteration(inContext, inPrintStatus, inNumIterationsMin,
                          inNumIterationsMax);

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
