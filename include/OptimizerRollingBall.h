#ifndef OPTIMIZERROLLINGBALL_H_
#define OPTIMIZERROLLINGBALL_H_

#include "Optimizer.h"
#include "OrderParameter.h"
#include <algorithm>

/*
See p. 12 of Polyak (1964):
https://www.researchgate.net/publication/243648538_Some_methods_of_speeding_up_the_convergence_of_iteration_methods
*/
template <typename T> class OptimizerRollingBall : public Optimizer<T> {
  typedef typename Grid<T>::GPU GridGPU;

public:
  // Constructor.
  OptimizerRollingBall();

  // Destructor.
  ~OptimizerRollingBall();

  // Take a step with the optimizer.
  void step(const OrderParameter<T> &rhsK, OrderParameter<T> &deltaK);

private:
  // The previous order parameter.
  OrderParameter<T> *_deltaKMinusOne;

  // Temporary order parameter needed for updating.
  OrderParameter<T> *_deltaTmp;

  // Whether the previous order parameter is initialized or not.
  bool _isDeltaInitialized;

  // Iteration index.
  int _iterationIndex;

  // The number of warmup iterations. The first step is always a pure
  // fixed-point step. But after that the weight of the ineratial term  is
  // gradually increased to its maximum value during this number of iterations.
  const int _nWarmupIterations = 10;

  // The maximum weight of the ineartial term.
  const T _ineartialWeight = static_cast<T>(0.9);
};

template <typename T>
OptimizerRollingBall<T>::OptimizerRollingBall()
    : Optimizer<T>(), _deltaKMinusOne(0), _deltaTmp(0),
      _isDeltaInitialized(false), _iterationIndex(0) {}

template <typename T> OptimizerRollingBall<T>::~OptimizerRollingBall() {
  if (_deltaKMinusOne) {
    delete _deltaKMinusOne;
  }
  if (_deltaTmp) {
    delete _deltaTmp;
  }
}

template <typename T>
void OptimizerRollingBall<T>::step(const OrderParameter<T> &rhsK,
                                   OrderParameter<T> &deltaK) {

  // Compute the residual.
  Optimizer<T>::computeResidual(rhsK, deltaK);

  if (_iterationIndex > 0) {
    const int nComponents = deltaK.getNumOrderParameterComponents();
    for (int i = 0; i < nComponents; ++i) {
      // Extract fields.
      const GridGPU &f =
          *(rhsK.getOrderParameterComponent(i)->getOrderParameterField());
      GridGPU &d =
          *(deltaK.getOrderParameterComponent(i)->getOrderParameterField());
      GridGPU &dp = *(_deltaKMinusOne->getOrderParameterComponent(i)
                          ->getOrderParameterField());
      GridGPU &dt =
          *(_deltaTmp->getOrderParameterComponent(i)->getOrderParameterField());

      // Compute ineartial weight.
      const T warmupFraction = static_cast<T>(_iterationIndex) /
                               static_cast<T>(_nWarmupIterations + 1);
      const T weight =
          _ineartialWeight * std::min(static_cast<T>(1.0), warmupFraction);

      // Update the order parameter with a rolling ball step.
      // Set tmp to new order parameter.
      dt = f + (d - dp) * weight;

      // Update previous order parameter.
      dp = d;

      // Set the order parameter to tmp.
      d = dt;
    }
  } else {
    if (_isDeltaInitialized) {
      // Update previous order parameter.
      *_deltaKMinusOne = deltaK;
    } else {
      // Delete previous order parameter.
      if (_deltaKMinusOne) {
        delete _deltaKMinusOne;
      }

      // Delete tmp order parameter.
      if (_deltaTmp) {
        delete _deltaTmp;
      }

      // Initialize the internal order parameters.
      _deltaKMinusOne = new OrderParameter<T>(deltaK);
      _deltaTmp = new OrderParameter<T>(deltaK);

      // Set flag to true.
      _isDeltaInitialized = true;
    }

    // Update the order parameter with a fixed point step.
    deltaK = rhsK;
  }

  // Increase iteration index.
  _iterationIndex++;
}

#endif // OPTIMIZERROLLINGBALL_H_
