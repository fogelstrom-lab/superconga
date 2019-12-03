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
  OrderParameter<T> *m_deltaKMinusOne;

  // Temporary order parameter needed for updating.
  OrderParameter<T> *m_deltaTmp;

  // Whether the previous order parameter is initialized or not.
  bool m_isDeltaInitialized;

  // Iteration index.
  int m_iterationIndex;

  // The number of warmup iterations. The first step is always a pure
  // fixed-point step. But after that the weight of the ineratial term  is
  // gradually increased to its maximum value during this number of iterations.
  const int m_nWarmupIterations = 10;

  // The maximum weight of the ineartial term.
  const T m_ineratialWeight = static_cast<T>(0.9);
};

template <typename T>
OptimizerRollingBall<T>::OptimizerRollingBall()
    : Optimizer<T>(), m_deltaKMinusOne(0), m_deltaTmp(0),
      m_isDeltaInitialized(false), m_iterationIndex(0) {}

template <typename T> OptimizerRollingBall<T>::~OptimizerRollingBall() {
  if (m_deltaKMinusOne) {
    delete m_deltaKMinusOne;
  }
  if (m_deltaTmp) {
    delete m_deltaTmp;
  }
}

template <typename T>
void OptimizerRollingBall<T>::step(const OrderParameter<T> &rhsK,
                                   OrderParameter<T> &deltaK) {

  // Compute the residual.
  Optimizer<T>::computeResidual(rhsK, deltaK);

  if (m_iterationIndex > 0) {
    const int nComponents = deltaK.getNumOrderParameterComponents();
    for (int i = 0; i < nComponents; ++i) {
      // Extract fields.
      const GridGPU &f =
          *(rhsK.getOrderParameterComponent(i)->getOrderParameterField());
      GridGPU &d =
          *(deltaK.getOrderParameterComponent(i)->getOrderParameterField());
      GridGPU &dp = *(m_deltaKMinusOne->getOrderParameterComponent(i)
                          ->getOrderParameterField());
      GridGPU &dt = *(
          m_deltaTmp->getOrderParameterComponent(i)->getOrderParameterField());

      // Compute ineartial weight.
      const T warmupFraction = static_cast<T>(m_iterationIndex) /
                               static_cast<T>(m_nWarmupIterations + 1);
      const T weight =
          m_ineratialWeight * std::min(static_cast<T>(1.0), warmupFraction);

      // Update the order parameter with a rolling ball step.
      // Set tmp to new order parameter.
      dt = f + (d - dp) * weight;

      // Update previous order parameter.
      dp = d;

      // Set the order parameter to tmp.
      d = dt;
    }
  } else {
    if (m_isDeltaInitialized) {
      // Update previous order parameter.
      *m_deltaKMinusOne = deltaK;
    } else {
      // Delete previous order parameter.
      if (m_deltaKMinusOne) {
        delete m_deltaKMinusOne;
      }

      // Delete tmp order parameter.
      if (m_deltaTmp) {
        delete m_deltaTmp;
      }

      // Initialize the internal order parameters.
      m_deltaKMinusOne = new OrderParameter<T>(deltaK);
      m_deltaTmp = new OrderParameter<T>(deltaK);

      // Set flag to true.
      m_isDeltaInitialized = true;
    }

    // Update the order parameter with a fixed point step.
    deltaK = rhsK;
  }

  // Increase iteration index.
  m_iterationIndex++;
}

#endif // OPTIMIZERROLLINGBALL_H_
