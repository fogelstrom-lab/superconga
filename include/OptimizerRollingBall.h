#ifndef OPTIMIZER_ROLLING_BALL_H_
#define OPTIMIZER_ROLLING_BALL_H_

#include "Grid.h"
#include "Optimizer.h"
#include "OrderParameter.h"
#include <algorithm>

// See p. 12 of Polyak (1964):
// https://www.researchgate.net/publication/243648538_Some_methods_of_speeding_up_the_convergence_of_iteration_methods

namespace scg {
// TODO: (Patric) Implement warmup/burnin and final relaxation at end of
// simulation, with getters and setters.
template <typename T> class OptimizerRollingBall : public Optimizer<T> {
public:
  // Constructor.
  OptimizerRollingBall(const T inIneartialWeight = static_cast<T>(0.5),
                       const int inNumWarmupIterations = 10);

  // Destructor.
  ~OptimizerRollingBall();

  // Take a step with the optimizer.
  void step(OrderParameter<T> *const inOutOrderParameter, const T inResidual);

private:
  // The previous order parameters.
  OrderParameter<T> *m_orderParameterKMinusOne;
  OrderParameter<T> *m_orderParameterKMinusTwo;

  // Whether the previous order parameters are initialized.
  bool m_isInitialized;

  // Iteration index.
  int m_iterationIndex;

  // The number of warmup iterations. The first step is always a pure
  // fixed-point step. But after that the weight of the ineratial term  is
  // gradually increased to its maximum value during this number of iterations.
  const int m_numWarmupIterations;

  // The maximum weight of the ineartial term.
  const T m_ineratialWeight;
};

template <typename T>
OptimizerRollingBall<T>::OptimizerRollingBall(const T inIneartialWeight,
                                              const int inNumWarmupIterations)
    : Optimizer<T>(), m_orderParameterKMinusOne(0),
      m_orderParameterKMinusTwo(0), m_isInitialized(false), m_iterationIndex(0),
      m_numWarmupIterations(inNumWarmupIterations),
      m_ineratialWeight(inIneartialWeight) {}

template <typename T> OptimizerRollingBall<T>::~OptimizerRollingBall() {
  if (m_orderParameterKMinusOne) {
    delete m_orderParameterKMinusOne;
  }
  if (m_orderParameterKMinusTwo) {
    delete m_orderParameterKMinusTwo;
  }
}

template <typename T>
void OptimizerRollingBall<T>::step(OrderParameter<T> *const inOutOrderParameter,
                                   const T inResidual) {
  // Unused.
  (void)inResidual;

  if (m_isInitialized) {
    // Compute ineartial weight.
    const T warmupFraction = static_cast<T>(m_iterationIndex) /
                             static_cast<T>(m_numWarmupIterations + 1);
    const T weight =
        m_ineratialWeight * std::min(static_cast<T>(1.0), warmupFraction);

    const int numComponents =
        inOutOrderParameter->getNumOrderParameterComponents();
    for (int i = 0; i < numComponents; ++i) {
      // The current order parameter,
      GridGPU<T> *d0 = inOutOrderParameter->getOrderParameterComponent(i)
                           ->getOrderParameterField();
      // The order parameter from one iteration ago.
      GridGPU<T> *d1 = m_orderParameterKMinusOne->getOrderParameterComponent(i)
                           ->getOrderParameterField();
      // The order parameter from two iterations ago.
      GridGPU<T> *d2 = m_orderParameterKMinusTwo->getOrderParameterComponent(i)
                           ->getOrderParameterField();

      // Update the order parameter with a rolling ball step.
      *d0 += (*d1 - *d2) * weight;

      // Update the previous order parameters.
      *d2 = *d1;
      *d1 = *d0;
    }
  } else {
    if (m_orderParameterKMinusOne) {
      delete m_orderParameterKMinusOne;
    }
    if (m_orderParameterKMinusTwo) {
      delete m_orderParameterKMinusTwo;
    }

    // Initialize the internal order parameters.
    m_orderParameterKMinusOne = new OrderParameter<T>(*inOutOrderParameter);
    m_orderParameterKMinusTwo = new OrderParameter<T>(*inOutOrderParameter);

    m_isInitialized = true;
  }

  // Increase iteration index.
  m_iterationIndex++;
}
} // namespace scg

#endif // OPTIMIZER_ROLLING_BALL_H_
