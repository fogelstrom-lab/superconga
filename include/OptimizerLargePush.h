#ifndef OPTIMIZER_LARGE_PUSH_H_
#define OPTIMIZER_LARGE_PUSH_H_

#include "Optimizer.h"
#include "OrderParameter.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace scg {
// TODO: (Patric) Implement warmup/burnin and final relaxation at end of
// simulation, with getters and setters.
template <typename T> class OptimizerLargePush : public Optimizer<T> {
public:
  // Constructor.
  OptimizerLargePush(
      const int inNumWarmupIterations = 10,
      const int inNumRelaxationIterations = 10,
      const T inMinConvergenceRateForPush = static_cast<T>(1.00001),
      const T inMaxConvergenceRateForPush = static_cast<T>(1.5),
      const T inMaxPushFactor = static_cast<T>(100.0));

  // Destructor.
  ~OptimizerLargePush();

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

  // To keep track of the iterations inbetween pushes.
  int m_relaxationIndex;

  // The number of warmup iterations.
  const int m_numWarmupIterations;

  // The number of relaxation iterations inbetween pushes.
  const int m_numRelaxationIterations;

  // Push parameters.
  const T m_minConvergenceRateForPush;
  const T m_maxConvergenceRateForPush;
  const T m_maxPushFactor;

  T m_residual;
  T m_convergenceRate;
};

template <typename T>
OptimizerLargePush<T>::OptimizerLargePush(const int inNumWarmupIterations,
                                          const int inNumRelaxationIterations,
                                          const T inMinConvergenceRateForPush,
                                          const T inMaxConvergenceRateForPush,
                                          const T inMaxPushFactor)
    : Optimizer<T>(), m_orderParameterKMinusOne(0),
      m_orderParameterKMinusTwo(0), m_isInitialized(false), m_iterationIndex(0),
      m_relaxationIndex(0), m_numWarmupIterations(inNumWarmupIterations),
      m_numRelaxationIterations(inNumRelaxationIterations),
      m_minConvergenceRateForPush(inMinConvergenceRateForPush),
      m_maxConvergenceRateForPush(inMaxConvergenceRateForPush),
      m_maxPushFactor(inMaxPushFactor), m_residual(static_cast<T>(0.0)),
      m_convergenceRate(static_cast<T>(0.0)) {}

template <typename T> OptimizerLargePush<T>::~OptimizerLargePush() {
  if (m_orderParameterKMinusOne) {
    delete m_orderParameterKMinusOne;
  }
  if (m_orderParameterKMinusTwo) {
    delete m_orderParameterKMinusTwo;
  }
}

template <typename T>
void OptimizerLargePush<T>::step(OrderParameter<T> *const inOutOrderParameter,
                                 const T inResidual) {
  // The current convergence rate.
  const T convergenceRate = m_residual / inResidual;

  // Check if the convergence rate is within accepable bounds.
  const bool isConvergenceRateOk =
      (m_minConvergenceRateForPush < convergenceRate) &&
      (convergenceRate < m_maxConvergenceRateForPush);

  // Check that enough iterations has passed.
  const bool isRelaxed = (m_relaxationIndex >= m_numRelaxationIterations) &&
                         (m_iterationIndex > m_numWarmupIterations);

  const bool doPush = isConvergenceRateOk && isRelaxed;

  const int numComponents =
      inOutOrderParameter->getNumOrderParameterComponents();

  if (m_isInitialized) {
    if (doPush) {
      // Reset relaxation index.
      m_relaxationIndex = 0;
    } else {
      // Increase relaxation index.
      m_relaxationIndex++;
    }

    // TODO(niclas): Why the factor of 1/4? I guess it just some empricially
    // good number. But from just looking at number of iterations needed for
    // convergence on our examples it seems better to set it to 1.
    const T numer = static_cast<T>(0.25);

    // TODO(niclas): Why not use the current covergence rate?
    const T denom = std::max(m_convergenceRate - static_cast<T>(1.0),
                             std::numeric_limits<T>::epsilon());

    const T pushFactor = std::min(numer / denom, m_maxPushFactor);

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

      if (doPush) {
        // Update the order parameter with a large push step.
        *d0 += (*d1 - *d2) * pushFactor;
      }

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

  m_residual = inResidual;
  m_convergenceRate = convergenceRate;

  // Increase iteration index.
  m_iterationIndex++;
}
} // namespace scg

#endif // OPTIMIZER_LARGE_PUSH_H_
