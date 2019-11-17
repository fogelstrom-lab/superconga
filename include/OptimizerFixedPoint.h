#ifndef OPTIMIZERFIXEDPOINT_H_
#define OPTIMIZERFIXEDPOINT_H_

#include "Optimizer.h"
#include "OrderParameter.h"

template <typename T> class OptimizerFixedPoint : public Optimizer<T> {

public:
  // Constructor.
  OptimizerFixedPoint() : Optimizer<T>(){};

  // Destructor.
  ~OptimizerFixedPoint(){};

  // Take a step with the optimizer.
  void step(const OrderParameter<T> &rhsK, OrderParameter<T> &deltaK);
};

template <typename T>
void OptimizerFixedPoint<T>::step(const OrderParameter<T> &rhsK,
                                  OrderParameter<T> &deltaK) {
  // Compute the residual.
  Optimizer<T>::computeResidual(rhsK, deltaK);

  // Update the order parameter.
  deltaK = rhsK;
}

#endif // OPTIMIZERFIXEDPOINT_H_
