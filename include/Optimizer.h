#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include "OrderParameter.h"

template <typename T> class Optimizer {
public:
  // Constructor.
  Optimizer(){};

  // Destructor.
  virtual ~Optimizer(){};

  // Take naive fixed-point step, i.e. do nothing here.
  virtual void step(OrderParameter<T> *const inOutOrderParameter,
                    const T inResidual) {
    // Unused.
    (void)inOutOrderParameter;
    (void)inResidual;
  };
};

#endif // OPTIMIZER_H_
