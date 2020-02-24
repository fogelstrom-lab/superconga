#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include "OrderParameter.h"

template <typename T> class Optimizer {
public:
  // Constructor.
  Optimizer(){};

  // Destructor.
  virtual ~Optimizer(){};

  // Take a step with the optimizer.
  virtual void step(OrderParameter<T> *const inOutOrderParameter){};
};

#endif // OPTIMIZER_H_
