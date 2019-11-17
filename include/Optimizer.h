#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include "Grid.h"
#include "OrderParameter.h"
#include <cmath>
#include <limits>

template <typename T> class Optimizer {
  typedef typename Grid<T>::GPU GridGPU;

public:
  // Constructor.
  Optimizer();

  // Destructor.
  virtual ~Optimizer() = 0;

  // Get the current residual.
  T getResidual() const { return _residual; }

  // Compute the residual.
  void computeResidual(const OrderParameter<T> &rhsK,
                       const OrderParameter<T> &deltaK);

  // Take a step with the optimizer.
  virtual void step(const OrderParameter<T> &rhsK, OrderParameter<T> &deltaK);

private:
  // The current residual.
  T _residual;
};

template <typename T>
Optimizer<T>::Optimizer() : _residual(std::numeric_limits<T>::max()) {}

template <typename T> Optimizer<T>::~Optimizer() {}

template <typename T>
void Optimizer<T>::computeResidual(const OrderParameter<T> &rhsK,
                                   const OrderParameter<T> &deltaK) {
  // Set residual to zero.
  _residual = static_cast<T>(0.0);

  // Set the number of elements to zero.
  int nElements = 0;

  // Get number of order parameter components.
  const int nComponents = deltaK.getNumOrderParameterComponents();

  // Loop over order parameter components and accumulate the square of
  // individual residuals.
  for (int i = 0; i < nComponents; ++i) {
    GridGPU *f = rhsK.getOrderParameterComponent(i)->getOrderParameterField();
    GridGPU *d = deltaK.getOrderParameterComponent(i)->getOrderParameterField();

    // Accumulate the number of elements.
    const int dimX = d->dimX();
    const int dimY = d->dimY();
    nElements += dimX * dimY;

    // Accumulate the square residual.
    _residual += (*f - *d).complexMagnitudeSq().sumField();
  }

  // Root mean squared. This gives larger weight to large residuals compared to
  // just summing the absolute values.
  _residual = std::sqrt(_residual / static_cast<T>(nElements));
}

#endif // OPTIMIZER_H_
