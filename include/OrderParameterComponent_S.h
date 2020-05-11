#ifndef ORDER_PARAMETER_COMPONENT_S_H_
#define ORDER_PARAMETER_COMPONENT_S_H_

#include "Grid.h"
#include "OrderParameterComponent.h"
#include "Parameters.h"
#include <string>

template <typename T>
class OrderParameterComponent_S : public OrderParameterComponent<T> {
  typedef typename Grid<T>::GPU GridGPU;

public:
  // Constructors and Destructors
  OrderParameterComponent_S();

  OrderParameterComponent_S(const OrderParameterComponent_S<T> &other);

  OrderParameterComponent_S(Parameters<T> *param, T Tcthis_Tc = 1.0);

  OrderParameterComponent_S<T> &operator=(OrderParameterComponent_S<T> other);

  OrderParameterComponent_S<T> *clone() const;

  // There is no momentum dependence.
  void computeMomentumDependence(GridGPU *target, const T pX, const T pY,
                                 bool conjugateBasis = false) {}

  // Because there is no momentum dependence the basis function normalization is
  // trivially satisfied.
  void initBasisNormalization(FermiSurface<T> *fermiSurface){};

  T deltaBulk() const;

  std::string getType() const { return m_type; }

  T getBasisNormalization() const { return m_basisNormalization; }

private:
  friend void swap(OrderParameterComponent_S<T> &A,
                   OrderParameterComponent_S<T> B) {
    swap(static_cast<OrderParameterComponent<T> &>(A),
         static_cast<OrderParameterComponent<T> &>(B));
  }

  // Factor to make <|eta|^2> = 1. Trivial in s-wave, included for generality.
  const T m_basisNormalization;

  const std::string m_type = std::string("s");
};

template <typename T>
OrderParameterComponent_S<T>::OrderParameterComponent_S()
    : OrderParameterComponent<T>(), m_basisNormalization(1.0) {}

template <typename T>
OrderParameterComponent_S<T>::OrderParameterComponent_S(
    const OrderParameterComponent_S<T> &other)
    : OrderParameterComponent<T>(other), m_basisNormalization(1.0) {}

template <typename T>
OrderParameterComponent_S<T>::OrderParameterComponent_S(Parameters<T> *param,
                                                        T Tcthis_Tc)
    : OrderParameterComponent<T>(param, Tcthis_Tc), m_basisNormalization(1.0) {}

template <typename T>
OrderParameterComponent_S<T> &
OrderParameterComponent_S<T>::operator=(OrderParameterComponent_S<T> other) {
  swap(*this, other);
  return *this;
}

template <typename T>
OrderParameterComponent_S<T> *OrderParameterComponent_S<T>::clone() const {
  return new OrderParameterComponent_S<T>(*this);
}

template <typename T> T OrderParameterComponent_S<T>::deltaBulk() const {
  return OrderParameterComponent<T>::deltaBulk();
}

#endif // ORDER_PARAMETER_COMPONENT_S_H_
