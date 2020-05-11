#ifndef ORDER_PARAMETER_COMPONENT_P_H_
#define ORDER_PARAMETER_COMPONENT_P_H_

#include "Grid.h"
#include "OrderParameterBasis.h"
#include "OrderParameterComponent.h"
#include "Parameters.h"
#include "cplx.h"
#include "utils.h"
#include <cmath>
#include <cuda.h>
#include <string>

template <typename T>
class OrderParameterComponent_P : public OrderParameterComponent<T> {
  typedef typename Grid<T>::GPU GridGPU;

public:
  // Constructors and Destructors
  OrderParameterComponent_P();

  OrderParameterComponent_P(const OrderParameterComponent_P<T> &other);

  OrderParameterComponent_P(Parameters<T> *param, T Tcthis_Tc = 1.0);

  OrderParameterComponent_P<T> &operator=(OrderParameterComponent_P<T> other);

  OrderParameterComponent_P<T> *clone() const;

  void computeMomentumDependence(GridGPU *target, const T pX, const T pY,
                                 bool conjugateBasis = false);

  void initBasisNormalization(FermiSurface<T> *fermiSurface);

  T deltaBulk() const;

  std::string getType() const { return m_type; }

  T getBasisNormalization() const { return m_basisNormalization; }

private:
  friend void swap(OrderParameterComponent_P<T> &A,
                   OrderParameterComponent_P<T> B) {
    swap(static_cast<OrderParameterComponent<T> &>(A),
         static_cast<OrderParameterComponent<T> &>(B));
  }

  // Factor to make <|eta|^2> = 1.
  T m_basisNormalization;

  const std::string m_type = std::string("p");
};

template <typename T>
OrderParameterComponent_P<T>::OrderParameterComponent_P()
    : OrderParameterComponent<T>(), m_basisNormalization(0.0) {}

template <typename T>
OrderParameterComponent_P<T>::OrderParameterComponent_P(
    const OrderParameterComponent_P<T> &other)
    : OrderParameterComponent<T>(other),
      m_basisNormalization(other.m_basisNormalization) {}

template <typename T>
OrderParameterComponent_P<T>::OrderParameterComponent_P(Parameters<T> *param,
                                                        T Tcthis_Tc)
    : OrderParameterComponent<T>(param, Tcthis_Tc), m_basisNormalization(0.0) {}

template <typename T>
OrderParameterComponent_P<T> &
OrderParameterComponent_P<T>::operator=(OrderParameterComponent_P<T> other) {
  swap(*this, other);
  return *this;
}

template <typename T>
OrderParameterComponent_P<T> *OrderParameterComponent_P<T>::clone() const {
  return new OrderParameterComponent_P<T>(*this);
}

template <typename T>
void OrderParameterComponent_P<T>::computeMomentumDependence(
    GridGPU *target, const T pX, const T pY, bool conjugateBasis) {
  const T crystalAxesRotation =
      OrderParameterComponent<T>::getCrystalAxesRotation();
  T pXRotated;
  T pYRotated;
  utils::rotate(crystalAxesRotation, pX, pY, pXRotated, pYRotated);
  const cplx<T> basis = Basis_P<T>::compute(pXRotated, pYRotated);
  (*target) *= m_basisNormalization * (conjugateBasis ? basis.conj() : basis);
}

template <typename T>
void OrderParameterComponent_P<T>::initBasisNormalization(
    FermiSurface<T> *fermiSurface) {
  const int numAngles = fermiSurface->getNumAngles();
  T avgBasisSquared = static_cast<T>(0.0);
  for (int angleIdx = 0; angleIdx < numAngles; ++angleIdx) {
    // Get Fermi momentum.
    T momentumX;
    T momentumY;
    fermiSurface->getFermiMomentum(angleIdx, momentumX, momentumY);

    // Evaluate the basis function.
    const cplx<T> eta = Basis_P<T>::compute(momentumX, momentumY);

    // Get integrand factor.
    const T integrandFactor = fermiSurface->getIntegrandFactor(angleIdx);

    // Accumulate.
    avgBasisSquared += eta.cabsSq() * integrandFactor;
  }
  m_basisNormalization = static_cast<T>(1.0) / std::sqrt(avgBasisSquared);
  printf("P-wave Basis Normalization: %f\n", m_basisNormalization);
}

template <typename T> T OrderParameterComponent_P<T>::deltaBulk() const {
  return OrderParameterComponent<T>::deltaBulk();
}

#endif // ORDER_PARAMETER_COMPONENT_P_H_
