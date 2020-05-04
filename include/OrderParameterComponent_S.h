#ifndef ORDER_PARAMETER_COMPONENT_S_H_
#define ORDER_PARAMETER_COMPONENT_S_H_

#include "Grid.h"
#include "OrderParameterBasis.h"
#include "OrderParameterComponent.h"
#include "Parameters.h"
#include <string>

template <typename T>
class OrderParameterComponent_S : public OrderParameterComponent<T> {
  typedef typename Grid<T>::GPU GridGPU;

public:
  // Constructors.
  OrderParameterComponent_S();
  OrderParameterComponent_S(const OrderParameterComponent_S<T> &other);
  OrderParameterComponent_S(Parameters<T> *param, T Tcthis_Tc = 1.0);

  OrderParameterComponent_S<T> &operator=(OrderParameterComponent_S<T> other);
  OrderParameterComponent_S<T> *clone() const override;

  // There is no momentum dependence.
  void
  computeMomentumDependence(GridGPU *target, const T pX, const T pY,
                            bool conjugateBasisFunction = false) const override;

  // TODO(niclas): Move to base class!
  void initBasisFunctionNormalization(FermiSurface<T> *fermiSurface) override;

  std::string getType() const override { return m_type; }

private:
  friend void swap(OrderParameterComponent_S<T> &A,
                   OrderParameterComponent_S<T> B) {
    swap(static_cast<OrderParameterComponent<T> &>(A),
         static_cast<OrderParameterComponent<T> &>(B));
  }

  const std::string m_type = std::string("s");
};

template <typename T>
OrderParameterComponent_S<T>::OrderParameterComponent_S()
    : OrderParameterComponent<T>() {}

template <typename T>
OrderParameterComponent_S<T>::OrderParameterComponent_S(
    const OrderParameterComponent_S<T> &other)
    : OrderParameterComponent<T>(other) {}

template <typename T>
OrderParameterComponent_S<T>::OrderParameterComponent_S(Parameters<T> *param,
                                                        T Tcthis_Tc)
    : OrderParameterComponent<T>(param, Tcthis_Tc) {}

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

template <typename T>
void OrderParameterComponent_S<T>::computeMomentumDependence(
    GridGPU *target, const T pX, const T pY,
    bool conjugateBasisFunction) const {
  const T crystalAxesRotation =
      OrderParameterComponent<T>::getCrystalAxesRotation();
  T pXRotated;
  T pYRotated;
  utils::rotate(crystalAxesRotation, pX, pY, pXRotated, pYRotated);
  const cplx<T> basisFunction =
      BasisFunction_S<T>::compute(pXRotated, pYRotated);
  const T basisFunctionNormalization =
      OrderParameterComponent<T>::getBasisFunctionNormalization();
  (*target) *= basisFunctionNormalization *
               (conjugateBasisFunction ? basisFunction.conj() : basisFunction);
}

template <typename T>
void OrderParameterComponent_S<T>::initBasisFunctionNormalization(
    FermiSurface<T> *fermiSurface) {
  const int numAngles = fermiSurface->getNumAngles();
  T avgBasisFunctionSquared = static_cast<T>(0.0);
  for (int angleIdx = 0; angleIdx < numAngles; ++angleIdx) {
    // Get Fermi momentum.
    T momentumX;
    T momentumY;
    fermiSurface->getFermiMomentum(angleIdx, momentumX, momentumY);

    // Evaluate the basis function.
    const cplx<T> basisFunction =
        BasisFunction_S<T>::compute(momentumX, momentumY);

    // Get integrand factor.
    const T integrandFactor = fermiSurface->getIntegrandFactor(angleIdx);

    // Accumulate.
    avgBasisFunctionSquared += basisFunction.cabsSq() * integrandFactor;
  }
  const T basisFunctionNormalization =
      static_cast<T>(1.0) / std::sqrt(avgBasisFunctionSquared);
  OrderParameterComponent<T>::setBasisFunctionNormalization(basisFunctionNormalization);
  printf("S-wave Basis Function Normalization: %f\n", basisFunctionNormalization);
}

#endif // ORDER_PARAMETER_COMPONENT_S_H_
