#ifndef ORDER_PARAMETER_COMPONENT_D_H_
#define ORDER_PARAMETER_COMPONENT_D_H_

#include "Grid.h"
#include "OrderParameterBasis.h"
#include "OrderParameterComponent.h"
#include "Parameters.h"
#include "cplx.h"
#include "defines.h"
#include "utils.h"
#include <cmath>
#include <cuda.h>
#include <string>

template <typename T>
class OrderParameterComponent_D : public OrderParameterComponent<T> {
  typedef typename Grid<T>::GPU GridGPU;

public:
  // Constructors.
  OrderParameterComponent_D();
  OrderParameterComponent_D(const OrderParameterComponent_D<T> &other);
  OrderParameterComponent_D(Parameters<T> *param, T Tcthis_Tc = 1.0);

  OrderParameterComponent_D<T> &operator=(OrderParameterComponent_D<T> other);
  OrderParameterComponent_D<T> *clone() const override;

  void
  computeMomentumDependence(GridGPU *target, const T pX, const T pY,
                            bool conjugateBasisFunction = false) const override;

  // TODO(niclas): Move to base class!
  void initBasisFunctionNormalization(FermiSurface<T> *fermiSurface) override;

  T deltaBulk() const override;

  std::string getType() const override { return m_type; }

private:
  friend void swap(OrderParameterComponent_D<T> &A,
                   OrderParameterComponent_D<T> B) {
    swap(static_cast<OrderParameterComponent<T> &>(A),
         static_cast<OrderParameterComponent<T> &>(B));
  }

  const std::string m_type = std::string("d");
};

template <typename T>
OrderParameterComponent_D<T>::OrderParameterComponent_D()
    : OrderParameterComponent<T>() {}

template <typename T>
OrderParameterComponent_D<T>::OrderParameterComponent_D(
    const OrderParameterComponent_D<T> &other)
    : OrderParameterComponent<T>(other) {}

template <typename T>
OrderParameterComponent_D<T>::OrderParameterComponent_D(Parameters<T> *param,
                                                        T Tcthis_Tc)
    : OrderParameterComponent<T>(param, Tcthis_Tc) {}

template <typename T>
OrderParameterComponent_D<T> &
OrderParameterComponent_D<T>::operator=(OrderParameterComponent_D<T> other) {
  swap(*this, other);
  return *this;
}

template <typename T>
OrderParameterComponent_D<T> *OrderParameterComponent_D<T>::clone() const {
  return new OrderParameterComponent_D<T>(*this);
}

template <typename T>
void OrderParameterComponent_D<T>::computeMomentumDependence(
    GridGPU *target, const T pX, const T pY,
    bool conjugateBasisFunction) const {
  const T crystalAxesRotation =
      OrderParameterComponent<T>::getCrystalAxesRotation();
  T pXRotated;
  T pYRotated;
  utils::rotate(crystalAxesRotation, pX, pY, pXRotated, pYRotated);
  const cplx<T> basisFunction =
      BasisFunction_D<T>::compute(pXRotated, pYRotated);
  const T basisFunctionNormalization =
      OrderParameterComponent<T>::getBasisFunctionNormalization();
  (*target) *= basisFunctionNormalization *
               (conjugateBasisFunction ? basisFunction.conj() : basisFunction);
}

template <typename T>
void OrderParameterComponent_D<T>::initBasisFunctionNormalization(
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
        BasisFunction_D<T>::compute(momentumX, momentumY);

    // Get integrand factor.
    const T integrandFactor = fermiSurface->getIntegrandFactor(angleIdx);

    // Accumulate.
    avgBasisFunctionSquared += basisFunction.cabsSq() * integrandFactor;
  }
  const T basisFunctionNormalization =
      static_cast<T>(1.0) / std::sqrt(avgBasisFunctionSquared);
  OrderParameterComponent<T>::setBasisFunctionNormalization(basisFunctionNormalization);
  printf("D-wave Basis Function Normalization: %f\n", basisFunctionNormalization);
}

template <typename T> T OrderParameterComponent_D<T>::deltaBulk() const {
  const T sWaveBulk = OrderParameterComponent<T>::deltaBulk();
  // It is faster to just solve for s-wave and rescale it by the known d-wave
  // value.
  const T scale = static_cast<T>(qcs::D_WAVE_BULK / qcs::S_WAVE_BULK);
  return sWaveBulk * scale;
}

#endif // ORDER_PARAMETER_COMPONENT_D_H_
