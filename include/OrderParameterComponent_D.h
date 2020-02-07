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

template <typename T>
class OrderParameterComponent_D : public OrderParameterComponent<T> {
  typedef typename Grid<T>::GPU GridGPU;

public:
  // Constructors and Destructors
  OrderParameterComponent_D();
  virtual ~OrderParameterComponent_D();

  OrderParameterComponent_D(const OrderParameterComponent_D<T> &other);

  OrderParameterComponent_D(Parameters<T> *param, T Tcthis_Tc = 1.0);

  OrderParameterComponent_D<T> &operator=(OrderParameterComponent_D<T> other);

  virtual OrderParameterComponent_D<T> *clone() const;

  virtual void computeMomentumDependence(GridGPU *target, const T pX,
                                         const T pY,
                                         bool conjugateBasis = false);

  virtual void initBasisNormalization(FermiSurface<T> *fermiSurface);

  virtual T deltaBulk() const;

  void setLatticeOrientation(T angle) { _latticeOrientation = angle; }
  T getLatticeOrientation() const { return _latticeOrientation; }

private:
  friend void swap(OrderParameterComponent_D<T> &A,
                   OrderParameterComponent_D<T> B) {
    swap(static_cast<OrderParameterComponent<T> &>(A),
         static_cast<OrderParameterComponent<T> &>(B));
    std::swap(A._latticeOrientation, B._latticeOrientation);
    std::swap(A._latticeOrientationGrid, B._latticeOrientationGrid);
  }

  // Factor to make <|eta|^2> = 1.
  T m_basisNormalization;

  // Specified in radians, as a rotational offset from the default.
  // A value of 0 (default) aligns the crystal lattice along the x- and y- axes.
  T _latticeOrientation;
  GridGPU *_latticeOrientationGrid;
};

template <typename T>
OrderParameterComponent_D<T>::OrderParameterComponent_D()
    : OrderParameterComponent<T>(), m_basisNormalization(0.0),
      _latticeOrientation(0.0), _latticeOrientationGrid(0) {
  OrderParameterComponent<T>::setType((char *)"d");
}

template <typename T>
OrderParameterComponent_D<T>::OrderParameterComponent_D(
    const OrderParameterComponent_D<T> &other)
    : OrderParameterComponent<T>(other),
      m_basisNormalization(other.m_basisNormalization),
      _latticeOrientation(other._latticeOrientation),
      _latticeOrientationGrid(0) {

  if (other._latticeOrientationGrid)
    _latticeOrientationGrid = new GridGPU(*other._latticeOrientationGrid);
}

template <typename T>
OrderParameterComponent_D<T>::OrderParameterComponent_D(Parameters<T> *param,
                                                        T Tcthis_Tc)
    : OrderParameterComponent<T>(param, Tcthis_Tc), m_basisNormalization(0.0),
      _latticeOrientation(0.0), _latticeOrientationGrid(0) {
  OrderParameterComponent<T>::setType((char *)"d");
}

template <typename T>
OrderParameterComponent_D<T>::~OrderParameterComponent_D() {
  if (_latticeOrientationGrid)
    delete _latticeOrientationGrid;
}

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
    GridGPU *target, const T pX, const T pY, bool conjugateBasis) {
  T pXRotated;
  T pYRotated;
  utils::rotate(_latticeOrientation, pX, pY, pXRotated, pYRotated);
  const cplx<T> basis = Basis_D<T>::compute(pXRotated, pYRotated);
  (*target) *= m_basisNormalization * (conjugateBasis ? basis.conj() : basis);
}

template <typename T>
void OrderParameterComponent_D<T>::initBasisNormalization(
    FermiSurface<T> *fermiSurface) {
  const int numAngles = fermiSurface->getNumAngles();
  T avgBasisSquared = static_cast<T>(0.0);
  for (int angleIdx = 0; angleIdx < numAngles; ++angleIdx) {
    // Get Fermi momentum.
    T momentumX;
    T momentumY;
    fermiSurface->getFermiMomentum(angleIdx, momentumX, momentumY);

    // Evaluate the basis function.
    const cplx<T> eta = Basis_D<T>::compute(momentumX, momentumY);

    // Get integrand factor.
    const T integrandFactor = fermiSurface->getIntegrandFactor(angleIdx);

    // Accumulate.
    avgBasisSquared += eta.cabsSq() * integrandFactor;
  }
  m_basisNormalization = static_cast<T>(1.0) / std::sqrt(avgBasisSquared);
  printf("D-wave Basis Normalization: %f\n", m_basisNormalization);
}

template <typename T> T OrderParameterComponent_D<T>::deltaBulk() const {
  const T sWaveBulk = OrderParameterComponent<T>::deltaBulk();
  // It is faster to just solve for s-wave and rescale it by the known d-wave
  // value.
  const T scale = static_cast<T>(qcs::D_WAVE_BULK / qcs::S_WAVE_BULK);
  return sWaveBulk * scale;
}

#endif // ORDER_PARAMETER_COMPONENT_D_H_
