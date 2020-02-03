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

template <typename T>
class OrderParameterComponent_P : public OrderParameterComponent<T> {
  typedef typename Grid<T>::GPU GridGPU;

public:
  // Constructors and Destructors
  OrderParameterComponent_P();
  virtual ~OrderParameterComponent_P();

  OrderParameterComponent_P(const OrderParameterComponent_P<T> &other);

  OrderParameterComponent_P(Parameters<T> *param, T Tcthis_Tc = 1.0);

  OrderParameterComponent_P<T> &operator=(OrderParameterComponent_P<T> other);

  virtual OrderParameterComponent_P<T> *clone() const;

  virtual void computeMomentumDependence(GridGPU *target, const T pX,
                                         const T pY,
                                         bool conjugateBasis = false);

  virtual void initBasisNormalization(FermiSurface<T> *fermiSurface);

  virtual double deltaBulk() const;

  void setLatticeOrientation(T angle) { _latticeOrientation = angle; }
  T getLatticeOrientation() const { return _latticeOrientation; }

private:
  friend void swap(OrderParameterComponent_P<T> &A,
                   OrderParameterComponent_P<T> B) {
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
OrderParameterComponent_P<T>::OrderParameterComponent_P()
    : OrderParameterComponent<T>(), m_basisNormalization(0.0),
      _latticeOrientation(0.0), _latticeOrientationGrid(0) {
  OrderParameterComponent<T>::setType((char *)"p");
}

template <typename T>
OrderParameterComponent_P<T>::OrderParameterComponent_P(
    const OrderParameterComponent_P<T> &other)
    : OrderParameterComponent<T>(other),
      m_basisNormalization(other.m_basisNormalization),
      _latticeOrientation(other._latticeOrientation),
      _latticeOrientationGrid(0) {

  if (other._latticeOrientationGrid)
    _latticeOrientationGrid = new GridGPU(*other._latticeOrientationGrid);
}

template <typename T>
OrderParameterComponent_P<T>::OrderParameterComponent_P(Parameters<T> *param,
                                                        T Tcthis_Tc)
    : OrderParameterComponent<T>(param, Tcthis_Tc), m_basisNormalization(0.0),
      _latticeOrientation(0.0), _latticeOrientationGrid(0) {
  OrderParameterComponent<T>::setType((char *)"p");
}

template <typename T>
OrderParameterComponent_P<T>::~OrderParameterComponent_P() {
  if (_latticeOrientationGrid)
    delete _latticeOrientationGrid;
}

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
  T pXRotated;
  T pYRotated;
  utils::rotate(_latticeOrientation, pX, pY, pXRotated, pYRotated);
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

template <typename T> double OrderParameterComponent_P<T>::deltaBulk() const {
  return OrderParameterComponent<T>::deltaBulk() * 1.5 / 1.76;
}

#endif // ORDER_PARAMETER_COMPONENT_P_H_
