#ifndef _ORDERPARAMETERCOMPONENT_P_H
#define _ORDERPARAMETERCOMPONENT_P_H

#include <Grid.h>
#include <OrderParameterComponent.h>
#include <OrderParameterComponent_Kernel.h>
#include <Parameters.h>

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

  virtual void computeMomentumDependence(GridGPU *target, const T angle);

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

  // Specified in radians, as a rotational offset from the default.
  // A value of 0 (default) aligns the crystal lattice along the x- and y- axes.
  T _latticeOrientation;
  GridGPU *_latticeOrientationGrid;
};

template <typename T>
OrderParameterComponent_P<T>::OrderParameterComponent_P()
    : OrderParameterComponent<T>(), _latticeOrientation(0.0),
      _latticeOrientationGrid(0) {
  OrderParameterComponent<T>::setType((char *)"p");
}

template <typename T>
OrderParameterComponent_P<T>::OrderParameterComponent_P(
    const OrderParameterComponent_P<T> &other)
    : OrderParameterComponent<T>(other),
      _latticeOrientation(other._latticeOrientation),
      _latticeOrientationGrid(0) {

  if (other._latticeOrientationGrid)
    _latticeOrientationGrid = new GridGPU(*other._latticeOrientationGrid);
}

template <typename T>
OrderParameterComponent_P<T>::OrderParameterComponent_P(Parameters<T> *param,
                                                        T Tcthis_Tc)
    : OrderParameterComponent<T>(param, Tcthis_Tc), _latticeOrientation(0.0),
      _latticeOrientationGrid(0) {
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
void OrderParameterComponent_P<T>::computeMomentumDependence(GridGPU *target,
                                                             const T angle) {
  // Add 0.5*PI to angle because zero rotation of order parameter field is
  // equivalent to momentum direction PI/2 (Riccati trajectory solver integrates
  // along positive y (for reasons of optimal memory access pattern))
  if (_latticeOrientationGrid) {
    int Nx = target->dimX();

    if (Nx == _latticeOrientationGrid->dimX()) {
      dim3 bpg, tpb;
      target->getKernelDimensions(bpg, tpb);

      computeMomentumDependence_kernel<T, Basis_P>
          <<<bpg, tpb>>>(target->getDataPointer(0, Type::real),
                         target->getDataPointer(0, Type::imag),
                         _latticeOrientationGrid->getDataPointer(),
                         _latticeOrientation, angle, Nx, Nx);
    }
  } else {
    (*target) *= Basis_P<T>::compute(angle + _latticeOrientation + 0.5 * M_PI);
  }
}

template <typename T> double OrderParameterComponent_P<T>::deltaBulk() const {
  return OrderParameterComponent<T>::deltaBulk() * 1.5 / 1.76;
}
#endif
