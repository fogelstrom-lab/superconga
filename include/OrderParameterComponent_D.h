#ifndef _ORDERPARAMETERCOMPONENT_D_H
#define _ORDERPARAMETERCOMPONENT_D_H

#include <Grid.h>
#include <OrderParameterComponent.h>
#include <OrderParameterComponent_Kernel.h>
#include <Parameters.h>
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

  virtual void computeMomentumDependence(GridGPU *target, const T angle);

  virtual double deltaBulk() const;

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

  // Specified in radians, as a rotational offset from the default.
  // A value of 0 (default) aligns the crystal lattice along the x- and y- axes.
  T _latticeOrientation;
  GridGPU *_latticeOrientationGrid;
};

template <typename T>
OrderParameterComponent_D<T>::OrderParameterComponent_D()
    : OrderParameterComponent<T>(), _latticeOrientation(0.0),
      _latticeOrientationGrid(0) {
  OrderParameterComponent<T>::setType((char *)"d");
}

template <typename T>
OrderParameterComponent_D<T>::OrderParameterComponent_D(
    const OrderParameterComponent_D<T> &other)
    : OrderParameterComponent<T>(other),
      _latticeOrientation(other._latticeOrientation),
      _latticeOrientationGrid(0) {

  if (other._latticeOrientationGrid)
    _latticeOrientationGrid = new GridGPU(*other._latticeOrientationGrid);
}

template <typename T>
OrderParameterComponent_D<T>::OrderParameterComponent_D(Parameters<T> *param,
                                                        T Tcthis_Tc)
    : OrderParameterComponent<T>(param, Tcthis_Tc), _latticeOrientation(0.0),
      _latticeOrientationGrid(0) {
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
void OrderParameterComponent_D<T>::computeMomentumDependence(GridGPU *target,
                                                             const T angle) {
  // Add 0.5*PI to angle because zero rotation of order parameter field is
  // equivalent to momentum direction PI/2 (Riccati trajectory solver integrates
  // along positive y (for reasons of optimal memory access pattern))
  if (_latticeOrientationGrid) {
    int Nx = target->dimX();

    if (Nx == _latticeOrientationGrid->dimX()) {
      dim3 bpg, tpb;
      target->getKernelDimensions(bpg, tpb);

      computeMomentumDependence_kernel<T, Basis_D>
          <<<bpg, tpb>>>(target->getDataPointer(0, Type::real),
                         target->getDataPointer(0, Type::imag),
                         _latticeOrientationGrid->getDataPointer(),
                         _latticeOrientation, angle, Nx, Nx);
    }
  } else {
    (*target) *= Basis_D<T>::compute(angle + _latticeOrientation + 0.5 * M_PI);
  }
}

template <typename T> double OrderParameterComponent_D<T>::deltaBulk() const {
  return OrderParameterComponent<T>::deltaBulk() * 1.5 / 1.76;
}
#endif
