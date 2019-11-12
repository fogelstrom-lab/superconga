#include "OrderParameterComponent_D.h"
#include "OrderParameterComponent_Kernel.cu"

template <typename T>
OrderParameterComponent_D<T>::OrderParameterComponent_D() :
	OrderParameterComponent<T>(), _latticeOrientation(0.0), _latticeOrientationGrid(0)
{
	OrderParameterComponent<T>::setType((char*)"d");
}


template <typename T>
OrderParameterComponent_D<T>::OrderParameterComponent_D( const OrderParameterComponent_D<T>& other ) :
	OrderParameterComponent<T>( other ), _latticeOrientation(other._latticeOrientation),
	_latticeOrientationGrid(0)
{

	if ( other._latticeOrientationGrid )
		_latticeOrientationGrid = new GridGPU(*other._latticeOrientationGrid);
}


template <typename T>
OrderParameterComponent_D<T>::OrderParameterComponent_D( Parameters<T>* param, T Tcthis_Tc ) :
	OrderParameterComponent<T>( param, Tcthis_Tc ), _latticeOrientation(0.0), _latticeOrientationGrid(0)
{
	OrderParameterComponent<T>::setType((char*)"d");
}


template <typename T>
OrderParameterComponent_D<T>::~OrderParameterComponent_D()
{
	if ( _latticeOrientationGrid ) delete _latticeOrientationGrid;
}


template <typename T>
OrderParameterComponent_D<T>& OrderParameterComponent_D<T>::operator=( OrderParameterComponent_D<T> other )
{
	swap(*this,other);
	return *this;
}

template <typename T>
OrderParameterComponent_D<T>* OrderParameterComponent_D<T>::clone() const
{
	return new OrderParameterComponent_D<T>( *this );
}


template <typename T>
void OrderParameterComponent_D<T>::computeMomentumDependence( GridGPU* target, const T angle )
{
	// Add 0.5*PI to angle because zero rotation of order parameter field is equivalent to momentum direction PI/2
	// (Riccati trajectory solver integrates along positive y (for reasons of optimal memory access pattern))
	if ( _latticeOrientationGrid )
	{
		int Nx = target->dimX();

		if ( Nx == _latticeOrientationGrid->dimX() )
		{
			dim3 bpg, tpb;
			target->getKernelDimensions(bpg,tpb);

			computeMomentumDependence_kernel<T, Basis_D><<<bpg,tpb>>>(
				target->getDataPointer(0,Type::real),
				target->getDataPointer(0,Type::imag),
				_latticeOrientationGrid->getDataPointer(),
				_latticeOrientation, angle, Nx, Nx );
		}
	}
	else
	{
		(*target) *= Basis_D<T>::compute( angle + _latticeOrientation + 0.5*M_PI );
	}
}


template <typename T>
double OrderParameterComponent_D<T>::deltaBulk() const
{
	return OrderParameterComponent<T>::deltaBulk() * 1.5 / 1.76;
}
