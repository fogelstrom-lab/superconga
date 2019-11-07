#include "OrderParameterComponent_P.h"
#include "OrderParameterComponent_Kernel.cu"

template <typename T>
OrderParameterComponent_P<T>::OrderParameterComponent_P() :
	OrderParameterComponent<T>(), _latticeOrientation(0.0), _latticeOrientationGrid(0)
{
	OrderParameterComponent<T>::setType((char*)"p");
}


template <typename T>
OrderParameterComponent_P<T>::OrderParameterComponent_P( const OrderParameterComponent_P<T>& other ) :
	OrderParameterComponent<T>( other ), _latticeOrientation(other._latticeOrientation),
	_latticeOrientationGrid(0)
{

	if ( other._latticeOrientationGrid )
		_latticeOrientationGrid = new GridGPU(*other._latticeOrientationGrid);
}


template <typename T>
OrderParameterComponent_P<T>::OrderParameterComponent_P( Parameters<T>* param, T Tcthis_Tc ) :
	OrderParameterComponent<T>( param, Tcthis_Tc ), _latticeOrientation(0.0), _latticeOrientationGrid(0)
{
	OrderParameterComponent<T>::setType((char*)"p");
}


template <typename T>
OrderParameterComponent_P<T>::~OrderParameterComponent_P()
{
	if ( _latticeOrientationGrid ) delete _latticeOrientationGrid;
}


template <typename T>
OrderParameterComponent_P<T>& OrderParameterComponent_P<T>::operator=( OrderParameterComponent_P<T> other )
{
	swap(*this,other);
	return *this;
}

template <typename T>
OrderParameterComponent_P<T>* OrderParameterComponent_P<T>::clone() const
{
	return new OrderParameterComponent_P<T>( *this );
}


/*template <typename T>
typename Grid<T>::GPU* OrderParameterComponent_P<T>::createLatticeOrientationGrid()
{
	if ( _latticeOrientationGrid == 0 )
	{
		int N =OrderParameterComponent<T>::getParameters()->getGridResolutionBase();
		_latticeOrientationGrid = new GridGPU(N,N,1,Type::real);
	}

	return _latticeOrientationGrid;
}


template <typename T>
void OrderParameterComponent_P<T>::setGrain_half( const T orientation )
{
	GridGPU* grain = createLatticeOrientationGrid();
	dim3 bpg, tpb;
	grain->getKernelDimensions(bpg,tpb);

	setGrains_kernel<<<bpg,tpb>>>( grain->getDataPointer(), orientation, grain->dimX(), grain->dimY() );
}*/


template <typename T>
void OrderParameterComponent_P<T>::computeMomentumDependence( GridGPU* target, const T angle )
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

			computeMomentumDependence_kernel<T, Basis_P><<<bpg,tpb>>>(
				target->getDataPointer(0,Type::real),
				target->getDataPointer(0,Type::imag),
				_latticeOrientationGrid->getDataPointer(),
				_latticeOrientation, angle, Nx, Nx );
		}
	}
	else
	{
		(*target) *= Basis_P<T>::compute( angle + _latticeOrientation + 0.5*M_PI );
	}
}


template <typename T>
double OrderParameterComponent_P<T>::deltaBulk() const
{
	return OrderParameterComponent<T>::deltaBulk() * 1.5 / 1.76;
}
