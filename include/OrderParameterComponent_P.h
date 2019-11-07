#ifndef _ORDERPARAMETERCOMPONENT_P_H
#define _ORDERPARAMETERCOMPONENT_P_H

//#include "util/csecond.c"
#include <Parameters.h>
#include <Grid.h>
#include <OrderParameterComponent.h>

//#include "OrderParameterComponent_Kernel.cu"

template <typename T>
class OrderParameterComponent_P : public OrderParameterComponent<T>
{
	typedef typename Grid<T>::GPU GridGPU;

public:
	// Constructors and Destructors
					OrderParameterComponent_P();
	virtual			~OrderParameterComponent_P();

					OrderParameterComponent_P( const OrderParameterComponent_P<T>& other );

					OrderParameterComponent_P( Parameters<T>* param, T Tcthis_Tc = 1.0 );

	OrderParameterComponent_P<T>& operator=( OrderParameterComponent_P<T> other );
	
	virtual OrderParameterComponent_P<T>* clone() const;

	virtual void	computeMomentumDependence( GridGPU* target, const T angle );

	virtual double	deltaBulk() const;
	
	void			setLatticeOrientation( T angle ) { _latticeOrientation = angle; }
	T				getLatticeOrientation() const { return _latticeOrientation; }

	/*GridGPU*		getLatticeOrientationGrid() const { return _latticeOrientationGrid; }
	GridGPU*		createLatticeOrientationGrid();
	
	void			setGrain_half( const T orientation );*/

private:
	friend void		swap( OrderParameterComponent_P<T>& A, OrderParameterComponent_P<T> B )
	{
		swap( static_cast<OrderParameterComponent<T>&>(A), static_cast<OrderParameterComponent<T>&>(B) );
		std::swap( A._latticeOrientation, B._latticeOrientation );
		std::swap( A._latticeOrientationGrid, B._latticeOrientationGrid );
	}

	// Specified in radians, as a rotational offset from the default.
	// A value of 0 (default) aligns the crystal lattice along the x- and y- axes.
	T			_latticeOrientation;
	GridGPU*	_latticeOrientationGrid;
};

#endif
		
