#ifndef _ORDERPARAMETERCOMPONENT_D_H
#define _ORDERPARAMETERCOMPONENT_D_H

#include <Parameters.h>
#include <Grid.h>
#include <OrderParameterComponent.h>

template <typename T>
class OrderParameterComponent_D : public OrderParameterComponent<T>
{
	typedef typename Grid<T>::GPU GridGPU;

public:
	// Constructors and Destructors
					OrderParameterComponent_D();
	virtual			~OrderParameterComponent_D();

					OrderParameterComponent_D( const OrderParameterComponent_D<T>& other );

					OrderParameterComponent_D( Parameters<T>* param, T Tcthis_Tc = 1.0 );

	OrderParameterComponent_D<T>& operator=( OrderParameterComponent_D<T> other );
	
	virtual OrderParameterComponent_D<T>* clone() const;

	virtual void	computeMomentumDependence( GridGPU* target, const T angle );

	virtual double	deltaBulk() const;
	
	void			setLatticeOrientation( T angle ) { _latticeOrientation = angle; }
	T				getLatticeOrientation() const { return _latticeOrientation; }

private:
	friend void		swap( OrderParameterComponent_D<T>& A, OrderParameterComponent_D<T> B )
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
		
