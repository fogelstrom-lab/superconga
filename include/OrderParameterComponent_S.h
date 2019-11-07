#ifndef _ORDERPARAMETERCOMPONENT_S_H
#define _ORDERPARAMETERCOMPONENT_S_H

//#include "util/csecond.c"
#include <Parameters.h>
#include <Grid.h>
#include <OrderParameterComponent.h>

//#include "OrderParameterComponent_Kernel.cu"

template <typename T>
class OrderParameterComponent_S : public OrderParameterComponent<T>
{
	typedef typename Grid<T>::GPU GridGPU;

public:
	// Constructors and Destructors
					OrderParameterComponent_S();
	virtual			~OrderParameterComponent_S() {}

					OrderParameterComponent_S( const OrderParameterComponent_S<T>& other );

					OrderParameterComponent_S( Parameters<T>* param, T Tcthis_Tc = 1.0 );

	OrderParameterComponent_S<T>& operator=( OrderParameterComponent_S<T> other );

	virtual OrderParameterComponent_S<T>* clone() const;
					
	virtual void	computeMomentumDependence( GridGPU* target, const T angle ) {}

	virtual double	deltaBulk() const;

	/* Added
	virtual void	setLatticeOrientation( T angle );
	virtual T		getLatticeOrientation() const; */

private:
	friend void		swap( OrderParameterComponent_S<T>& A, OrderParameterComponent_S<T> B )
	{
		swap( static_cast<OrderParameterComponent<T>&>(A), static_cast<OrderParameterComponent<T>&>(B) );
	}
};

#endif
		
