#ifndef _ORDERPARAMETERCOMPONENT_S_H
#define _ORDERPARAMETERCOMPONENT_S_H

#include <Parameters.h>
#include <Grid.h>
#include <OrderParameterComponent.h>

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

private:
	friend void		swap( OrderParameterComponent_S<T>& A, OrderParameterComponent_S<T> B )
	{
		swap( static_cast<OrderParameterComponent<T>&>(A), static_cast<OrderParameterComponent<T>&>(B) );
	}
};

template <typename T>
OrderParameterComponent_S<T>::OrderParameterComponent_S() :
	OrderParameterComponent<T>()
{
	OrderParameterComponent<T>::setType((char*)"s");
}


template <typename T>
OrderParameterComponent_S<T>::OrderParameterComponent_S( const OrderParameterComponent_S<T>& other ) :
	OrderParameterComponent<T>( other )
{
}


template <typename T>
OrderParameterComponent_S<T>::OrderParameterComponent_S( Parameters<T>* param, T Tcthis_Tc )
	:	OrderParameterComponent<T>( param, Tcthis_Tc )
{
	OrderParameterComponent<T>::setType((char*)"s");
}


template <typename T>
OrderParameterComponent_S<T>& OrderParameterComponent_S<T>::operator=( OrderParameterComponent_S<T> other )
{
	swap(*this,other);
	return *this;
}


template <typename T>
OrderParameterComponent_S<T>* OrderParameterComponent_S<T>::clone() const
{
	return new OrderParameterComponent_S<T>( *this );
}


template <typename T>
double OrderParameterComponent_S<T>::deltaBulk() const
{
	return OrderParameterComponent<T>::deltaBulk();
}
#endif
		
