#include "OrderParameterComponent_S.h"

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
