#include "cplx.h"

//template <typename T>
//const cplx<T> cplx<T>::i = cplx<T>((T)0.0,(T)1.0);


// MEMBER OPERATORS
//
template <typename T>
_DEV_ cplx<T> cplx<T>::operator+( const cplx<T>& r ) const
{
	cplx<T> c;
	c.x = x + r.x;
	c.y = y + r.y;
	return c;
}


template <typename T>
_DEV_ cplx<T> cplx<T>::operator-( const cplx<T>& r ) const
{
	cplx<T> c;
	c.x = x - r.x;
	c.y = y - r.y;
	return c;
}


template <typename T>
_DEV_ cplx<T> cplx<T>::operator*( const cplx<T>& r ) const
{
	cplx<T> c;
	c.x = x*r.x - y*r.y;
	c.y = x*r.y + y*r.x;
	return c;
}


template <typename T>
_DEV_ cplx<T> cplx<T>::operator/( const cplx<T>& r ) const
{
	cplx<T> c;
	T denom = (T)1.0/(r.x*r.x + r.y*r.y);
	c.x = (x*r.x + y*r.y) * denom;
	c.y = (y*r.x - x*r.y) * denom;
	return c;
}


template <typename T>
_DEV_ cplx<T> cplx<T>::operator+( const T & r ) const
{
	cplx<T> c;
	c.x = x + r;
	c.y = y;
	return c;
}


template <typename T>
_DEV_ cplx<T> cplx<T>::operator-( const T & r ) const
{
	cplx<T> c;
	c.x = x - r;
	c.y = y;
	return c;
}


template <typename T>
_DEV_ cplx<T> cplx<T>::operator*( const T & r ) const
{
	cplx<T> c;
	c.x = x*r;
	c.y = y*r;
	return c;
}


template <typename T>
_DEV_ cplx<T> cplx<T>::operator/( const T & r ) const
{
	cplx<T> c;
	T tmp = (T)1.0/r;
	c.x = x * tmp;
	c.y = y * tmp;
	return c;
}


template <typename T>
_DEV_ cplx<T>& cplx<T>::operator=( const cplx<T>& r )
{
	if ( this != &r )
	{
		x = r.x;
		y = r.y;
	}
	return *this;
}


template <typename T>
_DEV_ void cplx<T>::operator+=( const cplx<T>& r )
{
	x += r.x;
	y += r.y;
}


template <typename T>
_DEV_ void cplx<T>::operator-=( const cplx<T>& r )
{
	x -= r.x;
	y -= r.y;
}


template <typename T>
_DEV_ void cplx<T>::operator*=( const cplx<T>& r )
{
	cplx<T> c;
	c.x = x * r.x - y * r.y;
	c.y = x * r.y + y * r.x;

	x = c.x;
	y = c.y;
}


template <typename T>
_DEV_ void cplx<T>::operator*=( const T & r )
{
	x *= r;
	y *= r;
}


template <typename T>
_DEV_ cplx<T> cplx<T>::operator-() const
{
	cplx<T> c;
	c.x = -x;
	c.y = -y;
	return c;
}


// FRIEND OPERATORS
//
template <typename T1, typename T2>
_DEV_ cplx<T1> operator*( const T2 & l, const cplx<T1> & r )
{
	cplx<T1> c;
	c.x = (T1)l * r.x;
	c.y = (T1)l * r.y;
	return c;	
}


template <typename T1, typename T2>
_DEV_ cplx<T1> operator+( const T2 & l, const cplx<T1> & r )
{
	cplx<T1> c;
	c.x = (T1)l + r.x;
	c.y = r.y;
	return c;	
}


template <typename T1, typename T2>
_DEV_ cplx<T1> operator-( const T2 & l, const cplx<T1> & r )
{
	cplx<T1> c;
	c.x = (T1)l - r.x;
	c.y = -r.y;
	return c;	
}


// MEMBER FUNCTIONS
//
template <typename T>
_DEV_ cplx<T> cplx<T>::conj() const
{
	cplx<T> c;
	c.x = x;
	c.y = -y;
	return c;
}


template <typename T>
_DEV_ cplx<T> cplx<T>::imult() const
{
	cplx<T> c;
	c.x = -y;
	c.y = x;
	return c;
}


template <typename T>
_DEV_ T cplx<T>::cabs() const
{
	return sqrt( x*x + y*y );
}


template <typename T>
_DEV_ T cplx<T>::cabsSq() const
{
	return (x*x + y*y);
}


// STATIC MEMBER FUNCTIONS
//
template <typename T>
_DEV_ cplx<T> cplx<T>::csqrt( const cplx<T>& r )
{
	cplx<T> c;
	T tmp = sqrt(r.x*r.x + r.y*r.y);
	c.x = sqrt( (tmp + r.x)*(T)0.5 );
	c.y = sqrt( (tmp - r.x)*(T)0.5 );
	
	if ( r.y < (T)0.0 )
		c.y = -c.y;

	return c;
}


template <typename T>
_DEV_ cplx<T> cplx<T>::cexp( const cplx<T>& r )
{
	cplx<T> c;
	T tmp = exp(r.x);
	c.x = tmp * cos(r.y);
	c.y = tmp * sin(r.y);
	return c;
}
