#ifndef _CPLX_H
#define _CPLX_H

#ifdef __CUDACC__
#include <cuda.h>
#define _DEV_ __host__ __device__
#else
#define _DEV_
#endif


template <typename T>
struct cplx
{
	// Constructor
	_DEV_ cplx() {}
	_DEV_ cplx( const T& _x, const T& _y ) : x(_x), y(_y) {}

	_DEV_ ~cplx() {}

	// Variables
	T x, y;

	// Member operators
	_DEV_ const cplx<T> operator+( const cplx<T>& r ) const;
	_DEV_ const cplx<T> operator-( const cplx<T>& r ) const;
	_DEV_ const cplx<T> operator*( const cplx<T>& r ) const;
	_DEV_ const cplx<T> operator/( const cplx<T>& r ) const;

	_DEV_ const cplx<T> operator+( const T& r ) const;
	_DEV_ const cplx<T> operator-( const T& r ) const;
	_DEV_ const cplx<T> operator*( const T& r ) const;
	_DEV_ const cplx<T> operator/( const T& r ) const;

	_DEV_ const cplx<T> operator-() const;

	_DEV_ cplx<T>& operator=( const cplx<T>& r );

	_DEV_ void operator+=( const cplx<T>& r );
	_DEV_ void operator-=( const cplx<T>& r );
	_DEV_ void operator*=( const cplx<T>& r );

	_DEV_ void operator*=( const T & r );

	// Friend operators
	template <typename T1, typename T2>
	_DEV_ friend const cplx<T1> operator+( const T2& l, const cplx<T1>& r );

	template <typename T1, typename T2>
	_DEV_ friend const cplx<T1> operator-( const T2& l, const cplx<T1>& r );

	template <typename T1, typename T2>
	_DEV_ friend const cplx<T1> operator*( const T2& l, const cplx<T1>& r );


	// Member functions
	_DEV_ const cplx<T> conj() const;
	_DEV_ const cplx<T> imult() const;
	_DEV_ T cabs() const;
	_DEV_ T cabsSq() const;

	// Static member functions
	_DEV_ static cplx<T> csqrt( const cplx<T>& r );
	_DEV_ static cplx<T> cexp( const cplx<T>& r );
	
	// Static member variables
	_DEV_ static cplx<T> i() { return cplx<T>((T)0.0,(T)1.0); }
};

#endif
