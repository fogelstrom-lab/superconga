/////////////////////////////////////////////////////////////////////////////
// Functions to compute local contribution to energy from a vector potential
/////////////////////////////////////////////////////////////////////////////

#ifndef _LOCALVECTORPOTENTIAL_
#define _LOCALVECTORPOTENTIAL_

#ifndef _HD_
#define _HD_ __host__ __device__
#endif

namespace VectorPotential
{
	template <typename T>
	struct None
	{
		_HD_ static T computeVectorPotential_local(
			const T x_local, const T y_local, const T B,
			const T system_angle, const int N, const T h);
	};

	template <typename T>
	struct Linear
	{
		_HD_ static T computeVectorPotential_local(
			const T x_local, const T y_local, const T B,
			const T system_angle, const int N, const T h );
	};

	template <typename T>
	struct LinearCentered
	{
		_HD_ static T computeVectorPotential_local(
			const T x_local, const T y_local, const T B,
			const T system_angle, const int N, const T h );
	};

	template <typename T>
	struct Radial
	{
		_HD_ static T computeVectorPotential_local(
			const T x_local, const T y_local, const T B,
			const T system_angle, const int N, const T h );
	};

	template <typename T>
	struct Solenoid
	{
		_HD_ static T computeVectorPotential_local(
			const T x_local, const T y_local, const T B,
			const T system_angle, const int N, const T h );
	};
}



template <typename T>
_HD_ T VectorPotential::None<T>::computeVectorPotential_local(
	const T x_local, const T y_local,
	const T B, const T system_angle,
	const int N, const T h )
{
	return T(0.0);
}


template <typename T>
_HD_ T VectorPotential::Linear<T>::computeVectorPotential_local(
	const T x_local, const T y_local,
	const T B, const T system_angle,
	const int N, const T h )
{
	// Center system coordinates at origin
	const T offset = -T(0.5)*(N-1);

	const T x_local_centered = ( x_local + offset ) * h;
	const T y_local_centered = ( y_local + offset ) * h;

	// Rotate local coordinates to global frame
	const T x_global = x_local_centered * cos(system_angle) - y_local_centered * sin(system_angle) - offset*h;


	// Add energy shift according to vector potential
	return B*x_global*sin(system_angle + T(0.5)*T(M_PI));
}


template <typename T>
_HD_ T VectorPotential::LinearCentered<T>::computeVectorPotential_local(
	const T x_local, const T y_local,
	const T B, const T system_angle,
	const int N, const T h )
{
	// Center system coordinates at origin
	const T offset = -T(0.5)*(N-1);

	const T x_local_centered = ( x_local + offset ) * h;
	const T y_local_centered = ( y_local + offset ) * h;

	// Rotate local coordinates to global frame
	const T x_global = x_local_centered * cos(system_angle) - y_local_centered * sin(system_angle);


	// Add energy shift according to vector potential
	return B*x_global*sin(system_angle + T(0.5)*T(M_PI));
}


template <typename T>
_HD_ T VectorPotential::Radial<T>::computeVectorPotential_local(
	const T x_local, const T y_local,
	const T B, const T system_angle,
	const int N, const T h )
{
	// Center system coordinates at origin
	const T offset = -T(0.5)*(N-1);
	const T sinAngle = sin(system_angle);
	const T cosAngle = cos(system_angle);

	const T x_local_centered = ( x_local + offset ) * h;
	const T y_local_centered = ( y_local + offset ) * h;

	// Rotate local coordinates to global frame
	const T x_global = x_local_centered * cosAngle - y_local_centered * sinAngle;
	const T y_global = x_local_centered * sinAngle + y_local_centered * cosAngle;

	// Add energy shift according to vector potential
	return T(0.5)*B*( -y_global * cos(system_angle + T(0.5)*T(M_PI)) + x_global * sin(system_angle + T(0.5)*T(M_PI)) );
}


template <typename T>
_HD_ T VectorPotential::Solenoid<T>::computeVectorPotential_local(
	const T x_local, const T y_local,
	const T B, const T system_angle,
	const int N, const T h )
{
	// Center system coordinates at origin
	const T offset = -T(0.5)*(N-1);
	const T sinAngle = sin(system_angle);
	const T cosAngle = cos(system_angle);

	const T x_local_centered = ( x_local + offset ) * h;
	const T y_local_centered = ( y_local + offset ) * h;

	// Rotate local coordinates to global frame
	const T x_global = x_local_centered * cosAngle - y_local_centered * sinAngle;
	const T y_global = x_local_centered * sinAngle + y_local_centered * cosAngle;
	
	T r2 = x_global*x_global + y_global*y_global;
	
	if ( r2 < (T)1E-07 )
		r2 = (T)1E-07;
	
	// Add energy shift according to vector potential:
	// vF * A = vF * (Phi/(2*PI*r^2))*(-y,x)
	return B / (T(2.0)*T(M_PI)*r2) * ( -y_global * cos(system_angle + T(0.5)*T(M_PI)) + x_global * sin(system_angle + T(0.5)*T(M_PI)) );
}



#endif

