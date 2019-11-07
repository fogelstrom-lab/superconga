#include "Disc.h"

template <typename T>
Disc<T>::Disc() :
	_radius(0.5), _pos_x(0.0), _pos_y(0.0)
{}


template <typename T>
Disc<T>::Disc( const T radius, const T pos_x, const T pos_y ) :
	_radius(radius), _pos_x(pos_x), _pos_y(pos_y)
{}


template <typename T>
Disc<T>::Disc( const Disc<T>& other ) :
	_radius(other._radius), _pos_x(other._pos_x), _pos_y(other._pos_y)
{}


template <typename T>
Disc<T>& Disc<T>::operator=( Disc<T> other )
{
	swap(*this,other);
	return *this;
}


template <typename T>
void Disc<T>::setRadius( const T radius )
{
	_radius = radius;
}

template <typename T>
void Disc<T>::setPosX( const T pos_x )
{
	_pos_x = pos_x;
}

template <typename T>
void Disc<T>::setPosY( const T pos_y )
{
	_pos_y = pos_y;
}
