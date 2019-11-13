//
// This class provides a basic shape implementation for a disc. 
//
#ifndef _DISC_H
#define _DISC_H

template <typename T>
class Disc
{
public:
		Disc();
		~Disc() {};
	
		// Copy constructor
		Disc( const Disc& other );

		Disc( const T radius, const T pos_x = 0.0, const T pos_y = 0.0 );


	Disc&		operator=( Disc other );
					
	void		setRadius( const T radius );
	void		setPosX( const T pos_x );
	void		setPosY( const T pos_y );

	T 			getRadius() const { return _radius; }
	T 			getPosX() const { return _pos_x; }
	T 			getPosY() const { return _pos_y; }

private:

	friend void		swap( Disc& A, Disc& B )
	{
		std::swap(A._radius, B._radius);
		std::swap(A._pos_x, B._pos_x );
		std::swap(A._pos_y, B._pos_y );
	}

	T	_radius;		// Radius of disc
	T	_pos_x, _pos_y;	// Center of disc, stored in centered coordinates -0.5 < x,y < 0.5	
};

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
#endif

