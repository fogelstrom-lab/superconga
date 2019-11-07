//
// Handles the discrete representation of a disc shape primitive.
//
#ifndef _DISCGEOMETRY_H
#define _DISCGEOMETRY_H

#include <cuda.h>
#include <Disc.h>
#include "GeometryComponent.h"

template <typename T>
class DiscGeometry : public GeometryComponent<T>
{
public:
					DiscGeometry();
	virtual			~DiscGeometry();
	
	 				DiscGeometry( const DiscGeometry& other );

					DiscGeometry( const T radius, const T pos_x = 0.0, const T pos_y = 0.0 );
	

	DiscGeometry& 				operator=( DiscGeometry other );

	virtual DiscGeometry* 		clone() const { return new DiscGeometry(*this); }

	virtual void				create( const T angle );
	virtual Grid<char>::GPU*	createDomain( const int N );

	Disc<T>* 					getDisc() const { return _disc; }

private:

	friend void		swap( DiscGeometry& A, DiscGeometry& B )
	{
		swap( static_cast<GeometryComponent<T>&>(A), static_cast<GeometryComponent<T>&>(B) );
		std::swap( A._disc, B._disc );
	}

	Disc<T>*	_disc;
};

#endif

