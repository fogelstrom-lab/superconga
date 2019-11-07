#ifndef _POLYGONGEOMETRY_H
#define _POLYGONGEOMETRY_H

#include <cuda.h>
#include "Polygon.h"
#include "GeometryComponent.h"

template <typename T>
class PolygonGeometry : public GeometryComponent<T>
{
public:
			PolygonGeometry();
	virtual ~PolygonGeometry();

			PolygonGeometry( const PolygonGeometry& other );

			// Initialize the polygon to an approximation of a disc
			// numVertices = 4 will produce a square. cx, cy is the center coordinate
			PolygonGeometry( const int numVertices,
								T radius, T angle = 0.0, T cx = 0.0, T cy = 0.0 );		

			// Initialize the polygon to a predefined vertex set
			PolygonGeometry( const int numVertices, T* vertX, T* vertY );		


	PolygonGeometry&			operator=( PolygonGeometry other );

	virtual PolygonGeometry* 	clone() const { return new PolygonGeometry(*this); }

	virtual void				create( const T angle );
	virtual Grid<char>::GPU*	createDomain( const int N );

	Polygon<T>*					getPolygon() const { return _poly; }

private:

	friend void		swap( PolygonGeometry& A, PolygonGeometry& B )
	{
		swap( static_cast<GeometryComponent<T>&>(A), static_cast<GeometryComponent<T>&>(B) );
		std::swap( A._poly, B._poly );
	}

	Polygon<T>*	_poly;
};

#endif

