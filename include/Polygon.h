#ifndef _POLYGON_H
#define _POLYGON_H

#include <iostream>
#include <Grid.h>

// This class provides a basic shape implementation for a polygon. 
// Polygons must be defined in a counterclockwise fashion.

template <typename T>
class Polygon
{
	typedef typename Grid<T>::GPU GridGPU;

public:
		Polygon();
		~Polygon();
	
		// Copy constructor
		Polygon( const Polygon& other );

		// Initialize the polygon to an approximation of a disc
		// numVertices = 4 will produce a square. cx, cy is the center coordinate
		Polygon( const int numVertices,
				T radius, T angle = 0.0, T cx = 0.0, T cy = 0.0 );		

		// Initialize the polygon to a predefined vertex set
		Polygon( const int numVertices, T* vertX, T* vertY );

	
	Polygon&	operator=( Polygon other );

	int			getNumVertices() const { return _numVertices; }
	GridGPU*	getVertices() const { return _vertices; }

	// Copy vertices from arrays vertX and vertY
	void		setVertices( const T* vertX, const T* vertY );

	// Approximate a disc (this gets called from corresponding constructor)
	void		setVerticesRadial( const int N, T radius, T angle, T cx, T cy );

	// Polygon vertex coordinates needs to be in normalized coordinates (-0.5 to 0.5). This helper function transforms
	// coordinates in physical space to normalized space, should you want to enter coordinates in physical space.
	static void transformPhysicalToNormalized( const int numVertices, T* px, T* py, const T physical_width );

private:

	friend void		swap( Polygon& A, Polygon& B )
	{
		std::swap(A._numVertices, B._numVertices);
		std::swap(A._vertices, B._vertices );
	}

	int				_numVertices;
	GridGPU*		_vertices;
};

#endif
