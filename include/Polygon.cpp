#include "Polygon.h"


template <typename T>
Polygon<T>::Polygon() : _vertices(0), _numVertices(0)
{}


template <typename T>
Polygon<T>::Polygon( const Polygon<T>& other ) : _vertices(0), _numVertices(other._numVertices)
{
	if ( other._vertices ) _vertices = new GridGPU( *other._vertices );
}


template <typename T>
Polygon<T>::Polygon( const int numVertices, T radius, T angle, T cx, T cy )
	: _numVertices(numVertices)
{
	int Nx = numVertices;
	int Ny = 1;
	int dim = 2;
	
	_vertices = new GridGPU(Nx,Ny,dim,Type::real);

	setVerticesRadial( numVertices, radius, angle, cx, cy );	
}


template <typename T>
Polygon<T>::Polygon( const int numVertices, T* vertX, T* vertY )
	: _numVertices(numVertices)
{
	int dim = 2;
	
	_vertices = new GridGPU(numVertices,1,dim,Type::real);

	setVertices( vertX, vertY );	
}


template <typename T>
Polygon<T>::~Polygon()
{
	if ( _vertices ) delete _vertices;
}


template <typename T>
Polygon<T>& Polygon<T>::operator=( Polygon<T> other )
{
	swap(*this,other);
	return *this;
}


template <typename T>
void Polygon<T>::setVertices( const T* px, const T* py )
{
	typename Vector<T>::GPU& vert = *(_vertices->getVector());
	const int offset = _vertices->getFieldComponentOffset(1);

	for ( int i=0; i<_numVertices; ++i )
	{
		vert[i]			= px[i];
		vert[i+offset]	= py[i];
	}
}


template <typename T>
void Polygon<T>::setVerticesRadial( const int numVertices, T radius, T angle, T cx, T cy )
{
	if ( numVertices == _numVertices )
	{
		T px[numVertices], py[numVertices];
	
		for ( int p=0; p<numVertices; ++p )
		{
			T phi = M_PI/(T)numVertices * (1.0+2.0*p) + angle;
		
			px[p] = radius * cos(phi) + cx;
			py[p] = radius * sin(phi) + cy;
		}
		
		setVertices( px, py );
	}
}


template <typename T>
void Polygon<T>::transformPhysicalToNormalized( const int numVertices, T* px, T* py, const T w )
{
	const T wn = 0.35;

	for (int i=0; i<numVertices; ++i)
	{
		T x = px[i];
		T y = py[i];

		x = (x/w - 0.5)*2.0 * wn;
		y = (y/w - 0.5)*2.0 * wn;

		px[i] = x;
		py[i] = y;
	}
}
