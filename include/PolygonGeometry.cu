#include "PolygonGeometry.h"
#include "PolygonGeometry_Kernel.cu"

template <typename T>
PolygonGeometry<T>::PolygonGeometry() : GeometryComponent<T>(), _poly(0)
{}


template <typename T>
PolygonGeometry<T>::PolygonGeometry( const PolygonGeometry<T>& other ) : GeometryComponent<T>(other), _poly(0)
{
	if ( other._poly ) _poly = new Polygon<T>(*other._poly);
}


template <typename T>
PolygonGeometry<T>::PolygonGeometry( const int numVertices, T radius, T angle, T cx, T cy )
	: GeometryComponent<T>()
{
	_poly = new Polygon<T>(numVertices, radius, angle, cx, cy);
}


template <typename T>
PolygonGeometry<T>::PolygonGeometry( const int numVertices, T* vertX, T* vertY )
	: GeometryComponent<T>()
{
	_poly = new Polygon<T>(numVertices, vertX, vertY);
}


template <typename T>
PolygonGeometry<T>::~PolygonGeometry()
{
	if ( _poly ) delete _poly;
}


template <typename T>
PolygonGeometry<T>& PolygonGeometry<T>::operator=( PolygonGeometry<T> other )
{
	swap(*this,other);
	return *this;
}


template <typename T>
void PolygonGeometry<T>::create( const T angle )
{
	typedef GeometryComponent<T> G;
	
	G::allocateMem();

	int Nx, Ny;
	dim3 bpg, tpb;

	G::getGridDimensions(Nx,Ny);
	G::getKernelDimensions(bpg,tpb);

	// Calc inside/outside, distance, normals
	generatePolygonGeometry_Kernel<T,256><<<bpg, tpb>>>(
		G::getDomainLabels(), G::getNormals(0), G::getNormals(1), G::getBoundaryDistance(),
		Nx, Ny, angle, _poly->getNumVertices(),
		_poly->getVertices()->getDataPointer(0),
		_poly->getVertices()->getDataPointer(1) );
	
	G::labelBoundaryNodes();

	maskNormals_Kernel<T><<<bpg,tpb>>>(
		G::getDomainLabels(), G::getNormals(0), G::getNormals(1), G::getBoundaryDistance(), Nx, Ny );
}


template <typename T>
Grid<char>::GPU* PolygonGeometry<T>::createDomain( const int N )
{
	Grid<char>::GPU* dom_component = new Grid<char>::GPU(N,N,1,Type::real);
	dom_component->setZero();

	dim3 bpg, tpb;
	Grid<T>::GPU::getKernelDimensions(bpg,tpb,N,N);

	// Calc inside/outside, distance, normals
	generatePolygonGeometry_Kernel<T,256><<<bpg, tpb>>>(
		dom_component->getDataPointer(),
		_poly->getNumVertices(),
		_poly->getVertices()->getDataPointer(0),
		_poly->getVertices()->getDataPointer(1),
		N );
 
 	return dom_component;
}

