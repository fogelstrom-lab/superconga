#include "GeometryComponent.h"
#include "GeometryComponent_Kernel.cu"

template <typename T>
GeometryComponent<T>::GeometryComponent()
	:	_param(0), _domainLabels(0), _normals(0), _boundaryDistance(0)
{
	std::cout << "GeometryComponent()\n";
}


template <typename T>
GeometryComponent<T>::GeometryComponent( const GeometryComponent<T>& other ) :
	_param(other._param), _domainLabels(0), _normals(0), _boundaryDistance(0)
{
	if ( other._domainLabels )		_domainLabels = new Grid<char>::GPU( *other._domainLabels );
	if ( other._normals )			_normals = new GridGPU( *other._normals );
	if ( other._boundaryDistance )	_boundaryDistance = new GridGPU( *other._boundaryDistance );
}


template <typename T>
GeometryComponent<T>::~GeometryComponent()
{
	std::cout << "~GeometryComponent()\n";
	freeMem();
}


template <typename T>
void swap( GeometryComponent<T>& A, GeometryComponent<T>& B )
{
	std::swap(A._param, B._param);
	std::swap(A._domainLabels, B._domainLabels);
	std::swap(A._normals, B._normals);
	std::swap(A._boundaryDistance, B._boundaryDistance);
}


template <typename T>
void GeometryComponent<T>::allocateMem()
{
	int N;
	if ( _param )
		N = _param->getGridResolutionDeltaRotated();
	else
	{
		std::cout << "GeometryComponent<T>::allocateMem(): Error: No Parameters<T> object!\n";
		return;
	}

	GridGPU::initializeGrid( _normals, N, N, 2, Type::real );
	GridGPU::initializeGrid( _boundaryDistance, N, N, 1, Type::real );
	Grid<char>::GPU::initializeGrid( _domainLabels, N, N, 1, Type::real );
}


template <typename T>
void GeometryComponent<T>::freeMem()
{
	GridGPU::deleteGrid( _normals );
	GridGPU::deleteGrid( _boundaryDistance );
	Grid<char>::GPU::deleteGrid( _domainLabels );
}


template <typename T>
void GeometryComponent<T>::labelBoundaryNodes()
{
	int Nx, Ny;
	dim3 tpb, bpg;
	
	getGridDimensions( Nx, Ny );
	getKernelDimensions( bpg, tpb );
	
	// Label boundary interface nodes
	Grid<char>::GPU* oldLabels = _domainLabels;
	Grid<char>::GPU* newLabels(0);
	Grid<char>::GPU::initializeGrid(newLabels,oldLabels);

	labelEntryExit_Kernel<T><<<bpg, tpb>>>( oldLabels->getDataPointer(),
											newLabels->getDataPointer(),
											Nx, Ny );

	_domainLabels = newLabels;
	delete oldLabels;
}


template <typename T>
void GeometryComponent<T>::getGridDimensions( int& Nx, int& Ny ) const
{
	if ( _domainLabels )
	{
		Nx = _domainLabels->dimX();
		Ny = _domainLabels->dimY();
	}
	else
	{
		std::cout << "GeometryComponent<T>::getGridDimensions() Error: _domainLabels not initialized.\n";
		Nx = 0;
		Ny = 0;
	}
}


template <typename T>
void GeometryComponent<T>::getKernelDimensions( dim3& bpg, dim3& tpb ) const
{
	int Nx, Ny;
	getGridDimensions( Nx, Ny );

	int tx = 64;
	int bx = Nx/tx;
	if ( tx*bx != Nx )
		bx++;
		
	tpb = dim3(tx,1);
	bpg = dim3(bx,Ny);
}


template <typename T>
char* GeometryComponent<T>::getDomainLabels() const
{
	return _domainLabels->getDataPointer();
}


template <typename T>
T* GeometryComponent<T>::getNormals( const int dim ) const 
{
	return _normals->getDataPointer( dim );
}


template <typename T>
T* GeometryComponent<T>::getBoundaryDistance() const
{
	return _boundaryDistance->getDataPointer();
}
