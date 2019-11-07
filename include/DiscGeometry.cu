#include "DiscGeometry.h"
#include "DiscGeometry_Kernel.cu"

template <typename T>
DiscGeometry<T>::DiscGeometry() : GeometryComponent<T>(), _disc(0)
{}


template <typename T>
DiscGeometry<T>::DiscGeometry( const DiscGeometry<T>& other ) : GeometryComponent<T>(other), _disc(0)
{
	if ( other._disc ) _disc = new Disc<T>(*other._disc);
}


template <typename T>
DiscGeometry<T>::DiscGeometry( const T radius, const T pos_x, const T pos_y )
	: GeometryComponent<T>()
{
	_disc = new Disc<T>(radius,pos_x,pos_y);
}










template <typename T>
DiscGeometry<T>::~DiscGeometry()
{
	if ( _disc ) delete _disc;
}


template <typename T>
DiscGeometry<T>& DiscGeometry<T>::operator=( DiscGeometry<T> other )
{
	swap(*this,other);
	return *this;
}


template <typename T>
void DiscGeometry<T>::create( const T angle )
{
	typedef GeometryComponent<T> G;

	G::allocateMem();
	
	Parameters<T>* param = G::getParameters();

	if ( !param )
	{
		std::cout << "DiscGeometry::create(): ERROR - No parameters object!\n";
		return;
	}

	const int N = G::getParameters()->getGridResolutionBase();

	const T pos_x = _disc->getPosX();
	const T pos_y = _disc->getPosY();

	const T posRot_x = (cos(angle)*pos_x - sin(angle)*pos_y) * (N-1);
	const T posRot_y = (sin(angle)*pos_x + cos(angle)*pos_y) * (N-1);

	const T radius = _disc->getRadius() * T(N-1);
	
	int Nx, Ny;
	dim3 bpg, tpb;
	G::getGridDimensions( Nx, Ny );
	G::getKernelDimensions(bpg,tpb);

	// Defines discrete boundary domain and properties
	generateDisc_Kernel<T><<<bpg, tpb>>>(
		G::getDomainLabels(), G::getNormals(0), G::getNormals(1), G::getBoundaryDistance(),
		Nx, Ny, radius, posRot_x, posRot_y, angle );	
}


template <typename T>
Grid<char>::GPU* DiscGeometry<T>::createDomain( const int N )
{
	Grid<char>::GPU* dom_component = new Grid<char>::GPU(N,N,1,Type::real);
	dom_component->setZero();

	dim3 bpg, tpb;
	Grid<T>::GPU::getKernelDimensions(bpg,tpb,N,N);

//	const T pos_x = _disc->getPosX();
//	const T pos_y = _disc->getPosY();

//	const T posRot_x = pos_x * (N-1);
//	const T posRot_y = pos_y * (N-1);

//	const T radius = _disc->getRadius() * T(N-1);
	
	// Defines discrete boundary domain and properties
//	generateDisc_Kernel<T><<<bpg, tpb>>>(
//		dom_component->getDataPointer(),
//		N, N, radius, posRot_x, posRot_y, 0.0 );	

	return dom_component;
}
