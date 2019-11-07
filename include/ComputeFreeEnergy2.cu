#include "ComputeFreeEnergy2.h"

template <typename T>
ComputeFreeEnergy2<T>::ComputeFreeEnergy2()
	: ComputeProperty<T>(), _free_energy(0), _f(0), _f_tilde(0)
{}


template <typename T>
ComputeFreeEnergy2<T>::ComputeFreeEnergy2( Context<T>* ctx )
	: ComputeProperty<T>(ctx), _free_energy(0), _f(0), _f_tilde(0)
{
	ctx->add(this);	
}


template <typename T>
ComputeFreeEnergy2<T>::~ComputeFreeEnergy2()
{
	if ( _free_energy ) delete _free_energy;
	if ( _f ) delete _f;
	if ( _f_tilde ) delete _f_tilde;
}


template <typename T>
int ComputeFreeEnergy2<T>::initialize()
{
	std::cout << "ComputeFreeEnergy2<T>::initialize()\n";
	typedef ContextModule<T> C;
	
	// Get resolution
	const int N0 = C::getParameters()->getGridResolutionBase();
	const int N1 = C::getParameters()->getGridResolutionCoherence();

	GridGPU::initializeGrid( _free_energy, N0, N0 );
	GridGPU::initializeGrid( _f, N1, N1 );
	GridGPU::initializeGrid( _f_tilde, N1, N1 );

	_free_energy->setZero();

	ComputeProperty<T>::initGreensAngle( N1 );

	return 0;
}


template <typename T>
IntegrationIterator<T>* ComputeFreeEnergy2<T>::createIntegrationIterator( int numEnergiesPerBlock )
{
	return new IntegrationIteratorMatsubara<T>( ContextModule<T>::getContext(), numEnergiesPerBlock );
}


template <typename T>
void ComputeFreeEnergy2<T>::initializeIteration( int numFields )
{
	typedef ContextModule<T> C;
	
	if ( ComputeProperty<T>::doCompute() )
		_free_energy->setZero();
}


template <typename T>
void ComputeFreeEnergy2<T>::initializeGreensAngle()
{
	if ( ComputeProperty<T>::doCompute() )
	{
		//ComputeProperty<T>::initializeGreensAngle();	// Sets to zero
		_f->setZero();
		_f_tilde->setZero();
	}
}


template <typename T>
void ComputeFreeEnergy2<T>::computeGreensAngle(
	ParameterIterator<int>* iter, GridGPU* gamma, GridGPU* gammabar, GeometryGroup<T>* geom )
{
	if ( ComputeProperty<T>::doCompute() )
	{
		GreensFunctionStatic<T,GreensAnomalous>::computeGreensSumEnergy(
			_f, iter, gamma, gammabar, geom );

		GreensFunctionStatic<T,GreensAnomalousTilde>::computeGreensSumEnergy(
			_f_tilde, iter, gamma, gammabar, geom );
	}
}


template <typename T>
__global__ void free_energy_kernel( T* result_re, T* result_im, T* D_re, T* D_im, T* f_re, T* f_im, T* fb_re, T* fb_im, const int N )
{
	const int tx = blockDim.x * blockIdx.x + threadIdx.x;		// x index
	const int ty = blockDim.y * blockIdx.y + threadIdx.y;		// y index

	if ( tx < N && ty < N )
	{
		const int idx = ty*N + tx;

		const cplx<T> D( D_re[idx], D_im[idx] );
		const cplx<T> f( f_re[idx], f_im[idx] );
		const cplx<T> fb( fb_re[idx], fb_im[idx] );

		cplx<T> res = D * fb + D.conj() * f;

		result_re[idx] = res.x;
		result_im[idx] = res.y;
	}
}


template <typename T>
void ComputeFreeEnergy2<T>::integrateGreensAngle( GeometryGroup<T>* geom, const T angle )
{
	if ( ComputeProperty<T>::doCompute() )
	{
		GridGPU* greens = ComputeProperty<T>::getGreensCurrentAngle();
		GridGPU* delta_p = ComputeProperty<T>::getOrderParameter()->getOrderParameterFieldRotated();
		
		// Compensate for numerical diffusion from interpolation during rotation 
		geom->extendValuesAtBoundary( _f, 1 );
		geom->extendValuesAtBoundary( _f_tilde, 1 );

		// Kernel to compute: Delta(pF)*f_tilde + conjugate(Delta(pF))*f
		dim3 bpg, tpb;
		delta_p->getKernelDimensions(bpg,tpb);

		free_energy_kernel<<<bpg,tpb>>>(
			greens->getDataPointer(0,Type::real),
			greens->getDataPointer(0,Type::imag),
			delta_p->getDataPointer(0,Type::real),
			delta_p->getDataPointer(0,Type::imag),
			_f->getDataPointer(0,Type::real),
			_f->getDataPointer(0,Type::imag),
			_f_tilde->getDataPointer(0,Type::real),
			_f_tilde->getDataPointer(0,Type::imag),
			greens->dimX()	);

		// Rotate back to reference frame and add
		_free_energy->insertRotatedGridAdditive( greens, angle, 0 );
	}
}


template <typename T>
void ComputeFreeEnergy2<T>::computePropertyFromGreens()
{
	if ( ComputeProperty<T>::doCompute() )
	{
		const int numAngles = ComputeProperty<T>::getParameters()->getAngularResolution();
		const T T_Tc = ComputeProperty<T>::getParameters()->getTemperature();

		*_free_energy *= 2.0*M_PI*T_Tc/T(numAngles);
	}
}

