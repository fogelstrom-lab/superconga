#include "ComputeProperty.h"

template <typename T>
__global__ void addFermiFunction_kernel( T *data, T *e_re, T T_Tc, int Ne, int N )
{
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;
	const int te = blockIdx.z;
	
	if ( tx < N && ty < N && te < Ne )
	{
		const int idx = te*N*N + ty*N + tx;
		
		const T E = e_re[te];
		const T fermi = (T)1.0 / (exp(E/T_Tc) + (T)1.0);
		
		data[idx] *= fermi;
	}
}


template <typename T>
ComputeProperty<T>::ComputeProperty()
	: ContextModule<T>(), _greensCurrentAngle(0), _isConverged(false), _doCompute(true)
{}


template <typename T>
ComputeProperty<T>::ComputeProperty( Context<T>* ctx )
	: ContextModule<T>(ctx), _greensCurrentAngle(0), _isConverged(false), _doCompute(true)
{}


template <typename T>
ComputeProperty<T>::~ComputeProperty()
{
	std::cout << "~ComputeProperty()\n";
	if ( _greensCurrentAngle ) delete _greensCurrentAngle;
}


template <typename T>
void ComputeProperty<T>::initGreensAngle( const int N, const int numFields )
{
	GridGPU::initializeGrid( _greensCurrentAngle, N, N, numFields );
}


template <typename T>
__global__ void addGauge_Kernel( T *data_x, T *data_y, const T h, const T phi, const int N )
{
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ( tx < N )
	{
		const int idx = ty*N + tx;
		const T coord_x = (-(T)0.5*(N-1) + tx) * h;
		const T coord_y = (-(T)0.5*(N-1) + ty) * h;
		
		T r2 = coord_x*coord_x + coord_y*coord_y;

		if ( r2 < (T)1E-07 )
			r2 = (T)1E-07;
				
		const T mult = phi / ((T)2.0*M_PI*r2);
	
		data_x[idx] += mult * (-coord_y);
		data_y[idx] += mult * coord_x;
	}
}


template <typename T>
typename Grid<T>::GPU ComputeProperty<T>::computeSFMomentum(
	Parameters<T>* param, OrderParameter<T>* delta )
{
	const int N = param->getGridResolutionBase();
	const T h = param->getGridElementSize();
	const T Phi = param->getMagneticFieldStrength();
//	const T B = Phi / geom->area();
	
	GridGPU* delta_grid = delta->getOrderParameterField();

	GridGPU phaseGradient = delta_grid->phaseGradient( h );
	phaseGradient *= 0.5;

	// Add gauge
	dim3 bpg, tpb;
	phaseGradient.getKernelDimensions( bpg, tpb );
	
	addGauge_Kernel<<<bpg,tpb>>>(
		phaseGradient.getDataPointer(0),
		phaseGradient.getDataPointer(1),
		h, Phi, N );
	
	return phaseGradient;
}
