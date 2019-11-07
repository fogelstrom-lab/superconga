//#ifndef _VECTORPOTENTIAL_
//#define _VECTORPOTENTIAL_

template <typename T, template <typename> class POTENTIAL>
//template <typename T, typename POTENTIAL>
__global__ void computeEnergyShift_external_kernel( T* eShift, const T system_angle, const T B, const int N, const T h )
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	const int idx = y*N + x;

	if ( x < N && y < N )
	{
		//eShift[idx] = POTENTIAL<T>::computeVectorPotential_local(
		eShift[idx] = T(M_PI) * POTENTIAL<T>::computeVectorPotential_local(
//		eShift[idx] = POTENTIAL::computeVectorPotential_local(
			(T)x, (T)y, B, system_angle, N, h );
	}
}


template <typename T>
__global__ void computeEnergyShift_internal_kernel( T* eShift, const T* jx, const T* jy, const T system_angle, const int N )
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	const int idx = y*N + x;

	if ( x < N && y < N )
		eShift[idx] += T(M_PI) * (cos(system_angle)*jx[idx] + sin(system_angle)*jy[idx]);
	//	eShift[idx] += (cos(system_angle)*jx[idx] + sin(system_angle)*jy[idx]);
	//	eShift[idx] += cos(system_angle)*jx[idx] + sin(system_angle)*jy[idx];
}

//#endif
