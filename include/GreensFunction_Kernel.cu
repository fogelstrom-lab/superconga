#ifndef _GREENSKERNEL_
#define _GREENSKERNEL_

#include <cplx.h>
#include "LocalGreens.cu"

template <typename T, template <typename> class GREENS>
__global__ 
void computeGreens_kernel(
	const int N,
	T *greens_re, T *greens_im,
	T *gamma_re, T *gamma_im,
	T *gammabar_re, T *gammabar_im,
	char *domainLabels )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;		// x index
	const int y = blockIdx.y * blockDim.y + threadIdx.y;		// y index
	const int e_idx = blockIdx.z;								// energy index

	if ( x < N )
	{
		const int domain_idx = y*N + x;
		const int gamma_idx = e_idx*2*N*N + y*N + x;
	
		cplx<T> greens_local, gamma(gamma_re[gamma_idx], gamma_im[gamma_idx]), gammabar(gammabar_re[gamma_idx], gammabar_im[gamma_idx]);

		if ( domainLabels[domain_idx] > 0 )
		{
			GREENS<T>::computeGreens_local( gamma, gammabar, greens_local );
		}
	
		greens_re[gamma_idx] = greens_local.x;
		greens_im[gamma_idx] = greens_local.y;
	}
}


template <typename T, template <typename> class GREENS>
__global__ 
void computeGreensSumEnergy_kernel(
	const int N, const int Ne,
	T *greens_re, T *greens_im,
	T *gamma_re, T *gamma_im,
	T *gammabar_re, T *gammabar_im,
	char *domainLabels )
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;		// x index
	const int y = blockDim.y * blockIdx.y + threadIdx.y;		// y index
	
	int gamma_idx, greens_idx;
	int e_idx;
	
	if ( x < N )
	{
		greens_idx = y*N + x;
	
		cplx<T> greensTmp, greens_local, gamma, gammabar, greens_sum((T)0.0,(T)0.0);
	
		if ( domainLabels[greens_idx] > 0 )
		{
			for ( e_idx=0; e_idx<Ne; ++e_idx )
			{
				gamma_idx = e_idx*2*N*N + greens_idx;

				gamma = cplx<T>(gamma_re[gamma_idx], gamma_im[gamma_idx]);
				gammabar = cplx<T>(gammabar_re[gamma_idx], gammabar_im[gamma_idx]);

				GREENS<T>::computeGreens_local( gamma, gammabar, greensTmp );

				greens_sum += greensTmp;
			}
		}
	
		greens_re[greens_idx] += greens_sum.x;
		greens_im[greens_idx] += greens_sum.y;
	}
}


template <typename T, template <typename> class GREENS>
__global__ 
void computeGreensSumEnergy_kernel(
	const int N, const int Ne,
	const T *summandFactor,
	T *greens_re, T *greens_im,
	T *gamma_re, T *gamma_im,
	T *gammabar_re, T *gammabar_im,
	char *domainLabels )
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;		// x index
	const int y = blockDim.y * blockIdx.y + threadIdx.y;		// y index
	
	int gamma_idx, greens_idx;
	int e_idx;
	
	if ( x < N )
	{
		greens_idx = y*N + x;
	
		cplx<T> greensTmp, greens_local, gamma, gammabar, greens_sum((T)0.0,(T)0.0);
	
		if ( domainLabels[greens_idx] > 0 )
		{
			for ( e_idx=0; e_idx<Ne; ++e_idx )
			{
				gamma_idx = e_idx*2*N*N + greens_idx;

				gamma = cplx<T>(gamma_re[gamma_idx], gamma_im[gamma_idx]);
				gammabar = cplx<T>(gammabar_re[gamma_idx], gammabar_im[gamma_idx]);

				GREENS<T>::computeGreens_local( gamma, gammabar, greensTmp );

				greens_sum += summandFactor[e_idx] * greensTmp;
			}
		}
	
		greens_re[greens_idx] += greens_sum.x;
		greens_im[greens_idx] += greens_sum.y;
	}
}

#endif
