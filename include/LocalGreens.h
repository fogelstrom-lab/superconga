////////////////////////////////////////////////////////
// Functions to compute local value of Greens functions
///////////////////////////////////////////////////////

#ifndef _LOCALGREENS_H
#define _LOCALGREENS_H

#ifdef __CUDACC__
#ifndef _HD_
#define _HD_ __host__ __device__
#endif
#endif

#include <cplx.h>

template <typename T>
struct Greens
{
	_HD_ static void computeGreens_local( const cplx<T>& g, const cplx<T>& gb, cplx<T>& greens )
	{
		const cplx<T> g_gb = g*gb;
		
		// Calc greens function (g), -i * (1 - gamma*gamma_bar) / (1 + gamma*gamma_bar)
		greens = -cplx<T>::i() * ((T)1.0 - g_gb) / ((T)1.0 + g_gb);
	}
};


template <typename T>
struct GreensAnomalous
{
	_HD_ static void computeGreens_local( const cplx<T>& g, const cplx<T>& gb, cplx<T>& greens )
	{
		// Calc anomalous greens function (f), -2i*gamma / (1 + gamma*gamma_bar)
		greens = -(T)2.0 * cplx<T>::i() * g / ((T)1.0 + g*gb);	
	}
};


template <typename T>
struct GreensAnomalousTilde
{
	_HD_ static void computeGreens_local( const cplx<T>& g, const cplx<T>& gb, cplx<T>& greens )
	{
		// Calc f_tilde, 2i*gammabar / (1 + gamma*gamma_bar)
		greens = (T)2.0 * cplx<T>::i() * gb / ((T)1.0 + g*gb);
	}
};


template <typename T>
struct Greens_Delta
{
	_HD_ static void computeGreens_local( const cplx<T>& g, const cplx<T>& gb, cplx<T>& greens )
	{
		// Calc total greens function expression for order parameter: f + conj(f_tilde)
		cplx<T> mult = (T)2.0 * cplx<T>::i() / ((T)1.0 + g*gb);

		cplx<T> f = -mult*g;
		cplx<T> f_tilde_conj = (mult * gb).conj();

		greens = f + f_tilde_conj;
	}
};


template <typename T>
struct Greens_Current
{
	_HD_ static void computeGreens_local( const cplx<T>& g, const cplx<T>& gb, cplx<T>& greens )
	{
		cplx<T> g_gb = g*gb;

		greens = T(-2.0)*cplx<T>::i() * (1.0-g_gb)/(1.0+g_gb);
	}
};

#endif
