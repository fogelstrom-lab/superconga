#ifndef _GREENSFUNCTIONSTATIC_H
#define _GREENSFUNCTIONSTATIC_H

#include <Parameters.h>
#include <Grid.h>
#include <GreensFunction_Kernel.cu>

template<typename> class ParameterIterator;
template<typename> class GeometryGroup;

template <typename T, template <typename> class GREENS>
struct GreensFunctionStatic
{
	typedef typename Grid<T>::GPU GridGPU;

	// Greens function functions
	static void		computeGreens(
						GridGPU * result,
						const ParameterIterator<int> * iter,
						const GridGPU * gamma,
						const GridGPU * gammabar,
						const GeometryGroup<T> * geom );

	static void		computeGreensSumEnergy(
						GridGPU * result,
						const ParameterIterator<int> * iter,
						const GridGPU * gamma,
						const GridGPU * gammabar,
						const GeometryGroup<T> * geom );

	static void		computeGreensSumEnergy(
						GridGPU * result,
						const ParameterIterator<int> * iter,
						const GridGPU * summandFactor,
						const GridGPU * gamma,
						const GridGPU * gammabar,
						const GeometryGroup<T> * geom );
	
	// Helper functions
	static void		getKernelDimensions( dim3 & blocksPerGrid, dim3 & threadsPerBlock, const GridGPU * grid, int numEnergies = 0 );
		
};

#endif

