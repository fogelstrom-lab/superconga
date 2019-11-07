#include "GreensFunctionStatic.h"


// All these kernels can be optimized by using getKernelDimensionsOptim() !!!!


// Compute greens function for all energies, without integrating (over energies)
//
template <typename T, template <typename> class GREENS>
void GreensFunctionStatic<T,GREENS>::computeGreens(
					GridGPU * result,
					const ParameterIterator<int> * iter,
					const GridGPU * gamma,
					const GridGPU * gammabar,
					const GeometryGroup<T> * geom )
{
	int numEnergies = iter->getNumInterval();
	
	dim3 bpg, tpb;
	GreensFunctionStatic<T,GREENS>::getKernelDimensions( bpg, tpb, result, numEnergies );
	
	// Compute Greens function from gamma and gammabar
	computeGreens_kernel<T,GREENS><<<bpg,tpb>>>(
		result->dimX(),
		result->getDataPointer( 0, Type::real ),
		result->getDataPointer( 0, Type::imag ),
		gamma->getDataPointer( 0, Type::real ),
		gamma->getDataPointer( 0, Type::imag ),
		gammabar->getDataPointer( 0, Type::real ),
		gammabar->getDataPointer( 0, Type::imag ),
		geom->getDomainLabelsGrid()->getDataPointer( 0, Type::real )
		);
}


// Compute greens function for all energies and integrate (over energies)
//
template <typename T, template <typename> class GREENS>
void GreensFunctionStatic<T,GREENS>::computeGreensSumEnergy(
					GridGPU * result,
					const ParameterIterator<int> * iter,
					const GridGPU * gamma,
					const GridGPU * gammabar,
					const GeometryGroup<T> * geom )
{
	int numEnergies = iter->getNumInterval();
	
	dim3 bpg, tpb;
	GridGPU::getKernelDimensions( bpg, tpb, result->dimX(), result->dimY(), 1 );
		
	// Compute Greens function from gamma and gammabar
	computeGreensSumEnergy_kernel<T,GREENS><<<bpg,tpb>>>(
		result->dimX(), numEnergies,
		result->getDataPointer( 0, Type::real ),
		result->getDataPointer( 0, Type::imag ),
		gamma->getDataPointer( 0, Type::real ),
		gamma->getDataPointer( 0, Type::imag ),
		gammabar->getDataPointer( 0, Type::real ),
		gammabar->getDataPointer( 0, Type::imag ),
		geom->getDomainLabelsGrid()->getDataPointer( 0, Type::real )
		);
}


// Compute greens function for all energies and integrate (over energies)
//
template <typename T, template <typename> class GREENS>
void GreensFunctionStatic<T,GREENS>::computeGreensSumEnergy(
					GridGPU * result,
					const ParameterIterator<int> * iter,
					const GridGPU * summandFactor,
					const GridGPU * gamma,
					const GridGPU * gammabar,
					const GeometryGroup<T> * geom )
{
	int numEnergies = iter->getNumInterval();
	
	dim3 bpg, tpb;
	GridGPU::getKernelDimensions( bpg, tpb, result->dimX(), result->dimY(), 1 );
		
	// Compute Greens function from gamma and gammabar
	computeGreensSumEnergy_kernel<T,GREENS><<<bpg,tpb>>>(
		result->dimX(), numEnergies,
		summandFactor->getDataPointer( 0, Type::real ),
		result->getDataPointer( 0, Type::real ),
		result->getDataPointer( 0, Type::imag ),
		gamma->getDataPointer( 0, Type::real ),
		gamma->getDataPointer( 0, Type::imag ),
		gammabar->getDataPointer( 0, Type::real ),
		gammabar->getDataPointer( 0, Type::imag ),
		geom->getDomainLabelsGrid()->getDataPointer( 0, Type::real )
		);
}


template <typename T, template <typename> class GREENS>
void GreensFunctionStatic<T,GREENS>::getKernelDimensions( dim3& blocksPerGrid, dim3& threadsPerBlock, const GridGPU* grid, int numEnergies )
{
	const int N = grid->dimX();

	if ( numEnergies == 0 )
		numEnergies = grid->numFields();

	int tpb_x, tpb_y, bpg_x, bpg_y;

	tpb_x = N;
	tpb_y = 1;

	bpg_x = 1;
		
	if ( tpb_x > 1024 )
	{
		tpb_x = N/2 + 1;
		bpg_x = 2;
	}

	bpg_y = N;
		
	threadsPerBlock	= dim3( tpb_x, tpb_y, 1 );
	blocksPerGrid	= dim3( bpg_x, bpg_y, numEnergies );
}
