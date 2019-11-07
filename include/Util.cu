#include "Util.h"
#include "Util_Kernel.cu"

template <typename T>
void Util<T>::mirror_diag_lower_to_upper( typename Grid<T>::GPU* grid )
{
	const int N = grid->dimX();
		
	// GPU init
	dim3 blocksPerGrid, threadsPerBlock;
	grid->getKernelDimensions( blocksPerGrid, threadsPerBlock );
	
	// Kernel launch
	for ( int c=0; c<grid->numFields(); ++c )
	{
		mirror_diag_lower_to_upper_kernel<<< blocksPerGrid, threadsPerBlock >>>(
			grid->getDataPointer( c, Type::real ),	N );			

		if ( grid->isComplex() )
			mirror_diag_lower_to_upper_kernel<<< blocksPerGrid, threadsPerBlock >>>(
				grid->getDataPointer( c, Type::imag ),	N );			
	}
}

