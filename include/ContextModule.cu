#include "ContextModule.h"
#include <cuda.h>


template<typename T>
void ContextModule<T>::checkCudaError( std::string msg )
{
	cudaError_t err = cudaGetLastError();
	
	if ( err != cudaSuccess )
		std::cout << "CUDA ERROR: " << msg << ": " << cudaGetErrorString( err ) << std::endl;
}
