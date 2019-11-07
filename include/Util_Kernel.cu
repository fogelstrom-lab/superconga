
template <typename T>
__global__ void mirror_diag_lower_to_upper_kernel( T* data, const int N )
{
	// Coordinates to grid
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	const int idx = y*N + x;
	
	if ( y < x && x < N )
	{
		const int idx_dst = x*N + y;
		data[idx_dst] = data[idx];
	}
}
