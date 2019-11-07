
template <typename T>
__global__ void phaseWinding_radial_Kernel(
	T * data_re, T * data_im,
	const int Nx, const int Ny, const T h,
	const T numWinding, const T cutOffRadius, const T phaseShift, 
	const T center_x, const T center_y, T deltaBulk )
{
	// Coordinates to new resampling grid
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ( i < Nx && j < Ny )
	{
		const int idx = j*Nx + i;
	
		// Normalize coordinates to -0.5 <= x,y <= +0.5, and offset with center
		T x = ((T)i/(T)(Nx-1) - (T)0.5) - center_x;
		T y = ((T)j/(T)(Ny-1) - (T)0.5) - center_y;
	
		T phase;
	
		if ( x >= (T)0.0 && y >= (T)0.0 ) {
			if ( x == (T)0.0 )
				phase = M_PI*(T)0.5;
			else
				phase = atan(y/x);
		} else if ( x < (T)0.0 && y >= (T)0.0 ) {
			phase = M_PI + atan(y/x);
		} else if ( x < (T)0.0 && y < (T)0.0 ) {
			phase = M_PI + atan(y/x);
		} else {
			phase = (T)2.0*M_PI + atan(y/x);
		}

		phase += phaseShift;

		T radius = sqrt(x*x + y*y);

		if ( radius < cutOffRadius )
			phase = (T)0.0;

		if ( center_x == (T)0.0 && center_y == (T)0.0 && i == (Nx-1)/2 && j == (Ny-1)/2 )
			deltaBulk = (T)0.0;

		data_re[idx] = cos(phase*numWinding) * deltaBulk;
		data_im[idx] = sin(phase*numWinding) * deltaBulk;
	}
}

