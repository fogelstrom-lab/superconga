
template <typename T, typename T2>
__global__ void typeConvert_kernel( const T* data_src, T2* data_dst, const int Nx, const int Ny, const int complex )
{
	// Coordinates to grid
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;
	const int tf = blockIdx.z;
	
	const int idx = tf*(1+complex)*Nx*Ny + ty*Nx + tx;
	
	if ( tx < Nx && ty < Ny )
		data_dst[idx] = T2(data_src[idx]);
}


template <typename T>
__global__ void transpose_kernel( T* data, const int Nx )
{
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;

	const int idx_ij = ty*Nx + tx;
	const int idx_ji = tx*Nx + ty;

	if ( tx < Nx && tx > ty )
	{
		const T tmp = data[idx_ji];
		data[idx_ji] = data[idx_ij];
		data[idx_ij] = tmp;
	}
}


template <typename T>
__global__ void complexMagnitudeSq_kernel( const T* data_re, const T* data_im, T* result, const int Nx, const int Ny )
{
	// Coordinates to grid
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;
	const int tf = blockIdx.z;
	
	const int idx_src = tf*2*Nx*Ny + ty*Nx + tx;
	const int idx_dst = tf*Nx*Ny + ty*Nx + tx;
	
	if ( tx < Nx && ty < Ny )
	{
		T re = data_re[idx_src];
		T im = data_im[idx_src];
		
		result[idx_dst] = re*re + im*im;
	}
}


template <typename T>
__global__ void complexPhase_kernel( const T* data_src_re, const T* data_src_im, T* data_dst, const int Nx, const int Ny )
{
	// Coordinates to grid
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;
	const int tf = blockIdx.z;

	const int idx_src = tf*2*Nx*Ny + ty*Nx + tx;
	const int idx_dst = tf*Nx*Ny + ty*Nx + tx;

	T x, y, phase;
	
	if ( tx < Nx && ty < Ny )
	{
		x = data_src_re[idx_src];
		y = data_src_im[idx_src];
				
		phase = (T)0.0;

		// Interior of quadrants
		if ( x > (T)0.0 && y > (T)0.0 )
			phase = atan(y/x);
		else if ( x < (T)0.0 && y > (T)0.0 )
			phase = M_PI + atan(y/x);
		else if ( x < (T)0.0 && y < (T)0.0 )
			phase = M_PI + atan(y/x);
		else if ( x > (T)0.0 && y < (T)0.0 )
			phase = (T)2.0*M_PI + atan(y/x);
		// Boundaries
		else if ( x == (T)0.0 )
		{
			if ( y > (T)0.0 )
				phase = (T)0.5 * M_PI;
			if ( y < (T)0.0 )
				phase = (T)1.5 * M_PI;
		}
		else if ( y == (T)0.0 )
		{
			if ( x > (T)0.0 )
				phase = (T)0.0;
			if ( x < (T)0.0 )
				phase = M_PI;
		}
		
		data_dst[idx_dst] = phase;
	}
}


template <typename T>
__global__ void complexPhaseGradient_kernel( const T* data_src_re, const T* data_src_im, T* data_dst_x, T* data_dst_y, const T h, const int Nx, const int Ny )
{
	// Coordinates to grid
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;

	T f0, f1, f2, g0, g1, g2, denom, dPhase_dx, dPhase_dy;
	
	// Grid index
	const int idx = ty * Nx + tx;

	const int idx_x0 = ty * Nx + (tx-1);
	const int idx_x2 = ty * Nx + (tx+1);

	const int idx_y0 = (ty-1) * Nx + tx;
	const int idx_y2 = (ty+1) * Nx + tx;
	
	// Set gradient to zero at borders for now. Implement other gradient methods later!
	dPhase_dx = (T)0.0;
	dPhase_dy = (T)0.0;

	if ( tx >= 0 && tx <= Nx-1 && ty >= 0 && ty <= Ny-1 )
	{
		if ( tx > 0 && tx < Nx-1 && ty > 0 && ty < Ny-1 )
		{
			// Read elements at current grid point
			f1 = data_src_im[idx];
			g1 = data_src_re[idx];

			// Some precomputed quantities
			denom = (T)2.0*h*(f1*f1 + g1*g1);

			if ( denom > (T)0.00001 )
			{
				denom = (T)1.0 / denom;
			
				// Gradient in x
				f0 = data_src_im[idx_x0];
				g0 = data_src_re[idx_x0];

				f2 = data_src_im[idx_x2];
				g2 = data_src_re[idx_x2];

				dPhase_dx = ( (f2-f0)*g1 - f1*(g2-g0) ) * denom;

				// Gradient in y
				f0 = data_src_im[idx_y0];
				f2 = data_src_im[idx_y2];

				g0 = data_src_re[idx_y0];
				g2 = data_src_re[idx_y2];

				dPhase_dy = ( (f2-f0)*g1 - f1*(g2-g0) ) * denom;
			}
		}

		data_dst_x[idx] = dPhase_dx;
		data_dst_y[idx] = dPhase_dy;	
	}
}


template <typename T>
__global__ void fieldMagnitude_kernel( const T* data, T* data_dst, const int Nx, const int Ny, const int numFields, const int isComplex )
{
	// Coordinates to grid
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ( tx < Nx && ty < Ny )
	{
		// Array index to grid_sum
		const int idx = ty * Nx + tx;
	
		// Loop over components
		int i, idx_c;
	
		T val, sum = (T)0.0;

		for ( i=0; i<numFields; ++i )
		{
			idx_c = i*Nx*Ny*(1+isComplex) + idx;
			val = data[idx_c];
			sum += val*val;
		}
	
		data_dst[idx] = sqrt(sum);
	}
}


template <typename T>
__global__ void div_kernel( const T* data_x, const T* data_y, T* data_dst, const T h, const int Nx, const int Ny )
{
	// Coordinates to grid
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ( tx>0 && tx<Nx-1 && ty>0 && ty<Ny-1 )
	{
		// Grid index
		const int idx = ty * Nx + tx;

		data_dst[idx] = (data_x[idx+1]-data_x[idx]) + (data_y[idx+Nx]-data_y[idx])/h;
	}
}


template <typename T>
__global__ void curl_kernel( const T* data_x, const T* data_y, T* data_dst, const T h, const int Nx, const int Ny )
{
	// Coordinates to grid
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;

	// Grid index
	const int idx = ty * Nx + tx;
	
	if ( tx>0 && tx<Nx-1 && ty>0 && ty<Ny-1 )
	{
		const T Ay = data_y[ty*Nx + tx+1] - data_y[idx];
		const T Ax = data_x[(ty+1)*Nx + tx] - data_x[idx];

		data_dst[idx] = (Ay - Ax)/h;
	}
	else if ( tx==0 || ty==0 || tx==Nx-1 || ty==Ny-1 )
		data_dst[idx] = T(0.0);
}


template <typename T>
__global__ void del2_kernel( T* del2_f, const T* f, const T inv_h2, const int Nx, const int Ny )
{
	// Coordinates to grid
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;

	const int idx = ty*Nx + tx;

	if ( tx>0 && tx<Nx-1 && ty>0 && ty<Ny-1 )
		del2_f[idx] = (f[idx+1] + f[idx-1] + f[idx+Nx] + f[idx-Nx] - T(4.0)*f[idx]) * inv_h2;
	else if ( tx==0 || ty==0 || tx==Nx-1 || ty==Ny-1 )
		del2_f[idx] = T(0.0);
}


template <typename T>
__global__ void sumFieldComponents_kernel( const T* data, T* data_dst, const int Nx, const int Ny, const int numFields, const int isComplex )
{
	// Coordinates to grid
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ( tx < Nx && ty < Ny )
	{
		// Array index to grid_sum
		const int idx_result = ty * Nx + tx;
	
		// Loop over components
		int i, idx_source;
	
		T sum = (T)0.0;

		for ( i=0; i<numFields; ++i )
		{
			idx_source = i*Nx*Ny*(1+isComplex) + idx_result;
			sum += data[idx_source];
		}
	
		data_dst[idx_result] = sum;
	}
}


template <typename T>
__global__ void insertGrid_kernel(	T* data_this, const T* data_other,
									const int Nx_this, const int Ny_this,
									const int Nx_other, const int Ny_other,
									const int x_start_this, const int y_start_this,
									const int x_start_other, const int y_start_other )
{
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;

	const int tx_this = tx + x_start_this;
	const int ty_this = ty + y_start_this;

	const int tx_other = tx + x_start_other;
	const int ty_other = ty + y_start_other;

	if ( tx_this < Nx_this && ty_this < Ny_this && tx_other < Nx_other && ty_other < Ny_other )
	{
		const int idx_this = ty_this * Nx_this + tx_this;
		const int idx_other = ty_other * Nx_other + tx_other;
	
		data_this[idx_this] = data_other[idx_other];
	}
}


template <typename T>
__global__ void insertRotatedGrid_kernel(
	T* data, const T* data_other,
	const int Nx, const int Ny,	const int Nx_other, const int Ny_other,
	const float phi, const int x_offset, const int y_offset )
{
	// --> offset not implemented yet! <-- //

	// Coordinate transforms from current grid to lookup grid
	const T x_transform = (Nx_other - Nx)/(T)2.0;
	const T y_transform = (Ny_other - Ny)/(T)2.0;
	
	// Shifts lookup grid center to origin
	const T grid_origin_shift_x = (Nx_other-1)/(T)2.0;
	const T grid_origin_shift_y = (Ny_other-1)/(T)2.0;
	
	// Coordinates to current grid
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ( x < Nx && y < Ny )
	{
		// Array index to current grid
		const int idx_grid_out = y*Nx + x;
	
		// Coordinates to lookup grid
		const T X = x + x_transform - grid_origin_shift_x;
		const T Y = y + y_transform - grid_origin_shift_y;
	
		// Rotate centered coordinates, and put grid corner at origin
		const T Xr = (T)X*cos(-phi) - (T)Y*sin(-phi) + grid_origin_shift_x;
		const T Yr = (T)X*sin(-phi) + (T)Y*cos(-phi) + grid_origin_shift_y;
	
		// Bilinear interpolation
		const int X_int = Xr;
		const int Y_int = Yr;
	
		if ( (X_int>=0) && (X_int+1<Nx_other) && (Y_int>=0) && (Y_int+1<Ny_other) )
		{		
			const T X_frac = Xr - (T)X_int;
			const T Y_frac = Yr - (T)Y_int;

			T X1_interp, X2_interp;
	
			X1_interp =	((T)1.0-X_frac) * data_other[Y_int*Nx_other + X_int] +
						X_frac * data_other[Y_int*Nx_other + (X_int+1)];

			X2_interp =	((T)1.0-X_frac) * data_other[(Y_int+1)*Nx_other + X_int] +
						X_frac * data_other[(Y_int+1)*Nx_other + (X_int+1)];
	
			data[idx_grid_out] = ((T)1.0-Y_frac) * X1_interp + Y_frac * X2_interp;
		}
	}
}


template <typename T>
__global__ void insertRotatedGridAdditive_kernel(
	T* data, const T* data_other,
	const int Nx, const int Ny,	const int Nx_other, const int Ny_other,
	const float phi, const int x_offset, const int y_offset )
{
	// --> offset not implemented yet! <-- //

	// Coordinate transforms from current grid to lookup grid
	const T x_transform = (Nx_other - Nx)/(T)2.0;
	const T y_transform = (Ny_other - Ny)/(T)2.0;
	
	// Shifts lookup grid center to origin
	const T grid_origin_shift_x = (Nx_other-1)/(T)2.0;
	const T grid_origin_shift_y = (Ny_other-1)/(T)2.0;
	
	// Coordinates to current grid
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ( x < Nx && y < Ny )
	{
		// Array index to current grid
		const int idx_grid_out = y*Nx + x;
	
		// Coordinates to lookup grid
		const T X = x + x_transform - grid_origin_shift_x;
		const T Y = y + y_transform - grid_origin_shift_y;
	
		// Rotate centered coordinates, and put grid corner at origin
		const T Xr = (T)X*cos(-phi) - (T)Y*sin(-phi) + grid_origin_shift_x;
		const T Yr = (T)X*sin(-phi) + (T)Y*cos(-phi) + grid_origin_shift_y;
	
		// Bilinear interpolation
		const int X_int = Xr;
		const int Y_int = Yr;
	
		if ( (X_int>=0) && (X_int+1<Nx_other) && (Y_int>=0) && (Y_int+1<Ny_other) )
		{		
			const T X_frac = Xr - (T)X_int;
			const T Y_frac = Yr - (T)Y_int;

			T X1_interp, X2_interp;
	
			X1_interp =	((T)1.0-X_frac) * data_other[Y_int*Nx_other + X_int] +
						X_frac * data_other[Y_int*Nx_other + (X_int+1)];

			X2_interp =	((T)1.0-X_frac) * data_other[(Y_int+1)*Nx_other + X_int] +
						X_frac * data_other[(Y_int+1)*Nx_other + (X_int+1)];
	
			data[idx_grid_out] += ((T)1.0-Y_frac) * X1_interp + Y_frac * X2_interp;
		}
	}
}



template <typename T>
__global__ void resample_kernel(
	const T* data, T* data_tmp, const int Nx, const int Ny, const int Nx_new, const int Ny_new )
{
	// Coordinates to new resampling grid
	const int x_new = blockIdx.x * blockDim.x + threadIdx.x;
	const int y_new = blockIdx.y * blockDim.y + threadIdx.y;

	if ( x_new < Nx_new && y_new < Ny_new )
	{
		const int idx_new = y_new * Nx_new + x_new;

		// Scaling factor between grids
		const T sx = (T)(Nx-1) / (T)(Nx_new-1);
		const T sy = (T)(Ny-1) / (T)(Ny_new-1);
	
		// Coordinates to old grid
		const T x = sx * x_new;
		const T y = sy * y_new;
	
		const int x_int = x;
		const int y_int = y;
	
		const T x_frac = x - (T)x_int;
		const T y_frac = y - (T)y_int;

		// Bilinear interpolation
		int x_add = 1, y_add = 1;
		
		if ( x_new == Nx_new-1 )
			x_add = 0;
		if ( y_new == Ny_new-1 )
			y_add = 0;

		T x1_interp, x2_interp;
		
		x1_interp = ((T)1.0-x_frac) * data[y_int*Nx + x_int] +
					x_frac * data[y_int*Nx + (x_int+x_add)];		
		
		x2_interp = ((T)1.0-x_frac) * data[(y_int+y_add)*Nx + x_int] +
					x_frac * data[(y_int+y_add)*Nx + (x_int+x_add)];
	
		data_tmp[idx_new] = ((T)1.0-y_frac) * x1_interp + y_frac * x2_interp;
	}
}

