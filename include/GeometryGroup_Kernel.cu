#include <cuda.h>


template <typename T>
__device__ void normalize( T & vec_x, T & vec_y )
{
	const T norm = (T)1.0/sqrt(vec_x*vec_x + vec_y*vec_y);
	vec_x *= norm;
	vec_y *= norm;
}


// Adds *1 to existing *0
template <typename T>
__global__ void geomOp_kernel(
	char *dom0, T *norm0_x, T *norm0_y, T *dist0,
	char *dom1, T *norm1_x, T *norm1_y, T *dist1,
	char * geomOpMatrix,
	const int Nx, const int Ny )
{
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if ( tx < Nx && ty < Ny )
	{
		const int idx = ty*Nx + tx;
	
		char dom0_val = dom0[idx];
		char dom1_val = dom1[idx];

		const int geomOpMatrixIdx = dom0_val*4 + dom1_val;

		const T dist0_val = dist0[idx];
		const T dist1_val = dist1[idx];
	
		dom0_val = geomOpMatrix[ geomOpMatrixIdx ];
		
		T normal_x, normal_y;

		if ( dom0_val==2 || dom0_val==3 )
		{
			normal_x = norm0_x[idx] + norm1_x[idx];
			normal_y = norm0_y[idx] + norm1_y[idx];
			
			normalize( normal_x, normal_y );
		}
		else
		{
			normal_x = (T)0.0;
			normal_y = (T)0.0;
		}

		norm0_x[idx] = normal_x;
		norm0_y[idx] = normal_y;

		dom0[idx] = dom0_val;
		dist0[idx] = max( dist0_val, dist1_val );
	}
}


// Adds *1 to existing *0
template <typename T>
__global__ void geomOpAdd_kernel(
	char* dom0, char* dom1,
	const int Nx )
{
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if ( tx < Nx && ty < Nx )
	{
		const int idx = ty*Nx + tx;
	
		if ( dom1[idx] )
			dom0[idx] = 1;
	}
}

// Adds *1 to existing *0
template <typename T>
__global__ void geomOpRemove_kernel(
	char *dom0, T *norm0_x, T *norm0_y, T *dist0,
	char *dom1, T *norm1_x, T *norm1_y, T *dist1,
	char *domTmp,
	char * geomOpMatrix,
	const int Nx, const int Ny )
{
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if ( tx<Nx && ty>0 && ty<Ny-1 )
	{
		const int idxm = (ty-1)*Nx + tx;
		const int idx = ty*Nx + tx;
		const int idxp = (ty+1)*Nx + tx;

		char dom0_val = dom0[idx];

		const char dom1m_val = dom1[idxm];
		const char dom1_val = dom1[idx];
		const char dom1p_val = dom1[idxp];

		T normal_x, normal_y;
		
		if ( dom1_val > 0 )
		{
			dom0_val = 0;
		}
		else if ( dom1p_val == 2 )
		{
			if ( dom0_val == 1 )
			{
				dom0_val = 3;
				normal_x = norm0_x[idx] - norm1_x[idxp];
				normal_y = norm0_y[idx] - norm1_y[idxp];
				normalize( normal_x, normal_y );
				norm0_x[idx] = normal_x;
				norm0_y[idx] = normal_y;
				dist0[idx] = (T)1.0 - dist1[idxp];
			}
		}
		else if ( dom1m_val == 3 )
		{
			if ( dom0_val == 1 )
			{
				dom0_val = 2;
				normal_x = norm0_x[idx] - norm1_x[idxm];
				normal_y = norm0_y[idx] - norm1_y[idxm];
				normalize( normal_x, normal_y );
				norm0_x[idx] = normal_x;
				norm0_y[idx] = normal_y;
				dist0[idx] = (T)1.0 - dist1[idxm];				
			}
		}
	
		domTmp[idx] = dom0_val;
	}
}


template <typename T>
__global__ void geomOpRemove_kernel(
	char* dom0, char* dom1,
	const int Nx )
{
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if ( tx<Nx && ty>0 && ty<Nx-1 )
	{
		const int idx = ty*Nx + tx;

		const int dom0_val = int(dom0[idx]);
		const int dom1_val = int(dom1[idx]);

		dom0[idx] = char( dom0_val*(1-dom1_val) );
	}
}



// Slow operation, so we have two explicit kernels, one for complex and one for real valued.
template <typename T>
__global__ void extendValuesAtBoundary_kernel( //complex
	char *dom_ptr,
	T *grid_real, T *grid_imag,
	int Nx_geom, int Ny_geom,
	int Nx_target, int Ny_target,
	int Nx_geom_offset, int Ny_geom_offset,
	const int numExtend )
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int tf = blockIdx.z;
	
	if ( tx < Nx_target && ty < Ny_target )
	{
		int idx_geom = (ty+Ny_geom_offset) * Nx_geom + (tx+Nx_geom_offset);
	
		int i, j, idx_neighbor, idx_geom_neighbor;
	
		T interp_real = (T)0.0;
		T interp_imag = (T)0.0;
	
		int numNeighbors = 0;
	
		if ( dom_ptr[idx_geom] == 0 )
		{
			for ( j=-numExtend; j<=numExtend; ++j )
			{
				for ( i=-numExtend; i<=numExtend; ++i )
				{
					if ( (tx+i) > 0 && (tx+i) < Nx_target && (ty+j) > 0 && (ty+j) < Ny_target )
					{
						idx_geom_neighbor = (ty+j+Ny_geom_offset)*Nx_geom + (tx+i+Nx_geom_offset);

						if ( dom_ptr[idx_geom_neighbor] > 0 )
						{
							idx_neighbor = tf*2*Nx_target*Ny_target + (ty+j)*Nx_target + (tx+i);

							interp_real += grid_real[idx_neighbor];
							interp_imag += grid_imag[idx_neighbor];

							numNeighbors++;
						}
					}
				}
			}	
		}
	
		if ( numNeighbors )
		{
			const int idx = tf*2*Nx_target*Ny_target + ty*Nx_target + tx;
			grid_real[idx] = interp_real / (T)numNeighbors;
			grid_imag[idx] = interp_imag / (T)numNeighbors;
		}
	}
}


template <typename T>
__global__ void extendValuesAtBoundary_kernel( //real
	char *dom_ptr,
	T *grid_real,
	int Nx_geom, int Ny_geom,
	int Nx_target, int Ny_target,
	int Nx_geom_offset, int Ny_geom_offset,
	const int numExtend )
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int tf = blockIdx.z;

	if ( tx < Nx_target && ty < Ny_target )
	{
		int idx_geom = (ty+Ny_geom_offset) * Nx_geom + (tx+Nx_geom_offset);
	
		int i, j, idx_neighbor, idx_geom_neighbor;
	
		T interp_real = (T)0.0;
	
		int numNeighbors = 0;
	
		if ( dom_ptr[idx_geom] == 0 )
		{
			for ( j=-numExtend; j<=numExtend; ++j )
			{
				for ( i=-numExtend; i<=numExtend; ++i )
				{
					if ( (tx+i) > 0 && (tx+i) < Nx_target && (ty+j) > 0 && (ty+j) < Ny_target )
					{
						idx_geom_neighbor = (ty+j+Ny_geom_offset)*Nx_geom + (tx+i+Nx_geom_offset);

						if ( dom_ptr[idx_geom_neighbor] > 0 )
						{
							idx_neighbor = tf*Nx_target*Ny_target + (ty+j)*Nx_target + (tx+i);

							interp_real += grid_real[idx_neighbor];
						
							numNeighbors++;
						}
					}
				}
			}	
		}
	
		if ( numNeighbors )
		{
			const int idx = tf*Nx_target*Ny_target + ty*Nx_target + tx;
			grid_real[idx] = interp_real / (T)numNeighbors;
		}
	}
}


template <typename T>
__global__ void zeroValuesOutsideDomain_kernel(
	char *dom_ptr,
	T *target_ptr,
	int Nx_geom, int Ny_geom,
	int Nx_target, int Ny_target,
	int Nx_geom_offset, int Ny_geom_offset, int isComplex )
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int tf = blockIdx.z;
	
	if ( tx < Nx_target && ty < Ny_target )
	{
		int idx = tf*(Nx_target*Ny_target*(1+isComplex)) + ty * Nx_target + tx;

		int idx_geom = (ty+Ny_geom_offset) * Nx_geom + (tx+Nx_geom_offset);
	
		if ( dom_ptr[idx_geom] == 0 )
			target_ptr[idx] = (T)0.0;
	}
}

