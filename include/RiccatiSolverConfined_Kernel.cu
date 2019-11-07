#include <cplx.h>

template <typename T, typename GAMMA, int ENTRY, int EXIT, int DIR>
__global__ 
void computeGammaFunctionTemplate_new_kernel(
	const T h, const int N, const int Ne, const int energyIdx_offset,
	const int idx_x_min, const int idx_x_max, const int idx_y_min, const int idx_y_max,
	const int angle_idx, const T system_angle, const T momentum_angle, const T B,
	T *energies_re, T *energies_im,
	T *energy_shift,
	T *gamma_real, T *gamma_imag,
	T *delta_real, T *delta_imag,
	char *domainLabel, T *domainNormals_x, T *domainNormals_y, T *boundaryDistance,
	int *idx_in_gamma, int *idx_out_gamma,
	T *bnd_in_gamma_real, T *bnd_in_gamma_imag,
	T *bnd_out_gamma_real, T *bnd_out_gamma_imag )
{	
	const int x		= blockDim.x * blockIdx.x + threadIdx.x + idx_x_min;	// x-coordinate
	const int e_idx	= blockDim.y * blockIdx.y + threadIdx.y;				// energy coordinate

	cplx<T> g, g0((T)0.0,(T)0.0), g_bnd, energy;
	cplx<T> delta, D2, D_bnd;

	T dh;

	int idx, gamma_idx, idx_sparse;
	char nodeType;

	if ( x <= idx_x_max )
	{
		energy.x = energies_re[e_idx];
		energy.y = energies_im[e_idx];
		
		const T energy_zeroPotential = energy.x;

		int boundary_entry_count = 0;
		int boundary_exit_count = 0;
		
		const int y_start = idx_y_min*(int)(DIR==1) + idx_y_max*(int)(DIR==-1);
		int y;
		
		// OPTIMIZATION: PRECOMPUTE Y_MIN AND Y_MAX FOR EACH ROTATIONAL FRAME
		// Compute gamma along trajectory
		for ( int i=0; i<=(idx_y_max-idx_y_min); ++i )
		{
			y = y_start + DIR*i;
		
			// Index to 2d-grids (Delta)
			idx = y*N + x;

			// Add energy shift from vector potential
			energy.x = energy_zeroPotential + energy_shift[idx];

			// Index to gamma grid (for all energies)
			gamma_idx = e_idx*2*N*N + y*N + x;
			
			// What type of node are we on?
			nodeType = domainLabel[idx];
			
			// Read order parameter
			delta.x = delta_real[idx];
			delta.y = delta_imag[idx];

			g.x = (T)0.0;
			g.y = (T)0.0;

			if ( nodeType == 1 )	// Inside domain:
			{
				GAMMA::computeGamma_local(	delta, g0, energy, h, 1, g );			
			}
			else if ( nodeType == ENTRY )				// Domain boundary entry:
			{
				// Read gamma from boundary storage, and calc first gamma to grid,
				// using fractional step.
				idx_sparse = BoundaryStorage<T>::getSparseIndex(
								angle_idx, e_idx+energyIdx_offset, x, boundary_entry_count,
								Ne, N, idx_in_gamma );

				g0.x = bnd_in_gamma_real[idx_sparse];
				g0.y = bnd_in_gamma_imag[idx_sparse];

				// Distance to exact boundary location
				dh = h * abs( boundaryDistance[idx] );
				
				// Extrapolate delta at boundary from domain values
				D2.x = delta_real[(y+DIR)*N + x];
				D2.y = delta_imag[(y+DIR)*N + x];

				D_bnd = delta - (D2-delta) * (dh/h);

							
				// Compute gamma at current grid point, using exact distance to boundary
				GAMMA::computeGamma_local(	D_bnd, g0, energy, dh, 1, g );

				boundary_entry_count++;
			}			
			else if ( nodeType == EXIT )				// Domain boundary exit:
			{
				// Calc gamma to current grid node, then calc gamma with fractional
				// step at exact boundary location, and save in boundary storage.
				GAMMA::computeGamma_local(	delta, g0, energy, h, 1, g );
												
				// Distance to exact boundary location
				dh = h * abs( boundaryDistance[idx] );

				// Calc at exact boundary location
				GAMMA::computeGamma_local(	delta, g0, energy, dh+h, 1, g_bnd );

				// Store boundary gammas
				idx_sparse = BoundaryStorage<T>::getSparseIndex(
								angle_idx, e_idx+energyIdx_offset, x, boundary_exit_count,
								Ne, N, idx_out_gamma );

				bnd_out_gamma_real[idx_sparse] = g_bnd.x;
				bnd_out_gamma_imag[idx_sparse] = g_bnd.y;

				boundary_exit_count++;
			}
			
			// Store gammas
			gamma_real[gamma_idx] = g.x;
			gamma_imag[gamma_idx] = g.y;
		
			g0 = g;
		}
	}
}

/*
template <typename T, typename GAMMA, int ENTRY, int EXIT, int DIR>
__global__ 
void computeGammaFunctionTemplate_kernel(
	const T h, const int N, const int Ne, const int energyIdx_offset, const int idx_x_offset,
	const int angle_idx, const T system_angle, const T momentum_angle, const T B,
	T *energies_re, T *energies_im,
	T *energy_shift,
	T *gamma_real, T *gamma_imag,
	T *delta_real, T *delta_imag,
	char *domainLabel, T *domainNormals_x, T *domainNormals_y, T *boundaryDistance,
	int *idx_in_gamma, int *idx_out_gamma,
	T *bnd_in_gamma_real, T *bnd_in_gamma_imag,
	T *bnd_out_gamma_real, T *bnd_out_gamma_imag )
{	
	const int x		= blockDim.x * blockIdx.x + threadIdx.x + idx_x_offset;	// x-coordinate
	const int e_idx	= blockDim.y * blockIdx.y + threadIdx.y;				// energy coordinate

	cplx<T> g, g0((T)0.0,(T)0.0), g_bnd, energy;
	cplx<T> delta, D2, D_bnd;

	T dh;

	int idx, gamma_idx, idx_sparse;
	char nodeType;

	if ( x < N )
	{
		energy.x = energies_re[e_idx];
		energy.y = energies_im[e_idx];
		
		const T energy_zeroPotential = energy.x;

		int boundary_entry_count = 0;
		int boundary_exit_count = 0;
		
		const int yStart = (N-1)*(int)(DIR==-1);
		int y;
		
		// OPTIMIZATION: PRECOMPUTE Y_MIN AND Y_MAX FOR EACH ROTATIONAL FRAME
		// Compute gamma along trajectory
		for ( int i=0; i<N; ++i )
		{
			y = yStart + DIR*i;
		
			// Index to 2d-grids (Delta)
			idx = y*N + x;

			// Add energy shift from vector potential
			energy.x = energy_zeroPotential + energy_shift[idx];

			// Index to gamma grid (for all energies)
			gamma_idx = e_idx*2*N*N + y*N + x;
			
			// What type of node are we on?
			nodeType = domainLabel[idx];
			
			// Read order parameter
			delta.x = delta_real[idx];
			delta.y = delta_imag[idx];

			g.x = (T)0.0;
			g.y = (T)0.0;

			if ( nodeType == 1 )	// Inside domain:
			{
				GAMMA::computeGamma_local(	delta, g0, energy, h, i, g );			
			}
			else if ( nodeType == ENTRY )				// Domain boundary entry:
			{
				// Read gamma from boundary storage, and calc first gamma to grid,
				// using fractional step.
				idx_sparse = BoundaryStorage<T>::getSparseIndex(
								angle_idx, e_idx+energyIdx_offset, x, boundary_entry_count,
								Ne, N, idx_in_gamma );

				g0.x = bnd_in_gamma_real[idx_sparse];
				g0.y = bnd_in_gamma_imag[idx_sparse];

				// Distance to exact boundary location
				dh = h * abs( boundaryDistance[idx] );
				
				// Extrapolate delta at boundary from domain values
				D2.x = delta_real[(y+DIR)*N + x];
				D2.y = delta_imag[(y+DIR)*N + x];

				D_bnd = delta - (D2-delta) * (dh/h);

							
				// Compute gamma at current grid point, using exact distance to boundary
				GAMMA::computeGamma_local(	D_bnd, g0, energy, dh, i, g );

				boundary_entry_count++;
			}			
			else if ( nodeType == EXIT )				// Domain boundary exit:
			{
				// Calc gamma to current grid node, then calc gamma with fractional
				// step at exact boundary location, and save in boundary storage.
				GAMMA::computeGamma_local(	delta, g0, energy, h, i, g );
												
				// Distance to exact boundary location
				dh = h * abs( boundaryDistance[idx] );

				// Calc at exact boundary location
				GAMMA::computeGamma_local(	delta, g0, energy, dh+h, i, g_bnd );

				// Store boundary gammas
				idx_sparse = BoundaryStorage<T>::getSparseIndex(
								angle_idx, e_idx+energyIdx_offset, x, boundary_exit_count,
								Ne, N, idx_out_gamma );

				bnd_out_gamma_real[idx_sparse] = g_bnd.x;
				bnd_out_gamma_imag[idx_sparse] = g_bnd.y;

				boundary_exit_count++;
			}
			
			// Store gammas
			gamma_real[gamma_idx] = g.x;
			gamma_imag[gamma_idx] = g.y;
		
			g0 = g;
		}
	}
}
*/


/////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
template <typename T, typename GAMMA, int ENTRY, int EXIT, int DIR>
__global__ 
void computeGammaFunctionInitialBoundaryTemplate_kernel(
	const T h, const int N, const int Ne, const int energyIdx_offset, const int idx_x_offset,
	const int angle_idx, const T system_angle, const T momentum_angle, const T B,
	T *energies_re, T *energies_im,
	T *energy_shift,
	T *gamma_real, T *gamma_imag,
	T *delta_real, T *delta_imag,
	char *domainLabel, T *domainNormals_x, T *domainNormals_y, T *boundaryDistance,
	int *idx_in_gamma, int *idx_out_gamma,
	T *bnd_in_gamma_real, T *bnd_in_gamma_imag,
	T *bnd_out_gamma_real, T *bnd_out_gamma_imag )
{
	const int x		= blockDim.x * blockIdx.x + threadIdx.x + idx_x_offset;	// x-coordinate
	const int e_idx	= blockDim.y * blockIdx.y + threadIdx.y;				// energy coordinate

	cplx<T> delta, g_bnd, energy;

	int idx, idx_sparse;
	char nodeType;

	if ( x < N )
	{
		energy.x = energies_re[e_idx];
		energy.y = energies_im[e_idx];

		const T energy_zeroPotential = energy.x;

//		int boundary_entry_count = 0;
		int boundary_exit_count = 0;
		
		const int yStart = (N-1)*(int)(DIR==-1);
		int y;
		
		// Compute gamma along trajectory
		for ( int i=0; i<N; ++i )
		{
			y = yStart + DIR*i;
			
			// Index to 2d-grids (Delta)
			idx = y*N + x;

			// Add energy shift from vector potential
			energy.x = energy_zeroPotential + energy_shift[idx];
	
			// What type of node are we on?
			nodeType = domainLabel[idx];

			if ( nodeType == EXIT )
			{
				// Read order parameter
				delta.x = delta_real[idx];
				delta.y = delta_imag[idx];

				// Compute gammas and store							
				idx_sparse = BoundaryStorage<T>::getSparseIndex(
								angle_idx, e_idx+energyIdx_offset, x, boundary_exit_count,
								Ne, N, idx_out_gamma );

					
				// Compute gamma at current grid point
				GAMMA::computeGamma_local(	delta,
											cplx<T>(0.0,0.0),
											energy,
											(T)0.0, 0,
											g_bnd );

				bnd_out_gamma_real[idx_sparse] = g_bnd.x;
				bnd_out_gamma_imag[idx_sparse] = g_bnd.y;

				boundary_exit_count++;
			}		
		}
	}
}

