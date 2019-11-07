template <typename T, typename GAMMA, template <typename> class POTENTIAL,
			int ENTRY, int EXIT, int DIR>
void computeGammaFunctionTemplate_CPU(
	const T h, const int N, const int Ne, const int energyIdx_offset, const int idx_x_offset,
	const int angle_idx, const T system_angle, const T momentum_angle, const T B,
	T *energies_re, T *energies_im,
	T *gamma_real, T *gamma_imag,
	T *delta_real, T *delta_imag,
	char *domainLabel, T *domainNormals_x, T *domainNormals_y, T *boundaryDistance,
	int *idx_in_gamma, int *idx_out_gamma,
	T *bnd_in_gamma_real, T *bnd_in_gamma_imag,
	T *bnd_out_gamma_real, T *bnd_out_gamma_imag )
{	
//	const int x		= blockDim.x * blockIdx.x + threadIdx.x + idx_x_offset;	// x-coordinate
//	const int e_idx	= blockDim.y * blockIdx.y + threadIdx.y;				// energy coordinate

	T dh, g_real, g_imag, g_bnd_real, g_bnd_imag, g0_real=0.0, g0_imag=0.0;
	T D1_re, D1_im, D2_re, D2_im, Dbnd_re, Dbnd_im;

	T energy_re, potential;

	int idx, gamma_idx, idx_sparse;
	char nodeType;

	int x, y, e_idx, y_start;

	T energy_zeroPotential_re, energy_im;
	int boundary_entry_count, boundary_exit_count;

//	if ( x < N && e_idx < Ne )
	for ( x=x_min; x<N; ++x )
	{
		for ( e_idx=e_min; e_idx<e_max; ++e_idx )
		{	
			energy_zeroPotential_re = energies_re[e_idx];
			energy_im = energies_im[e_idx];
			
			boundary_entry_count = 0;
			boundary_exit_count = 0;
			
			yStart = (N-1)*(int)(DIR==-1);
//			int y;
			
			// Compute gamma along trajectory
			for ( int i=0; i<N; ++i )
			{
				y = yStart + DIR*i;
			
				// Set gammas to zero
				g_real = (T)0.0;			
				g_imag = (T)0.0;	

				// (THIS COULD BE PRE-COMPUTED IN A SEPARATE KERNEL)
				// Add energy contribution from vector potential
				potential = POTENTIAL<T>::computeVectorPotential_local(
					(T)x, (T)y, energy_zeroPotential_re, B, system_angle, momentum_angle, N, h );

				energy_re = energy_zeroPotential_re + potential;

				// Index to 2d-grids (Delta)
				idx = y*N + x;
				
				// Index to gamma grid (for all energies)
				gamma_idx = e_idx*2*N*N + y*N + x;
				
				// What type of node are we on?
				nodeType = domainLabel[idx];
								
				if ( nodeType == 1 )	// Inside domain:
				{
					GAMMA::computeGamma_local(	delta_real[idx], delta_imag[idx],
												g0_real, g0_imag,
												energy_re, energy_im,
												h, i,
												g_real, g_imag );			
				}
				else if ( nodeType == ENTRY )				// Domain boundary entry:
				{
					// Read gamma from boundary storage, and calc first gamma to grid,
					// using fractional step.
									
					idx_sparse = BoundaryStorage<T>::getSparseIndex(
									angle_idx, e_idx+energyIdx_offset, x, boundary_entry_count,
									Ne, N, idx_in_gamma );

					g0_real = bnd_in_gamma_real[idx_sparse];
					g0_imag = bnd_in_gamma_imag[idx_sparse];

					// Distance to exact boundary location
					dh = h * abs( boundaryDistance[idx] );
					
					// Extrapolate delta at boundary from domain values
					D1_re = delta_real[idx];
					D1_im = delta_imag[idx];

					D2_re = delta_real[(y+DIR)*N + x];
					D2_im = delta_imag[(y+DIR)*N + x];

					Dbnd_re = D1_re - ((D2_re-D1_re)/h) * dh;
					Dbnd_im = D1_im - ((D2_im-D1_im)/h) * dh;

								
					// Compute gamma at current grid point, using exact distance to boundary
					GAMMA::computeGamma_local(	Dbnd_re, Dbnd_im,
												g0_real, g0_imag,
												energy_re, energy_im,
												dh, i,
												g_real, g_imag );

					boundary_entry_count++;
				}			
				else if ( nodeType == EXIT )				// Domain boundary exit:
				{
					// Calc gamma to current grid node, then calc gamma with fractional
					// step at exact boundary location, and save in boundary storage.
					GAMMA::computeGamma_local(	delta_real[idx], delta_imag[idx],
												g0_real, g0_imag,
												energy_re, energy_im,
												h, i,
												g_real, g_imag );
													
					// Distance to exact boundary location
					dh = h * abs( boundaryDistance[idx] );

					// Calc at exact boundary location
					GAMMA::computeGamma_local(	delta_real[idx], delta_imag[idx],
												g0_real, g0_imag,
												energy_re, energy_im,
												dh+h, i,
												g_bnd_real, g_bnd_imag );

					// Store boundary gammas
					idx_sparse = BoundaryStorage<T>::getSparseIndex(
									angle_idx, e_idx+energyIdx_offset, x, boundary_exit_count,
									Ne, N, idx_out_gamma );

					bnd_out_gamma_real[idx_sparse] = g_bnd_real;
					bnd_out_gamma_imag[idx_sparse] = g_bnd_imag;

					boundary_exit_count++;
				}
				
				// Store gammas
				gamma_real[gamma_idx] = g_real;
				gamma_imag[gamma_idx] = g_imag;
			
				g0_real = g_real;
				g0_imag = g_imag;
			}
		}
	}
}



/////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
template <typename T, typename GAMMA, template <typename> class POTENTIAL,
			int ENTRY, int EXIT, int DIR>
__global__ 
void computeGammaFunctionInitialBoundaryTemplate_CPU(
	const T h, const int N, const int Ne, const int energyIdx_offset, const int idx_x_offset,
	const int angle_idx, const T system_angle, const T momentum_angle, const T B,
	T *energies_re, T *energies_im,
	T *gamma_real, T *gamma_imag,
	T *delta_real, T *delta_imag,
	char *domainLabel, T *domainNormals_x, T *domainNormals_y, T *boundaryDistance,
	int *idx_in_gamma, int *idx_out_gamma,
	T *bnd_in_gamma_real, T *bnd_in_gamma_imag,
	T *bnd_out_gamma_real, T *bnd_out_gamma_imag )
{
	const int x		= blockDim.x * blockIdx.x + threadIdx.x + idx_x_offset;	// x-coordinate
	const int e_idx	= blockDim.y * blockIdx.y + threadIdx.y;				// energy coordinate

	T g_bnd_real, g_bnd_imag, energy_re;

	int idx, idx_sparse;
	char nodeType;

	if ( x < N && e_idx < Ne )
	{
		T energy_zeroPotential_re = energies_re[e_idx];
		T energy_im = energies_im[e_idx];
		
//		int boundary_entry_count = 0;
		int boundary_exit_count = 0;
		
		const int yStart = (N-1)*(int)(DIR==-1);
		int y;
		
		// Compute gamma along trajectory
		for ( int i=0; i<N; ++i )
		{
			y = yStart + DIR*i;
			
			// (THIS COULD BE PRE-COMPUTED IN A SEPARATE KERNEL)
			// Add energy contribution from vector potential
			energy_re = POTENTIAL<T>::computeVectorPotential_local(
				(T)x, (T)y, energy_zeroPotential_re, B, system_angle, momentum_angle, N, h );

			// Index to 2d-grids (Delta)
			idx = y*N + x;
				
			// What type of node are we on?
			nodeType = domainLabel[idx];
/*
			if ( nodeType == ENTRY )
			{
				// Compute gammas and store							
				idx_sparse = BoundaryStorage<T>::getSparseIndex(
								angle_idx, e_idx, x, boundary_entry_count,
								Ne, N, idx_in_gamma );

					
				// Compute gamma at current grid point
				GAMMA<T>::computeGamma_local(	delta_real[idx], delta_imag[idx],
												(T)0.0, (T)0.0,
												energy_re, energy_im,
												(T)0.0, 0,
												g_bnd_real, g_bnd_imag );

				bnd_in_gamma_real[idx_sparse] = g_bnd_real;
				bnd_in_gamma_imag[idx_sparse] = g_bnd_imag;

				boundary_entry_count++;
			}
*/
			if ( nodeType == EXIT )
			{
				// Compute gammas and store							
				idx_sparse = BoundaryStorage<T>::getSparseIndex(
								angle_idx, e_idx+energyIdx_offset, x, boundary_exit_count,
								Ne, N, idx_out_gamma );

					
				// Compute gamma at current grid point
				GAMMA::computeGamma_local(	delta_real[idx], delta_imag[idx],
												(T)0.0, (T)0.0,
												energy_re, energy_im,
												(T)0.0, 0,
												g_bnd_real, g_bnd_imag );

				bnd_out_gamma_real[idx_sparse] = g_bnd_real;
				bnd_out_gamma_imag[idx_sparse] = g_bnd_imag;

				boundary_exit_count++;
			}		
		}
	}
}

