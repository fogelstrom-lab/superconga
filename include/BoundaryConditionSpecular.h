#ifndef _BOUNDARYCONDITIONSPECULAR_H
#define _BOUNDARYCONDITIONSPECULAR_H

#include <BoundaryCondition.h>
#include <cuda.h>

template <typename T>
class BoundaryConditionSpecular : public BoundaryCondition<T>
{
	typedef typename Grid<T>::GPU GridGPU;
	typedef typename Grid<T>::CPU GridCPU;

public:
					BoundaryConditionSpecular() : BoundaryCondition<T>() {}
	virtual			~BoundaryConditionSpecular() {}
					
	virtual void	computeBoundaryCondition( Context<T>* ctx );

private:
	
};

namespace
{
template <typename T>
__device__ T calcReflection( T dir_x, T dir_y, T n_x, T n_y )
{
	T dir_dot_n = dir_x*n_x + dir_y*n_y;
	T alpha = acos( dir_x*dir_x + dir_y*dir_y - (T)2.0*(dir_dot_n * dir_dot_n) );
	
	if ( n_x < (T)0.0 )
		alpha = (T)2.0*M_PI - alpha;
	
	return alpha;
}

template <typename T>
__device__ T calcReflectionNegative( T dir_x, T dir_y, T n_x, T n_y )
{
	T dir_dot_n = dir_x*n_x + dir_y*n_y;
	T alpha = acos( dir_x*dir_x + dir_y*dir_y - (T)2.0*(dir_dot_n * dir_dot_n) );
	
	if ( n_x > (T)0.0 )
		alpha = (T)2.0*M_PI - alpha;
	
	return alpha;
}

template <typename T>
__global__ void boundaryConditionSpecular_gamma_kernel(
	const int Nx, const int Ne, const int Na, const int angle,
	const int idx_x_min, const int idx_x_max, const int idx_y_min, const int idx_y_max,
	const int N_bnd_in_gamma, const int N_bnd_out_gamma,
	char * domainLabel,
	T * boundaryNormals_x, T * boundaryNormals_y,
	T * boundaryDistance,
	int * sparse_idx_in_gamma, int * sparse_idx_out_gamma,
	int * numBoundaryIntersect_gamma,
	int * coordBoundaryIntersect_gamma,
	T * bnd_in_gamma_real, T * bnd_in_gamma_imag,
	T * bnd_out_gamma_real, T * bnd_out_gamma_imag )
{
	const int x = threadIdx.x + blockDim.x * blockIdx.x + idx_x_min;		// x-coordinate
	const int e = threadIdx.y + blockDim.y * blockIdx.y;							// energy coordinate
	
	T h, phi, angle_norm, pt_idx_frac, angle_idx_frac;
	T boundary_pt_x, boundary_pt_y, boundary_pt_rot_x, boundary_pt_rot_y;
	T val_bnd_real, val_bnd_imag, val_bnd_interp_real, val_bnd_interp_imag;

	T total_weight, y_coord_diff;

	T rot_pt_angle[2], angle_weight[2], pt_weight[2];

	int y, rot, pt;
	int idx, idx_sparse_in, idx_sparse_out, angle_idx[2], idx_interp[2];
	int num_isect, isect, isect_current, sparse_idx;
	
	char nodeType;
	
	// Use variable 'offset' when transforming grid index to coordinates with origin in center
	// of grid matrix.
	const T coord_offset = (Nx-1)*(T)0.5;

	if ( x <= idx_x_max )
	{
		int boundary_entry_count = 0;

		// Reflect gamma
		//
		for ( y = idx_y_min; y <= idx_y_max; ++y )
		{
			// Index to 2d-grids
			idx = y*Nx + x;
	
			nodeType = domainLabel[idx];
	
			// At boundary, going in. Now we need to find where trajectory comes from
			if ( nodeType == 2 )
			{
				// Distance to exact boundary location
				h = abs( boundaryDistance[idx] );
								
                // Calc angle between outgoing and incoming trajectory                  
				phi = calcReflection( (T)0.0, (T)1.0, boundaryNormals_x[idx], boundaryNormals_y[idx] );  
                
                // Exact angle (in reference frame, normalized 0 to 1) of trajectory towards boundary
				angle_norm = (T)angle/(T)Na + phi/((T)2.0*PI);

                // Wrap angle to stay inside range
				if ( angle_norm < (T)0.0 )
					angle_norm = angle_norm + (T)1.0;

				if ( angle_norm >= (T)1.0 )
					angle_norm = angle_norm - (T)1.0;
 
 				// Find corresponding indices of trajectories towards boundary
 				// (must interpolate from two angles)
				angle_idx[0] = angle_norm * Na;
				angle_idx[1] = angle_idx[0] + 1;

				angle_idx_frac = angle_norm * Na - (T)angle_idx[0];
				              
				// Wrap index to stay inside range                  
				if ( angle_idx[0] < 0 )
					angle_idx[0] = angle_idx[0] + Na;

				if ( angle_idx[0] > (Na-1) )
					angle_idx[0] = angle_idx[0] - Na;

				if ( angle_idx[1] < 0 )
					angle_idx[1] = angle_idx[1] + Na;

				if ( angle_idx[1] > (Na-1) )
					angle_idx[1] = angle_idx[1] - Na;


                // Weights for interpolation of discrete angles

                // Interpolate gammas from outgoing trajectory from stored
                // boundary values. 
                
                // Rotate point on boundary
				boundary_pt_x = x - coord_offset;
				boundary_pt_y = y - coord_offset - h;
                
				angle_weight[0] = (T)1.0 - angle_idx_frac;
				angle_weight[1] = angle_idx_frac;

  				rot_pt_angle[0] = (T)(angle - angle_idx[0])/(T)Na * (T)2.0*M_PI;
 				rot_pt_angle[1] = (T)(angle - angle_idx[1])/(T)Na * (T)2.0*M_PI;
						
				// Rotated coordinate above will generally fall between two
				// discrete grid points, so an interpolation from these needs
				// to be made. 
				
				// Only works for one entry/exit per trajectory
				//
				
				// NEW ALGORITHM:
				// Make energy innermost loop. Need to make Ne-sized arrays for
				// intermediate interpolation values.
				// Sparse storage majority: Angle, energy, trajectory

	            val_bnd_real = (T)0.0;
	            val_bnd_imag = (T)0.0;

	            val_bnd_interp_real = (T)0.0;
	            val_bnd_interp_imag = (T)0.0;
            
            	total_weight = (T)0.0;
            
         	
				for ( rot=0; rot<2; ++rot )
	            {						            	
	            	// Rotate boundary point to originating frame
					boundary_pt_rot_x = boundary_pt_x * cos(rot_pt_angle[rot]) - boundary_pt_y * sin(rot_pt_angle[rot]);
					boundary_pt_rot_y = boundary_pt_x * sin(rot_pt_angle[rot]) + boundary_pt_y * cos(rot_pt_angle[rot]);
					
					// Transform coordinate to index
					boundary_pt_rot_x += coord_offset;
					boundary_pt_rot_y += coord_offset;
			        
			        // Interpolation (points) indices
					idx_interp[0] = boundary_pt_rot_x;
					idx_interp[1] = idx_interp[0] + 1;

	                // Interpolation (points) weights
                    pt_idx_frac = boundary_pt_rot_x - idx_interp[0];

					pt_weight[0] = (T)1.0 - pt_idx_frac;
					pt_weight[1] = pt_idx_frac;


					for ( pt=0; pt<2; ++pt )
					{
						// Find number of boundary intersections for current trajectory
						num_isect = numBoundaryIntersect_gamma[ angle_idx[rot]*Nx + idx_interp[pt] ];

						isect_current = 0;

						// Find which intersection corresponds to current point
						for ( isect=0; isect<num_isect; ++isect )
						{
							sparse_idx = sparse_idx_out_gamma[ angle_idx[rot]*Nx + idx_interp[pt] ] + isect;
							
							y_coord_diff = abs( (T)coordBoundaryIntersect_gamma[ sparse_idx ] - boundary_pt_rot_y );

							if ( y_coord_diff < (T)2.0 )
								isect_current = isect;
						}

						// Get sparse index for intersection
						idx_sparse_out = BoundaryStorage<T>::getSparseIndex(
							angle_idx[rot], e, idx_interp[pt], isect_current,
							Ne, Nx, sparse_idx_out_gamma );


						if ( idx_sparse_out > -1 )
						{							
							val_bnd_real = bnd_out_gamma_real[idx_sparse_out]; // Maybe pre-load these to shared mem
							val_bnd_imag = bnd_out_gamma_imag[idx_sparse_out];

							total_weight += pt_weight[pt] * angle_weight[rot];
						}
						else 
						{
							val_bnd_real = (T)0.0;
							val_bnd_imag = (T)0.0;						
						}
					
			            // Add interpolation contribution to final value
						val_bnd_interp_real += val_bnd_real * pt_weight[pt] * angle_weight[rot];
						val_bnd_interp_imag += val_bnd_imag * pt_weight[pt] * angle_weight[rot];				
					}
				}

				idx_sparse_in = BoundaryStorage<T>::getSparseIndex(
							angle, e, x, boundary_entry_count,
							Ne, Nx, sparse_idx_in_gamma );
				
				if ( total_weight > (T)0.0 )
				{
					bnd_in_gamma_real[idx_sparse_in] = val_bnd_interp_real / total_weight;
					bnd_in_gamma_imag[idx_sparse_in] = val_bnd_interp_imag / total_weight;
				}
				else
				{
					bnd_in_gamma_real[idx_sparse_in] = (T)0.0;
					bnd_in_gamma_imag[idx_sparse_in] = (T)0.0;				
				}		

				boundary_entry_count++;
 			}
		}
	}
}

template <typename T>
__global__ void boundaryConditionSpecular_gammabar_kernel(
	const int Nx, const int Ne, const int Na, const int angle,
	const int idx_x_min, const int idx_x_max, const int idx_y_min, const int idx_y_max,
	int N_bnd_in_gammabar, int N_bnd_out_gammabar,
	char * domainLabel,
	T * boundaryNormals_x, T * boundaryNormals_y,
	T * boundaryDistance,
	int * sparse_idx_in_gammabar, int * sparse_idx_out_gammabar,
	int * numBoundaryIntersect_gammabar,
	int * coordBoundaryIntersect_gammabar,
	T * bnd_in_gammabar_real, T * bnd_in_gammabar_imag,
	T * bnd_out_gammabar_real, T * bnd_out_gammabar_imag )
{

	const int x = threadIdx.x + blockDim.x * blockIdx.x + idx_x_min;		// x-coordinate
	const int e = threadIdx.y + blockDim.y * blockIdx.y;					// energy coordinate
	
	T h, phi, angle_norm, pt_idx_frac, angle_idx_frac;
	T boundary_pt_x, boundary_pt_y, boundary_pt_rot_x, boundary_pt_rot_y;
	T val_bnd_real, val_bnd_imag, val_bnd_interp_real, val_bnd_interp_imag;

	T total_weight, y_coord_diff;

	T rot_pt_angle[2], angle_weight[2], pt_weight[2];

	int y, rot, pt;
	int idx, idx_sparse_in, idx_sparse_out, angle_idx[2], idx_interp[2];
	int num_isect, isect, isect_current, sparse_idx;
	
	char nodeType;
	
	// Use variable 'offset' when transforming grid index to coordinates with origin in center
	// of grid matrix.
	const T coord_offset = (Nx-1)*(T)0.5;

	if ( x <= idx_x_max )
	{
		int boundary_entry_count = 0;
				
		// Reflect gammabar
		//
		for ( y = idx_y_max; y >= idx_y_min; --y )
		{
			// Index to 2d-grids
			idx = y*Nx + x;
	
			nodeType = domainLabel[idx];
	
			// At boundary, going in. Now we need to find where trajectory comes from
			if ( nodeType == 3 )
			{
				// Distance to exact boundary location
				h = abs( boundaryDistance[idx] );
								
                // Calc angle between outgoing and incoming trajectory                  
				phi = calcReflection( (T)0.0, -(T)1.0, boundaryNormals_x[idx], boundaryNormals_y[idx] );  
                
                // Exact angle (in reference frame, normalized 0 to 1) of trajectory towards boundary
				angle_norm = (T)angle/(T)Na - phi/((T)2.0*M_PI);

                // Wrap angle to stay inside range
				if ( angle_norm < (T)0.0 )
					angle_norm = angle_norm + (T)1.0;

				if ( angle_norm >= (T)1.0 )
					angle_norm = angle_norm - (T)1.0;
 
 				// Find corresponding indices of trajectories towards boundary
 				// (must interpolate from two angles)
				angle_idx[0] = angle_norm * Na;
				angle_idx[1] = angle_idx[0] + 1;

                angle_idx_frac = angle_norm * Na - (T)angle_idx[0];
				               
				// Wrap index to stay inside range                  
				if ( angle_idx[0] < 0 )
					angle_idx[0] = angle_idx[0] + Na;

				if ( angle_idx[0] > (Na-1) )
					angle_idx[0] = angle_idx[0] - Na;
                
				if ( angle_idx[1] < 0 )
					angle_idx[1] = angle_idx[1] + Na;

				if ( angle_idx[1] > (Na-1) )
					angle_idx[1] = angle_idx[1] - Na;
                
                  
                // Weights for interpolation of discrete angles
				angle_weight[0] = (T)1.0 - angle_idx_frac;
				angle_weight[1] = angle_idx_frac;


                // Interpolate gammas from outgoing trajectory from stored
                // boundary values. 
                
                // Rotate point on boundary
				boundary_pt_x = x - coord_offset;
				boundary_pt_y = y - coord_offset + h;
                
  				rot_pt_angle[0] = (T)(angle - angle_idx[0])/(T)Na * (T)2.0*M_PI;
 				rot_pt_angle[1] = (T)(angle - angle_idx[1])/(T)Na * (T)2.0*M_PI;
                
			
				// Rotated coordinate above will generally fall between two
				// discrete grid points, so an interpolation from these needs
				// to be made. 				
	            val_bnd_real = (T)0.0;
	            val_bnd_imag = (T)0.0;

	            val_bnd_interp_real = (T)0.0;
	            val_bnd_interp_imag = (T)0.0;
            
            	total_weight = (T)0.0;
            
				for ( rot=0; rot<2; ++rot )
	            {					
	            	
	            	// Rotate boundary point to discrete frame
					boundary_pt_rot_x = boundary_pt_x * cos(rot_pt_angle[rot]) - boundary_pt_y * sin(rot_pt_angle[rot]);
					boundary_pt_rot_y = boundary_pt_x * sin(rot_pt_angle[rot]) + boundary_pt_y * cos(rot_pt_angle[rot]);

					// Transform coordinate to index
					boundary_pt_rot_x += coord_offset;
					boundary_pt_rot_y += coord_offset;
			        
			        // Interpolation (points) indices
					idx_interp[0] = boundary_pt_rot_x;
					idx_interp[1] = idx_interp[0] + 1;

	                // Interpolation (points) weights
                    pt_idx_frac = boundary_pt_rot_x - idx_interp[0];

					pt_weight[0] = (T)1.0 - pt_idx_frac;
					pt_weight[1] = pt_idx_frac;


					for ( pt=0; pt<2; ++pt )
					{
						// Find number of boundary intersections for current trajectory
						num_isect = numBoundaryIntersect_gammabar[ angle_idx[rot]*Nx + idx_interp[pt] ];

						isect_current = 0;

						// Find which intersection corresponds to current point
						for ( isect=0; isect<num_isect; ++isect )
						{
							sparse_idx = sparse_idx_out_gammabar[ angle_idx[rot]*Nx + idx_interp[pt] ] + isect;
							
							y_coord_diff = abs( (T)coordBoundaryIntersect_gammabar[ sparse_idx ] - boundary_pt_rot_y );

							if ( y_coord_diff < (T)2.0 )
								isect_current = isect;
						}

						// Get sparse index for intersection
						idx_sparse_out = BoundaryStorage<T>::getSparseIndex(
							angle_idx[rot], e, idx_interp[pt], isect_current,
							Ne, Nx, sparse_idx_out_gammabar );

						if ( idx_sparse_out > -1 )
						{							
							val_bnd_real = bnd_out_gammabar_real[idx_sparse_out]; // Maybe pre-load these to shared mem
							val_bnd_imag = bnd_out_gammabar_imag[idx_sparse_out];
							
							total_weight += pt_weight[pt] * angle_weight[rot];					
						}
						else 
						{
							val_bnd_real = (T)0.0;
							val_bnd_imag = (T)0.0;						
						}
					
			            // Add interpolation contribution to final value
			            val_bnd_interp_real += val_bnd_real * pt_weight[pt] * angle_weight[rot];
			            val_bnd_interp_imag += val_bnd_imag * pt_weight[pt] * angle_weight[rot];				
					}
				}
					
				idx_sparse_in = BoundaryStorage<T>::getSparseIndex(
							angle, e, x, boundary_entry_count,
							Ne, Nx, sparse_idx_in_gammabar );
							
				if ( total_weight > (T)0.0 )
				{
					bnd_in_gammabar_real[idx_sparse_in] = val_bnd_interp_real / total_weight;
					bnd_in_gammabar_imag[idx_sparse_in] = val_bnd_interp_imag / total_weight;		
				}
				else
				{
					bnd_in_gammabar_real[idx_sparse_in] = (T)0.0;
					bnd_in_gammabar_imag[idx_sparse_in] = (T)0.0;	
				}

				boundary_entry_count++;			
			}				
		}

	}
}
} // namespace anonymous

template <typename T>
void BoundaryConditionSpecular<T>::computeBoundaryCondition( Context<T>* ctx )
{
	IntegrationIterator<T>* iterator = ctx->getIntegrationIterator();
	GeometryGroup<T>* geom = ctx->getGeometry();
	RiccatiSolver<T>* riccati = ctx->getRiccatiSolver();
	BoundaryStorage<T>* bnd = riccati->getBoundaryStorage();

	const int Na = ctx->getParameters()->getAngularResolution();
	const int Ne = iterator->getNumEnergiesInterval(); 
	
	T angle_rad;
	const int angle_idx = iterator->getCurrentAngleIndex();
	
	int Nx, Ny;
	geom->getGridDimensions( Nx, Ny );
	
	const int N_bnd_in_gamma = bnd->getBoundaryIn_gamma()->dimX() / Ne;
	const int N_bnd_out_gamma = bnd->getBoundaryOut_gamma()->dimX() / Ne;
	const int N_bnd_in_gammabar = bnd->getBoundaryIn_gammabar()->dimX() / Ne;
	const int N_bnd_out_gammabar = bnd->getBoundaryOut_gammabar()->dimX() / Ne;


	// Somewhat faster with streams
	cudaStream_t s1, s2;
	cudaStreamCreate( &s1 );
	cudaStreamCreate( &s2 );

	for ( int angle_idx=0; angle_idx<Na; ++angle_idx )
	{
		// Current angle in radians
		angle_rad = (T)angle_idx/(T)Na * ((T)2.0*M_PI);

		// Set per angle optimized kernel dimensions
		int idx_x_min = (bnd->getTrajectoryRangeX()->getDataPointer(0))[angle_idx];
		int idx_x_max = (bnd->getTrajectoryRangeX()->getDataPointer(1))[angle_idx];
		int idx_y_min = (bnd->getTrajectoryRangeY()->getDataPointer(0))[angle_idx];
		int idx_y_max = (bnd->getTrajectoryRangeY()->getDataPointer(1))[angle_idx];

		int numTrajectories = idx_x_max - idx_x_min + 1;

		dim3 bpg, tpb;
		GridGPU::getKernelDimensions(bpg,tpb,numTrajectories,Ne);
	
		// Create geometry for current angle
		geom->create( -angle_rad );
	
		// Compute reflection
		boundaryConditionSpecular_gamma_kernel<T><<<bpg,tpb,0,s1>>>(
			// Model parameters
			Nx, Ne, Na, angle_idx,
			idx_x_min, idx_x_max, idx_y_min, idx_y_max,
			// Array dimensions for sparse storage
			N_bnd_in_gamma, N_bnd_out_gamma,
			// Grids for domain info/data
			geom->getDomainLabelsGrid()->getDataPointer(),
			geom->getNormalsGrid()->getDataPointer( 0 ),
			geom->getNormalsGrid()->getDataPointer( 1 ),
			geom->getBoundaryDistanceGrid()->getDataPointer(),
			// Data relating to sparse storage for gammas
			bnd->getIndexSparseIn_gamma()->getDataPointer(),
			bnd->getIndexSparseOut_gamma()->getDataPointer(),
			bnd->getNumBoundaryIntersect_gamma()->getDataPointer(),
			bnd->getBoundaryIntersectCoords_gamma()->getDataPointer(),
			bnd->getBoundaryIn_gamma()->getDataPointer( 0, Type::real ),
			bnd->getBoundaryIn_gamma()->getDataPointer( 0, Type::imag ),
			bnd->getBoundaryOut_gamma()->getDataPointer( 0, Type::real ),
			bnd->getBoundaryOut_gamma()->getDataPointer( 0, Type::imag )
			);


		boundaryConditionSpecular_gammabar_kernel<T><<<bpg,tpb,0,s2>>>(
			// Model parameters
			Nx, Ne, Na, angle_idx,
			idx_x_min, idx_x_max, idx_y_min, idx_y_max,
			// Array dimensions for sparse storage
			N_bnd_in_gammabar, N_bnd_out_gammabar,
			// Grids for domain info/data
			geom->getDomainLabelsGrid()->getDataPointer(),
			geom->getNormalsGrid()->getDataPointer( 0 ),
			geom->getNormalsGrid()->getDataPointer( 1 ),
			geom->getBoundaryDistanceGrid()->getDataPointer(),
			// Data relating to sparse storage for gammas
			bnd->getIndexSparseIn_gammabar()->getDataPointer(),
			bnd->getIndexSparseOut_gammabar()->getDataPointer(),
			bnd->getNumBoundaryIntersect_gammabar()->getDataPointer(),
			bnd->getBoundaryIntersectCoords_gammabar()->getDataPointer(),
			bnd->getBoundaryIn_gammabar()->getDataPointer( 0, Type::real ),
			bnd->getBoundaryIn_gammabar()->getDataPointer( 0, Type::imag ),
			bnd->getBoundaryOut_gammabar()->getDataPointer( 0, Type::real ),
			bnd->getBoundaryOut_gammabar()->getDataPointer( 0, Type::imag )				
			);
	}
	
	cudaStreamDestroy( s1 );
	cudaStreamDestroy( s2 );
}

#endif
