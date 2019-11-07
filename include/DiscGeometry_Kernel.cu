#include <cuda.h>


template <typename T>
__global__ void generateDisc_Kernel(
	char *dom_p, T *normX_p, T *normY_p, T *dist_p,
	const int Nx, const int Ny, const T radius, const T pos_x, const T pos_y, const T angle )
{
	int i;

	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if ( tx < Nx && ty < Ny )
	{
		const int idx = ty*Nx + tx;
	
		T coord_x, coord_y[3], distSq[3];
		T radiusSq = radius*radius;
		
		coord_x = (T)tx - (Nx-1)*(T)0.5 - pos_x;

		for ( i=0; i<3; ++i )
		{
			coord_y[i] = (T)(ty-1+i) - (Ny-1)*(T)0.5 - pos_y;
			distSq[i] = coord_x*coord_x + coord_y[i]*coord_y[i];
		}	

		// Compute domain labels
		char label = 0;
		
		if ( distSq[1] <= radiusSq )
		{
			label = 1;
			if ( distSq[0] > radiusSq )
				label = 2;
			else if ( distSq[2] > radiusSq )
				label = 3;
			
			if ( distSq[0] > radiusSq && distSq[2] > radiusSq )
				label = 0;
		}


		// Distance along y (trajectory path direction) to exact boundary location
		T dist_intersect = (T)0.0;
		T norm_x = (T)0.0;
		T norm_y = (T)0.0;
		
		if ( label > 1 )
		{
			// Determine boundary normal at exact boundary location
			dist_intersect = sqrt( radiusSq - coord_x*coord_x ) - abs(coord_y[1]);
			norm_x = -coord_x / radius;

			if ( coord_y[1] <= (T)0.0 )
				norm_y = -(coord_y[1] - dist_intersect) / radius;
			else
				norm_y = -(coord_y[1] + dist_intersect) / radius;
		}

		// Store computed data
		//
		// (Boundary intersection is regarded as inside.
		// Maybe implement special case for this, to avoid dividing by zero, etc.)
		dom_p[idx] = label;
		dist_p[idx] = dist_intersect;
		normX_p[idx] = norm_x;
		normY_p[idx] = norm_y;
	}
}


/*
template <typename T>
__global__ void generateDisc_Kernel(
	char *dom_p, T *normX_p, T *normY_p, T *dist_p,
	const int Nx, const int Ny, const T radius, const T pos_x, const T pos_y, const T angle )
{
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if ( tx < Nx && ty < Ny )
	{
		const int idx = ty*Nx + tx;
	
		const T coord_x = (T)tx - (Nx-1)*(T)0.5 - pos_x;
		const T coord_y = (T)ty - (Ny-1)*(T)0.5 - pos_y;

		const T distSq = coord_x*coord_x + coord_y*coord_y;
		const T radiusSq = radius*radius;

		int isInside = (int)(distSq <= radiusSq);
			
		// Distance along y (trajectory path direction) to exact boundary location
		T dist_intersect = (T)10.0;

		if ( coord_x * coord_x <= radiusSq )
			dist_intersect = sqrt( radiusSq - coord_x*coord_x ) - abs(coord_y);
	
		// Determine boundary normal at exact boundary location
		T norm_x = -coord_x / radius;
		T norm_y;

		if ( coord_y <= (T)0.0 )
			norm_y = -(coord_y-dist_intersect) / radius;
		else if ( coord_y > (T)0.0 )
			norm_y = -(coord_y+dist_intersect) / radius;
	
		// Store computed data
		//
		// (Boundary intersection is regarded as inside.
		// Maybe implement special case for this, to avoid dividing by zero, etc.)

		
		dom_p[idx] = (char)isInside;
		dist_p[idx] = dist_intersect;
		normX_p[idx] = norm_x * isInside;
		normY_p[idx] = norm_y * isInside;
	}
}
*/
