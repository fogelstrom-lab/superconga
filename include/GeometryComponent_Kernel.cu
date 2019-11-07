#include <cuda.h>

// Label boundary interface nodes
// Codes:
// 0 - Outside domain
// 1 - Inside domain
// 2 - Trajectory entering domain
// 3 - Trajectory exiting domain

// 4 - Trajectory entering AND exiting domain at the same grid point (not implemented)


// Label type 2 and 3 and 4 nodes
template <typename T>
__global__ void labelEntryExit_Kernel( char *oldLabels_p, char *newLabels_p, const int Nx, const int Ny )
{
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ( tx < Nx && ty < Ny  )
	{
		const int idx = ty*Nx + tx;
		
		char newLabel = 0;
		char oldLabel[3];		

		if ( ty > 0 && ty < Ny-1 )
		{		
			// Read current and adjacent labels
			// Maybe use shared memory here for more efficient data loading and memory coalescing
			for ( int i=0; i<3; ++i )
				oldLabel[i] = oldLabels_p[ (ty-1+i)*Nx + tx ];

			// Label interface nodes
			if ( oldLabel[1] == 1 )
			{
				newLabel = 1;
				
				if ( oldLabel[0] == 0 )
				{
					newLabel = 2;
				}
				else if ( oldLabel[2] == 0 )
				{
					newLabel = 3;
				}
			
				if ( oldLabel[0] == 0 && oldLabel[2] == 0 )
					newLabel = 0;
			}
		}		
		newLabels_p[idx] = newLabel;
	}
}

