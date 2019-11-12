#include <cuda.h>

template <typename T, int NV>
__global__ void generatePolygonGeometry_Kernel(
	char *dom_p, T *normX_p, T *normY_p, T *dist_p,
	const int Nx, const int Ny, const T phi,
	const int numVertices, T * vertX, T * vertY )
{
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;

	int i;
	
	T ptX_R[NV], ptY_R[NV], lineVectX, lineVectY, crossZ, normX, normY, norm_tmp;
	
	T isect_out, normX_out, normY_out;
	
	T m, k, isect;
	T cosPhi, sinPhi;
	
	if ( tx < Nx && ty < Ny )
	{
		const int idx = ty*Nx + tx;
	
		const T coordX = (T)tx - (T)(Nx-1)*(T)0.5;
		const T coordY = (T)ty - (T)(Ny-1)*(T)0.5;

		// Rotate vertices to local frame
		cosPhi = cos(phi);
		sinPhi = sin(phi);
		
		for ( i=0; i<numVertices; ++i )
		{
			ptX_R[i] = (cosPhi*vertX[i] - sinPhi*vertY[i]) * (Nx-1);
			ptY_R[i] = (sinPhi*vertX[i] + cosPhi*vertY[i]) * (Nx-1);
		}
		
		// Compute stuff
		int ip, sum_cross = 0;

		isect_out = (T)100000.0;
		
		for ( i=0; i<numVertices; ++i )
		{
			ip = i+1;
			if ( ip == numVertices )
				ip = 0;
			
			// Vector for polygon line segment
			lineVectX = ptX_R[ip] - ptX_R[i];
			lineVectY = ptY_R[ip] - ptY_R[i];


			// z-component of cross product (used to determine inside/outside)
			crossZ = lineVectX*(coordY-ptY_R[i]) - lineVectY*(coordX-ptX_R[i]);
			
			if ( crossZ >= (T)0.0 )
				sum_cross++;
			
			
			// Normals
			normX = -lineVectY;
			normY = lineVectX;
			
			norm_tmp = (T)1.0/sqrt(normX*normX + normY*normY);
			
			normX *= norm_tmp;
			normY *= norm_tmp;
			
			
			// Compute distance to boundary
			if ( abs(lineVectX) > (T)0.00001 )
			{
				k = lineVectY / lineVectX;
				m = ptY_R[i] - k*ptX_R[i];
			
				isect = abs( k*coordX + m - coordY );
			}
			else
			{
				isect = (T)10000.0;
			}
			
			if ( isect < isect_out )
			{
				isect_out = isect;
				normX_out = normX;
				normY_out = normY;
			}
			
		} 

		// Store computed data
		dom_p[idx] = (char)(int)( sum_cross == numVertices );
		dist_p[idx] = isect_out;
		normX_p[idx] = normX_out;
		normY_p[idx] = normY_out;
	}
}


template <typename T, int NV>
__global__ void generatePolygonGeometry_Kernel(
	char *dom_p,
	const int numVertices,
	T* vertX, T* vertY,
	const int Nx )
{
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;

	T ptX_R[NV], ptY_R[NV], lineVectX, lineVectY, crossZ;
	
	if ( tx < Nx && ty < Nx )
	{
		const int idx = ty*Nx + tx;
	
		const T coordX = (T)tx - (T)(Nx-1)*(T)0.5;
		const T coordY = (T)ty - (T)(Nx-1)*(T)0.5;

		for ( int i=0; i<numVertices; ++i )
		{
			ptX_R[i] = vertX[i] * (Nx-1);
			ptY_R[i] = vertY[i] * (Nx-1);
		}
		
		// Compute stuff
		int ip, sum_cross = 0;
		
		for ( int i=0; i<numVertices; ++i )
		{
			ip = i+1;
			if ( ip == numVertices )
				ip = 0;
			
			// Vector for polygon line segment
			lineVectX = ptX_R[ip] - ptX_R[i];
			lineVectY = ptY_R[ip] - ptY_R[i];

			// z-component of cross product (used to determine inside/outside)
			crossZ = lineVectX*(coordY-ptY_R[i]) - lineVectY*(coordX-ptX_R[i]);
			
			if ( crossZ >= (T)0.0 )
				sum_cross++;	
		} 

		// Store computed data
		dom_p[idx] = (char)(int)( sum_cross == numVertices );
	}
}


template <typename T>
__global__ void maskNormals_Kernel(
	char *dom_p, T *normX_p, T *normY_p, T *dist_p, const int Nx, const int Ny )
{
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;

	const int idx = ty*Nx + tx;

	if ( tx < Nx && ty < Ny )
	{
		T mult = T(int(dom_p[idx]) > 1);

		// Store computed data
		dist_p[idx] *= mult;
		normX_p[idx] *= mult;
		normY_p[idx] *= mult;
	}
}
