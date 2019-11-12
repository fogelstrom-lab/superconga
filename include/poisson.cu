#include <cuda.h>
#include <Grid.h>

// Solves a set of poisson equations in the "y"-direction, ie each column is a 1-d poisson equation.
// This means that if you want to solve along x, the solution and the right hand side vector
// have to be transposed first (and then transposed again when finished). The reason for this is
// to make sure memory is read in a coalesced manner.

// This routine can probably be made quite a bit faster, but it it fast enough by a good margin
// not to be performance critical for the current project.
template <typename T>
__global__ void poisson_tridiag( T* A, const T* b, const int Nx )
{
	const int x = threadIdx.x;

	// Initialize
	A[Nx+x] += A[x];
	A[(Nx-2)*Nx+x] += A[(Nx-1)*Nx+x];

	// GE forward
	for (int j=2; j<Nx-2; ++j)
		A[j*Nx+x] += (T)(j-1)/(T)j * A[(j-1)*Nx+x];

	// GE backward
	for (int j=Nx-2; j>=2; --j)
	{
		A[j*Nx+x] *= (T)j/(T)(j+1);
		A[(j-1)*Nx+x] += A[j*Nx+x];
	}

	// Compute remaining element
	A[Nx+x] *= (T)0.5;
}


template <typename T>
__global__ void poisson_tridiag( T* A, const T* b, const char* geom, const int Nx )
{
	const int x = threadIdx.x;

	int idx;

	// Initialize
	for (int y=1; y<Nx-1; ++y)
	{
		idx = y*Nx + x;

		if ( geom[idx] == 2 )
			A[(y+1)*Nx+x] += A[y*Nx+x];
		else if ( geom[idx] == 3 )
			A[(y-1)*Nx+x] += A[y*Nx+x];
	}

	// Storage for FD lattice site count
	short Nk[2] = {0,0};
	short s = 0;
	char g0, g1;

	// GE forward
	for (int y=1; y<Nx-1; ++y)
	{
		idx = y*Nx + x;

		g0 = geom[idx];
		g1 = geom[(y+1)*Nx+x];

		if ( g0 == 1 )
		{
			Nk[s]++;
			if ( g1 == 1 )
				A[(y+1)*Nx+x] += (T)(Nk[s])/(T)(Nk[s]+1) * A[y*Nx+x];
		}
		else if ( g0 == 3 )
			s++;
	}

	// GE backward
	s--;
	for (int y=Nx-2; y>=1; --y)
	{
		idx = y*Nx + x;

		g0 = geom[idx];
		g1 = geom[(y-1)*Nx+x];	

		if ( g0 == 1 )
		{
			if ( g1 == 1 )
			{
				A[y*Nx+x] *= (T)Nk[s]/(T)(Nk[s]+1);
				A[(y-1)*Nx+x] += A[y*Nx+x];
				Nk[s]--;
			}
			else if ( g1 == 2 )
			{
				s--;
				A[y*Nx+x] *= (T)0.5;
			}
		}
	}
}



// Set dirichlet boundary condition
template <typename T>
__global__ void poisson_bc_dirichlet( T* A, const T val, const int Nx )
{
	const int x = threadIdx.x;
	const int y = blockIdx.y;

	if ( y==0 )
		A[x] = val;
	else if ( y==1 )
		A[(Nx-1)*Nx+x] = val;
}


// Routine for solving the poisson equation with the Jacobi iterative method. Slow, but memory efficient
// compared to a direct solve.
template <typename T>
__global__ void poisson_x( T* A_src, T* A_dst, const T* b, const int Nx )
{
	const int x = threadIdx.x + 1;
	const int y = blockIdx.y + 1;

	const int idx = y*Nx + x;

	T A_0, A_2;

	A_0 = A_src[idx-1];
	A_2 = A_src[idx+1];

	A_dst[idx] = (T)0.5*(-b[idx] + A_0 + A_2);
}


// Resampling routine required by a multigrid iterative method. The function below is not good at all,
// as it is non-conserving.
template <typename T>
__global__ void poisson_resample(
	T* data, T* data_tmp, const int Nx, const int Ny, const int Nx_new, const int Ny_new )
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


template <typename T>
void solve_poisson_tridiag(
	GridData<T,thrust::device_vector<T> >* sol,
	GridData<T,thrust::device_vector<T> >* rhs,
	const T h, const T c, const T bc,
	GridData<char,thrust::device_vector<char> >* geom = 0 )
{
	const int N = sol->dimX();

	dim3 bpg_bc(1,2,1);
	dim3 tpb_bc(N,1,1);

	dim3 bpg(1,1,1);
	dim3 tpb(N,1,1);

	(*sol) = (*rhs)*(-h*h*c);

	sol->transpose(0);
	rhs->transpose(0);

	poisson_bc_dirichlet<<<bpg_bc,tpb_bc>>>( sol->getDataPointer(0), bc, N );
	poisson_bc_dirichlet<<<bpg_bc,tpb_bc>>>( sol->getDataPointer(1), bc, N );

	if ( geom )
	{
		poisson_tridiag<<<bpg,tpb>>>( sol->getDataPointer(0), rhs->getDataPointer(0), geom->getDataPointer(0), N );
		poisson_tridiag<<<bpg,tpb>>>( sol->getDataPointer(1), rhs->getDataPointer(1), geom->getDataPointer(0), N );
	}
	else
	{
		poisson_tridiag<<<bpg,tpb>>>( sol->getDataPointer(0), rhs->getDataPointer(0), N );
		poisson_tridiag<<<bpg,tpb>>>( sol->getDataPointer(1), rhs->getDataPointer(1), N );
	}

	sol->transpose(0);
	rhs->transpose(0);
}
