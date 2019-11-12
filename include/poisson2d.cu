#include <cuda.h>
#include <Grid.h>
#include <GeometryGroup.h>

// Compute vector potential by integrating greens function kernel
template <typename T>
__global__ void computeVectorPotential2d_kernel( T* Ax, T* Ay, const T* jx, const T* jy, const char* geom, const int px, const int py, const T h, const T c, const int Nx )
{
	const int x = threadIdx.x;
	const int y = blockIdx.y;

	const int idx = y*Nx + x;
	const int idxp = py*Nx + px;

	// This block was in a commented if-statement
	// if ( geom[idx] )
	T mult;

	if ( x == px && y == py )
	{
		mult = c*0.25*h*h*(-6.0 + M_PI - log(4.0));
	}
	else
	{
		const T dx = px - x;
		const T dy = py - y;
		mult = c*h*h*log(h*sqrt(dx*dx + dy*dy));
	}

	Ax[idx] += mult*jx[idxp];
	Ay[idx] += mult*jy[idxp];
}



// Compute vector potential by integrating greens function kernel
template <typename T>
__global__ void laplace2d_kernel( T* del_f, const T* f, const T h, const int Nx )
{
	const int x = threadIdx.x + 1;
	const int y = blockIdx.y + 1;

	const int idx = y*Nx + x;

	del_f[idx] = (f[idx+1] + f[idx-1] + f[idx+Nx] + f[idx-Nx] - T(4.0)*f[idx]) / (h*h);
}


template <typename T>
__global__ void laplace2d_kernel( T* del_f, const T* f, const char* geom, const T h, const int Nx )
{
	const int x = threadIdx.x + 1;
	const int y = blockIdx.y + 1;

	const int idx = y*Nx + x;

	// Determine which edge we are at
	const char bottom	= geom[(y-1)*Nx+x];
	const char top		= geom[(y+1)*Nx+x];
	const char left		= geom[y*Nx+x-1];
	const char right	= geom[y*Nx+x+1];

	T dfx=1.0, dfy=1.0;

	if ( !bottom || !top )
		dfy = 0.0;
	if ( !left || !right )
		dfx = 0.0;

	del_f[idx] = ( dfx*(f[idx+1] + f[idx-1]) + dfy*(f[idx+Nx] + f[idx-Nx]) - (dfx+dfy)*T(2.0)*f[idx]) / (h*h);
}


template <typename T>
void computeVectorPotential(
	GridData<T,thrust::device_vector<T> >* A,
	const GridData<T,thrust::device_vector<T> >* j,
	GridData<char,thrust::device_vector<char> >* geom,
	const T h, const T c, const int N )
{
	A->setZero();

	dim3 bpg(1,N);
	dim3 tpb(N,1);

	GridData<char,thrust::host_vector<char> > geom_host = *geom;
	char* g = geom_host.getDataPointer(0);
	
	for ( int y=0; y<N; ++y )
	{
		for ( int x=0; x<N; ++x )
		{
			if ( g[y*N+x] )
			{
				computeVectorPotential2d_kernel<<<bpg,tpb>>>(
					A->getDataPointer(0), A->getDataPointer(1),
					j->getDataPointer(0), j->getDataPointer(1),
					geom->getDataPointer(0),
					x, y, h, c, N );
			}
		}
	}
	*A *= (-1.0/(2.0*M_PI));
}


template <typename T>
GridData<T,thrust::device_vector<T> >* laplace2d(
	const GridData<T,thrust::device_vector<T> >* A,
	const T h,
	const GridData<char,thrust::device_vector<char> >* geom = 0
 )
{
	typedef typename Grid<T>::GPU GridGPU;

	GridGPU* j = 0;
	GridGPU::initializeGrid(j,A);

	j->setZero();

	const int N = A->dimX();

	dim3 bpg(1,N-2);
	dim3 tpb(N-2,1);

	for ( int c=0; c<A->numFields(); ++c )
	{
		if ( geom )
			laplace2d_kernel<<<bpg,tpb>>>(j->getDataPointer(c), A->getDataPointer(c), geom->getDataPointer(0), h, N);
		else
			laplace2d_kernel<<<bpg,tpb>>>(j->getDataPointer(c), A->getDataPointer(c), h, N);
	}

	return j;
}



// Set dirichlet boundary condition
template <typename T>
__global__ void poisson_bc_dirichlet( T* A, const T val, const int Nx )
{
	const int x = threadIdx.x;
	const int y = blockIdx.y;

	if ( x==0 || x==Nx-1 || y==0 || y==Nx-1 )
		A[y*Nx+x] = val;
}


template <typename T>
__global__ void poisson_bc_dirichlet( T* A, const char* geom, const T val, const int Nx )
{
	const int idx = blockIdx.y*Nx + threadIdx.x;

	if ( geom[idx] == 0 )
		A[idx] = val;
}


template <typename T>
__global__ void poisson_bc_B0( const T* Ax_src, const T* Ay_src, T* Ax_dst, T* Ay_dst, const char* geom, const int Nx )
{
	const int x = threadIdx.x + 1;
	const int y = blockIdx.y + 1;
	const int idx = y*Nx + x;

	// Determine which edge we are at
	const char bottom	= geom[(y-1)*Nx+x];
	const char top		= geom[(y+1)*Nx+x];
	const char left		= geom[y*Nx+x-1];
	const char right	= geom[y*Nx+x+1];

	// Set BC's, as per the discrete version of d(Ay)/dx = d(Ax)/dy
	if ( geom[idx] == 0 && (bottom||top||left||right) )
	{
		int t=0, r=0;
		T mult_x = 1.0, mult_y = 1.0;

		if ( top )
		{
			t = 1;
			mult_x = -1.0;
		}

		if ( right )
		{
			r = 1;
			mult_y = -1.0;
		}

		Ax_dst[idx] = Ax_src[(y-1+2*t)*Nx+(x)] + mult_x*(Ay_src[(y)*Nx+(x+r)] - Ay_src[(y)*Nx+(x-1+r)]);
		Ay_dst[idx] = Ay_src[(y)*Nx+(x-1+2*r)] + mult_y*(Ax_src[(y+t)*Nx+(x)] - Ax_src[(y-1+t)*Nx+(x)]);
	}
}


// Routine for solving the poisson equation with the Jacobi iterative method. Slow, but memory efficient
// compared to a direct solve.
template <typename T>
__global__ void poisson_jacobi( const T* A_src, T* A_dst, const T* b, const int Nx )
{
	const int x = threadIdx.x;
	const int y = blockIdx.y;

	const int idx = y*Nx + x;

	T Ax_0, Ax_2, Ay_0, Ay_2;

	Ax_0 = A_src[y*Nx + x - 1];
	Ax_2 = A_src[y*Nx + x + 1];

	Ay_0 = A_src[(y-1)*Nx + x];
	Ay_2 = A_src[(y+1)*Nx + x];

	A_dst[idx] = (T)0.25*(-b[idx] + Ax_0 + Ax_2 + Ay_0 + Ay_2);
}


template <typename T>
__global__ void poisson_jacobi( const T* A_src, T* A_dst, const T* b, const char* geom, const int Nx )
{
	const int x = threadIdx.x + 1;
	const int y = blockIdx.y + 1;

	const int idx = y*Nx + x;

	if ( geom[idx] )
	{
		T Ax_0 = A_src[y*Nx + x - 1];
		T Ax_2 = A_src[y*Nx + x + 1];

		T Ay_0 = A_src[(y-1)*Nx + x];
		T Ay_2 = A_src[(y+1)*Nx + x];

		A_dst[idx] = (T)0.25*(-b[idx] + Ax_0 + Ax_2 + Ay_0 + Ay_2);
	}
}


template <typename T>
__global__ void shrink( T* data, const char* geom, const int n, const int Nx )
{
	const int x = threadIdx.x + 1;
	const int y = blockIdx.y + 1;

	const int idx = y*Nx + x;

	if ( geom[idx] )
	{
		//const int n = 5;
		short sum = 0, sumc = 0;

		for ( int ny=-n; ny<n; ++ny )
		{
			for ( int nx=-n; nx<n; ++nx )
			{
				if ( (x+nx)>=0 && (x+nx)<Nx && (y+ny)>=0 && (y+ny)<Nx )
				{
					sum += short(geom[(y+ny)*Nx+x+nx]);
					sumc++;
				}
			}
		}

		if ( sum != sumc )
			data[idx] = 0.0;
	}
	else
		data[idx] = 0.0;
}


template <typename T>
__global__ void poisson_jacobi( const T* A_src, T* A_dst, const T* b, const char* geom, T b_mult, const int Nx )
{
	const int x = threadIdx.x + 1;
	const int y = blockIdx.y + 1;

	const int idx = y*Nx + x;

	if ( geom[idx] )
	{
		T Ax_0 = A_src[y*Nx + x-1];
		T Ax_2 = A_src[y*Nx + x+1];

		T Ay_0 = A_src[(y-1)*Nx + x];
		T Ay_2 = A_src[(y+1)*Nx + x];

		A_dst[idx] = (T)0.25*(b[idx]*b_mult + Ax_0 + Ax_2 + Ay_0 + Ay_2);
	}
}





template <typename T>
void solve_poisson_jacobi(
	GridData<T,thrust::device_vector<T> >* sol,
	const GridData<T,thrust::device_vector<T> >* rhs,
	const T h, const T c, const T bc,
	GridData<char,thrust::device_vector<char> >* geom, const T conv, int saveIter=0, int applyBC=1 )
{
	typedef typename Grid<T>::GPU GridGPU;

	GridGPU *sol_new=0, *sol_tmp;
	GridGPU::initializeGrid(sol_new,rhs);

	int N = rhs->dimX();
	const T b_mult = h*h*c;

	dim3 bpg(1,N-2,1);
	dim3 tpb(N-2,1,1);

	dim3 bpg_bc(1,N,1);
	dim3 tpb_bc(N,1,1);

	for ( int i=0; i<sol->numFields(); ++i )
		poisson_bc_dirichlet<<<bpg_bc,tpb_bc>>>( sol->getDataPointer(i), geom->getDataPointer(0), bc, N );

	*sol_new = *sol;

	T res = 1000.0;
	int i=0;
	int ic=0;

	while ( res > conv )
	{
		for ( int f=0; f<sol->numFields(); ++f )
			poisson_jacobi<<<bpg,tpb>>>( sol->getDataPointer(f), sol_new->getDataPointer(f), rhs->getDataPointer(f), geom->getDataPointer(0), b_mult, N );

		// Enforce BC
		if ( applyBC )
			poisson_bc_B0<<<bpg,tpb>>>( sol->getDataPointer(0), sol->getDataPointer(1), sol_new->getDataPointer(0), sol_new->getDataPointer(1), geom->getDataPointer(0), N );

		sol_tmp = sol;
		sol = sol_new;
		sol_new = sol_tmp;

		i++;

		if ( !(i%100) )
		{
			res = (*sol_new - *sol).fieldMagnitude().sumField() / (N*N);
			//std::cout << "conv = " << conv << ", res = " << res << "\n";
			if (saveIter && ic < 200 )
				sol->write("A",ic++);
		}
	}
	
	std::cout << "solve_poisson_jacobi: converged after " << i << " iterations.\n";
	
	GridGPU::deleteGrid(sol_new);
}


// Multigrid (or rather, dual grid) solver for the jacobi poisson solver.
// Use when grids are larger than 200 lattice sites.
template <typename T>
GridData<T,thrust::device_vector<T> >* solve_poisson_mg(
	const GridData<T,thrust::device_vector<T> >* rhs,
	const T h, const T c, const T bc,
	GeometryGroup<T>* geom, const T conv = 5.0E-10 )
{
	typedef typename Grid<T>::GPU GridGPU;

	int N0;
	const int N1 = rhs->dimX();

	Grid<char>::GPU dom, dom_r;

	// Set resolution for small grid
	if ( N1%2 )	// N1 odd
		N0 = (N1-1)/2 + 1;
	else		// N1 even
		N0 = N1/2;

	const T h0 = h*(N1-1) / T(N0-1);

	// Small grid
	//

	// Resample RHS to small grid
	GridGPU rhs0(*rhs);
	rhs0.resample(N0,N0);

	// Init solution for small grid
	GridGPU* sol = 0;
	GridGPU::initializeGrid(sol,&rhs0);

	// Create geometry for small grid
	const int N0_r = geom->getParameters()->calcGridResolutionRotated(N0);

	dom_r = geom->createDomain(N0_r);

	dom = Grid<char>::GPU(N0,N0,1,Type::real);
	dom.setZero();
	dom.insertGrid(&dom_r);

	// Solve for small grid
	solve_poisson_jacobi<T>(sol, &rhs0, h0, c, bc, &dom, conv, 0, 0);

	sol->write("small.grd");


	// Regular grid
	//
	sol->resample(N1,N1);

	// Create geometry for regular grid
	const int N1_r = geom->getParameters()->calcGridResolutionRotated(N1);

	dom_r = geom->createDomain(N1_r);
	
	dom = Grid<char>::GPU(N1,N1,1,Type::real);
	dom.setZero();
	dom.insertGrid(&dom_r);

	// Solve for regular grid
	solve_poisson_jacobi<T>(sol, rhs, h, c, bc, &dom, conv, 0, 0);

	return sol;
}


template <typename T>
//void solve_poisson_tridiag( Grid<T>::GPU* sol, Grid<T>::GPU* rhs, const T h, const T c, const T bc )
void solve_poisson_jacobi(
	GridData<T,thrust::device_vector<T> >* sol_src,
	GridData<T,thrust::device_vector<T> >* sol_dst,
	GridData<T,thrust::device_vector<T> >* rhs,
	const T h, const T c, const T bc,
	GridData<char,thrust::device_vector<char> >* geom = 0 )
{
	const int N = sol_src->dimX();

	if ( geom )
	{
		dim3 bpg(1,N-2,1);
		dim3 tpb(N-2,1,1);

		for ( int i=0; i<sol_src->numFields(); ++i )
			poisson_jacobi<<<bpg,tpb>>>( sol_src->getDataPointer(i), sol_dst->getDataPointer(i), rhs->getDataPointer(i), geom->getDataPointer(0), N );
	}
	else
	{
		dim3 bpg(1,N-2,1);
		dim3 tpb(N-2,1,1);
		for ( int i=0; i<sol_src->numFields(); ++i )
			poisson_jacobi<<<bpg,tpb>>>( sol_src->getDataPointer(i), sol_dst->getDataPointer(i), rhs->getDataPointer(i), N );
	}
}