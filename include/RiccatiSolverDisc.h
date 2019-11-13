#ifndef _RICCATISOLVERDISC_H
#define _RICCATISOLVERDISC_H

#include <RiccatiSolver.h>
#include <cplx.h>

template <typename T, template <typename> class GAMMA, template <typename> class POTENTIAL>
class RiccatiSolverDisc : public RiccatiSolver<T>
{
	typedef typename Grid<T>::GPU GridGPU;

public:

	// Constructors and Destructors

					RiccatiSolverDisc();
	virtual			~RiccatiSolverDisc();

					RiccatiSolverDisc( Context<T>* ctx );

	virtual int 	initialize();

	void			initializeBoundaryStorage() {}

	// Functions
	virtual void	computeInitialCondition(
						const ParameterIterator<int>& energy_iter, const GridGPU* energies,
						const GridGPU* delta, const int angle_idx ) {}

	virtual void	computeBoundaryCondition() {}

	virtual void	computeCoherenceFunctions(
						const ParameterIterator<int>& energy_iter, const GridGPU* energies,
						const GridGPU* delta, const int angle_idx ) {}

private:

};

namespace
{
template <typename T, typename GAMMA, template <typename> class POTENTIAL,
			int ENTRY, int EXIT, int DIR>
__global__ 
void computeGammaFunctionDisc_kernel(
	const T h, const int N, const int Ne, const int idx_x_offset,
	const int angle_idx, const T system_angle, const T momentum_angle, const T B,
	T *energies_re, T *energies_im,
	T *gamma_real, T *gamma_imag,
	T *delta_real, T *delta_imag,
	char *domainLabel, T *domainNormals_x, T *domainNormals_y, T *boundaryDistance )
{	
	const int x		= blockDim.x * blockIdx.x + threadIdx.x + idx_x_offset;	// x-coordinate
	const int e_idx	= blockDim.y * blockIdx.y + threadIdx.y;				// energy coordinate

	cplx<T> delta, g((T)0.0,(T)0.0), g0((T)0.0,(T)0.0), energy;
	T potential;

	int idx, gamma_idx, y;
	char nodeType;

	if ( x < N && e_idx < Ne )
	{
		energy.x = energies_re[e_idx];
		energy.y = energies_im[e_idx];
		
		const T energy_zeroPotential = energy.x;
		
		const int yStart = (N-1)*(int)(DIR==-1);
		
		// Compute gamma along trajectory
		for ( int i=0; i<N; ++i )
		{
			y = yStart + DIR*i;
			
			// Add energy contribution from vector potential
			potential = POTENTIAL<T>::computeVectorPotential_local(
				(T)x, (T)y, energy_zeroPotential, B, system_angle, momentum_angle, N, h );

			energy.x = energy_zeroPotential + potential;

			// Index to 2d-grids (Delta)
			idx = y*N + x;
			
			// Index to gamma grid (for all energies)
			gamma_idx = e_idx*2*N*N + y*N + x;
			
			// What type of node are we on?
			nodeType = domainLabel[idx];
			
			// Read order parameter
			delta.x = delta_real[idx];
			delta.y = delta_imag[idx];

			if ( nodeType == 1 )	// Inside domain:
			{
				GAMMA::computeGamma_local(	delta, g0, energy, h, i * (int)(nodeType!=1), g );
			}
			
			gamma_real[gamma_idx] = g.x;
			gamma_imag[gamma_idx] = g.y;

			g0 = g;
		}
	}
}
} // namespace anonymous

template <typename T, template <typename> class GAMMA, template <typename> class POTENTIAL>
RiccatiSolverDisc<T,GAMMA,POTENTIAL>::RiccatiSolverDisc()
	:	RiccatiSolver<T>()
{}


template <typename T, template <typename> class GAMMA, template <typename> class POTENTIAL>
RiccatiSolverDisc<T,GAMMA,POTENTIAL>::~RiccatiSolverDisc()
{
	std::cout << "~RiccatiSolverDisc()\n";
}

template <typename T, template <typename> class GAMMA, template <typename> class POTENTIAL>
RiccatiSolverDisc<T,GAMMA,POTENTIAL>::RiccatiSolverDisc( Context<T>* ctx )
	:	RiccatiSolver<T>( ctx )
{
	std::cout << "RiccatiSolverDisc::RiccatiSolverDisc( Context<T>* )\n";
	ctx->set( this );
}


template <typename T, template <typename> class GAMMA, template <typename> class POTENTIAL>
int RiccatiSolverDisc<T,GAMMA,POTENTIAL>::initialize()
{
	std::cout << "RiccatiSolverDisc::initialize()\n";
	RiccatiSolver<T>::initialize();

	return 0;
}
#endif
