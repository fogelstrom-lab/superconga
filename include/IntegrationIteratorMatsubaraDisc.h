#ifndef _INTEGRATIONITERATORMATSUBARADISC_H
#define _INTEGRATIONITERATORMATSUBARADISC_H

#include <cuda.h>
#include <cplx.h>
#include <GeometryGroup.h>
#include <OrderParameter.h>
#include <RiccatiSolver.h>
#include <ComputeProperty.h>
#include <ParameterIterator.h>
#include <Context.h>

template <typename T>
class IntegrationIteratorMatsubaraDisc : public IntegrationIterator<T>
{
	typedef typename Grid<T>::GPU GridGPU;
	typedef typename Grid<T>::CPU GridCPU;

public:
				IntegrationIteratorMatsubaraDisc();
	virtual		~IntegrationIteratorMatsubaraDisc();

				IntegrationIteratorMatsubaraDisc(	Context<T>* ctx,
													int numEnergiesPerBlock = -1 );

	virtual int		initialize();

	// Configures energy iterators
	void			initIterators( int numEnergiesPerBlock );
	virtual void	initEnergies();

	virtual int 	getNumEnergiesInterval();
	virtual int 	getEnergyIdxOffset_boundaryStorage() { return 0; }

	// Call this once before starting self consistent iteration, or when changing initial condition.
	virtual void	setBoundaryInitialCondition() {}

	// Computes all properties added with addComputeProperty(). Iterate this self consistently.
	virtual void	compute();		

private:

};

template <typename T>
__global__ void gen_trajectories( T* traj_x, T* traj_y, T* cos2Theta, T p0x, T p0y, const T h, const T R2, const int Na, const int Ns )
{
	T p1x, p1y, t;

	const int a = threadIdx.x;

	T theta = a*(T)2.0*M_PI / Na;
	T vFx = cos(theta);
	T vFy = sin(theta);

	traj_x[a*Ns] = p0x;
	traj_y[a*Ns] = p0y;

	for (int s=1; s<Ns; ++s)
	{
    	p1x = p0x + vFx*h;
    	p1y = p0y + vFy*h;

    	if ( p1x*p1x+p1y*p1y >= R2 )
    	{
    		const T pv = p0x*vFx + p0y*vFy;
    		
    		t = -pv + sqrt(R2+pv*pv-(p0x*p0x+p0y*p0y));
    		
    		const T pRx = p0x + vFx*t;
    		const T pRy = p0y + vFy*t;

    		t = (T)2.0*((p1x*pRx+p1y*pRy)/R2 - (T)1.0);

    		p1x = p1x - t*pRx;
    		p1y = p1y - t*pRy;

    		t = (T)2.0/R2*(vFx*pRx + vFy*pRy);

    		vFx = vFx - t*pRx;
    		vFy = vFy - t*pRy;

	  		if ( vFx >= (T)0.0 && vFy >= (T)0.0 ) {
				if ( vFx == (T)0.0 )
					theta = M_PI*(T)0.5;
				else
					theta = atan(vFy/vFx);
			} else if ( vFx < (T)0.0 && vFy >= (T)0.0 ) {
				theta = M_PI + atan(vFy/vFx);
			} else if ( vFx < (T)0.0 && vFy < (T)0.0 ) {
				theta = M_PI + atan(vFy/vFx);
			} else {
				theta = (T)2.0*M_PI + atan(vFy/vFx);
			}  		
	  	}

    	traj_x[a*Ns+s] = p1x;
    	traj_y[a*Ns+s] = p1y;

    	cos2Theta[a*Ns+s] = cos(2.0*theta);

    	p0x = p1x;
    	p0y = p1y;
	}            
}


template <typename T>
__device__ void interpDelta( cplx<T>& delta_interp, const T* delta_re, const T* delta_im, const T px, const T py, const int Nx)
{
    const int px_i = px;
    const int py_i = py;

    const T dp_x = px - px_i;
    const T dp_y = py - py_i;

    const T y0_re = (1.0-dp_x)*delta_re[py_i*Nx+px_i] + dp_x*delta_re[py_i*Nx+px_i+1];
    const T y0_im = (1.0-dp_x)*delta_im[py_i*Nx+px_i] + dp_x*delta_im[py_i*Nx+px_i+1];

    const T y1_re = (1.0-dp_x)*delta_re[(py_i+1)*Nx+px_i] + dp_x*delta_re[(py_i+1)*Nx+px_i+1];
    const T y1_im = (1.0-dp_x)*delta_im[(py_i+1)*Nx+px_i] + dp_x*delta_im[(py_i+1)*Nx+px_i+1];

    delta_interp.x = (1.0-dp_y)*y0_re + dp_y*y1_re;
    delta_interp.y = (1.0-dp_y)*y0_im + dp_y*y1_im;
}


template <typename T>
__global__ void gamma_trajectories(
	T* ga_re, T* ga_im, T* gb_re, T* gb_im,
	const T* traj_x, const T* traj_y, const T* cos2Theta, const T* delta_re, const T* delta_im, const T* e_im,
	const T h, const int Nx, const int Na, const int Ne, const int Ns )
{
	const int e = threadIdx.x;
	const int a = blockIdx.y;

	const int idx_a = a*Ne + e;
	const int idx_b = ((a+Na/2)%Na)*Ne + e;

	cplx<T> ga, gb, ga0, gb0;
	cplx<T> D, delta, en(0.0,e_im[e]);

	T px, py;

	for (int s=0; s<Ns; ++s)
	{
		// Get position
		px = traj_x[a*Ns + (Ns-s-1)];
		py = traj_y[a*Ns + (Ns-s-1)];

		px = px/h + (Nx-1)*0.5;
		py = py/h + (Nx-1)*0.5;

		// Get delta
		interpDelta(D, delta_re, delta_im, px, py, Nx);

		D *= sqrt(2.0) * cos2Theta[a*Ns + (Ns-s-1)];

		// compute gamma
		Gamma::General<T>::Gamma::computeGamma_local(D, ga0, en, h, s, ga);
		Gamma::General<T>::Gammabar::computeGamma_local(D, gb0, en, h, s, gb);

		ga0 = ga;
		gb0 = gb;
	}

	ga_re[idx_a] = ga.x;
	ga_im[idx_a] = ga.y;

	gb_re[idx_b] = gb.x;
	gb_im[idx_b] = gb.y;
}


template <typename T>
__global__ void gamma_trajectories_debug(
	T* ga_re, T* ga_im, T* gb_re, T* gb_im,
	const T* traj_x, const T* traj_y, const T* delta_re, const T* delta_im, const T* e_im,
	const T h, const int Nx, const int Na, const int Ne, const int Ns )
{
	const int e = threadIdx.x;
	const int a = blockIdx.y;

	const int idx_a = a*Ne + e;
	const int idx_b = ((a+Na/2)%Na)*Ne + e;

	cplx<T> ga, gb, ga0, gb0;
	cplx<T> D, delta, en(0.0,e_im[e]);

	T px, py;
	int x,y;

	// Get position
	px = traj_x[a*Ns];
	py = traj_y[a*Ns];

	px = px/h + (Nx-1)*0.5;
	py = py/h + (Nx-1)*0.5;

	x = px;
	y = py;

	D.x = delta_re[y*Nx+x];
	D.y = delta_im[y*Nx+x];

	Gamma::General<T>::Gamma::computeGamma_local(D, ga0, en, h, 0, ga);
	Gamma::General<T>::Gammabar::computeGamma_local(D, gb0, en, h, 0, gb);

	ga_re[idx_a] = D.x;
	ga_im[idx_a] = D.y;

	gb_re[idx_b] = D.x;
	gb_im[idx_b] = D.y;

}


template <typename T>
__global__ void greens(
	T* G_re, T* G_im,
	const T* ga_re, const T* ga_im, const T* gb_re, const T* gb_im,
	const int Na, const int Ne )
{
	const int e = threadIdx.x;
	const int a = blockIdx.y;

	const T theta = 2.0*M_PI * (T)a / (T)Na;

	const int idx = a*Ne + e;

	const cplx<T> ga(ga_re[idx],ga_im[idx]);
	const cplx<T> gb(gb_re[idx],gb_im[idx]);
	cplx<T> G;

	GreensAnomalous<T>::computeGreens_local(ga,gb,G);

	G *= sqrt(2.0) * cos(2.0*theta);

	G_re[idx] = G.x;
	G_im[idx] = G.y;
}


template <typename T>
IntegrationIteratorMatsubaraDisc<T>::IntegrationIteratorMatsubaraDisc()
	:	IntegrationIterator<T>()
{}

template <typename T>
IntegrationIteratorMatsubaraDisc<T>::IntegrationIteratorMatsubaraDisc(
	Context<T>* ctx, int numEnergiesPerBlock )
	:	IntegrationIterator<T>( ctx, numEnergiesPerBlock )
{
	std::cout << "IntegrationIteratorMatsubaraDisc<T>()\n";
	ctx->set( this );
}


template <typename T>
IntegrationIteratorMatsubaraDisc<T>::~IntegrationIteratorMatsubaraDisc()
{
	std::cout << "~IntegrationIteratorMatsubaraDisc()\n";
}


template <typename T>
int IntegrationIteratorMatsubaraDisc<T>::initialize()
{
	std::cout << "IntegrationIteratorMatsubaraDisc::initialize()\n";
	IntegrationIterator<T>::initialize();

	initIterators( IntegrationIterator<T>::getNumEnergiesPerBlock() );
	
	return 0;
}


template <typename T>
void IntegrationIteratorMatsubaraDisc<T>::initEnergies()
{
	const Parameters<T>* param = ContextModule<T>::getParameters();

	T T_Tc = param->getTemperature();
	int numEnergies = param->getNumMatsubaraFrequencies();

	// Set matsubara energies
	GridCPU* energies_cpu = new GridCPU(numEnergies,1);
	
	T *energies_re = energies_cpu->getDataPointer(0,Type::real);
	T *energies_im = energies_cpu->getDataPointer(0,Type::imag);

	for ( int i=0; i<numEnergies; ++i )
	{
		energies_re[i] = 0.0;
		energies_im[i] = (M_PI*T_Tc) * (2.0*i + 1.0);
	}

	GridGPU* energies = new GridGPU;
	*energies = *energies_cpu;

	IntegrationIterator<T>::setEnergies( energies );	
}



template <typename T>
void IntegrationIteratorMatsubaraDisc<T>::initIterators( int numEnergiesPerBlock )
{
	typedef IntegrationIterator<T> I;

	initEnergies();

	int numEnergies = I::getEnergies()->dimX();

	ParameterIterator<int>* iter = new ParameterIterator<int>(0,numEnergies);
	I::setEnergyIterator( iter );

	std::cout << "Initializing matsubara iterator with Ne = " << numEnergies <<
		", NeBlock = " << numEnergiesPerBlock << "\n";

	if ( numEnergiesPerBlock < 1 )
		iter->setIncrement( numEnergies );
	else
		iter->setIncrement( numEnergiesPerBlock );

	for ( iter->begin(); !iter->isDone(); iter->next() )
	{
		std::cout << "min = " << iter->getMin() << "\n";
		std::cout << "max = " << iter->getMax() << "\n";
		std::cout << "inc = " << iter->getIncrement() << "\n";
		std::cout << "num = " << iter->getNumInterval() << "\n\n";
	}

	iter->begin();
}


template <typename T>
int IntegrationIteratorMatsubaraDisc<T>::getNumEnergiesInterval()
{
	return IntegrationIterator<T>::getEnergies()->dimX();
}


template <typename T>
void IntegrationIteratorMatsubaraDisc<T>::compute()
{
	typedef IntegrationIterator<T> I;

	Context<T>* ctx = ContextModule<T>::getContext();

	// Get some parameters and variables
	const int Ns = 50;
	const int Nx = ctx->getParameters()->getGridResolutionBase();
	const int Na = ctx->getParameters()->getAngularResolution();
	const T h = ctx->getParameters()->getGridElementSize();
	const T W = ctx->getParameters()->getGridWidth();
	const T R2 = (W*0.5)*(W*0.5);

	ParameterIterator<int>* energy_iter = I::getEnergyIterator();
	GridGPU* energies = I::getEnergies();

	const int Ne = energies->dimX();

	OrderParameter<T>*	delta = ctx->getOrderParameter();
	GridGPU* delta_grid = delta->getOrderParameterComponentField(0);

	std::cout << delta_grid->dimX() << "\n";

	ComputeOrderParameterDisc<T>* deltaComp = static_cast<ComputeOrderParameterDisc<T>*>(ctx->getComputeObject(0));
	OrderParameter<T>* delta_new = deltaComp->getOrderParameter();

	OrderParameterComponent<T>* s_wave = delta_new->getOrderParameterComponent(0);
	GridGPU* s_wave_grid = s_wave->getOrderParameterField();

	std::cout << s_wave_grid->dimX() << "\n";

	const int NC = ctx->getNumComputeObjects();

	// Initialize compute objects (usually just make sure mem is allocated and zeroed)
	for ( int i=0; i<NC; ++i )
		ctx->getComputeObject(i)->initializeIteration( energy_iter->getNumInterval() );

	T px, py;
	dim3 bpg_g(1,Na,1);
	dim3 tpb_g(Ne,1,1);
	dim3 tpb(Na,1,1);
	dim3 bpg(1,1,1);

	GridGPU traj_x(Ns,1,Na,Type::real);
	GridGPU traj_y(Ns,1,Na,Type::real);
	GridGPU cos2Theta(Ns,1,Na,Type::real);

	GridGPU ga(Ne,Na,1,Type::complex);
	GridGPU gb(Ne,Na,1,Type::complex);

	GridGPU G(Ne,Na,1,Type::complex);

	GridCPU G_sum(Nx,Nx,1,Type::complex);
	G_sum.setZero();

	T* G_re_ptr = G_sum.getDataPointer(0,Type::real);
	T* G_im_ptr = G_sum.getDataPointer(0,Type::imag);

	T G_re, G_im;
	int idx;

	GridCPU tx,ty;

	int it = I::getNumIterations();

	it = it%(Nx*Nx);

	// Iterate over real space lattice points
	for ( int y=0; y<Nx; ++y )
	{
		std::cout << "Row " << y << "\n";

		for ( int x=0; x<Nx; ++x )
		{
			idx = y*Nx + x;

			px = (x-(Nx-1)*0.5)*h;
			py = (y-(Nx-1)*0.5)*h;

			if ( (px*px + py*py) < R2 )
			{
				// Generate trajectories for fermi circle
				gen_trajectories<<<bpg,tpb>>>(
					traj_x.getDataPointer(), traj_y.getDataPointer(), cos2Theta.getDataPointer(),
					px, py, h, R2, Na, Ns);

 				// Compute gammas along trajectories
 				gamma_trajectories<<<bpg_g,tpb_g>>>(
 					ga.getDataPointer(0,Type::real), ga.getDataPointer(0,Type::imag),
 					gb.getDataPointer(0,Type::real), gb.getDataPointer(0,Type::imag),
 					traj_x.getDataPointer(), traj_y.getDataPointer(), cos2Theta.getDataPointer(),
 					delta_grid->getDataPointer(0,Type::real), delta_grid->getDataPointer(0,Type::imag),
 					energies->getDataPointer(0,Type::imag),
					h, Nx, Na, Ne, Ns  );

 				// Compute greens
 				greens<<<bpg_g,tpb_g>>>(
 					G.getDataPointer(0,Type::real), G.getDataPointer(0,Type::imag),
 					ga.getDataPointer(0,Type::real), ga.getDataPointer(0,Type::imag),
 					gb.getDataPointer(0,Type::real), gb.getDataPointer(0,Type::imag),
 					Na, Ne );

 				// Sum over greens
 				G.sumField(G_re,G_im);

 				if ( x==100 && y==100)
 				{
 					G.writeNumpyArray("G.txt");
 					G.write("G.grd");
 					ga.writeNumpyArray("ga.txt");
 					ga.write("ga.grd");
 					gb.writeNumpyArray("gb.txt");
 					gb.write("gb.grd");
 				}

 				G_re_ptr[idx] = G_re;
 				G_im_ptr[idx] = G_im;
			}
		}
	}

	*s_wave_grid = G_sum;
	*s_wave_grid *= 2.1;

	G_sum.write("G_sum.grd");

	// Compute updated order parameter (components)
	// Add coupling constant and angle integration normalization factor here
	for ( int i=0; i<NC; ++i )
		ctx->getComputeObject(i)->computePropertyFromGreens();

	
	// Print CUDA error, if any
	ContextModule<T>::checkCudaError( "IntegrationIteratorMatsubaraDisc<T>::compute()" );
	
	I::increaseIterationCount();
}
#endif

