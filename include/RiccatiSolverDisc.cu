#include "RiccatiSolverDisc.h"
#include "RiccatiSolverDisc_Kernel.cu"

template <typename T, template <typename> class GAMMA, template <typename> class POTENTIAL>
RiccatiSolverDisc<T,GAMMA,POTENTIAL>::RiccatiSolverDisc()
	:	RiccatiSolver<T>()
{}


template <typename T, template <typename> class GAMMA, template <typename> class POTENTIAL>
RiccatiSolverDisc<T,GAMMA,POTENTIAL>::~RiccatiSolverDisc()
{
	std::cout << "~RiccatiSolverDisc()\n";
}


/*
template <typename T, template <typename> class GAMMA, template <typename> class POTENTIAL>
RiccatiSolverDisc<T,GAMMA,POTENTIAL>::RiccatiSolverDisc( Parameters<T> & param, GeometryGroup<T> * geom )
	:	RiccatiSolver<T>( param, geom )
{
	std::cout << "RiccatiSolverDisc::RiccatiSolverDisc(Parameters<T> & param, GeometryGroup<T> * geom)\n";
}
*/
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

/*
template <typename T, template <typename> class GAMMA, template <typename> class POTENTIAL>
void RiccatiSolverDisc<T,GAMMA,POTENTIAL>::computeCoherenceFunctions(
	const ParameterIterator<int>& energy_iter, const GridGPU* energies,
	const GridGPU* deltaRot, const int current_angle_idx )
{
	typedef RiccatiSolver<T> R;
	typedef ContextModule<T> C;

	// Get model parameters
	Parameters<T> * param = C::getParameters();

	// Determine some scalars
	const T h				= param->getGridElementSize();
	const T B				= param->getMagneticFieldStrength();
	const int N				= param->getGridResolutionCoherence();
	const int numAngles		= param->getAngularResolution();
	

	// Get pointers to grids
	GridGPU* gamma		= R::getGamma();
	GridGPU* gammabar	= R::getGammabar();

	GeometryGroup<T>* geom	= C::getGeometry();

	dim3 blocksPerGrid, threadsPerBlock;

	int energyIdx_start = energy_iter.getCurrentValue();
	int numEnergiesPerBlock = energy_iter.getNumInterval();

	R::getKernelDimensions( blocksPerGrid, threadsPerBlock, numEnergiesPerBlock );
	int trajectoryIdx_start = 0;
//	int trajectoryIdx_start = R::getStorage()->getTrajectoryStartIndexAngle( current_angle_idx );
//	R::getKernelDimensionsOptim( blocksPerGrid, threadsPerBlock, numEnergiesPerBlock, current_angle_idx );

	// System and gamma angle
	T system_angle		= ((T)current_angle_idx/(T)numAngles) * (2.0*PI);
	T gamma_angle		= system_angle + 0.5*PI;

	if ( gamma_angle >= 2.0*PI )
		gamma_angle -= 2.0*PI;
	

	// Gives 10-15% speedup with streams
	cudaStream_t s1, s2;
	cudaStreamCreate( &s1 );
	cudaStreamCreate( &s2 );

	// 	Solve gamma by propagating riccati trajectories along positive y
	computeGammaFunctionBulk_kernel<T,typename GAMMA<T>::Gamma,POTENTIAL,2,3,1>
	<<<blocksPerGrid,threadsPerBlock,0,s1>>>(
		// Model parameters and energy values
		h, N, numEnergiesPerBlock, trajectoryIdx_start, current_angle_idx, system_angle, gamma_angle, B,
		energies->getDataPointer( 0, Type::real ) + energyIdx_start,
		energies->getDataPointer( 0, Type::imag ) + energyIdx_start,
		// Grids for gamma and Delta
		gamma->getDataPointer( 0, Type::real ),
		gamma->getDataPointer( 0, Type::imag ),
		deltaRot->getDataPointer( 0, Type::real ),
		deltaRot->getDataPointer( 0, Type::imag ),
		// Grids for domain info/data
		geom->getDomainLabelsGrid()->getDataPointer( 0, Type::real ),
		geom->getNormalsGrid()->getDataPointer( 0, Type::real ),
		geom->getNormalsGrid()->getDataPointer( 1, Type::real ),
		geom->getBoundaryDistanceGrid()->getDataPointer( 0, Type::real )
		);
	
	// 	Solve gammabar by propagating riccati trajectories along negative y
	computeGammaFunctionBulk_kernel<T,typename GAMMA<T>::Gammabar,POTENTIAL,3,2,-1>
	<<<blocksPerGrid,threadsPerBlock,0,s2>>>(
		// Model parameters and energy values
		h, N, numEnergiesPerBlock, trajectoryIdx_start, current_angle_idx, system_angle, gamma_angle, B,
		energies->getDataPointer( 0, Type::real ) + energyIdx_start,
		energies->getDataPointer( 0, Type::imag ) + energyIdx_start,
		// Grids for gammabar and Delta
		gammabar->getDataPointer( 0, Type::real ),
		gammabar->getDataPointer( 0, Type::imag ),
		deltaRot->getDataPointer( 0, Type::real ),
		deltaRot->getDataPointer( 0, Type::imag ),
		// Grids for domain info/data
		geom->getDomainLabelsGrid()->getDataPointer( 0, Type::real ),
		geom->getNormalsGrid()->getDataPointer( 0, Type::real ),
		geom->getNormalsGrid()->getDataPointer( 1, Type::real ),
		geom->getBoundaryDistanceGrid()->getDataPointer( 0, Type::real )
		);


	cudaStreamDestroy(s1);
	cudaStreamDestroy(s2);
}

*/