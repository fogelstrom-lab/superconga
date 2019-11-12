#include "RiccatiSolverConfined.h"
#include "RiccatiSolverConfined_Kernel.cu"


template <typename T, template <typename> class POTENTIAL>
void loc_computeEnergyShift( T* data, const int current_angle_idx)
{	
	dim3 bpg, tpb;
	computeEnergyShift_external_kernel<T,POTENTIAL><<<bpg,tpb>>>( data, 1.0, 1.0, 100, 0.1 );
}


template <typename T, template <typename> class GAMMA, template <typename> class POTENTIAL>
RiccatiSolverConfined<T,GAMMA,POTENTIAL>::RiccatiSolverConfined()
	:	RiccatiSolver<T>()
{}


template <typename T, template <typename> class GAMMA, template <typename> class POTENTIAL>
RiccatiSolverConfined<T,GAMMA,POTENTIAL>::~RiccatiSolverConfined()
{}


template <typename T, template <typename> class GAMMA, template <typename> class POTENTIAL>
RiccatiSolverConfined<T,GAMMA,POTENTIAL>::RiccatiSolverConfined( Context<T>* ctx )
	:	RiccatiSolver<T>( ctx )
{
	std::cout << "RiccatiSolverConfined::RiccatiSolverConfined( Context<T>* )\n";
	ctx->set( this );
}


template <typename T, template <typename> class GAMMA, template <typename> class POTENTIAL>
int RiccatiSolverConfined<T,GAMMA,POTENTIAL>::initialize()
{
	RiccatiSolver<T>::initialize();
	RiccatiSolver<T>::initializeBoundaryStorage();

	return 0;
}


template <typename T, template <typename> class GAMMA, template <typename> class POTENTIAL>
void RiccatiSolverConfined<T,GAMMA,POTENTIAL>::computeCoherenceFunctions(
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

	GeometryGroup<T>* geom	= R::getGeometry();
	BoundaryStorage<T>* storage = R::getBoundaryStorage();

	dim3 blocksPerGrid, threadsPerBlock;

	int numEnergiesCurrentBlock = energy_iter.getNumInterval();
	int energyIdxOffset = energy_iter.getCurrentValue();

	int numEnergies_storage = C::getIntegrationIterator()->getNumEnergiesInterval();
	int energyIdxOffset_storage = C::getIntegrationIterator()->getEnergyIdxOffset_boundaryStorage();


	R::getKernelDimensionsOptim( blocksPerGrid, threadsPerBlock, numEnergiesCurrentBlock, current_angle_idx );

	int idx_x_min = (storage->getTrajectoryRangeX()->getDataPointer(0))[current_angle_idx];
	int idx_x_max = (storage->getTrajectoryRangeX()->getDataPointer(1))[current_angle_idx];

	int idx_y_min = (storage->getTrajectoryRangeY()->getDataPointer(0))[current_angle_idx];
	int idx_y_max = (storage->getTrajectoryRangeY()->getDataPointer(1))[current_angle_idx];

	// System and gamma angle
	T system_angle = ((T)current_angle_idx/(T)numAngles) * (2.0*M_PI);
	T gamma_angle = system_angle + 0.5*M_PI;

	if ( gamma_angle >= 2.0*M_PI )
		gamma_angle -= 2.0*M_PI;

	// Gives 10-15% speedup with streams
	cudaStream_t s1, s2;
	cudaStreamCreate( &s1 );
	cudaStreamCreate( &s2 );

	//Make riccati kernel read vector potential via precomputed field.
	//To allow for efficient computation with no vector potential, it is accessed
	//via a templated function object. 

	// Compute energy shift from magnetic field
	GridGPU* eShift = R::getEnergyShift();
	GridGPU* ps = R::getInducedVectorPotential();

	int doInducedBfield = R::doComputeInducedMagneticField();

	computeEnergyShift<T,POTENTIAL>(this,ps,eShift,doInducedBfield,current_angle_idx);

	// 	Solve gamma by propagating riccati trajectories along positive y
	computeGammaFunctionTemplate_new_kernel<T,typename GAMMA<T>::Gamma,2,3,1>
	<<<blocksPerGrid,threadsPerBlock,0,s1>>>(
		// Model parameters and energy values
		h, N, numEnergies_storage, energyIdxOffset_storage,
		idx_x_min, idx_x_max, idx_y_min, idx_y_max,
		current_angle_idx, system_angle, gamma_angle, B,
		energies->getDataPointer( 0, Type::real ) + energyIdxOffset,
		energies->getDataPointer( 0, Type::imag ) + energyIdxOffset,
		eShift->getDataPointer(),
		// Grids for gamma and Delta
		gamma->getDataPointer( 0, Type::real ),
		gamma->getDataPointer( 0, Type::imag ),
		deltaRot->getDataPointer( 0, Type::real ),
		deltaRot->getDataPointer( 0, Type::imag ),
		// Grids for domain info/data
		geom->getDomainLabelsGrid()->getDataPointer( 0, Type::real ),
		geom->getNormalsGrid()->getDataPointer( 0, Type::real ),
		geom->getNormalsGrid()->getDataPointer( 1, Type::real ),
		geom->getBoundaryDistanceGrid()->getDataPointer( 0, Type::real ),
		// Data relating to sparse storage for boundary gammas
		storage->getIndexSparseIn_gamma()->getDataPointer( 0, Type::real ),
		storage->getIndexSparseOut_gamma()->getDataPointer( 0, Type::real ),
		storage->getBoundaryIn_gamma()->getDataPointer( 0, Type::real ),
		storage->getBoundaryIn_gamma()->getDataPointer( 0, Type::imag ),
		storage->getBoundaryOut_gamma()->getDataPointer( 0, Type::real ),
		storage->getBoundaryOut_gamma()->getDataPointer( 0, Type::imag )
		);

	// 	Solve gammabar by propagating riccati trajectories along negative y
	computeGammaFunctionTemplate_new_kernel<T,typename GAMMA<T>::Gammabar,3,2,-1>
	<<<blocksPerGrid,threadsPerBlock,0,s2>>>(
		// Model parameters and energy values
		h, N, numEnergies_storage, energyIdxOffset_storage,
		idx_x_min, idx_x_max, idx_y_min, idx_y_max,
		current_angle_idx, system_angle, gamma_angle, B,
		energies->getDataPointer( 0, Type::real ) + energyIdxOffset,
		energies->getDataPointer( 0, Type::imag ) + energyIdxOffset,
		eShift->getDataPointer(),
		// Grids for gamma and Delta
		gammabar->getDataPointer( 0, Type::real ),
		gammabar->getDataPointer( 0, Type::imag ),
		deltaRot->getDataPointer( 0, Type::real ),
		deltaRot->getDataPointer( 0, Type::imag ),
		// Grids for domain info/data
		geom->getDomainLabelsGrid()->getDataPointer( 0, Type::real ),
		geom->getNormalsGrid()->getDataPointer( 0, Type::real ),
		geom->getNormalsGrid()->getDataPointer( 1, Type::real ),
		geom->getBoundaryDistanceGrid()->getDataPointer( 0, Type::real ),
		// Data relating to sparse storage for boundary gammas
		storage->getIndexSparseIn_gammabar()->getDataPointer( 0, Type::real ),
		storage->getIndexSparseOut_gammabar()->getDataPointer( 0, Type::real ),
		storage->getBoundaryIn_gammabar()->getDataPointer( 0, Type::real ),
		storage->getBoundaryIn_gammabar()->getDataPointer( 0, Type::imag ),
		storage->getBoundaryOut_gammabar()->getDataPointer( 0, Type::real ),
		storage->getBoundaryOut_gammabar()->getDataPointer( 0, Type::imag )
		);

	cudaStreamDestroy(s1);
	cudaStreamDestroy(s2);
}


template <typename T, template <typename> class GAMMA, template <typename> class POTENTIAL>
void RiccatiSolverConfined<T,GAMMA,POTENTIAL>::gammaSymmetryTransform()
{
	typedef RiccatiSolver<T> R;
	
	GridGPU* gamma		= R::getGamma();
	GridGPU* gammabar	= R::getGammabar();
	
	// Do this ONLY if testing negative vF
	GridGPU gammaTmp( gamma );
	gammaTmp = *gamma;

	*gamma = *gammabar;
	gamma->conjugate();

	*gammabar = gammaTmp;
	gammabar->conjugate();	
}


template <typename T, template <typename> class GAMMA, template <typename> class POTENTIAL>
void RiccatiSolverConfined<T,GAMMA,POTENTIAL>::computeInitialCondition(
	const ParameterIterator<int>& energy_iter, const GridGPU* energies,
	const GridGPU* deltaRot, const int current_angle_idx )
{
	typedef RiccatiSolver<T> R;
	typedef ContextModule<T> C;

	// Get model parameters
	Parameters<T>* param = C::getParameters();

	// Determine some scalars
	const T h				= param->getGridElementSize();
	const T B				= param->getMagneticFieldStrength();
	const int N				= param->getGridResolutionCoherence();
	const int numAngles		= param->getAngularResolution();
	
	// Get pointers to grids
	GridGPU* gamma		= R::getGamma();
	GridGPU* gammabar	= R::getGammabar();

	GeometryGroup<T>* geom	= R::getGeometry();
	BoundaryStorage<T>* storage = R::getBoundaryStorage();

	dim3 blocksPerGrid, threadsPerBlock;

	int numEnergiesCurrentBlock = energy_iter.getNumInterval();
	int energyIdxOffset = energy_iter.getCurrentValue();

	int numEnergies_storage = C::getIntegrationIterator()->getNumEnergiesInterval();
	int energyIdxOffset_storage = C::getIntegrationIterator()->getEnergyIdxOffset_boundaryStorage();
	
	// Get kernel dimensions
	int trajectoryIdx_start = storage->getTrajectoryStartIndexAngle( current_angle_idx );
	R::getKernelDimensionsOptim( blocksPerGrid, threadsPerBlock, numEnergiesCurrentBlock, current_angle_idx );


	// System and gamma angle
	T system_angle = ((T)current_angle_idx/(T)numAngles) * (2.0*M_PI);
	T gamma_angle = system_angle + 0.5*M_PI;

	if ( gamma_angle >= 2.0*M_PI )
		gamma_angle -= 2.0*M_PI;

	// Get energy shift from magnetic field
	GridGPU* eShift = R::getEnergyShift();


	// 	Solve gamma by propagating riccati trajectories along positive y
	computeGammaFunctionInitialBoundaryTemplate_kernel<T,typename GAMMA<T>::Gamma,2,3,1>
	<<<blocksPerGrid,threadsPerBlock>>>(
		// Model parameters and energy values
		h, N, numEnergies_storage, energyIdxOffset_storage, trajectoryIdx_start, current_angle_idx,
		system_angle, gamma_angle, B,
		energies->getDataPointer( 0, Type::real ) + energyIdxOffset,
		energies->getDataPointer( 0, Type::imag ) + energyIdxOffset,
		eShift->getDataPointer(),
		// Grids for gamma and Delta
		gamma->getDataPointer( 0, Type::real ),
		gamma->getDataPointer( 0, Type::imag ),
		deltaRot->getDataPointer( 0, Type::real ),
		deltaRot->getDataPointer( 0, Type::imag ),
		// Grids for domain info/data
		geom->getDomainLabelsGrid()->getDataPointer( 0, Type::real ),
		geom->getNormalsGrid()->getDataPointer( 0, Type::real ),
		geom->getNormalsGrid()->getDataPointer( 1, Type::real ),
		geom->getBoundaryDistanceGrid()->getDataPointer( 0, Type::real ),
		// Data relating to sparse storage for boundary gammas
		storage->getIndexSparseIn_gamma()->getDataPointer( 0, Type::real ),
		storage->getIndexSparseOut_gamma()->getDataPointer( 0, Type::real ),
		storage->getBoundaryIn_gamma()->getDataPointer( 0, Type::real ),
		storage->getBoundaryIn_gamma()->getDataPointer( 0, Type::imag ),
		storage->getBoundaryOut_gamma()->getDataPointer( 0, Type::real ),
		storage->getBoundaryOut_gamma()->getDataPointer( 0, Type::imag )
		);
		

	// 	Solve gammabar by propagating riccati trajectories along negative y
	computeGammaFunctionInitialBoundaryTemplate_kernel<T,typename GAMMA<T>::Gammabar,3,2,-1>
	<<<blocksPerGrid,threadsPerBlock>>>(
		// Model parameters and energy values
		h, N, numEnergies_storage, energyIdxOffset_storage, trajectoryIdx_start, current_angle_idx,
		system_angle, gamma_angle, B,
		energies->getDataPointer( 0, Type::real ) + energyIdxOffset,
		energies->getDataPointer( 0, Type::imag ) + energyIdxOffset,
		eShift->getDataPointer(),
		// Grids for gammabar and Delta
		gammabar->getDataPointer( 0, Type::real ),
		gammabar->getDataPointer( 0, Type::imag ),
		deltaRot->getDataPointer( 0, Type::real ),
		deltaRot->getDataPointer( 0, Type::imag ),
		// Grids for domain info/data
		geom->getDomainLabelsGrid()->getDataPointer( 0, Type::real ),
		geom->getNormalsGrid()->getDataPointer( 0, Type::real ),
		geom->getNormalsGrid()->getDataPointer( 1, Type::real ),
		geom->getBoundaryDistanceGrid()->getDataPointer( 0, Type::real ),
		// Data relating to sparse storage for boundary gammas
		storage->getIndexSparseIn_gammabar()->getDataPointer( 0, Type::real ),
		storage->getIndexSparseOut_gammabar()->getDataPointer( 0, Type::real ),
		storage->getBoundaryIn_gammabar()->getDataPointer( 0, Type::real ),
		storage->getBoundaryIn_gammabar()->getDataPointer( 0, Type::imag ),
		storage->getBoundaryOut_gammabar()->getDataPointer( 0, Type::real ),
		storage->getBoundaryOut_gammabar()->getDataPointer( 0, Type::imag )
		);

}




