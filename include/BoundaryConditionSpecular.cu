#include <BoundaryConditionSpecular.h>
#include <BoundaryConditionSpecular_kernel.cu>

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
