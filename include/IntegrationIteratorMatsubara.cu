#include "IntegrationIteratorMatsubara.h"

template <typename T>
IntegrationIteratorMatsubara<T>::IntegrationIteratorMatsubara()
	:	IntegrationIterator<T>()
{}

template <typename T>
IntegrationIteratorMatsubara<T>::IntegrationIteratorMatsubara(
	Context<T>* ctx, int numEnergiesPerBlock )
	:	IntegrationIterator<T>( ctx, numEnergiesPerBlock )
{
	std::cout << "IntegrationIteratorMatsubara<T>()\n";
	ctx->set( this );
}


template <typename T>
IntegrationIteratorMatsubara<T>::~IntegrationIteratorMatsubara()
{
	std::cout << "~IntegrationIteratorMatsubara()\n";
}


template <typename T>
int IntegrationIteratorMatsubara<T>::initialize()
{
	std::cout << "IntegrationIteratorMatsubara::initialize()\n";
	IntegrationIterator<T>::initialize();

	initIterators( IntegrationIterator<T>::getNumEnergiesPerBlock() );
	
	return 0;
}


template <typename T>
void IntegrationIteratorMatsubara<T>::initEnergies()
{
	const Parameters<T>* param = ContextModule<T>::getParameters();

	T T_Tc = param->getTemperature();
	//int numEnergies = param->getNumMatsubaraFrequencies();
	int numEnergies = param->getNumFrequencies();

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

	/*GridCPU* ones = new GridCPU(numEnergies, 1, 1, Type::real);
	ones->setValue(1.0, Type::real);
	GridGPU* residues = new GridGPU;
	*residues = *ones;*/
	GridGPU* residues = new GridGPU(numEnergies, 1, 1, Type::real);
	residues->setValue(1.0, Type::real);

	IntegrationIterator<T>::setEnergies( energies );
	IntegrationIterator<T>::setResidues( residues );
}


template <typename T>
void IntegrationIteratorMatsubara<T>::initIterators( int numEnergiesPerBlock )
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
int IntegrationIteratorMatsubara<T>::getNumEnergiesInterval()
{
	return IntegrationIterator<T>::getEnergies()->dimX();
}


template <typename T>
int IntegrationIteratorMatsubara<T>::getEnergyIdxOffset_boundaryStorage()
{
	return IntegrationIterator<T>::getEnergyIterator()->getCurrentValue();
}


template <typename T>
void IntegrationIteratorMatsubara<T>::setBoundaryInitialCondition()
{
	std::cout << "IntegrationIteratorMatsubara<T>::setBoundaryInitialCondition(): Begin...\n";

	typedef IntegrationIterator<T> I;

	Context<T>*	ctx = ContextModule<T>::getContext();

	// Get some parameters and variables
	const int numAngles = ctx->getParameters()->getAngularResolution();
	T angle_rad;

	GridGPU* deltaRotated;		// Pointer to temporary grid for order parameter in local frame

	ParameterIterator<int>* energy_iter = I::getEnergyIterator();
	GridGPU* energies = I::getEnergies();

	RiccatiSolver<T>*	riccati = ctx->getRiccatiSolver();
	GeometryGroup<T>*	geom = ctx->getGeometry();
	OrderParameter<T>*	delta = ctx->getOrderParameter();

	// Do angle loop (integrate over momentum angles)
	//
	for ( int angle_idx=0; angle_idx<numAngles; ++angle_idx )
	{
		// Angle (in radians) of system in local frame
		I::setCurrentAngleIndex( angle_idx );
		angle_rad = I::getCurrentAngleRadians();

		// Rotate order parameter field to current angle
		deltaRotated = delta->getOrderParameterFieldRotated( -angle_rad );

		// Create discrete geometry data for current angle
		geom->create( -angle_rad );

		// Compute coherence functions and store on boundary as initial condition
		for ( energy_iter->begin(); !energy_iter->isDone(); energy_iter->next() )
		{	
			riccati->computeInitialCondition( *energy_iter, energies, deltaRotated, angle_idx );
		}				
	}

	std::cout << "IntegrationIteratorMatsubara<T>::setBoundaryInitialCondition(): Done!\n";
	
	// Print CUDA error, if any
	ContextModule<T>::checkCudaError( "IntegrationIteratorMatsubara<T>::setBoundaryInitialCondition()" );
}




template <typename T>
void IntegrationIteratorMatsubara<T>::compute()
{
	typedef IntegrationIterator<T> I;

	Context<T>* ctx = ContextModule<T>::getContext();

	// Get some parameters and variables
	const int numAngles = ctx->getParameters()->getAngularResolution();
	T angle_rad;

	GridGPU* deltaRotated;				// Pointer to temporary grid for order parameter in local frame
	GridGPU deltaRotatedMultFactor;		// This is different to one when computing free energy according to Sauls et al.

	ParameterIterator<int>* energy_iter = I::getEnergyIterator();
	GridGPU* energies = I::getEnergies();
	GridGPU* residues = I::getResidues();

	RiccatiSolver<T>*	riccati = ctx->getRiccatiSolver();
	GeometryGroup<T>*	geom = ctx->getGeometry();
	OrderParameter<T>*	delta = ctx->getOrderParameter();

	const int NC = ctx->getNumComputeObjects();


	// Set initial condition on boundary, if 
	if ( !I::getNumIterations() )
		setBoundaryInitialCondition();


	// Initialize compute objects (usually just make sure mem is allocated and zeroed)
	for ( int i=0; i<NC; ++i )
		ctx->getComputeObject(i)->initializeIteration( energy_iter->getNumInterval() );


	// Apply boundary condition(s)
	riccati->computeBoundaryConditions();


	// Include magnetic field induced by currents
	riccati->computeInducedMagneticField();


	// Do angle loop (=momentum integral, and sum over energies)
	//
	for ( int angle_idx=0; angle_idx<numAngles; ++angle_idx )
	{
		// Angle (in radians) of system in local frame
		I::setCurrentAngleIndex( angle_idx );
		angle_rad = I::getCurrentAngleRadians();

		// Rotate order parameter field to current angle
		deltaRotated = delta->getOrderParameterFieldRotated( -angle_rad );

		// If we are computing free energy with Luttinger-Ward functional, the order parameter
		// is multiplied by a constant (which is orderParameterMultFactor). In all other cases
		// the constant is 1.
		deltaRotatedMultFactor = (*deltaRotated) * riccati->getOrderParameterMultFactor();

		// Create discrete geometry data for current angle
		geom->create( -angle_rad );


		// Solve riccati equations for current angle and compute appropriate greens function
		for ( int i=0; i<NC; ++i )
			ctx->getComputeObject(i)->initializeGreensAngle();

		// Iterate over (possibly) several blocks of energies.
		for ( energy_iter->begin(); !energy_iter->isDone(); energy_iter->next() )
		{
			riccati->computeCoherenceFunctions( *energy_iter, energies, &deltaRotatedMultFactor, angle_idx );

			for ( int i=0; i<NC; ++i )
				ctx->getComputeObject(i)->computeGreensAngle(
					energy_iter, riccati->getGamma(), riccati->getGammabar(), geom );
			//for ( int i=0; i<NC; ++i )
			//	ctx->getComputeObject(i)->computeGreensAngle(
			//		energy_iter, residues, riccati->getGamma(), riccati->getGammabar(), geom );

		}		
			
		// Add integration contribution of greens function from current angle
		for ( int i=0; i<NC; ++i )
			ctx->getComputeObject(i)->integrateGreensAngle( geom, angle_rad );
	}

	// Compute updated order parameter (components)
	// Add coupling constant and angle integration normalization factor here
	for ( int i=0; i<NC; ++i )
		ctx->getComputeObject(i)->computePropertyFromGreens();
	
	// Print CUDA error, if any
	ContextModule<T>::checkCudaError( "IntegrationIteratorMatsubara<T>::compute()" );
	
	I::increaseIterationCount();
}
