#include "ComputeOrderParameter.h"

template <typename T>
ComputeOrderParameter<T>::ComputeOrderParameter()
	: ComputeProperty<T>(), _delta(0), _deltaTmp(0), _delta_fields(0)
{}


template <typename T>
ComputeOrderParameter<T>::ComputeOrderParameter( Context<T>* ctx )
	: ComputeProperty<T>(ctx), _delta(0), _deltaTmp(0), _delta_fields(0)
{
	ctx->add(this);
}


template <typename T>
ComputeOrderParameter<T>::~ComputeOrderParameter()
{
	std::cout << "~ComputeOrderParameter<T>\n";
	if ( _delta ) delete _delta;
	if ( _deltaTmp ) delete _deltaTmp;
	if ( _delta_fields ) delete _delta_fields;
}


template <typename T>
int ComputeOrderParameter<T>::initialize()
{
	std::cout << "ComputeOrderParameter<T>::initialize()\n";

	typedef ContextModule<T> C;

	// Make a local copy of order parameter, store new order parameter in this.
	setOrderParameter( C::getOrderParameter() );

	const int N0 = C::getParameters()->getGridResolutionBase();
	const int N1 = C::getParameters()->getGridResolutionCoherence();

	const int Nc = _delta->getNumOrderParameterComponents();

	GridGPU::initializeGrid( _deltaTmp, N0, N0 );
	GridGPU::initializeGrid( _delta_fields, N0, N0, Nc, Type::complex );

	ComputeProperty<T>::initGreensAngle(N1);

	ComputeProperty<T>::setConverged(false);


	GridCPU energies_cpu = *(C::getIntegrationIterator()->getEnergies());
	GridCPU residues_cpu = *(C::getIntegrationIterator()->getResidues());

	// Initialize coupling constant
	for ( int i=0; i<Nc; ++i )
	{
		OrderParameterComponent<T> *component = _delta->getOrderParameterComponent(i);
		component->initCouplingConstant( energies_cpu.getDataPointer( 0, Type::imag ),
										 residues_cpu.getDataPointer( 0, Type::real ) );
	}

	return 0;
}


template <typename T>
IntegrationIterator<T>* ComputeOrderParameter<T>::createIntegrationIterator( int numEnergiesPerBlock )
{
	//return new IntegrationIteratorMatsubara<T>( ContextModule<T>::getContext(), numEnergiesPerBlock );
	return new IntegrationIteratorOzaki<T>( ContextModule<T>::getContext(), numEnergiesPerBlock );
}


template <typename T>
void ComputeOrderParameter<T>::setOrderParameter( const OrderParameter<T>* delta_in )
{
	if ( delta_in )
	{
		if ( _delta ) delete _delta;
		_delta = new OrderParameter<T>( *delta_in );
	}
}


template <typename T>
void ComputeOrderParameter<T>::initializeIteration( int numFields )
{
	typedef ContextModule<T> C;
	
	if ( ComputeProperty<T>::doCompute() )
	{
		// Temporary grids for new order parameter components
		const int numDelta = _delta->getNumOrderParameterComponents();

		for ( int i=0; i<numDelta; ++i )
			_delta->getOrderParameterComponent(i)->getOrderParameterField()->setZero();
	}
}


template <typename T>
void ComputeOrderParameter<T>::initializeGreensAngle()
{
	if ( ComputeProperty<T>::doCompute() )
		ComputeProperty<T>::initializeGreensAngle();	// Sets to zero
}


template <typename T>
void ComputeOrderParameter<T>::computeGreensAngle(
	ParameterIterator<int> * iter, GridGPU * gamma, GridGPU * gammabar, GeometryGroup<T> * geom )
{
	if ( ComputeProperty<T>::doCompute() )
	{
		typedef ComputeProperty<T> C;
		IntegrationIterator<T>* ii = ContextModule<T>::getIntegrationIterator();
		GridGPU* residues = ii->getResidues();

		GreensFunctionStatic<T,Greens_Delta>::computeGreensSumEnergy(
			C::getGreensCurrentAngle(), iter, residues, gamma, gammabar, geom );
	}
}


template <typename T>
void ComputeOrderParameter<T>::integrateGreensAngle( GeometryGroup<T> * geom, const T angle )
{
	typedef ComputeProperty<T> C;

	if ( ComputeProperty<T>::doCompute() )
	{
		GridGPU* greens = C::getGreensCurrentAngle();

		for ( int i=0; i<_delta->getNumOrderParameterComponents(); ++i )
		{
			_deltaTmp->setZero();
			
			// Compensate for numerical diffusion from interpolation during rotation 
			geom->extendValuesAtBoundary( greens, 1 );

			// Rotate back to reference frame and add momentum dependence
			_deltaTmp->insertRotatedGrid( greens, angle );
			_delta->getOrderParameterComponent(i)->computeMomentumDependence( _deltaTmp, angle );

			// Add angle contribution (angle integration)
			*( _delta->getOrderParameterComponent(i)->getOrderParameterField() ) += *_deltaTmp;
		}
	}
}


template <typename T>
void ComputeOrderParameter<T>::computePropertyFromGreens()
{
	typedef ComputeProperty<T> C;
	
	if ( C::doCompute() )
	{
		Parameters<T>* param = C::getParameters();
		const int Na = param->getAngularResolution();
		int Nc = _delta->getNumOrderParameterComponents();
		int numEnergies = param->getNumFrequencies();

		//GridCPU energies_cpu = *(ContextModule<T>::getIntegrationIterator()->getEnergies());
		//GridCPU residues_cpu = *(ContextModule<T>::getIntegrationIterator()->getResidues());
		
		for ( int i=0; i<Nc; ++i )
		{
			// Compute new delta from anomalous greens function
			OrderParameterComponent<T> *component = _delta->getOrderParameterComponent(i);
			
			component->computeCouplingConstant();

			//component->computeCouplingConstant( energies_cpu.getDataPointer( 0, Type::imag ),
			//									residues_cpu.getDataPointer( 0, Type::real ) );

			*(component->getOrderParameterField()) *= 1.0/(T)Na;	
		}

		// Compute residual
		OrderParameter<T>* deltaOld = ContextModule<T>::getOrderParameter();
		T residual = computeResidual( deltaOld );
		C::setResidual( residual );

		if ( residual < param->getConvergenceCriterion() )
			C::setConverged();


		// Update main order parameter
		*deltaOld = *_delta;

		//T a = 1.0;
		/*for( int i=0; i<Nc; ++i )
		{
			GridGPU* newField = _delta->getOrderParameterComponent(i)->getOrderParameterField();
			GridGPU* oldField = deltaOld->getOrderParameterComponent(i)->getOrderParameterField();

			*(oldField) *= (1.0-a);
			*(newField) *= a;
			*(oldField) += *(newField);
		}*/

		// Copy results to grid with one field per order parameter component
		for ( int i=0; i<Nc; ++i )
			_delta->getOrderParameterComponentField(i)->copyFieldTo( _delta_fields, 0, i );
	}
}


template <typename T>
T ComputeOrderParameter<T>::computeResidual( OrderParameter<T>* old ) const
{
	T residual = 0.0;
	const int numDelta = _delta->getNumOrderParameterComponents();

	for ( int i=0; i<numDelta; ++i )
	{
		// Compute residual for each component
		GridGPU* deltaNew = _delta->getOrderParameterComponent(i)->getOrderParameterField();
		GridGPU* deltaOld = old->getOrderParameterComponent(i)->getOrderParameterField();

		residual += GridGPU::differenceMagnitude( *deltaOld, *deltaNew ) / sqrt(numDelta);		
	}
	return residual;	
}





