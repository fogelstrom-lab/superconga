#include "ComputeOrderParameterDisc.h"

template <typename T>
ComputeOrderParameterDisc<T>::ComputeOrderParameterDisc()
	: ComputeProperty<T>(), _delta(0), _deltaTmp(0), _delta_fields(0)//, _greensAtomic(0)
{}


template <typename T>
ComputeOrderParameterDisc<T>::ComputeOrderParameterDisc( Context<T>* ctx )
	: ComputeProperty<T>(ctx), _delta(0), _deltaTmp(0), _delta_fields(0)//, _greensAtomic(0)
{
	ctx->add(this);
}


template <typename T>
ComputeOrderParameterDisc<T>::~ComputeOrderParameterDisc()
{
	std::cout << "~ComputeOrderParameterDisc<T>\n";
	if ( _delta ) delete _delta;
	if ( _deltaTmp ) delete _deltaTmp;
	if ( _delta_fields ) delete _delta_fields;
//	if ( _greensAtomic ) delete _greensAtomic;
}


template <typename T>
int ComputeOrderParameterDisc<T>::initialize()
{
	std::cout << "ComputeOrderParameterDisc<T>::initialize()\n";

	typedef ContextModule<T> C;

	// Make a local copy of order parameter, store new order parameter in this.
	setOrderParameter( C::getOrderParameter() );

	const int N0 = C::getParameters()->getGridResolutionBase();
	const int N1 = C::getParameters()->getGridResolutionCoherence();

	const int Nc = _delta->getNumOrderParameterComponents();

	GridGPU::initializeGrid( _deltaTmp, N0, N0 );
	GridGPU::initializeGrid( _delta_fields, N0, N0, Nc, Type::complex );

//	GridData<float>::initializeGrid( &_greensAtomic, N1, N1 );
	ComputeProperty<T>::initGreensAngle(N1);

	return 0;
}


template <typename T>
IntegrationIterator<T>* ComputeOrderParameterDisc<T>::createIntegrationIterator( int numEnergiesPerBlock )
{
	return new IntegrationIteratorMatsubaraDisc<T>( ContextModule<T>::getContext(), numEnergiesPerBlock );
}


template <typename T>
void ComputeOrderParameterDisc<T>::setOrderParameter( const OrderParameter<T>* delta_in )
{
	if ( delta_in )
	{
		if ( _delta ) delete _delta;
		_delta = new OrderParameter<T>( *delta_in );
	}
}


template <typename T>
void ComputeOrderParameterDisc<T>::initializeIteration( int numFields )
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

/*
template <typename T>
void ComputeOrderParameterDisc<T>::initializeGreensAngle()
{
	if ( ComputeProperty<T>::doCompute() )
		ComputeProperty<T>::initializeGreensAngle();	// Sets to zero
}


template <typename T>
void ComputeOrderParameterDisc<T>::computeGreensAngle(
	ParameterIterator<int> * iter, GridGPU * gamma, GridGPU * gammabar, GeometryGroup<T> * geom )
{
	typedef ComputeProperty<T> C;

	if ( ComputeProperty<T>::doCompute() )
		GreensFunctionStatic<T,GreensAnomalous>::computeGreensSumEnergy(
			C::getGreensCurrentAngle(), iter, gamma, gammabar, geom );

//	GreensFunctionStatic<T,GreensAnomalous>::computeGreensSumEnergyReduction(
//		_greensAtomic, iter, gamma, gammabar, geom );
}


template <typename T>
void ComputeOrderParameterDisc<T>::integrateGreensAngle( GeometryGroup<T> * geom, const T angle )
{
	typedef ComputeProperty<T> C;

	if ( ComputeProperty<T>::doCompute() )
	{
		GridGPU * greens = C::getGreensCurrentAngle();

		// Transfer float griddata to T griddata
	//	dim3 bpg, tpb;
	//	_greensAtomic->getKernelDimensions(bpg,tpb);
	//	
	//	transfer_Kernel<<<bpg,tpb>>>( _greensAtomic->getGridDimX(),
	//							_greensAtomic->getDataPointerGPU(0,Type::real),
	//							_greensAtomic->getDataPointerGPU(0,Type::imag),
	//							greens->getDataPointerGPU(0,Type::real),				
	//							greens->getDataPointerGPU(0,Type::imag) );

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
*/

template <typename T>
void ComputeOrderParameterDisc<T>::computePropertyFromGreens()
{
	typedef ComputeProperty<T> C;
	
	if ( C::doCompute() )
	{
		Parameters<T>* param = C::getParameters();
		const int Na = param->getAngularResolution();
		int Nc = _delta->getNumOrderParameterComponents();
		
		for ( int i=0; i<Nc; ++i )
		{
			// Compute new delta from anomalous greens function
			OrderParameterComponent<T> *component = _delta->getOrderParameterComponent(i);
			
			component->computeCouplingConstant();
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

		// Copy results to grid with one field per order parameter component
		for ( int i=0; i<Nc; ++i )
			_delta->getOrderParameterComponentField(i)->copyFieldTo( _delta_fields, 0, i );
	}
}


template <typename T>
T ComputeOrderParameterDisc<T>::computeResidual( OrderParameter<T>* old ) const
{
	T residual = 0.0;
	const int numDelta = _delta->getNumOrderParameterComponents();

	for ( int i=0; i<numDelta; ++i )
	{
		// Compute residual for each component
		GridGPU* deltaNew = _delta->getOrderParameterComponent(i)->getOrderParameterField();
		GridGPU* deltaOld = old->getOrderParameterComponent(i)->getOrderParameterField();

		//std::cout << "cr: " << deltaNew->dimX() << ", " << deltaOld->dimX();
		//GridGPU::printDimensions( *deltaNew );
		//GridGPU::printDimensions( *deltaOld );

		residual += GridGPU::differenceMagnitude( *deltaOld, *deltaNew ) / sqrt(numDelta);		
	}
	return residual;	
}





