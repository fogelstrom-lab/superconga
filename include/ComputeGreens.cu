#include "ComputeGreens.h"

template <typename T>
ComputeGreens<T>::ComputeGreens()
	: ComputeProperty<T>(), _delta(0), _greens_interval(0), _greens_all(0)
{}


template <typename T>
ComputeGreens<T>::ComputeGreens( Context<T>* ctx )
	: ComputeProperty<T>(ctx), _delta(0), _greens_interval(0), _greens_all(0)
{
	ctx->add(this);
}


template <typename T>
ComputeGreens<T>::~ComputeGreens()
{
	std::cout << "~ComputeGreens<T>\n";
	if ( _delta ) delete _delta;
	if ( _deltaTmp ) delete _deltaTmp;
	if ( _greens_interval ) delete _greens_interval;
	if ( _greens_all ) delete _greens_all;
}


template <typename T>
int ComputeGreens<T>::initialize()
{
	std::cout << "ComputeGreens<T>::initialize()\n";

	typedef ContextModule<T> C;

	// Make a local copy of order parameter, store new order parameter in this.
	setOrderParameter( C::getOrderParameter() );

	const int N0 = C::getParameters()->getGridResolutionBase();
	const int N1 = C::getParameters()->getGridResolutionCoherence();

	const int Nc = _delta->getNumOrderParameterComponents();
	const int Ne = C::getParameters()->getNumFrequencies();
	const int Ne_interval = C::getIntegrationIterator()->getEnergyIterator()->getNumInterval();

	GridGPU::initializeGrid( _deltaTmp, N0, N0 );

	GridGPU::initializeGrid( _greens_interval, N0, N0, Ne_interval, Type::complex );
	GridGPU::initializeGrid( _greens_all, N0, N0, Ne, Type::complex );

	ComputeProperty<T>::initGreensAngle(N1, Ne_interval);

	return 0;
}


template <typename T>
IntegrationIterator<T>* ComputeGreens<T>::createIntegrationIterator( int numEnergiesPerBlock )
{
	return new IntegrationIteratorKeldysh<T>( ContextModule<T>::getContext(), numEnergiesPerBlock );
}


template <typename T>
void ComputeGreens<T>::setOrderParameter( const OrderParameter<T>* delta_in )
{
	if ( delta_in )
	{
		if ( _delta ) delete _delta;
		_delta = new OrderParameter<T>( *delta_in );
	}
}


template <typename T>
void ComputeGreens<T>::initializeIteration( int numFields )
{
	typedef ContextModule<T> C;
	
	if ( ComputeProperty<T>::doCompute() )
	{
		_greens_interval->setZero();
	}
}


template <typename T>
void ComputeGreens<T>::initializeGreensAngle()
{
	if ( ComputeProperty<T>::doCompute() )
		ComputeProperty<T>::initializeGreensAngle();	// Sets to zero
}


template <typename T>
void ComputeGreens<T>::computeGreensAngle(
	ParameterIterator<int> * iter, GridGPU * gamma, GridGPU * gammabar, GeometryGroup<T> * geom )
{
	typedef ComputeProperty<T> C;

	if ( ComputeProperty<T>::doCompute() )
	{
		GreensFunctionStatic<T,Greens_Delta>::computeGreens(
			C::getGreensCurrentAngle(), iter, gamma, gammabar, geom );
	}
}


template <typename T>
void ComputeGreens<T>::integrateGreensAngle( GeometryGroup<T> * geom, const T angle )
{
	typedef ComputeProperty<T> C;

	if ( ComputeProperty<T>::doCompute() )
	{
		GridGPU * greens = C::getGreensCurrentAngle();

		for ( int i=0; i<_delta->getNumOrderParameterComponents(); ++i )
		{
			// Compensate for numerical diffusion from interpolation during rotation 
			geom->extendValuesAtBoundary( greens, 1 );

			// Rotate back to reference frame and add momentum dependence
			_delta->getOrderParameterComponent(i)->computeMomentumDependence( greens, angle );

			// Add angle contribution (angle integration)
			_greens_interval->insertRotatedGridAdditive( greens, angle );
		}
	}
}


template <typename T>
void ComputeGreens<T>::computePropertyFromGreens()
{
	typedef ComputeProperty<T> C;
	
	if ( C::doCompute() )
	{
		Parameters<T>* param = C::getParameters();
		const int Na = param->getAngularResolution();

		const ParameterIterator<int>* iter = C::getIntegrationIterator()->getEnergyIterator();

		const int numEnergiesCurrentBlock = iter->getNumInterval();
		const int energyIdx_offset = iter->getCurrentValue();
		
		// Copy temp LDOS to full grid
		for ( int i=0; i<numEnergiesCurrentBlock; ++i )
			_ldos->copyFieldTo( _ldos_all, i, i+energyIdx_offset );


	}
}


template <typename T>
T ComputeGreens<T>::computeResidual( OrderParameter<T>* old ) const
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





