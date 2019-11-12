#ifndef _COMPUTEGREENS_H
#define _COMPUTEGREENS_H

//#include "Parameters.h"
//#include "GridData.h"
#include <ComputeProperty.h>
#include <IntegrationIteratorMatsubara.h>


template <typename T>
class ComputeGreens : public ComputeProperty<T>
{
	typedef typename Grid<T>::GPU GridGPU;

public:
	// Constructors and Destructors
							ComputeGreens();
	virtual					~ComputeGreens();

							ComputeGreens( Context<T>* ctx );

	virtual int 					initialize();
	virtual IntegrationIterator<T>* createIntegrationIterator( int numEnergiesPerBlock = -1 );

	OrderParameter<T>*		getOrderParameter() { return _delta; }
	void					setOrderParameter( const OrderParameter<T>* delta_in );
	

	// Virtual functions
	virtual GridGPU*		getResult() { return _greens_all; }


	// This should be called once for every iteration, before any integration takes place.
	virtual void			initializeIteration( int numFields = 1 );
	
	// This should be called once for every angle, before any integration takes place.
	virtual void			initializeGreensAngle();
	
	
	// Computes greens function for current angle. This may be called multiple times
	// as we integrate over many energies. Result is in local frame and will be stored in _greensCurrentAngle.
	virtual void			computeGreensAngle(
								ParameterIterator<int> * iter,
								GridGPU * gamma,
								GridGPU * gammabar,
								GeometryGroup<T> * geom );

	// Add the result from computeGreensAngle() to total. Rotates _greensCurrentAngle to global frame
	// and adds momentum dependence.
	virtual void			integrateGreensAngle( GeometryGroup<T>* geom, const T angle );

	// Compute physical property from greens function.
	virtual void			computePropertyFromGreens();

	// Helper function to compute residual
	T 						computeResidual( OrderParameter<T>* old ) const;

private:

	// Computed order parameter will be stored here at end of IntegrationIterator::compute() function.
	OrderParameter<T>*		_delta;

	GridGPU*				_greens_interval;
	GridGPU*				_greens_all;
};

#endif

