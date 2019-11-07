#ifndef _COMPUTECURRENT_H
#define _COMPUTECURRENT_H

//#include "Parameters.h"
//#include "GridData.h"
#include <ComputeProperty.h>
#include <IntegrationIteratorMatsubara.h>
#include <IntegrationIteratorOzaki.h>


template <typename T>
class ComputeCurrent : public ComputeProperty<T>
{
	typedef typename Grid<T>::GPU GridGPU;
public:
	// Constructors and Destructors
							ComputeCurrent();
	virtual					~ComputeCurrent();

//							ComputeCurrent( Parameters<T> & param );
							ComputeCurrent( Context<T>* ctx );
	
	virtual GridGPU*		getResult() { return _current_real; }

	virtual int				initialize();
	
	virtual IntegrationIterator<T>* createIntegrationIterator( int numEnergiesPerBlock = -1 );

	// This should be called once for every iteration, before any integration takes place.
	virtual void			initializeIteration( int numFields = 1 );

	// This should be called once for every angle, before any integration takes place.
	virtual void			initializeGreensAngle();
	
	// Computes greens function for current angle. This may be called multiple times
	// as we integrate over many energies. Result is in local frame and will be stored in _greensCurrentAngle.
	virtual void			computeGreensAngle(
								ParameterIterator<int>* iter,
								GridGPU* gamma,
								GridGPU* gammabar,
								GeometryGroup<T>* geom );

	// Add the result from computeGreensAngle() to total. Rotates _greensCurrentAngle to global frame
	// and adds momentum dependence.
	virtual void			integrateGreensAngle( GeometryGroup<T>* geom, const T angle );

	// Compute physical property from greens function.
	virtual void			computePropertyFromGreens();
	
	// Compute residual
//	virtual T				computeResidual( const ComputeProperty<T> * other ) const;
	
private:

	// Computed current will be stored here at end of IntegrationIterator::compute() function.
	GridGPU*				_current;
	GridGPU*				_current_real;

	// Temporary grid.
	GridGPU*				_tmp;

};

#endif

