#ifndef _COMPUTEFREEENERGY2_H
#define _COMPUTEFREEENERGY2_H

//#include "Parameters.h"
//#include "GridData.h"
#include <ComputeProperty.h>
#include <IntegrationIteratorMatsubara.h>


template <typename T>
class ComputeFreeEnergy2 : public ComputeProperty<T>
{
	typedef typename Grid<T>::GPU GridGPU;
public:
	// Constructors and Destructors
							ComputeFreeEnergy2();
	virtual					~ComputeFreeEnergy2();

//							ComputeFreeEnergy2( Parameters<T> & param );
							ComputeFreeEnergy2( Context<T>* ctx );
	
	virtual GridGPU*		getResult() { return _free_energy; }

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
	GridGPU*				_free_energy;

	// Temporary grids.
	GridGPU*				_f;
	GridGPU*				_f_tilde;

};

#endif

