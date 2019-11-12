#ifndef _COMPUTECURRENTDENSITY_H
#define _COMPUTECURRENTDENSITY_H

//#include "Parameters.h"
//#include "GridData.h"
#include <ComputeProperty.h>
#include <IntegrationIteratorKeldysh.h>

template <typename T>
class ComputeCurrentDensity : public ComputeProperty<T>
{	
	typedef typename Grid<T>::GPU GridGPU;
	typedef typename Grid<T>::CPU GridCPU;
public:
	// Constructors and Destructors
							ComputeCurrentDensity();
	virtual					~ComputeCurrentDensity();

							ComputeCurrentDensity( Context<T>* ctx );
	

	virtual GridGPU*		getResult() { return _lcurrent_all; }
	virtual IntegrationIterator<T>* createIntegrationIterator( int numEnergiesPerBlock = -1 );

	// Context initialize function
	virtual int				initialize();

	// This should be called once for every iteration, before any integration takes place.
	virtual void			initializeIteration( int numFields = 1 );
	
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

	GridGPU*				getResult_interval( int field=0 ) { if (field==0) return _lcurrent_x; else return _lcurrent_y; }

	GridCPU*				getCurrentDensity() const { return _current; }

	int 					getMultFermiFunction() const { return _multFermiFunction; }
	void					setMultFermiFunction( const int mult ) { _multFermiFunction = mult; }


private:

	// Computed local current density will be stored here at end of IntegrationIterator::compute() function.
	GridGPU*			_lcurrent_x;
	GridGPU*			_lcurrent_y;

	// Total current density for system, for each energy level
	GridGPU*			_lcurrent_all;

	// Integrated current, per energy level
	GridCPU*			_current;
	
	// Temporary grid.
	GridGPU*			_tmp;

	// Multiply with fermi function
	int 				_multFermiFunction;
};

#endif
