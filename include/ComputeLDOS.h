#ifndef _COMPUTELDOS_H
#define _COMPUTELDOS_H

//#include "Parameters.h"
//#include "GridData.h"
#include <ComputeProperty.h>
#include <IntegrationIteratorKeldysh.h>


template <typename T>
class ComputeLDOS : public ComputeProperty<T>
{	
	typedef typename Grid<T>::GPU GridGPU;
	typedef typename Grid<T>::CPU GridCPU;
public:
	// Constructors and Destructors
							ComputeLDOS();
	virtual					~ComputeLDOS();

							ComputeLDOS( Context<T>* ctx );
	
	virtual GridGPU*		getResult() { return _ldos_all; }
	GridGPU*				getResult_interval() { return _ldos; }
	GridCPU*				getDOS() { return _dos; }

	virtual IntegrationIterator<T>* createIntegrationIterator( int numEnergiesPerBlock = -1 );

	virtual int				initialize();


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
	
	// Set wheter to multiply result with fermi function or not.
	int 					getMultFermiFunction() const { return _multFermiFunction; }
	void					setMultFermiFunction( const int mult ) { _multFermiFunction = mult; }

private:

	// Computed LDOS will be stored here at end of IntegrationIterator::compute() function.
	GridGPU*				_ldos;
	GridGPU*				_ldos_all;

	// Integrated DOS for system, for each energy level
	GridCPU*				_dos;

	int 					_multFermiFunction;
};

#endif
