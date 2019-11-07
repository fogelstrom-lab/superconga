#ifndef _COMPUTEFREEENERGY_H
#define _COMPUTEFREEENERGY_H

//#include "Parameters.h"
//#include "GridData.h"
#include <Initial.h>
#include <ComputeProperty.h>
#include <IntegrationIteratorMatsubara.h>
#include <IntegrationIteratorOzaki.h>


template <typename T>
class ComputeFreeEnergy : public ComputeProperty<T>
{
	typedef typename Grid<T>::GPU GridGPU;
	typedef typename Grid<T>::CPU GridCPU;
public:
	// Constructors and Destructors
							ComputeFreeEnergy();
	virtual					~ComputeFreeEnergy();

							ComputeFreeEnergy( Context<T>* ctx );
	
	virtual GridGPU*		getResult() { return _freeEnergyGrid; }
	T						getFreeEnergy() const { return _freeEnergy; }
	T						getFreeEnergy_unitArea() const { return _freeEnergy_unitArea; }

	// Experimental!!
	T						getFreeEnergyIm() const { return _freeEnergy_im; }
	T						getFreeEnergy_noB() const { return _freeEnergy_noB; }
	T						getBterm() const { return _Bterm; }



	virtual int				initialize();
	virtual IntegrationIterator<T>* createIntegrationIterator( int numEnergiesPerBlock = -1 );
	
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

	static T				freeEnergy_S(
								Parameters<T>* param, GeometryGroup<T>* geom,
								OrderParameter<T>* s_wave );

	static T				freeEnergyT0_S();


private:

	// Total free energy
	T						_freeEnergy;
	T						_freeEnergy_unitArea;
	T						_freeEnergy_im;
	T						_freeEnergy_noB;
	T						_Bterm;
	
	GridGPU*				_freeEnergyGrid;

	// Expression for free energy is divided into two terms for computational
	// simplicity. Only one term needs to be computed in angle integration procedure.
	T						_freeEnergy_angle;
};

#endif

